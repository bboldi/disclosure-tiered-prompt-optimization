"""Resumable public-data model-selection pilot; never opens confirmatory partitions."""

from __future__ import annotations

import argparse
import math
import shutil
import statistics
import uuid
from pathlib import Path
from typing import Any

from ..domain import metrics
from ..runner import RunPaused, utc_now
from ..storage import IntegrityError, Store, digest
from .adapters import normalize
from .campaign import Campaign, prepare, profiles_from, verify_runtime
from .conditions import NAIVE, TASK, feedback_request, optimizer_request, selected_profiles
from .execution import Execution, StageExhausted, ledger
from .progress import Progress
from .scheduling import allocate, main_forecast, pareto_and_promote, utility
from .telemetry import Telemetry


def measurements(store: Store, prefix: str) -> dict[str, Any]:
    latencies, costs = [], []
    input_tokens, output_tokens = [], []
    for name in store.names("work/*/spec.json"):
        spec = store.get(name)
        if spec["key"] != prefix and not spec["key"].startswith(prefix + "/"):
            continue
        folder = name.rsplit("/", 1)[0]
        elapsed = 0.0
        for request_name in store.names(folder + "/attempts/*/request.json"):
            request = store.get(request_name)
            response_name = request_name.replace("request.json", "response.json")
            if store.exists(response_name):
                response = store.get(response_name)
                elapsed += response["duration_ns"] / 1e9
                try:
                    reply = normalize(response, spec["condition"]["provider"])
                except (ValueError, KeyError, TypeError):
                    continue
                if reply["reported_api_cost_usd"] is not None:
                    costs.append(float(reply["reported_api_cost_usd"]))
                input_tokens.append(reply["input_tokens"])
                output_tokens.append(reply["output_tokens"])
            else:
                elapsed += request["timeout_seconds"]
        if store.exists(folder + "/result.json"):
            latencies.append(elapsed + 0.5)  # conservative worker/journal overhead allowance
    ordered = sorted(latencies)
    return {
        "latency_p90_seconds": ordered[math.ceil(len(ordered) * 0.9) - 1] if ordered else None,
        "latency_mean_seconds": statistics.mean(ordered) if ordered else None,
        "observed_calls": len(ordered),
        "reported_cost_usd": sum(costs) if costs else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


class Pilot:
    def __init__(self, execution: Execution):
        self.execution, self.store = execution, execution.store
        self.plan = self.store.get("plan.json")
        self.campaign = Campaign(execution)
        self.profiles = profiles_from(self.store)
        self.conditions = self.store.get("inputs/conditions.json")
        self.local = {c["id"]: c for c in self.conditions["local"]}
        self.hosted = {c["id"]: c for c in self.conditions["hosted"]}
        self.prompts = self.store.get("inputs/prompts.json")

    def progress_schedule(
        self, milestone: str, future: dict[str, int], *, final: bool = False
    ) -> None:
        progress = self.execution.progress
        if progress is None:
            return
        counts = {key: 0 for key in progress.sizes}
        for name in self.store.names("work/*/result.json"):
            counts[self.store.get(name)["key"].split("/")[0]] += 1
        sizes = {key: future.get(key, count) for key, count in counts.items()}
        sizes["report"] = 1
        progress.sizes, progress.estimated = sizes, not final
        self.store.put(
            f"schedules/{milestone}.json",
            {
                "sizes": sizes,
                "estimated": not final,
                "note": "Future work estimates are revised after admission/allocation; censored arms are recorded separately. Final 100% means the resolved pilot has ended, not every originally proposed arm ran.",
            },
        )

    def summary(
        self, prefix: str, condition: dict[str, Any], rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "condition_id": condition["id"],
            "family": condition.get("family"),
            "metrics": metrics(rows),
            **measurements(self.store, prefix),
        }

    def stage(self, name: str, seconds: float) -> None:
        self.store.put(
            f"stages/{name}/started.json",
            self.store.get(f"stages/{name}/started.json")
            if self.store.exists(f"stages/{name}/started.json")
            else {
                "accounted_seconds": self.plan["max_seconds"] - self.execution.remaining_seconds(),
                "budget_seconds": seconds,
            },
        )
        print(
            f"Starting/resuming {name}; cumulative budget remaining {self.execution.remaining_seconds() / 3600:.2f}h",
            flush=True,
        )
        self.execution.stage_time_left = (
            (lambda: self.stage_remaining(name)) if name in ("P0", "P1") else None
        )

    def stage_remaining(self, name: str) -> float:
        start = self.store.get(f"stages/{name}/started.json")
        used = (
            self.plan["max_seconds"]
            - self.execution.remaining_seconds()
            - start["accounted_seconds"]
        )
        return float(start["budget_seconds"] - used)

    def propose(self, key: str, hosted: dict[str, Any], permitted: dict[str, Any]) -> str | None:
        # Canonical slug is provenance; use the catalog request ID and recheck resolution.
        self.campaign.checked.discard(hosted["id"])
        self.campaign.check(hosted)
        condition, body = optimizer_request(hosted, permitted)
        self.store.put(
            "disclosures/" + digest(key) + ".json",
            {"key": key, "request": body, "sha256": digest(body), "tier": permitted.get("tier", 2)},
        )
        result = self.execution.call(key, condition, body, role="optimizer")
        if result["status"] == "valid":
            self.campaign.reconcile_generation(result["attempt"], hosted["canonical_model"])
        return str(result["value"]) if result["status"] == "valid" else None

    def admission(self) -> dict[str, Any]:
        name = "stages/P0/report.json"
        if self.store.exists(name):
            return dict(self.store.get(name))
        self.stage("P0", 2700)
        profiles = selected_profiles(self.profiles, "pilot_development", 6)
        summaries, excluded, hosted = [], [], []
        for condition in self.local.values():
            prefix = "P0/" + condition["id"]
            cached = "stages/P0/conditions/" + digest(condition["id"]) + ".json"
            if self.store.exists(cached):
                summaries.append(self.store.get(cached))
                continue
            if self.stage_remaining("P0") < 6 * 20:
                excluded.append(
                    {
                        "condition_id": condition["id"],
                        "reason": "P0 time allocation exhausted; not evaluated",
                    }
                )
                continue
            try:
                rows = self.campaign.evaluate(prefix, condition, NAIVE, profiles)
            except StageExhausted:
                excluded.append(
                    {
                        "condition_id": condition["id"],
                        "reason": "P0 time exhausted during condition; partial raw evidence retained",
                    }
                )
                continue
            summary = self.summary(prefix, condition, rows)
            self.store.put(cached, summary)
            summaries.append(summary)
            if summary["metrics"]["coverage"] < 5 / 6:
                excluded.append(
                    {
                        "condition_id": condition["id"],
                        "reason": "fewer than 5/6 schema-valid final answers under explicit cap",
                    }
                )
        for condition in self.hosted.values():
            try:
                prompt = self.propose(
                    "P0/hosted/" + condition["id"],
                    condition,
                    {"task": TASK, "current_prompt": NAIVE, "feedback": None, "tier": 2},
                )
            except StageExhausted:
                excluded.append(
                    {
                        "condition_id": condition["id"],
                        "reason": "P0 hosted admission censored by stage clock",
                    }
                )
                continue
            if prompt:
                hosted.append(condition["id"])
            else:
                excluded.append(
                    {"condition_id": condition["id"], "reason": "bounded protocol probe failed"}
                )
        result = {
            "local_summaries": summaries,
            "excluded": excluded,
            "admitted_local": [
                s["condition_id"] for s in summaries if s["metrics"]["coverage"] >= 5 / 6
            ],
            "admitted_hosted": hosted,
        }
        self.store.put(name, result)
        self.execution.stage_time_left = None
        return result

    def screen(self, admission: dict[str, Any]) -> dict[str, Any]:
        name = "stages/P1/report.json"
        if self.store.exists(name):
            return dict(self.store.get(name))
        self.stage("P1", 4500)
        profiles = selected_profiles(self.profiles, "pilot_development", 24)
        summaries, censored = [], []
        for identifier in admission["admitted_local"]:
            condition = self.local[identifier]
            prefix = "P1/" + identifier
            cached = "stages/P1/conditions/" + digest(identifier) + ".json"
            if self.store.exists(cached):
                summaries.append(self.store.get(cached))
                continue
            if self.stage_remaining("P1") < 120:
                censored.append(
                    {"condition_id": identifier, "reason": "baseline stage allocation exhausted"}
                )
                continue
            rows = []
            baselines = {}
            for label, prompt in self.prompts.items():
                try:
                    evaluated = self.campaign.evaluate(
                        prefix + "/" + label, condition, prompt, profiles
                    )
                except StageExhausted:
                    break
                rows.extend(evaluated)
                baselines[label] = metrics(evaluated)
            if len(baselines) != len(self.prompts):
                censored.append(
                    {
                        "condition_id": identifier,
                        "reason": "P1 time allocation expired during baseline; partial evidence retained",
                    }
                )
                continue
            summary = {
                **self.summary(prefix, condition, rows),
                "baselines": baselines,
                "selection_note": "Pooled across equal-size naive/historical baselines; these are repeated profiles, not independent samples.",
            }
            self.store.put(cached, summary)
            summaries.append(summary)
        result = {
            "summaries": summaries,
            "censored": censored,
            "promotion": pareto_and_promote(summaries),
            "predeclared_family_reference": "granite4.2",
        }
        self.store.put(name, result)
        self.execution.stage_time_left = None
        return result

    def trajectory(
        self,
        stage: str,
        executor: dict[str, Any],
        hosted: dict[str, Any],
        seed: int,
        design: dict[str, Any],
        *,
        fresh: bool = False,
    ) -> dict[str, Any]:
        prefix = f"{stage}/{executor['id']}/{hosted['id']}/{seed}"
        report_name = "trajectories/" + digest(prefix) + ".json"
        if self.store.exists(report_name):
            return dict(self.store.get(report_name))
        offset = 48 if fresh else (0 if seed == 11 else 6)
        batch = selected_profiles(
            self.profiles, "pilot_development", design["batch_size"], offset=offset
        )
        incumbent = NAIVE
        initial = self.campaign.evaluate(prefix + "/baseline", executor, incumbent, batch)
        trace = [{"round": -1, "metrics": metrics(initial), "prompt_sha256": digest(incumbent)}]
        for iteration in range(design["iterations"]):
            batch = selected_profiles(
                self.profiles,
                "pilot_development",
                design["batch_size"],
                offset=offset + iteration * design["batch_size"],
            )
            rows = self.campaign.evaluate(
                f"{prefix}/r{iteration}/incumbent", executor, incumbent, batch
            )
            candidates = [(incumbent, rows, -1)]
            for candidate in range(2):
                key = f"{prefix}/r{iteration}/proposal{candidate}"
                proposed = self.propose(
                    key,
                    hosted,
                    feedback_request(
                        incumbent, rows, batch, tier=2, seed=seed, candidate=candidate
                    ),
                )
                if proposed is None:
                    self.store.put(
                        "candidate-failures/" + digest(key) + ".json",
                        {
                            "key": key,
                            "reason": "no valid prompt after bounded protocol attempts; incumbent retained",
                        },
                    )
                    continue
                evaluated = self.campaign.evaluate(
                    f"{prefix}/r{iteration}/candidate{candidate}", executor, proposed, batch
                )
                candidates.append((proposed, evaluated, candidate))
            # Incumbent first wins exact ties; all challengers share its feedback batch.
            incumbent, best, slot = max(
                candidates,
                key=lambda item: metrics(item[1])["failure_aware_lower_bound"]["micro_f1"],
            )
            decision = {
                "round": iteration,
                "slot": slot,
                "prompt_sha256": digest(incumbent),
                "metrics": metrics(best),
                "compared_prompt_hashes": [digest(c[0]) for c in candidates],
            }
            self.store.put("decisions/" + digest(f"{prefix}/{iteration}") + ".json", decision)
            trace.append(decision)
        validation = selected_profiles(
            self.profiles, "pilot_validation", design["validation_size"], offset=24 if fresh else 0
        )
        shortlist = []
        for index, prompt in enumerate((NAIVE, incumbent)):
            rows = self.campaign.evaluate(
                f"{prefix}/selection{index}", executor, prompt, validation
            )
            shortlist.append({"index": index, "prompt": prompt, "metrics": metrics(rows)})
        selected = max(shortlist, key=utility)
        result = {
            "prefix": prefix,
            "executor": executor["id"],
            "optimizer": hosted["id"],
            "seed": seed,
            "design": {**design, "seeds": [seed]},
            "trace": trace,
            "shortlist": shortlist,
            "selected": selected,
            "fresh_confirmation": fresh,
            **measurements(self.store, prefix),
        }
        self.store.put(report_name, result)
        return result

    def execute(self) -> dict[str, Any]:
        admission = self.admission()
        self.progress_schedule(
            "after-P0",
            {"P1": 48 * len(admission["admitted_local"]), "P2": 1392, "P3": 50, "P4": 100},
        )
        screen = self.screen(admission)
        promoted = screen["promotion"]["selected"]
        optimizers = [
            m
            for m in ("deepseek/deepseek-v4-pro", "z-ai/glm-5.3")
            if m in admission["admitted_hosted"]
        ]
        if len(promoted) != 2 or len(optimizers) != 2:
            return {"status": "inconclusive_admission", "admission": admission, "screen": screen}
        local_times = [
            next(
                s["latency_p90_seconds"]
                for s in screen["summaries"]
                if s["condition_id"] == identifier
            )
            for identifier in promoted
        ]
        host_measurements = [measurements(self.store, "P0/hosted/" + m) for m in optimizers]
        host_time = max(m["latency_p90_seconds"] for m in host_measurements)
        if any(m["reported_cost_usd"] is None for m in host_measurements):
            return {
                "status": "inconclusive_unpriced_hosted_admission",
                "hosted_measurements": host_measurements,
                "costs": self.execution.accounts(),
            }
        mean_cost = sum(m["reported_cost_usd"] for m in host_measurements) / len(host_measurements)
        account = self.execution.accounts()
        allocation_name = "stages/P2/allocation.json"
        if not self.store.exists(allocation_name):
            allocation = allocate(
                self.execution.remaining_seconds(),
                local_times,
                host_time,
                available_usd=float(self.execution.max_cost)
                - float(account["reported_cost_usd"])
                - float(account["reserved_unknown_usd"])
                - 0.5,
                mean_hosted_cost=mean_cost,
            )
            self.store.put(allocation_name, allocation)
        allocation = self.store.get(allocation_name)
        if allocation["status"] != "admitted":
            return {"status": "inconclusive_schedule", "screen": screen, "allocation": allocation}
        design = allocation["selected"]
        self.progress_schedule(
            "after-P1",
            {
                "P2": design.get(
                    "executor_calls",
                    8
                    * (
                        (1 + 3 * design["iterations"]) * design["batch_size"]
                        + 2 * design["validation_size"]
                    ),
                )
                + 16 * design["iterations"],
                "P3": 50,
                "P4": 100,
            },
        )
        self.stage("P2", allocation["selected"]["conservative_seconds"])
        trajectories = [
            self.trajectory("P2", self.local[e], self.hosted[o], seed, design)
            for e in promoted
            for o in optimizers
            for seed in design["seeds"]
        ]
        cells = []
        for executor in promoted:
            for optimizer in optimizers:
                matching = [
                    r
                    for r in trajectories
                    if r["executor"] == executor and r["optimizer"] == optimizer
                ]
                cells.append(
                    {
                        "executor": executor,
                        "optimizer": optimizer,
                        "mean_selected_f1": statistics.mean(
                            utility(r["selected"]) for r in matching
                        ),
                        "seed_values": [utility(r["selected"]) for r in matching],
                    }
                )
        ranked = sorted(
            cells, key=lambda r: (-r["mean_selected_f1"], r["executor"], r["optimizer"])
        )
        self.store.put(
            "stages/P2/report.json",
            {
                "cells": cells,
                "ranked": ranked,
                "interpretation": "Exploratory two-seed pilot rankings; no significance or full-study winner claim.",
            },
        )
        self.progress_schedule("after-P2", {"P3": 50, "P4": 100})
        references: list[dict[str, Any]] = []
        reference_design = {
            "iterations": 1,
            "batch_size": 6,
            "validation_size": 12,
            "seeds": [43],
            "candidate_slots": 2,
        }
        opus = "anthropic/claude-opus-5"
        if opus in admission["admitted_hosted"] and self.execution.remaining_seconds() > 1500:
            references.append(
                self.trajectory(
                    "P3", self.local[promoted[0]], self.hosted[opus], 43, reference_design
                )
            )
        harness = {
            "pi_executable": shutil.which("pi"),
            "opencode_executable": shutil.which("opencode"),
            "isolation_executable": shutil.which("bwrap"),
            "status": "not_admitted",
            "reason": "A matched-access Pi tool sandbox and OpenCode recovery adapter are not implemented. No agent gets the evaluator, test data or study key. Direct/harness comparison remains unexecuted; no feasibility success claimed.",
        }
        self.store.put("stages/P3/harness-admission.json", harness)
        confirmations = []
        # Fixed first eligible cell other than the leader; not selected by its apparent loss.
        control = next(
            c
            for c in cells
            if (c["executor"], c["optimizer"]) != (ranked[0]["executor"], ranked[0]["optimizer"])
        )
        needed = 1.25 * (2 * 48 * max(local_times) + 4 * host_time) + 120
        if self.execution.remaining_seconds() > needed:
            for cell in (ranked[0], control):
                confirmations.append(
                    self.trajectory(
                        "P4",
                        self.local[cell["executor"]],
                        self.hosted[cell["optimizer"]],
                        47,
                        reference_design,
                        fresh=True,
                    )
                )
        else:
            self.store.put(
                "stages/P4/censored.json",
                {
                    "reason": "paired fresh confirmation does not fit; neither arm dispatched",
                    "conservative_seconds": needed,
                },
            )
        study = ledger(self.store.root.parent, study=True)
        self.execution.refresh_key()
        available = (
            60
            - max(float(self.execution.key_usage), float(study["reported_cost_usd"]))
            - float(study["reserved_unknown_usd"])
        )
        forecast = main_forecast(local_times, host_time, mean_cost, min(42, max(0, available - 10)))
        return {
            "status": "completed_exploratory_pilot",
            "screen": screen,
            "allocation": allocation,
            "cells": cells,
            "references": references,
            "harness": harness,
            "fresh_confirmations": confirmations,
            "main_forecast": forecast,
            "costs": self.execution.accounts(),
            "key_usage_usd": str(self.execution.key_usage),
            "study_unknown_reservation_usd": study["reserved_unknown_usd"],
        }


def run(root: Path, env_file: Path) -> int:
    store = Store(root)
    with Store(root.parent).lock(), store.lock():
        plan = verify_runtime(store)
        if plan.get("source_quality", {}).get("status") != "admitted_after_source_review":
            print(
                "NOT LAUNCHABLE: benchmark source-quality review remains unresolved. Prepared artifacts are for engineering review; no inference dispatched.",
                flush=True,
            )
            return 2
        if store.exists("reports/complete.json"):
            print(
                "Pilot already complete; zero new inference. See reports/complete.json.", flush=True
            )
            return 0
        progress = Progress(store, plan["initial_step_estimate"])
        progress.estimated = True
        execution = Execution(
            store,
            env_file,
            max_seconds=plan["max_seconds"],
            max_cost_usd=plan["max_cost_usd"],
            max_attempts=plan["max_attempts"],
            study_root=root.parent,
            runtime=root / "runtime",
            progress=progress,
        )
        monitor = Telemetry(store, interval=1)
        progress.clock = lambda: {
            "remaining_seconds": max(0, execution.remaining_seconds()),
            "accounted_seconds": plan["max_seconds"] - execution.remaining_seconds(),
        }
        execution.monitor = monitor
        monitor.start()
        monitor_stopped = False
        try:
            pilot = Pilot(execution)
            result = pilot.execute()
            result["accounted_running_seconds"] = (
                plan["max_seconds"] - execution.remaining_seconds()
            )
            execution.checkpoint()
            monitor.stop()
            monitor_stopped = True
            store.put("reports/complete.json", result)
            pilot.progress_schedule("final", {}, final=True)
            progress.emit("report", "resolved pilot", result["status"])
            print(f"Pilot status: {result['status']}. Raw data and decisions: {root}", flush=True)
            return 0
        except KeyboardInterrupt:
            print("Interrupted; run the same resume.py command to continue.", flush=True)
            return 130
        except (RunPaused, IntegrityError, OSError, ValueError) as exc:
            store.put(
                f"pauses/{uuid.uuid4().hex}.json", {"reason": str(exc), "timestamp_utc": utc_now()}
            )
            print(f"PAUSED: {exc}. Evidence retained; do not delete or reset the run.", flush=True)
            return 2
        finally:
            execution.checkpoint()
            if not monitor_stopped:
                monitor.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--registry-dir", type=Path)
    parser.add_argument("--expert-prompt", type=Path)
    parser.add_argument(
        "--env-file", type=Path, default=Path(__file__).resolve().parents[2] / ".env"
    )
    args = parser.parse_args()
    if args.action == "prepare":
        if args.benchmark_dir is None or args.registry_dir is None or args.expert_prompt is None:
            parser.error("prepare requires benchmark, registry and historical expert prompt paths")
        prepare(
            args.run_dir,
            args.benchmark_dir,
            args.registry_dir,
            {
                "kind": "public_selection_pilot",
                "max_seconds": 21600,
                "target_seconds": 14400,
                "max_cost_usd": "8",
                "max_attempts": 10000,
                "initial_step_estimate": {
                    "P0": 63,
                    "P1": 480,
                    "P2": 1392,
                    "P3": 50,
                    "P4": 100,
                    "report": 1,
                },
                "progress_note": "Initial work estimate; adaptive allocation and censored work reported by stage. Percent is work, not within-generation progress.",
                "env_file": str(args.env_file.resolve()),
                "expert_source": str(args.expert_prompt.resolve()),
            },
        )
        Store(args.run_dir).put(
            "inputs/prompts.json",
            {"naive": NAIVE, "historical_expert": args.expert_prompt.read_text()},
        )
        prepared = Store(args.run_dir)
        prepared.put(
            "preparation.json",
            {
                "plan_sha256": digest(prepared.get("plan.json")),
                "input_sha256": {
                    name: digest(prepared.get(name)) for name in prepared.names("inputs/*.json")
                },
            },
        )
        print(
            "Prepared pilot. Launch/resume with this run's resume.py and --env-file /absolute/path/to/code/.env."
        )
        return 0
    return run(args.run_dir.resolve(), args.env_file.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
