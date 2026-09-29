"""Phase 3 replicated tier campaign: frozen manifest, paired schedules, sealed panels.

One run directory holds one frozen design. Every trajectory is `arm/executor/T<tier>/R<rep>`.
Repetitions differ only by a seeded feedback-batch schedule shared across tiers and executors,
and by a hosted sampling seed where the pinned endpoint accepts one. Executors stay at
temperature 0, seed 11. Selection uses the validation partition; each selected prompt is
evaluated once per sealed panel, shared across trajectories that select identical text.
Optimizer calls are serial with local evaluation; no pipelining is claimed.
"""

from __future__ import annotations

import argparse
import random
import statistics
import uuid
from pathlib import Path
from typing import Any

from ..benchmark.model import BenchmarkProfile
from ..domain import metrics
from ..runner import RunPaused, utc_now
from ..storage import IntegrityError, Store, digest
from .campaign import Campaign, freeze_runtime, prepare, profiles_from, verify_runtime
from .conditions import NAIVE, optimizer_request, study_feedback_request
from .execution import Execution
from .progress import Progress
from .scheduling import utility
from .taskbudget import SCOPE, ScopedExecution, scope_usage
from .telemetry import Telemetry
from .transport import OLLAMA

TIERS = (1, 2, 3)
PANELS = ("test", "temporal", "product_heldout")
PARTITIONS = ("optimization", "validation", *PANELS)
ARMS = ("main", "gepa", "reference")
TASKS = (*ARMS, "selection", "sealed", "report")
PROTOCOL = (
    "docs/v2_addressing_limitations.md",
    "docs/adr/0033-freeze-researcher-optimizer-choice-and-task3-handoff.md",
    "docs/adr/0034-pin-gepa-with-tier-restricted-feedback-and-exact-budgets.md",
    "docs/adr/0036-defer-pi-and-exclude-opencode-and-magnitude.md",
)


def batch_schedule(
    profiles: list[BenchmarkProfile], repetitions: int, depth: int, batch_size: int, *, seed: int
) -> list[list[list[str]]]:
    """Per-repetition balanced batches from the optimization partition.

    Each repetition shuffles every (stratum, positive) bucket with its own seeded generator and
    interleaves them, so every batch keeps the partition's difficulty/label balance. Batches are
    drawn without replacement until the partition is exhausted, then the interleaving restarts.
    """
    pool = sorted((p for p in profiles if p.partition == "optimization"), key=lambda p: p.id)
    if not pool or min(repetitions, depth, batch_size) < 1:
        raise ValueError("schedule requires optimization profiles and positive sizes")
    schedules = []
    for rep in range(1, repetitions + 1):
        rng = random.Random(f"{seed}/{rep}")  # noqa: S311 - deterministic schedule, not security
        buckets = []
        for stratum in ("easy", "medium", "hard"):
            for positive in (False, True):
                bucket = [
                    p.id for p in pool if p.stratum == stratum and bool(p.expected) == positive
                ]
                rng.shuffle(bucket)
                buckets.append(bucket)
        ordered: list[str] = []
        while any(buckets):
            for bucket in buckets:
                if bucket:
                    ordered.append(bucket.pop())
        needed = depth * batch_size
        cycle = list(ordered)
        while len(cycle) < needed:
            cycle.extend(ordered)
        schedules.append([cycle[i * batch_size : (i + 1) * batch_size] for i in range(depth)])
    return schedules


def trajectory_calls(design: dict[str, Any]) -> dict[str, int]:
    """Upper-bound logical calls for one trajectory; shared evaluations may reduce actual counts."""
    depth, slots, batch = design["depth"], design["candidates_per_round"], design["batch_size"]
    return {
        "local_search": batch * depth * (slots + 1),
        "local_selection": 2 * design["validation_size"],
        "local_sealed": sum(design["panels"].values()),
        "hosted": depth * slots,
    }


def build_manifest(
    design: dict[str, Any], profiles: list[BenchmarkProfile], conditions: dict[str, Any]
) -> dict[str, Any]:
    local = {c["id"]: c for c in conditions["local"]}
    hosted = {c["id"]: c for c in conditions["hosted"]}
    if any(e not in local for e in design["executors"]) or design["optimizer"] not in hosted:
        raise IntegrityError("design names a condition absent from the frozen registry")
    if len(set(design["executors"])) != len(design["executors"]):
        raise IntegrityError("duplicate executor in design")
    if any(t not in TIERS for t in design["tiers"]) or not design["tiers"]:
        raise IntegrityError("design tiers must be a subset of 1, 2, 3")
    if min(design["repetitions"], design["depth"], design["candidates_per_round"]) < 1:
        raise IntegrityError("design requires positive repetitions, depth and candidate slots")
    counts = {name: sum(p.partition == name for p in profiles) for name in PARTITIONS}
    if counts["validation"] < design["validation_size"] or any(
        counts[panel] < size for panel, size in design["panels"].items()
    ):
        raise IntegrityError("frozen partitions are smaller than the design requires")
    for arm in ("gepa", "reference"):
        extra = design.get(arm)
        if extra is None:
            continue
        if extra["executor"] not in local or extra["tier"] not in TIERS or extra["repetitions"] < 1:
            raise IntegrityError(f"{arm} arm is misconfigured")
        if arm == "reference" and design.get("reference_optimizer") not in hosted:
            raise IntegrityError("reference arm needs a frozen reference optimizer")
    schedules = batch_schedule(
        profiles,
        max(
            [design["repetitions"]]
            + [design[a]["repetitions"] for a in ("gepa", "reference") if design.get(a)]
        ),
        design["depth"],
        design["batch_size"],
        seed=design["schedule_seed"],
    )
    per = trajectory_calls(design)
    trajectories = []
    first = int(design.get("first_repetition", 1))
    if first < 1 or first > design["repetitions"]:
        raise IntegrityError("first_repetition must lie within 1..repetitions")
    for rep in range(first, design["repetitions"] + 1):
        for executor in design["executors"]:
            for tier in design["tiers"]:
                trajectories.append(
                    {"arm": "main", "executor": executor, "tier": tier, "repetition": rep}
                )
    for arm in ("gepa", "reference"):
        extra = design.get(arm)
        if extra:
            for rep in range(1, extra["repetitions"] + 1):
                trajectories.append(
                    {
                        "arm": arm,
                        "executor": extra["executor"],
                        "tier": extra["tier"],
                        "repetition": rep,
                    }
                )
    endpoint = hosted[design["optimizer"]]["endpoint"]
    return {
        "design": design,
        "schedules": {str(i + 1): s for i, s in enumerate(schedules)},
        "trajectories": trajectories,
        "per_trajectory_calls": per,
        "upper_bound_calls": {
            "local": len(trajectories)
            * (per["local_search"] + per["local_selection"] + per["local_sealed"]),
            "hosted": len(trajectories) * per["hosted"],
        },
        "hosted_seed_supported": "seed" in endpoint.get("supported_parameters", []),
        "partition_counts": counts,
        "profile_sha256": digest([p.record() for p in sorted(profiles, key=lambda p: p.id)]),
        "replication_contract": (
            "Repetition r varies only the seeded optimization batch schedule (shared across tiers "
            "and executors) and the hosted sampling seed hosted_seed_base + r when the pinned "
            "endpoint lists 'seed'. Executors use temperature 0 and seed 11. Selection-validation "
            "and sealed-panel evaluations are shared per (executor, prompt text) because the "
            "Executor is deterministic by configuration; backend nondeterminism is measured "
            "separately, not by repeating panels."
        ),
        "execution_order": "repetition outer, executor middle, tier inner; then gepa, then reference",
        "pipelining": "none; hosted Optimizer calls are serial with local evaluation",
    }


def default_design(promoted: list[str]) -> dict[str, Any]:
    return {
        "executors": list(promoted),
        "optimizer": "z-ai/glm-5.3",
        "reference_optimizer": "deepseek/deepseek-v4-pro",
        "tiers": list(TIERS),
        "repetitions": 5,
        "depth": 8,
        "candidates_per_round": 2,
        "batch_size": 24,
        "validation_size": 96,
        "panels": {"test": 192, "temporal": 96, "product_heldout": 96},
        "output_tokens": 2048,
        "hosted_seed_base": 1000,
        "schedule_seed": 20260913,
        "gepa": None,
        "reference": None,
        "defer_generation_metadata": True,
        "first_repetition": 1,
    }


def prepare_study(
    root: Path,
    benchmark: Path,
    registry: Path,
    calibration: Path,
    env_file: Path,
    design: dict[str, Any],
    *,
    phase: str,
    max_seconds: float,
    max_cost_usd: str,
) -> dict[str, Any]:
    if root.exists():
        raise IntegrityError("study requires a fresh run identity")
    if phase not in ("repilot", "phase3"):
        raise IntegrityError("phase must be repilot or phase3")
    parent = Store(calibration)
    verdict = parent.get("reports/complete.json")["verdict"]
    if verdict.get("passed") is not True:
        raise IntegrityError("study requires a passing calibration gate")
    if parent.get("plan.json")["benchmark_sha256"] != digest(Store(benchmark).get("manifest.json")):
        raise IntegrityError("study must use the benchmark admitted by calibration")
    if int(design.get("first_repetition", 1)) == 1 and any(
        e not in design["executors"] for e in verdict["promoted"]
    ):
        raise IntegrityError("design must include every gate-promoted executor")
    config: dict[str, Any] = {
        "kind": "main_study",
        "phase": phase,
        "runner_module": "promptbench.live.study",
        "max_seconds": max_seconds,
        "max_cost_usd": max_cost_usd,
        "max_attempts": 200000,
        "input_partitions": list(PARTITIONS),
        "initial_step_estimate": {task: 1 for task in TASKS},
        "env_file": str(env_file.resolve()),
        "benchmark_root": str(benchmark.resolve()),
        "calibration_root": str(calibration.resolve()),
    }
    if phase == "repilot":
        config["budget_scope"] = SCOPE
        config["scope_gpu_ceiling_seconds"] = 16 * 3600
        config["prior_scope_usage"] = scope_usage(root.parent)
    prepare(root, benchmark, registry, config)
    store = Store(root)
    store.put(
        "inputs/calibration.json",
        {"verdict": verdict, "report_sha256": digest(parent.get("reports/complete.json"))},
    )
    project = Path(__file__).resolve().parents[3]
    store.put(
        "inputs/protocol.json",
        {name: (project / name).read_text() for name in PROTOCOL if (project / name).exists()},
    )
    conditions = store.get("inputs/conditions.json")
    for condition in conditions["local"]:
        if condition["id"] in design["executors"] or condition["id"] in {
            design[a]["executor"] for a in ("gepa", "reference") if design.get(a)
        }:
            condition["num_predict"] = design["output_tokens"]
            condition["timeout_seconds"] = 300
    store.put("inputs/study_conditions.json", conditions)
    manifest = build_manifest(design, profiles_from(store), conditions)
    steps = manifest["upper_bound_calls"]
    per = manifest["per_trajectory_calls"]
    count = len(manifest["trajectories"])
    manifest["initial_step_estimate"] = {
        "main": per["local_search"] * sum(t["arm"] == "main" for t in manifest["trajectories"]),
        "gepa": per["local_search"] * sum(t["arm"] == "gepa" for t in manifest["trajectories"]),
        "reference": per["local_search"]
        * sum(t["arm"] == "reference" for t in manifest["trajectories"]),
        "selection": per["local_selection"] * count,
        "sealed": per["local_sealed"] * count,
        "report": 1,
    }
    manifest["initial_step_estimate"] = {
        k: max(1, v) for k, v in manifest["initial_step_estimate"].items()
    }
    manifest["upper_bound_calls"] = steps
    manifest["plan_sha256"] = digest(store.get("plan.json"))
    manifest["input_sha256"] = {n: digest(store.get(n)) for n in store.names("inputs/*.json")}
    store.put("manifest.json", manifest)
    return manifest


def timing(store: Store, keys: list[str]) -> dict[str, Any]:
    durations, costs = [], []
    for key in keys:
        pointer = "logical-keys/" + digest(key) + ".json"
        if not store.exists(pointer):
            continue
        work = "work/" + store.get(pointer)["spec_sha256"]
        for name in store.names(work + "/attempts/*/response.json"):
            response = store.get(name)
            durations.append(response["duration_ns"] / 1e9)
            normalized = name.replace("response.json", "normalized.json")
            if store.exists(normalized):
                cost = store.get(normalized).get("reported_api_cost_usd")
                if cost is not None:
                    costs.append(float(cost))
    ordered = sorted(durations)
    return {
        "physical_attempts": len(ordered),
        "mean_seconds": statistics.mean(ordered) if ordered else None,
        "p90_seconds": ordered[max(0, int(len(ordered) * 0.9) - 1)] if ordered else None,
        "total_seconds": sum(ordered),
        "reported_cost_usd": sum(costs) if costs else None,
    }


ADOPTED = (
    "work",
    "logical-keys",
    "evaluations",
    "selections",
    "sealed-panels",
    "trajectories",
    "decisions",
    "disclosures",
    "generations",
    "generation-pending",
    "candidate-failures",
    "gepa-runs",
    "hosted-key-usage",
    "checks",
)


def prepare_continuation(
    parent: Path,
    root: Path,
    *,
    max_seconds: float,
    max_cost_usd: str,
    reason: str,
    refresh_runtime: bool = False,
    accept_tariff: str | None = None,
) -> dict[str, Any]:
    """Adopt a paused study's journal into a fresh identity with a new time allocation.

    Records are copied byte for byte, so hosted attempts keep their dispatch identity and the
    study-wide ledger deduplicates them; nothing is billed twice. The parent stays intact.
    """
    import shutil

    if root.exists():
        raise IntegrityError("continuation requires a fresh run identity")
    source = Store(parent)
    with source.lock():
        if source.exists("reports/complete.json") or not source.names("pauses/*.json"):
            raise IntegrityError("only a paused, incomplete study can be continued")
        plan, manifest = source.get("plan.json"), source.get("manifest.json")
        if plan["kind"] != "main_study":
            raise IntegrityError("continuation requires a study run")
        root.mkdir(parents=True)
        for folder in ("inputs", "runtime", *ADOPTED):
            if (parent / folder).exists():
                shutil.copytree(parent / folder, root / folder)
        runtime_change: dict[str, Any] | None = None
        if refresh_runtime:
            # Explicit, recorded runtime replacement: the current repository source is frozen
            # into the child and every changed module is listed with old and new hashes.
            from .preflight import sources

            old_source = source.get("inputs/source.json")
            new_source = sources()
            changed = sorted(
                n
                for n in set(old_source) | set(new_source)
                if old_source.get(n) != new_source.get(n)
            )
            shutil.rmtree(root / "runtime")
            freeze_runtime(root, new_source)
            (root / "inputs/source.json").unlink()
            Store(root).put("inputs/source.json", new_source)
            runtime_change = {
                "changed_modules": {
                    n: {
                        "old_sha256": digest(old_source.get(n)),
                        "new_sha256": digest(new_source.get(n)),
                    }
                    for n in changed
                },
                "note": "Runtime refreshed on continuation; scientific inputs, manifest and design unchanged.",
            }
        if (parent / "resume.py").exists():
            shutil.copy2(parent / "resume.py", root / "resume.py")
        target = Store(root)
        tariff_change: dict[str, Any] | None = None
        if accept_tariff is not None:
            # Explicit acceptance of a hosted price increase for one pinned endpoint, taken from
            # the parent's latest retained endpoint check. Only that endpoint's pricing changes.
            # Check names are content hashes, so "latest" must be decided by the retained
            # completion time, not by name order.
            checks = sorted(
                source.names("checks/*/endpoints-*/response.json"),
                key=lambda n: str(source.get(n).get("finished_utc", "")),
            )
            if not checks:
                raise IntegrityError("no retained endpoint check to accept a tariff from")
            import json as _json

            latest = _json.loads(source.get(checks[-1])["body_text"])["data"]["endpoints"]
            for name in ("inputs/conditions.json", "inputs/study_conditions.json"):
                conditions = target.get(name)
                hosted = next((h for h in conditions["hosted"] if h["id"] == accept_tariff), None)
                if hosted is None:
                    raise IntegrityError("tariff acceptance names an unknown hosted condition")
                current = next(e for e in latest if e["tag"] == hosted["endpoint"]["tag"])
                old_pricing = dict(hosted["endpoint"]["pricing"])
                hosted["endpoint"]["pricing"] = {**old_pricing, **current["pricing"]}
                (root / name).unlink()
                target.put(name, conditions)
                tariff_change = {
                    "hosted": accept_tariff,
                    "endpoint": hosted["endpoint"]["tag"],
                    "old_pricing": old_pricing,
                    "new_pricing": hosted["endpoint"]["pricing"],
                    "source_check": checks[-1],
                }
        new_plan = {
            **plan,
            **(
                {"source_sha256": digest(Store(root).get("inputs/source.json"))}
                if refresh_runtime
                else {}
            ),
            "max_seconds": max_seconds,
            "max_cost_usd": max_cost_usd,
            "parent_run": str(parent.resolve()),
            "parent_plan_sha256": digest(plan),
            "continuation_reason": reason,
        }
        if plan.get("budget_scope") == SCOPE:
            new_plan["prior_scope_usage"] = scope_usage(root.parent, exclude=root)
        target.put("plan.json", new_plan)
        target.put(
            "manifest.json",
            {
                **manifest,
                "plan_sha256": digest(new_plan),
                "input_sha256": {n: digest(target.get(n)) for n in target.names("inputs/*.json")},
            },
        )
        lineage = {
            "parent": str(parent.resolve()),
            "parent_manifest_sha256": digest(manifest),
            "adopted": {
                folder: sum(1 for _ in (root / folder).rglob("*") if _.is_file())
                for folder in ADOPTED
                if (root / folder).exists()
            },
            "parent_pauses": [source.get(n) for n in source.names("pauses/*.json")],
            "reason": reason,
            "runtime_change": runtime_change,
            "tariff_change": tariff_change,
            "timestamp_utc": utc_now(),
        }
        target.put("parent-lineage.json", lineage)
        return lineage


def _without_price_ceiling(payload: dict[str, Any]) -> dict[str, Any]:
    request = dict(payload.get("request", {}))
    provider = dict(request.get("provider", {}))
    provider.pop("max_price", None)
    request["provider"] = provider
    return {**payload, "request": request, "sha256": None}


def record_disclosure(store: Store, name: str, payload: dict[str, Any]) -> None:
    """Write the Optimizer-view record for a proposal, tolerating an accepted tariff change.

    A disclosure documents what the Optimizer may learn; the price ceiling sent with the
    request is run metadata, and after an explicit tariff acceptance a replayed proposal
    differs from its retained record only there. The retained record stands, and the ceiling
    actually sent with every wire request remains in the attempt journal. Any other
    difference is still an immutability conflict.
    """
    if store.exists(name):
        existing = store.get(name)
        if _without_price_ceiling(existing) == _without_price_ceiling(payload):
            return
    store.put(name, payload)


class Study:
    def __init__(self, execution: Execution):
        self.execution, self.store = execution, execution.store
        self.plan = self.store.get("plan.json")
        self.manifest = self.store.get("manifest.json")
        self.design = self.manifest["design"]
        self.campaign = Campaign(execution)
        self.profiles = {p.id: p for p in profiles_from(self.store)}
        conditions = self.store.get("inputs/study_conditions.json")
        self.local = {c["id"]: c for c in conditions["local"]}
        self.hosted = {c["id"]: c for c in conditions["hosted"]}
        self.keys: dict[str, list[str]] = {}

    # ----- evaluation helpers -------------------------------------------------------------
    def batch(self, identifiers: list[str]) -> list[BenchmarkProfile]:
        return [self.profiles[i] for i in identifiers]

    def partition(self, name: str, size: int) -> list[BenchmarkProfile]:
        chosen = sorted(
            (p for p in self.profiles.values() if p.partition == name), key=lambda p: p.id
        )
        if len(chosen) < size:
            raise IntegrityError(f"partition {name} smaller than design")
        return chosen[:size]

    def evaluate(
        self, prefix: str, executor: dict[str, Any], prompt: str, profiles: list[BenchmarkProfile]
    ) -> list[dict[str, Any]]:
        self.keys.setdefault(prefix.split("/", 1)[0], []).extend(
            f"{prefix}/{p.id}" for p in profiles
        )
        return self.campaign.evaluate(prefix, executor, prompt, profiles)

    def hosted_seed(self, hosted: dict[str, Any], repetition: int) -> int | None:
        if "seed" in hosted["endpoint"].get("supported_parameters", []):
            return int(self.design["hosted_seed_base"]) + repetition
        return None

    def propose(
        self, key: str, hosted: dict[str, Any], permitted: dict[str, Any], seed: int | None
    ) -> str | None:
        if any(
            r["partition"] != "optimization"
            for r in (permitted.get("feedback") or {}).get("rows", [])
        ):
            raise IntegrityError("optimizer feedback derived from a sealed partition")
        # Hosted identity is verified once per controller session; every response is still
        # checked against the frozen model/provider by Execution.call.
        self.campaign.check(hosted)
        condition, body = optimizer_request(hosted, permitted, seed=seed)
        record_disclosure(
            self.store,
            "disclosures/" + digest(key) + ".json",
            {
                "key": key,
                "request": body,
                "sha256": digest(body),
                "tier": permitted["tier"],
                "hosted_seed_sent": seed,
            },
        )
        self.keys.setdefault(key.split("/", 1)[0], []).append(key)
        result = self.execution.call(key, condition, body, role="optimizer")
        if result["status"] == "valid":
            if self.design.get("defer_generation_metadata", False):
                # Metadata only; fetched offline by the `reconcile` action after the campaign.
                self.store.put(
                    "generation-pending/" + digest(result["attempt"]) + ".json",
                    {
                        "attempt": result["attempt"],
                        "canonical_model": hosted["canonical_model"],
                        "reason": "deferred by design; not an error",
                    },
                )
            else:
                self.campaign.reconcile_generation(result["attempt"], hosted["canonical_model"])
            return str(result["value"])
        self.store.put(
            "candidate-failures/" + digest(key) + ".json",
            {"key": key, "status": result["status"], "reason": "no valid prompt; slot forfeited"},
        )
        return None

    # ----- shared selection and sealed evaluation ---------------------------------------
    def selection(self, executor: dict[str, Any], prompt: str) -> dict[str, Any]:
        prefix = f"selection/{executor['id']}/{digest(prompt)}"
        name = "selections/" + digest(prefix) + ".json"
        if self.store.exists(name):
            return dict(self.store.get(name))
        rows = self.evaluate(
            prefix, executor, prompt, self.partition("validation", self.design["validation_size"])
        )
        summary = {"prefix": prefix, "prompt_sha256": digest(prompt), "metrics": metrics(rows)}
        self.store.put(name, summary)
        return summary

    def sealed(self, executor: dict[str, Any], prompt: str) -> dict[str, Any]:
        result = {}
        for panel, size in self.design["panels"].items():
            prefix = f"sealed/{executor['id']}/{panel}/{digest(prompt)}"
            name = "sealed-panels/" + digest(prefix) + ".json"
            if self.store.exists(name):
                result[panel] = self.store.get(name)
                continue
            rows = self.evaluate(prefix, executor, prompt, self.partition(panel, size))
            by_stratum = {
                s: metrics([r for r in rows if r["stratum"] == s])
                for s in ("easy", "medium", "hard")
            }
            summary = {
                "prefix": prefix,
                "panel": panel,
                "prompt_sha256": digest(prompt),
                "metrics": metrics(rows),
                "by_stratum": by_stratum,
            }
            self.store.put(name, summary)
            result[panel] = summary
        return result

    def finish(
        self, prefix: str, executor: dict[str, Any], incumbent: str, extra: dict[str, Any]
    ) -> dict[str, Any]:
        shortlist = [{"index": 0, "prompt": NAIVE, **self.selection(executor, NAIVE)}]
        if incumbent != NAIVE:
            shortlist.append(
                {"index": 1, "prompt": incumbent, **self.selection(executor, incumbent)}
            )
        selected = max(shortlist, key=utility)  # first (naive) wins exact ties
        report = {
            "prefix": prefix,
            "executor": executor["id"],
            **extra,
            "shortlist": shortlist,
            "selected": selected,
            "prompt_changed": selected["prompt"] != NAIVE,
            "sealed": self.sealed(executor, selected["prompt"]),
            "timing": {
                task: timing(self.store, [k for k in keys if k.startswith(prefix)])
                for task, keys in self.keys.items()
            },
        }
        self.store.put("trajectories/" + digest(prefix) + ".json", report)
        return report

    # ----- arms --------------------------------------------------------------------------
    def trajectory(self, arm: str, item: dict[str, Any]) -> dict[str, Any]:
        executor, tier, rep = self.local[item["executor"]], item["tier"], item["repetition"]
        hosted = self.hosted[
            self.design["reference_optimizer"] if arm == "reference" else self.design["optimizer"]
        ]
        prefix = f"{arm}/{executor['id']}/T{tier}/R{rep}"
        name = "trajectories/" + digest(prefix) + ".json"
        if self.store.exists(name):
            return dict(self.store.get(name))
        schedule = self.manifest["schedules"][str(rep)]
        seed = self.hosted_seed(hosted, rep)
        incumbent, rows = NAIVE, None
        trace = []
        for round_index in range(self.design["depth"]):
            batch = self.batch(schedule[round_index])
            key = f"{prefix}/r{round_index}/incumbent"
            rows = self.evaluate(key, executor, incumbent, batch)
            candidates = [(incumbent, rows, -1)]
            for slot in range(self.design["candidates_per_round"]):
                proposed = self.propose(
                    f"{prefix}/r{round_index}/proposal{slot}",
                    hosted,
                    study_feedback_request(incumbent, rows, batch, tier=tier, candidate=slot),
                    seed,
                )
                if proposed is None:
                    continue
                evaluated = self.evaluate(
                    f"{prefix}/r{round_index}/candidate{slot}", executor, proposed, batch
                )
                candidates.append((proposed, evaluated, slot))
            incumbent, best, slot = max(
                candidates,
                key=lambda c: metrics(c[1])["failure_aware_lower_bound"]["micro_f1"],
            )
            decision = {
                "round": round_index,
                "slot": slot,
                "prompt_sha256": digest(incumbent),
                "metrics": metrics(best),
                "compared_prompt_hashes": [digest(c[0]) for c in candidates],
                "batch": schedule[round_index],
            }
            self.store.put("decisions/" + digest(f"{prefix}/{round_index}") + ".json", decision)
            trace.append(decision)
        return self.finish(
            prefix,
            executor,
            incumbent,
            {
                "arm": arm,
                "optimizer": hosted["id"],
                "tier": tier,
                "repetition": rep,
                "hosted_seed_sent": seed,
                "trace": trace,
                "final_prompt": incumbent,
            },
        )

    def gepa_trajectory(self, item: dict[str, Any]) -> dict[str, Any]:
        from .gepa_adapter import TierAdapter, run_gepa

        executor, tier, rep = self.local[item["executor"]], item["tier"], item["repetition"]
        hosted = self.hosted[self.design["optimizer"]]
        prefix = f"gepa/{executor['id']}/T{tier}/R{rep}"
        name = "trajectories/" + digest(prefix) + ".json"
        if self.store.exists(name):
            return dict(self.store.get(name))
        schedule = [tuple(b) for b in self.manifest["schedules"][str(rep)]]
        seed = self.hosted_seed(hosted, rep)
        counters = {"evaluate": 0, "propose": 0}

        def run_executor(prompt: str, profile: BenchmarkProfile) -> tuple[list[str] | None, str]:
            ordinal = counters["evaluate"]
            counters["evaluate"] += 1
            rows = self.evaluate(f"{prefix}/e{ordinal:05d}", executor, prompt, [profile])
            row = rows[0]
            return row["prediction"], row["status"]

        def run_proposer(request: dict[str, Any]) -> str:
            ordinal = counters["propose"]
            counters["propose"] += 1
            proposed = self.propose(f"{prefix}/p{ordinal:04d}", hosted, request, seed)
            if proposed is None:
                raise RuntimeError("GEPA proposal forfeited; engine stops")
            return proposed

        attempt = len(self.store.names(f"gepa-runs/{digest(prefix)}/*/manifest.json"))
        sub = Store(self.store.root / "gepa-runs" / digest(prefix) / f"{attempt:02d}")
        adapter = TierAdapter(
            [self.profiles[i] for b in schedule for i in b if i in self.profiles],
            run_executor,
            run_proposer,
            tier=tier,
            depth=self.design["depth"],
            candidates_per_round=self.design["candidates_per_round"],
            batch_size=self.design["batch_size"],
            store=sub,
        )
        summary = run_gepa(adapter, schedule, seed=int(self.design["schedule_seed"]) + rep)
        final = str(summary["selected_candidate"]["prompt"])
        return self.finish(
            prefix,
            executor,
            final,
            {
                "arm": "gepa",
                "optimizer": hosted["id"],
                "tier": tier,
                "repetition": rep,
                "hosted_seed_sent": seed,
                "gepa": {**summary, "run_dir": str(sub.root.relative_to(self.store.root))},
                "final_prompt": final,
            },
        )

    def execute(self) -> dict[str, Any]:
        before = self.execution.refresh_key()
        self.store.put(
            f"hosted-key-usage/{self.execution.session}-before.json",
            {"project_dedicated_key_total_usd": str(before), "timestamp_utc": utc_now()},
        )
        reports = []
        for item in self.manifest["trajectories"]:
            if item["arm"] == "gepa":
                reports.append(self.gepa_trajectory(item))
            else:
                reports.append(self.trajectory(item["arm"], item))
        if self.campaign.last_local:
            self.campaign.metadata(
                "unload",
                OLLAMA + "/api/generate",
                {"model": self.campaign.last_local, "keep_alive": 0, "stream": False},
            )
            self.campaign.last_local = None
        after = self.execution.refresh_key()
        self.store.put(
            f"hosted-key-usage/{self.execution.session}-after.json",
            {"project_dedicated_key_total_usd": str(after), "timestamp_utc": utc_now()},
        )
        from .analysis import analysis, forecast

        exported = analysis(self.store)
        self.store.put("reports/analysis.json", exported)
        return {
            "status": "completed",
            "phase": self.plan["phase"],
            "trajectories": len(reports),
            "prompt_changed": sum(r["prompt_changed"] for r in reports),
            "forecast": forecast(self.store, self.manifest),
            "project_hosted_spend_usd": str(after),
            "costs": self.execution.accounts(),
            "analysis_sha256": digest(exported),
        }


def reconcile(root: Path, env_file: Path) -> dict[str, Any]:
    """Offline hosted-metadata pass: resolves deferred generation records, no inference."""
    store = Store(root)
    with store.lock():
        # The journal clock includes every prior session of this run; metadata lookups must
        # not be starved by the finished campaign's own running time.
        execution = Execution(
            store,
            env_file,
            max_seconds=259200,
            max_cost_usd="0.01",
            max_attempts=1,
            study_root=root.parent,
            runtime=root / "runtime",
        )
        campaign = Campaign(execution)
        resolved, failed = 0, 0
        for name in store.names("generation-pending/*.json"):
            record = store.get(name)
            if "attempt" not in record:
                continue  # legacy records from in-run failures keep their own lookups
            try:
                campaign.reconcile_generation(record["attempt"], record["canonical_model"])
                resolved += 1
            except (ValueError, OSError):
                failed += 1
        summary = {
            "resolved": resolved,
            "failed": failed,
            "generations": len(store.names("generations/*.json")),
            "timestamp_utc": utc_now(),
        }
        store.put(f"reconciliations/{uuid.uuid4().hex}.json", summary)
        return summary


def run(root: Path, env_file: Path) -> int:
    store = Store(root)
    with Store(root.parent).lock(), store.lock():
        plan, manifest = verify_runtime(store), store.get("manifest.json")
        if (
            plan["kind"] != "main_study"
            or digest(plan) != manifest["plan_sha256"]
            or any(digest(store.get(n)) != h for n, h in manifest["input_sha256"].items())
        ):
            raise IntegrityError("study inputs or runtime changed")
        if store.exists("reports/complete.json"):
            print("Study already complete; no new inference.", flush=True)
            return 0
        progress = Progress(store, manifest["initial_step_estimate"])
        progress.estimated = True
        engine_type = ScopedExecution if plan.get("budget_scope") == SCOPE else Execution
        execution = engine_type(
            store,
            env_file,
            max_seconds=plan["max_seconds"],
            max_cost_usd=plan["max_cost_usd"],
            max_attempts=plan["max_attempts"],
            study_root=root.parent,
            runtime=root / "runtime",
            progress=progress,
        )
        progress.clock = lambda: {
            "remaining_seconds": max(0, execution.remaining_seconds()),
            "accounted_seconds": plan["max_seconds"] - execution.remaining_seconds(),
        }
        monitor = Telemetry(store)
        execution.monitor = monitor
        monitor.start()
        try:
            result = Study(execution).execute()
            execution.checkpoint()
            result["telemetry"] = monitor.stop()
            result["accounted_running_seconds"] = (
                plan["max_seconds"] - execution.remaining_seconds()
            )
            store.put("reports/complete.json", result)
            progress.emit("report", "study", "complete")
            print(f"Study complete: {result['trajectories']} trajectories. {root}", flush=True)
            return 0
        except KeyboardInterrupt:
            print("Interrupted; run the same resume.py command to continue.", flush=True)
            return 130
        except (RunPaused, IntegrityError, OSError, ValueError) as exc:
            store.put(
                "pauses/" + uuid.uuid4().hex + ".json",
                {"reason": str(exc), "timestamp_utc": utc_now()},
            )
            print(f"STUDY PAUSED: {exc}. Evidence retained; resume with resume.py.", flush=True)
            return 2
        finally:
            execution.checkpoint()
            if monitor.thread.is_alive():
                monitor.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "continue", "reconcile"))
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument(
        "--reason", default="time allocation exhausted; continuation authorized by the operator"
    )
    parser.add_argument("--refresh-runtime", action="store_true")
    parser.add_argument(
        "--accept-tariff", help="hosted condition ID whose current endpoint price is accepted"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--registry-dir", type=Path)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--phase", choices=("repilot", "phase3"), default="repilot")
    parser.add_argument("--executors", nargs="*", help="extra executor condition IDs")
    parser.add_argument(
        "--only-executors", nargs="*", help="restrict an extension run to these executors"
    )
    parser.add_argument("--optimizer", default="z-ai/glm-5.3")
    parser.add_argument("--reference-optimizer", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--tiers", nargs="*", type=int, default=list(TIERS))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--first-repetition",
        type=int,
        default=1,
        help="run only repetitions first..repetitions (schedules for lower indices are identical)",
    )
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--output-tokens", type=int, default=2048)
    parser.add_argument("--gepa-repetitions", type=int, default=0)
    parser.add_argument("--reference-repetitions", type=int, default=0)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--max-cost-usd")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.action == "reconcile":
        env = (args.env_file or Path(Store(root).get("plan.json")["env_file"])).resolve()
        print(reconcile(root, env))
        return 0
    if args.action == "continue":
        if args.parent_run is None or args.max_hours is None:
            parser.error("continue requires --parent-run and --max-hours")
        lineage = prepare_continuation(
            args.parent_run.resolve(),
            root,
            max_seconds=args.max_hours * 3600,
            max_cost_usd=args.max_cost_usd or "4",
            reason=args.reason,
            refresh_runtime=args.refresh_runtime,
            accept_tariff=args.accept_tariff,
        )
        print(f"Prepared continuation of {lineage['parent']}: adopted {lineage['adopted']}.")
        return 0
    if args.action == "prepare":
        if any(
            p is None
            for p in (args.benchmark_dir, args.registry_dir, args.calibration_dir, args.env_file)
        ):
            parser.error("prepare requires benchmark, registry, passing calibration and env file")
        promoted = Store(args.calibration_dir).get("reports/complete.json")["verdict"]["promoted"]
        design = default_design(promoted)
        design["executors"] = list(promoted) + [
            e for e in (args.executors or []) if e not in promoted
        ]
        if args.only_executors:
            design["executors"] = [e for e in design["executors"] if e in args.only_executors]
        design.update(
            optimizer=args.optimizer,
            reference_optimizer=args.reference_optimizer,
            tiers=args.tiers,
            repetitions=args.repetitions,
            first_repetition=args.first_repetition,
            depth=args.depth,
            candidates_per_round=args.candidates,
            batch_size=args.batch_size,
            output_tokens=args.output_tokens,
        )
        if args.gepa_repetitions:
            design["gepa"] = {
                "executor": promoted[0],
                "tier": 2,
                "repetitions": args.gepa_repetitions,
            }
        if args.reference_repetitions:
            design["reference"] = {
                "executor": promoted[0],
                "tier": 2,
                "repetitions": args.reference_repetitions,
            }
        hours = args.max_hours or (4 if args.phase == "repilot" else 72)
        cost = args.max_cost_usd or ("4" if args.phase == "repilot" else "15")
        manifest = prepare_study(
            root,
            args.benchmark_dir,
            args.registry_dir,
            args.calibration_dir,
            args.env_file,
            design,
            phase=args.phase,
            max_seconds=hours * 3600,
            max_cost_usd=cost,
        )
        print(
            f"Prepared {args.phase}: {len(manifest['trajectories'])} trajectories, "
            f"upper bound {manifest['upper_bound_calls']}. Launch with {root}/resume.py."
        )
        return 0
    return run(root, (args.env_file or Path(Store(root).get("plan.json")["env_file"])).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
