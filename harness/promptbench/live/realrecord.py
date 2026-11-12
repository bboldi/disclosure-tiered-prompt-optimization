"""Real-Record Panel (round 1 revision): fixed prompts on real CVE records, one pass each.

Per Executor: the naive prompt, the RQ4 output scaffold (its own schema and 4,096-token cap)
and every changed selected prompt of the campaign's main arm; plus hosted reference rows with
the naive prompt. Never used for optimization or selection. The rows stored here are scored by
the harness against `expected`, which counts a "yes" on the hand-labelled undecided cell as a
false positive; the prespecified scoring (matcher-decided stratum, hand-labelled stratum) is offline in
`scripts/g34_analysis.py`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from ..benchmark.model import restore_advisory, restore_profile, score
from ..domain import metrics
from ..runner import RunPaused, utc_now
from ..storage import IntegrityError, Store, digest
from .campaign import Campaign, prepare, profiles_from, verify_runtime
from .ceiling import ceiling_request
from .conditions import NAIVE, executor_request
from .execution import Execution
from .progress import Progress
from .scaffolding import SCAFFOLD_APPEND
from .telemetry import Telemetry

EXECUTORS = ("granite4.2:30b/off", "qwen3.8:27b/off")
MAIN_OPTIMIZER = "z-ai/glm-5.3"
CAMPAIGN_OUTPUT_TOKENS = 2048  # the campaign's Executor cap (study design output_tokens)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def selected_prompts(data: Path) -> dict[str, list[dict[str, Any]]]:
    """Distinct changed selected prompts of the main arm per Executor, checked against the shas."""
    shas = {
        (r["release_section"], r["executor"], r["tier"], r["repetition"]): r[
            "selected_prompt_sha256"
        ]
        for r in csv.DictReader((data / "exports" / "trajectories.csv").open())
        if r["arm"] == "main" and r["optimizer"] == MAIN_OPTIMIZER
    }
    result: dict[str, list[dict[str, Any]]] = {e: [] for e in EXECUTORS}
    for r in read_jsonl(data / "exports" / "selected_prompts.jsonl"):
        if (
            r["arm"] != "main"
            or r["optimizer"] != MAIN_OPTIMIZER
            or r["release_section"] not in ("campaign", "campaign-extension")
            or r["executor"] not in result
        ):
            continue
        sha = digest(r["selected_prompt"])
        key = (r["release_section"], r["executor"], str(r["tier"]), str(r["repetition"]))
        if shas[key] != sha:
            raise IntegrityError(f"selected prompt does not match its recorded sha: {key}")
        if not r["prompt_changed"]:
            if r["selected_prompt"] != NAIVE:
                raise IntegrityError(f"unchanged selection is not the naive prompt: {key}")
            continue
        chosen = next((p for p in result[r["executor"]] if p["sha256"] == sha), None)
        if chosen is None:
            chosen = {"sha256": sha, "prompt": r["selected_prompt"], "selected_by": []}
            result[r["executor"]].append(chosen)
        chosen["selected_by"].append(
            {"section": r["release_section"], "tier": r["tier"], "repetition": r["repetition"]}
        )
    return result


def load_panel(panel: Path) -> tuple[dict[str, Any], list[dict], dict[str, dict]]:
    manifest = json.loads((panel / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((panel / name).read_bytes()).hexdigest() != expected:
            raise IntegrityError("panel file changed since it was built: " + name)
    advisories = {
        row["id"]: restore_advisory(row) for row in read_jsonl(panel / "advisories.jsonl")
    }
    profiles = [restore_profile(row, advisories) for row in read_jsonl(panel / "profiles.jsonl")]
    if len(profiles) != manifest["counts"]["subprofiles"]:
        raise IntegrityError("panel profile count differs from its manifest")
    return (
        manifest,
        [p.record() for p in profiles],
        {a.id: a.record() for p in profiles for a in p.advisories},
    )


def prepare_realrecord(
    root: Path,
    benchmark: Path,
    registry: Path,
    env_file: Path,
    panel: Path,
    data: Path,
    *,
    hosted_models: list[str],
    executors: list[str],
    max_seconds: float,
    max_cost_usd: str,
) -> dict[str, Any]:
    if root.exists():
        raise IntegrityError("real-record run requires a fresh identity")
    panel_manifest, profile_rows, advisory_rows = load_panel(panel)
    prompts_by_executor = selected_prompts(data)
    prepare(
        root,
        benchmark,
        registry,
        {
            "kind": "real_record_panel",
            "runner_module": "promptbench.live.realrecord",
            "max_seconds": max_seconds,
            "max_cost_usd": max_cost_usd,
            "max_attempts": 5000,
            "input_partitions": [],  # the panel replaces the benchmark profiles below
            "initial_step_estimate": {"real_record": 1, "report": 1},
            "env_file": str(env_file.resolve()),
            "benchmark_root": str(benchmark.resolve()),
            "panel_root": str(panel.resolve()),
            "data_root": str(data.resolve()),
        },
    )
    store = Store(root)
    for name, payload in (
        ("inputs/profiles.json", profile_rows),
        ("inputs/advisories.json", advisory_rows),
    ):
        (root / name).unlink()
        store.put(name, payload)
    store.put("inputs/panel_manifest.json", panel_manifest)
    profiles = profiles_from(store)
    if {p.partition for p in profiles} != {"real_record"}:
        raise IntegrityError("inputs must hold only the real-record panel")
    conditions = store.get("inputs/conditions.json")
    local = {c["id"]: c for c in conditions["local"]}
    hosted = {c["id"]: c for c in conditions["hosted"]}
    if any(m not in hosted for m in hosted_models) or any(e not in local for e in executors):
        raise IntegrityError("model absent from the frozen registry")

    prompts = {"naive": NAIVE, "output_scaffold": NAIVE + SCAFFOLD_APPEND}
    runs: list[dict[str, Any]] = []  # one (condition, prompt) pass over the whole panel
    run_conditions: dict[str, dict[str, Any]] = {}
    for executor in executors:
        campaign = {
            **local[executor],
            "num_predict": CAMPAIGN_OUTPUT_TOKENS,
            "timeout_seconds": 300,
        }
        run_conditions[executor] = campaign
        # the RQ4 scaffold condition, as in scaffolding.control_design
        run_conditions[executor + "/output_scaffold"] = {
            **local[executor],
            "num_ctx": 16384,
            "num_predict": 4096,
            "timeout_seconds": 300,
            "output_scaffold": True,
        }
        passes = [("naive", executor), ("output_scaffold", executor + "/output_scaffold")]
        for p in prompts_by_executor[executor]:
            prompts[p["sha256"]] = p["prompt"]
            passes.append((p["sha256"], executor))
        # the longest prompt first, so the largest request is the first call (context check)
        passes.sort(key=lambda x: -len(prompts[x[0]]))
        runs += [{"condition": c, "prompt": k, "executor": executor} for k, c in passes]
    for model in hosted_models:
        run_conditions[model] = {
            **hosted[model],
            "schema_enum": False,
            "structured_inventory": False,
        }
        runs.append({"condition": model, "prompt": "naive", "executor": model})
    size = {p.id: len(canonical_input(p)) for p in profiles}
    order = sorted(size, key=lambda i: (-size[i], i))
    manifest = {
        "prompts": prompts,
        "selected_prompts": prompts_by_executor,
        "conditions": run_conditions,
        "runs": runs,
        "profile_order": order,
        "logical_calls": len(runs) * len(order),
        "initial_step_estimate": {"real_record": len(runs) * len(order), "report": 1},
        "plan_sha256": digest(store.get("plan.json")),
        "input_sha256": {n: digest(store.get(n)) for n in store.names("inputs/*.json")},
    }
    store.put("manifest.json", manifest)
    return manifest


def canonical_input(profile) -> str:
    return json.dumps(profile.executor_input(), sort_keys=True)


def check_context(store: Store, result: dict[str, Any], condition: dict[str, Any]) -> None:
    """The first (largest) request of each local condition must fit its context with the cap."""
    normalized = result["attempt"] + "/normalized.json"
    tokens = store.get(normalized).get("input_tokens") if store.exists(normalized) else None
    if tokens is None:
        raise IntegrityError(f"largest request has no input token count ({result['status']})")
    if tokens + condition["num_predict"] > condition["num_ctx"]:
        raise IntegrityError(
            f"largest request does not fit: {tokens} input + {condition['num_predict']} "
            f"output > num_ctx {condition['num_ctx']}"
        )


def execute(execution: Execution) -> dict[str, Any]:
    store, campaign = execution.store, Campaign(execution)
    manifest = store.get("manifest.json")
    by_id = {p.id: p for p in profiles_from(store)}
    before = execution.refresh_key()
    store.put(
        f"hosted-key-usage/{execution.session}-before.json",
        {"project_dedicated_key_total_usd": str(before), "timestamp_utc": utc_now()},
    )
    summaries: dict[str, Any] = {}
    contexts: dict[str, Any] = {}
    for item in manifest["runs"]:
        condition, prompt = (
            manifest["conditions"][item["condition"]],
            manifest["prompts"][item["prompt"]],
        )
        prefix = f"real_record/{item['condition']}/{digest(prompt)}"
        name = "panels/" + digest(prefix) + ".json"
        if store.exists(name):
            summaries[prefix] = store.get(name)
            continue
        campaign.check(condition)
        rows = []
        for identifier in manifest["profile_order"]:
            profile = by_id[identifier]
            if condition["provider"] == "ollama":
                admitted, body = condition, executor_request(condition, prompt, profile)
            else:
                admitted, body = ceiling_request(condition, prompt, profile)
            key = f"{prefix}/{identifier}"
            result = execution.call(
                key, admitted, body, role="executor", allowed_ids={a.id for a in profile.advisories}
            )
            if condition["provider"] == "ollama" and identifier == manifest["profile_order"][0]:
                check_context(store, result, condition)
                contexts[item["condition"]] = store.get(result["attempt"] + "/normalized.json").get(
                    "input_tokens"
                )
            row = score(profile, result["value"], result["status"])
            store.put(
                "evaluations/" + digest(key) + ".json",
                {
                    "key": key,
                    "condition_id": item["condition"],
                    "prompt_sha256": digest(prompt),
                    "row": row,
                },
            )
            rows.append(row)
            if condition["provider"] != "ollama" and result["status"] in (
                "valid",
                "invalid_output",
            ):
                store.put(
                    "generation-pending/" + digest(result["attempt"]) + ".json",
                    {
                        "attempt": result["attempt"],
                        "canonical_model": condition["canonical_model"],
                        "reason": "deferred by design; not an error",
                    },
                )
        summary = {
            "prefix": prefix,
            "condition": item["condition"],
            "executor": item["executor"],
            "prompt_key": item["prompt"],
            "prompt_sha256": digest(prompt),
            "harness_metrics_all_cells": metrics(rows),
        }
        store.put(name, summary)
        summaries[prefix] = summary
    after = execution.refresh_key()
    store.put(
        f"hosted-key-usage/{execution.session}-after.json",
        {"project_dedicated_key_total_usd": str(after), "timestamp_utc": utc_now()},
    )
    return {
        "status": "completed",
        "runs": summaries,
        "largest_request_input_tokens": contexts,
        "project_hosted_spend_usd": str(after),
        "costs": execution.accounts(),
        "note": (
            "harness_metrics_all_cells counts the hand-labelled undecided cell against `expected`; "
            "the panel's strata are scored offline by scripts/g34_analysis.py."
        ),
    }


def run(root: Path, env_file: Path) -> int:
    store = Store(root)
    with Store(root.parent).lock(), store.lock():
        plan, manifest = verify_runtime(store), store.get("manifest.json")
        if (
            plan["kind"] != "real_record_panel"
            or digest(plan) != manifest["plan_sha256"]
            or any(digest(store.get(n)) != h for n, h in manifest["input_sha256"].items())
        ):
            raise IntegrityError("real-record inputs or runtime changed")
        if store.exists("reports/complete.json"):
            print("Real-record run already complete; no new inference.", flush=True)
            return 0
        progress = Progress(store, manifest["initial_step_estimate"])
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
        monitor = Telemetry(store)
        execution.monitor = monitor
        monitor.start()
        try:
            result = execute(execution)
            execution.checkpoint()
            result["telemetry"] = monitor.stop()
            store.put("reports/complete.json", result)
            progress.emit("report", "real-record panel", "complete")
            print(f"Real-record run complete. {root}", flush=True)
            return 0
        except KeyboardInterrupt:
            print("Interrupted; run resume.py to continue.", flush=True)
            return 130
        except (RunPaused, IntegrityError, OSError, ValueError) as exc:
            store.put(
                "pauses/" + uuid.uuid4().hex + ".json",
                {"reason": str(exc), "timestamp_utc": utc_now()},
            )
            print(f"REAL-RECORD RUN PAUSED: {exc}. Resume with resume.py.", flush=True)
            return 2
        finally:
            execution.checkpoint()
            if monitor.thread.is_alive():
                monitor.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--registry-dir", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--panel-dir", type=Path, help="built by scripts/build_real_record_panel.py"
    )
    parser.add_argument(
        "--data-dir", type=Path, help="released data (exports/selected_prompts.jsonl)"
    )
    parser.add_argument("--hosted", nargs="*", default=["anthropic/claude-opus-5"])
    # a hosted-only run: reservation() holds a whole advertised context per request (Opus:
    # USD 5.10), so the hosted pass may need its own cap (2026-10-29)
    parser.add_argument("--executors", nargs="*", default=list(EXECUTORS))
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--max-cost-usd", default="5")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.action == "prepare":
        required = (
            args.benchmark_dir,
            args.registry_dir,
            args.env_file,
            args.panel_dir,
            args.data_dir,
        )
        if any(p is None for p in required):
            parser.error("prepare requires benchmark, registry, env file, panel and data dirs")
        manifest = prepare_realrecord(
            root,
            args.benchmark_dir,
            args.registry_dir,
            args.env_file,
            args.panel_dir,
            args.data_dir,
            hosted_models=args.hosted,
            executors=args.executors,
            max_seconds=args.max_hours * 3600,
            max_cost_usd=args.max_cost_usd,
        )
        print(
            f"Prepared {root}: {len(manifest['runs'])} passes x "
            f"{len(manifest['profile_order'])} sub-profiles = {manifest['logical_calls']} calls",
            flush=True,
        )
        return 0
    return run(root, (args.env_file or Path(Store(root).get("plan.json")["env_file"])).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
