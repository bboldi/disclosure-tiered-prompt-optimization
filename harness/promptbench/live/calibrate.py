"""Frozen fixed-prompt calibration on pilot-development data only; no prompt search."""

from __future__ import annotations

import argparse
import json
import math
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from ..benchmark.model import BenchmarkProfile, score
from ..config import CALIBRATION_CONDITIONS as LOCAL_IDS
from ..config import CALIBRATION_REFERENCE as REFERENCE
from ..domain import metrics
from ..runner import RunPaused, utc_now
from ..storage import IntegrityError, Store, digest
from .adapters import reservation
from .calibration_recovery import adopt_local_calibration
from .campaign import Campaign, prepare, profiles_from, verify_runtime
from .conditions import NAIVE, executor_request, optimizer_request, selected_profiles
from .progress import Progress
from .taskbudget import CALIBRATION_GPU_CEILING_SECONDS, SCOPE, ScopedExecution, scope_usage
from .telemetry import Telemetry
from .transport import OLLAMA

GATE = {
    "naive_f1_min": 0.30,
    "naive_f1_max": 0.70,
    "hosted_f1_min": 0.90,
    "coverage_min": 0.95,
    "nonzero_batch_fraction_min": 0.80,
    "families_min": 2,
}


def reference_request(
    condition: dict[str, Any], prompt: str, profile: BenchmarkProfile
) -> tuple[dict[str, Any], dict[str, Any]]:
    view = executor_request(
        {**condition, "think": None, "num_ctx": 16384, "num_predict": 4096}, prompt, profile
    )
    admitted, body = optimizer_request(condition, {})
    body.update(messages=view["messages"], max_tokens=4096, temperature=0)
    body["response_format"]["json_schema"] = {
        "name": "applicable_cves",
        "strict": True,
        "schema": view["format"],
    }
    admitted["reservation_usd"] = str(reservation(body, condition["endpoint"]))
    return admitted, body


def summarize_condition(
    store: Store, prefix: str, condition: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    durations, loads = [], []
    for name in store.names("work/*/spec.json"):
        spec = store.get(name)
        if spec["key"].startswith(prefix + "/"):
            for response_name in store.names(
                name.removesuffix("spec.json") + "attempts/*/response.json"
            ):
                response = store.get(response_name)
                durations.append(response["duration_ns"])
                try:
                    raw = json.loads(response["body_text"])
                    loads.append(raw.get("load_duration"))
                except ValueError:
                    loads.append(None)
    batches = [
        dict(Counter(c for row in rows[i : i + 6] for c in row["categories"]))
        for i in range(0, len(rows), 6)
    ]
    nonzero = sum(bool(batch) for batch in batches)
    return {
        "condition_id": condition["id"],
        "family": condition.get("family"),
        "metrics": metrics(rows),
        "rows": rows,
        "duration_ns": durations,
        "load_duration_ns": loads,
        "latency_p90_seconds": sorted(durations)[math.ceil(0.9 * len(durations)) - 1] / 1e9
        if durations
        else None,
        "six_profile_error_counts": batches,
        "nonzero_batch_fraction": nonzero / len(batches) if batches else 0,
    }


def calibration_gate(local: dict[str, dict[str, Any]], hosted: dict[str, Any]) -> dict[str, Any]:
    eligible, reasons = [], {}
    complete = set(local) == set(LOCAL_IDS) and len(hosted["rows"]) == 48
    for identifier in LOCAL_IDS:
        cell = local.get(identifier)
        if cell is None:
            complete = False
            continue
        naive, expert = cell["naive"], cell["historical_expert"]
        complete = complete and len(naive["rows"]) == 48 and len(expert["rows"]) == 48
        f1 = naive["metrics"]["failure_aware_lower_bound"]["micro_f1"]
        failures = []
        if not GATE["naive_f1_min"] <= f1 <= GATE["naive_f1_max"]:
            failures.append("naive_f1_outside_frozen_headroom_range")
        if min(naive["metrics"]["coverage"], expert["metrics"]["coverage"]) < GATE["coverage_min"]:
            failures.append("coverage_below_gate")
        if naive["nonzero_batch_fraction"] < GATE["nonzero_batch_fraction_min"]:
            failures.append("too_few_nonzero_feedback_batches")
        if naive["latency_p90_seconds"] is None:
            failures.append("missing_timing")
        reasons[identifier] = failures
        if not failures:
            eligible.append(naive)
    selected, families = [], set()
    for cell in sorted(eligible, key=lambda c: (c["latency_p90_seconds"], c["condition_id"])):
        if cell["family"] not in families:
            selected.append(cell["condition_id"])
            families.add(cell["family"])
    reference_passed = (
        hosted["metrics"]["failure_aware_lower_bound"]["micro_f1"] >= GATE["hosted_f1_min"]
    )
    passed = complete and reference_passed and len(families) >= GATE["families_min"]
    return {
        "passed": passed,
        "complete_screen": complete,
        "criteria": GATE,
        "hosted_reference_passed": reference_passed,
        "eligible_conditions": [c["condition_id"] for c in eligible],
        "condition_exclusion_reasons": reasons,
        "promoted": selected[:2] if passed else [],
        "next_action": "controls and campaign"
        if passed
        else "rebuild benchmark and rerun without lowering gates",
    }


def prepare_calibration(
    root: Path,
    benchmark: Path,
    registry: Path,
    expert: Path,
    env_file: Path,
    parent_root: Path | None = None,
) -> None:
    if root.exists():
        raise IntegrityError("calibration requires a fresh run identity")
    used = scope_usage(root.parent)
    remaining = CALIBRATION_GPU_CEILING_SECONDS - used["charged_gpu_seconds"]
    if remaining < 60:
        raise RunPaused("pre-pilot calibration time allocation exhausted")
    prepare(
        root,
        benchmark,
        registry,
        {
            "kind": "calibration",
            "runner_module": "promptbench.live.calibrate",
            "budget_scope": SCOPE,
            "scope_gpu_ceiling_seconds": CALIBRATION_GPU_CEILING_SECONDS,
            "max_seconds": remaining,
            "max_cost_usd": "2",
            "max_attempts": 1872,
            "input_partitions": ["pilot_development"],
            "local_ids": list(LOCAL_IDS),
            "profiles": 48,
            "initial_step_estimate": {"calibration": 624, "report": 1},
            "gate": GATE,
            "env_file": str(env_file.resolve()),
            "benchmark_root": str(benchmark.resolve()),
            "prior_scope_usage": used,
        },
    )
    store = Store(root)
    store.put("inputs/prompts.json", {"naive": NAIVE, "historical_expert": expert.read_text()})
    project = Path(__file__).resolve().parents[3]
    store.put(
        "inputs/protocol.json",
        {
            name: (project / name).read_text()
            for name in (
                "docs/v2_addressing_limitations.md",
                "docs/adr/0026-freeze-prestudy-calibration-and-resource-gates.md",
                "docs/adr/0027-continue-calibration-with-measured-time-allocation.md",
                "docs/adr/0028-revise-prestudy-resource-ceilings.md",
                "docs/adr/0029-preserve-naive-prompt-with-exact-schema-overlap-exception.md",
                "docs/adr/0030-recalibrate-with-weighted-near-misses.md",
            )
            if (project / name).exists()
        },
    )
    profiles = selected_profiles(profiles_from(store), "pilot_development", 48)
    conditions = store.get("inputs/conditions.json")
    selected = {
        c["id"]: {**c, "num_predict": 4096 if c["think"] else 512}
        for c in conditions["local"]
        if c["id"] in LOCAL_IDS
    }
    if set(selected) != set(LOCAL_IDS):
        raise IntegrityError("required calibration model conditions are missing from registry")
    reference = next(c for c in conditions["hosted"] if c["id"] == REFERENCE)
    store.put(
        "manifest.json",
        {
            "plan_sha256": digest(store.get("plan.json")),
            "input_sha256": {n: digest(store.get(n)) for n in store.names("inputs/*.json")},
            "profile_ids": [p.id for p in profiles],
            "profile_sha256": digest([p.record() for p in profiles]),
            "local": selected,
            "hosted": {**reference, "schema_enum": False, "structured_inventory": False},
            "gate": GATE,
            "local_calls": 576,
            "hosted_calls": 48,
            "total_calls": 624,
        },
    )

    if parent_root is not None:
        adopt_local_calibration(parent_root, root)


def execute(execution: ScopedExecution) -> dict[str, Any]:
    store, campaign = execution.store, Campaign(execution)
    manifest, prompts = store.get("manifest.json"), store.get("inputs/prompts.json")
    by_id = {p.id: p for p in profiles_from(store)}
    profiles = [by_id[i] for i in manifest["profile_ids"]]
    if (
        any(p.partition != "pilot_development" for p in profiles)
        or digest([p.record() for p in profiles]) != manifest["profile_sha256"]
    ):
        raise IntegrityError("calibration must use the frozen pilot-development sample only")
    local: dict[str, dict[str, Any]] = {}
    for identifier in LOCAL_IDS:
        condition = manifest["local"][identifier]
        local[identifier] = {}
        for prompt_name, prompt in prompts.items():
            prefix = f"calibration/{identifier}/{prompt_name}"
            name = "conditions/" + digest(prefix) + ".json"
            if not store.exists(name):
                rows = campaign.evaluate(prefix, condition, prompt, profiles)
                store.put(name, summarize_condition(store, prefix, condition, rows))
            local[identifier][prompt_name] = store.get(name)
            summary = local[identifier][prompt_name]
            print(
                f"CALIBRATION {identifier} {prompt_name}: F1={summary['metrics']['failure_aware_lower_bound']['micro_f1']:.4f}, coverage={summary['metrics']['coverage']:.3f}",
                flush=True,
            )
    if campaign.last_local:
        campaign.metadata(
            "unload",
            OLLAMA + "/api/generate",
            {"model": campaign.last_local, "keep_alive": 0, "stream": False},
        )
        campaign.last_local = None
    before = execution.refresh_key()
    store.put(
        f"hosted-key-usage/{execution.session}-before.json",
        {"usage_usd": str(before), "timestamp_utc": utc_now()},
    )
    print(
        f"HOSTED REFERENCE: key usage before USD {before}; calibration combined cap USD 2",
        flush=True,
    )
    reference, rows, attempts = manifest["hosted"], [], []
    prefix = "calibration/" + REFERENCE + "/naive"
    try:
        campaign.check(reference)
        for profile in profiles:
            condition, body = reference_request(reference, prompts["naive"], profile)
            result = execution.call(
                prefix + "/" + profile.id,
                condition,
                body,
                role="executor",
                allowed_ids={a.id for a in profile.advisories},
            )
            row = score(profile, result["value"], result["status"])
            store.put(
                "evaluations/" + digest(prefix + "/" + profile.id) + ".json",
                {
                    "key": prefix + "/" + profile.id,
                    "condition_id": REFERENCE,
                    "prompt_sha256": digest(prompts["naive"]),
                    "row": row,
                },
            )
            rows.append(row)
            if result["response_sha256"] is not None:
                attempts.append(result["attempt"])
        for attempt in attempts:
            campaign.reconcile_generation(attempt, reference["canonical_model"])
    finally:
        after = execution.refresh_key()
        store.put(
            f"hosted-key-usage/{execution.session}-after.json",
            {"usage_usd": str(after), "delta_usd": str(after - before), "timestamp_utc": utc_now()},
        )
        print(
            f"HOSTED REFERENCE: key usage after USD {after}; block delta USD {after - before}",
            flush=True,
        )
    hosted = summarize_condition(store, prefix, reference, rows)
    verdict = calibration_gate(local, hosted)
    return {
        "benchmark_root": execution.store.get("plan.json")["benchmark_root"],
        "local": local,
        "hosted": hosted,
        "verdict": verdict,
        "accounted_running_seconds": execution.max_seconds - execution.remaining_seconds(),
        "scope_usage": scope_usage(execution.study_root),
    }


def run(root: Path, env_file: Path) -> int:
    store = Store(root)
    with Store(root.parent).lock(), store.lock():
        plan = verify_runtime(store)
        manifest = store.get("manifest.json")
        if digest(plan) != manifest["plan_sha256"] or any(
            digest(store.get(n)) != h for n, h in manifest["input_sha256"].items()
        ):
            raise IntegrityError("calibration frozen inputs changed")
        if store.exists("reports/complete.json"):
            print("Calibration already complete; no new inference.", flush=True)
            return 0
        progress = Progress(store, plan["initial_step_estimate"])
        execution = ScopedExecution(
            store,
            env_file,
            max_seconds=plan["max_seconds"],
            max_cost_usd=plan["max_cost_usd"],
            max_attempts=plan["max_attempts"],
            study_root=root.parent,
            runtime=root / "runtime",
            progress=progress,
        )
        progress.clock = lambda: {"remaining_seconds": max(0, execution.remaining_seconds())}
        monitor = Telemetry(store)
        execution.monitor = monitor
        monitor.start()
        try:
            result = execute(execution)
            execution.checkpoint()
            monitor.stop()
            store.put("reports/complete.json", result)
            progress.emit(
                "report", "calibration gate", "PASS" if result["verdict"]["passed"] else "FAIL"
            )
            return 0
        except (RunPaused, IntegrityError, OSError, ValueError) as exc:
            store.put(
                "pauses/" + uuid.uuid4().hex + ".json",
                {"reason": str(exc), "timestamp_utc": utc_now()},
            )
            print(f"CALIBRATION PAUSED: {exc}", flush=True)
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
    parser.add_argument("--expert-prompt", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--parent-run", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.action == "prepare":
        if any(
            x is None
            for x in (args.benchmark_dir, args.registry_dir, args.expert_prompt, args.env_file)
        ):
            parser.error("prepare requires benchmark, registry, expert prompt and env file")
        prepare_calibration(
            root,
            args.benchmark_dir,
            args.registry_dir,
            args.expert_prompt,
            args.env_file,
            args.parent_run,
        )
        return 0
    env_file = args.env_file or Path(Store(root).get("plan.json")["env_file"])
    return run(root, env_file.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
