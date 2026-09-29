"""Fixed paired scaffolding controls on the sealed test panel; no prompt optimization."""

from __future__ import annotations

import argparse
import math
import uuid
from pathlib import Path
from typing import Any

from ..benchmark.model import BenchmarkProfile
from ..config import ABLATION_EXECUTORS as EXECUTORS
from ..domain import metrics
from ..runner import RunPaused, utc_now
from ..storage import IntegrityError, Store, digest
from .campaign import Campaign, prepare, profiles_from, verify_runtime
from .conditions import NAIVE, executor_request, selected_profiles
from .execution import Execution
from .progress import Progress
from .taskbudget import SCOPE, scope_usage
from .telemetry import Telemetry
from .transport import OLLAMA

MAX_ATTEMPTS = 1920 * 3
# Existing journal constructor accepts at most 72 hours; retain that operational guard.
CONTROL_WATCHDOG_SECONDS = min(259200, MAX_ATTEMPTS * 330 + 3600)

VARIANTS = ("naive", "output_scaffold", "structured_inventory", "schema_enum", "neutral")
SCAFFOLD_APPEND = (
    ' Also include "advisory_decisions", an array with one object for each supplied advisory: '
    '{"id":"CVE-...","applicable":true} or {"id":"CVE-...","applicable":false}. '
    "Include every supplied ID exactly once. The applicable_cves array must contain exactly "
    "the IDs marked true."
)
# Whole neutral sentences plus spaces match bytes without truncating a sentence into an instruction.
NEUTRAL_SENTENCE = " This task is part of a collection of independent items."
NEUTRAL_APPEND = (NEUTRAL_SENTENCE * (len(SCAFFOLD_APPEND) // len(NEUTRAL_SENTENCE))).ljust(
    len(SCAFFOLD_APPEND)
)
PROTOCOL = (
    "docs/v2_addressing_limitations.md",
    "docs/adr/0033-freeze-researcher-optimizer-choice-and-task3-handoff.md",
    "docs/adr/0035-freeze-task3-fixed-scaffolding-controls.md",
)


def control_design(
    profiles: list[BenchmarkProfile], local: list[dict[str, Any]], verdict: dict[str, Any]
) -> dict[str, Any]:
    if verdict.get("passed") is not True or tuple(verdict.get("promoted", [])) != EXECUTORS:
        raise IntegrityError("controls require the two gate-promoted Executors")
    if (
        len(profiles) != 192
        or len({p.id for p in profiles}) != 192
        or any(p.partition != "test" for p in profiles)
    ):
        raise IntegrityError("controls require all 192 unique sealed-test profiles")
    by_id = {c["id"]: c for c in local}
    prompts = {
        variant: NAIVE
        + (
            SCAFFOLD_APPEND
            if variant == "output_scaffold"
            else NEUTRAL_APPEND
            if variant == "neutral"
            else ""
        )
        for variant in VARIANTS
    }
    conditions, schedule = {}, []
    for identifier in EXECUTORS:
        original = by_id[identifier]
        if original["provider"] != "ollama" or original["think"] is not False:
            raise IntegrityError("controls require local thinking-off conditions")
        for variant in VARIANTS:
            key = f"scaffolding/{identifier}/{variant}"
            conditions[key] = {
                **original,
                "num_ctx": 16384,
                "num_predict": 4096,
                "timeout_seconds": 300,
                "output_scaffold": variant == "output_scaffold",
                "structured_inventory": variant == "structured_inventory",
                "schema_enum": variant == "schema_enum",
            }
        for i, profile in enumerate(profiles):
            offset = i % len(VARIANTS)
            for variant in VARIANTS[offset:] + VARIANTS[:offset]:
                prefix = f"scaffolding/{identifier}/{variant}"
                schedule.append(
                    {
                        "prefix": prefix,
                        "variant": variant,
                        "profile_id": profile.id,
                        "body_sha256": digest(
                            executor_request(conditions[prefix], prompts[variant], profile)
                        ),
                    }
                )
    return {
        "profile_ids": [p.id for p in profiles],
        "profile_sha256": digest([p.record() for p in profiles]),
        "prompts": prompts,
        "conditions": conditions,
        "schedule": schedule,
        "logical_calls": len(schedule),
        "optimizer_calls": 0,
        "neutral_matching": {
            "unit": "ASCII characters/bytes",
            "append_length": len(SCAFFOLD_APPEND),
        },
    }


def prepare_controls(
    root: Path, benchmark: Path, registry: Path, calibration: Path, env_file: Path
) -> None:
    if root.exists():
        raise IntegrityError("controls require a fresh run identity")
    parent = Store(calibration)
    report, manifest = parent.get("reports/complete.json"), parent.get("manifest.json")
    if parent.get("plan.json")["benchmark_sha256"] != digest(Store(benchmark).get("manifest.json")):
        raise IntegrityError("controls must use the benchmark admitted by calibration")
    # Validate gate before creating the new identity. Full input checks follow preparation.
    if (
        report["verdict"].get("passed") is not True
        or tuple(report["verdict"]["promoted"]) != EXECUTORS
    ):
        raise IntegrityError("controls require the passing calibration promotion")
    # Finite journal watchdog follows physical request bounds, not a research-time allocation.
    prepare(
        root,
        benchmark,
        registry,
        {
            "kind": "fixed_scaffolding_controls",
            "runner_module": "promptbench.live.scaffolding",
            "budget_scope": SCOPE,
            "max_seconds": CONTROL_WATCHDOG_SECONDS,
            "max_cost_usd": "0.01",
            "max_attempts": MAX_ATTEMPTS,
            "input_partitions": ["test"],
            "initial_step_estimate": {"scaffolding": 1920, "report": 1},
            "env_file": str(env_file.resolve()),
            "benchmark_root": str(benchmark.resolve()),
            "calibration_root": str(calibration.resolve()),
            "prior_scope_usage": scope_usage(root.parent),
            "time_guard_basis": "physical-attempt-derived watchdog clipped to existing journal 72h maximum; continuation required if reached, not a research hour stop",
        },
    )
    store = Store(root)
    store.put(
        "inputs/calibration.json",
        {"manifest": manifest, "verdict": report["verdict"], "report_sha256": digest(report)},
    )
    project = Path(__file__).resolve().parents[3]
    store.put(
        "inputs/protocol.json",
        {name: (project / name).read_text() for name in PROTOCOL if (project / name).exists()},
    )
    profiles = selected_profiles(profiles_from(store), "test", 192)
    design = control_design(
        profiles, store.get("inputs/conditions.json")["local"], report["verdict"]
    )
    store.put(
        "manifest.json",
        {
            **design,
            "plan_sha256": digest(store.get("plan.json")),
            "input_sha256": {n: digest(store.get(n)) for n in store.names("inputs/*.json")},
        },
    )


def distribution(values: list[int | float | None]) -> dict[str, Any]:
    known = sorted(v for v in values if v is not None)
    return {
        "values": values,
        "known": len(known),
        "missing": len(values) - len(known),
        "total": sum(known) if len(known) == len(values) else None,
        "known_total": sum(known),
        "mean_known": sum(known) / len(known) if known else None,
        "p90_known": known[math.ceil(0.9 * len(known)) - 1] if known else None,
    }


def observations(
    store: Store, prefix: str, profiles: list[BenchmarkProfile]
) -> list[dict[str, Any]]:
    result = []
    for profile in profiles:
        key = prefix + "/" + profile.id
        work = "work/" + store.get("logical-keys/" + digest(key) + ".json")["spec_sha256"]
        committed = store.get(work + "/result.json")
        attempt = committed["attempt"]
        response = (
            store.get(attempt + "/response.json")
            if store.exists(attempt + "/response.json")
            else {}
        )
        normalized = (
            store.get(attempt + "/normalized.json")
            if store.exists(attempt + "/normalized.json")
            else {}
        )
        result.append(
            {
                "profile_id": profile.id,
                "status": committed["status"],
                "input_tokens": normalized.get("input_tokens"),
                "output_tokens": normalized.get("output_tokens"),
                "latency_seconds": response["duration_ns"] / 1e9
                if response.get("duration_ns") is not None
                else None,
                "attempt": attempt,
                "physical_attempts": len(store.names(work + "/attempts/*/request.json")),
                "physical_attempt_latency_seconds": [
                    store.get(n)["duration_ns"] / 1e9
                    for n in store.names(work + "/attempts/*/response.json")
                ],
            }
        )
    return result


def paired_summary(
    rows: list[dict[str, Any]], observed: list[dict[str, Any]], naive: list[dict[str, Any]]
) -> dict[str, Any]:
    if [o["profile_id"] for o in observed] != [o["profile_id"] for o in naive]:
        raise IntegrityError("control contrasts require identical profile order")
    fields = ("input_tokens", "output_tokens", "latency_seconds")
    return {
        "metrics": metrics(rows),
        "rows": rows,
        "observations": observed,
        "usage_and_latency": {
            field: distribution([o[field] for o in observed]) for field in fields
        },
        "paired_difference_from_naive": {
            field: distribution(
                [
                    o[field] - n[field] if o[field] is not None and n[field] is not None else None
                    for o, n in zip(observed, naive, strict=True)
                ]
            )
            for field in fields
        },
        "pairing_note": "All committed outcomes, including invalid output. Missing measurements remain null. Latency contrasts use committed attempts; physical attempts are also retained.",
    }


def execute(execution: Execution) -> dict[str, Any]:
    store, campaign = execution.store, Campaign(execution)
    manifest = store.get("manifest.json")
    by_id = {p.id: p for p in profiles_from(store)}
    profiles = [by_id[i] for i in manifest["profile_ids"]]
    design = control_design(
        profiles,
        store.get("inputs/conditions.json")["local"],
        store.get("inputs/calibration.json")["verdict"],
    )
    if any(manifest[k] != value for k, value in design.items()):
        raise IntegrityError("fixed control manifest or sealed data changed")
    before = execution.refresh_key()
    store.put(
        f"hosted-key-usage/{execution.session}-before.json",
        {"project_dedicated_key_total_usd": str(before), "timestamp_utc": utc_now()},
    )
    print(
        f"PROJECT HOSTED SPEND before local controls: USD {before} total dedicated key; no hosted inference",
        flush=True,
    )
    for item in manifest["schedule"]:
        campaign.evaluate(
            item["prefix"],
            manifest["conditions"][item["prefix"]],
            manifest["prompts"][item["variant"]],
            [by_id[item["profile_id"]]],
        )
    if campaign.last_local:
        campaign.metadata(
            "unload",
            OLLAMA + "/api/generate",
            {"model": campaign.last_local, "keep_alive": 0, "stream": False},
        )
        campaign.last_local = None
    after = execution.refresh_key()
    store.put(
        f"hosted-key-usage/{execution.session}-after.json",
        {"project_dedicated_key_total_usd": str(after), "timestamp_utc": utc_now()},
    )
    print(f"PROJECT HOSTED SPEND after local controls: USD {after} total dedicated key", flush=True)
    summaries = {}
    for identifier in EXECUTORS:
        naive = observations(store, f"scaffolding/{identifier}/naive", profiles)
        for variant in VARIANTS:
            prefix = f"scaffolding/{identifier}/{variant}"
            rows = [
                store.get("evaluations/" + digest(prefix + "/" + p.id) + ".json")["row"]
                for p in profiles
            ]
            summary = paired_summary(rows, observations(store, prefix, profiles), naive)
            summaries[prefix] = summary
            store.put("conditions/" + digest(prefix) + ".json", summary)
    return {
        "conditions": summaries,
        "logical_calls": len(manifest["schedule"]),
        "project_hosted_spend_usd": str(after),
        "optimizer_calls": 0,
        "next_action": "closeout; the campaign runner is promptbench.live.study",
    }


def run(root: Path, env_file: Path) -> int:
    store = Store(root)
    with Store(root.parent).lock(), store.lock():
        plan, manifest = verify_runtime(store), store.get("manifest.json")
        if (
            plan["kind"] != "fixed_scaffolding_controls"
            or digest(plan) != manifest["plan_sha256"]
            or any(digest(store.get(n)) != h for n, h in manifest["input_sha256"].items())
        ):
            raise IntegrityError("fixed control inputs or runtime changed")
        if store.exists("reports/complete.json"):
            print("Fixed controls already complete; no new inference.", flush=True)
            return 0
        progress = Progress(store, plan["initial_step_estimate"])
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
            result["scope_usage"] = scope_usage(root.parent)
            store.put("reports/complete.json", result)
            progress.emit("report", "fixed paired controls", "complete")
            return 0
        except (RunPaused, IntegrityError, OSError, ValueError) as exc:
            store.put(
                "pauses/" + uuid.uuid4().hex + ".json",
                {"reason": str(exc), "timestamp_utc": utc_now()},
            )
            print(f"FIXED CONTROLS PAUSED: {exc}", flush=True)
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
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.action == "prepare":
        if any(
            p is None
            for p in (args.benchmark_dir, args.registry_dir, args.calibration_dir, args.env_file)
        ):
            parser.error("prepare requires benchmark, registry, passing calibration and env file")
        prepare_controls(
            root, args.benchmark_dir, args.registry_dir, args.calibration_dir, args.env_file
        )
        return 0
    return run(root, (args.env_file or Path(Store(root).get("plan.json")["env_file"])).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
