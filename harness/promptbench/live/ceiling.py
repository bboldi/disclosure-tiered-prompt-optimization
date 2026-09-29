"""Sealed-panel reference rows: hosted or local models do the task with the unchanged naive prompt.

One pass per model, no optimization. Hosted rows are the ceiling references; local
rows are naive baselines at the campaign output cap. Metadata reconciliation is deferred to
`study reconcile`.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any

from ..benchmark.model import BenchmarkProfile, score
from ..domain import metrics
from ..runner import RunPaused, utc_now
from ..storage import IntegrityError, Store, digest
from .adapters import reservation
from .campaign import Campaign, prepare, profiles_from, verify_runtime
from .conditions import NAIVE, executor_request, optimizer_request
from .execution import Execution
from .progress import Progress
from .telemetry import Telemetry

PANELS = ("test", "temporal", "product_heldout")
PROTOCOL = ("docs/adr/0040-hosted-executor-ceiling-and-opus-optimizer-reference.md",)


def ceiling_request(
    condition: dict[str, Any], prompt: str, profile: BenchmarkProfile
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Same contract as the calibration hosted reference; temperature only where supported."""
    view = executor_request(
        {**condition, "think": None, "num_ctx": 16384, "num_predict": 4096}, prompt, profile
    )
    admitted, body = optimizer_request(condition, {})
    body.update(messages=view["messages"], max_tokens=4096)
    supported = condition["endpoint"].get("supported_parameters", [])
    if "temperature" in supported:
        body["temperature"] = 0
    body["response_format"]["json_schema"] = {
        "name": "applicable_cves",
        "strict": True,
        "schema": view["format"],
    }
    admitted["reservation_usd"] = str(reservation(body, condition["endpoint"]))
    admitted["temperature_supported"] = "temperature" in supported
    return admitted, body


def prepare_ceiling(
    root: Path,
    benchmark: Path,
    registry: Path,
    env_file: Path,
    *,
    models: list[str],
    max_seconds: float,
    max_cost_usd: str,
    endpoint_overrides: dict[str, str] | None = None,
    output_tokens: int = 2048,
) -> dict[str, Any]:
    if root.exists():
        raise IntegrityError("ceiling run requires a fresh identity")
    prepare(
        root,
        benchmark,
        registry,
        {
            "kind": "hosted_ceiling",
            "runner_module": "promptbench.live.ceiling",
            "max_seconds": max_seconds,
            "max_cost_usd": max_cost_usd,
            "max_attempts": 5000,
            "input_partitions": list(PANELS),
            "initial_step_estimate": {"ceiling": 1, "report": 1},
            "env_file": str(env_file.resolve()),
            "benchmark_root": str(benchmark.resolve()),
        },
    )
    store = Store(root)
    project = Path(__file__).resolve().parents[3]
    store.put(
        "inputs/protocol.json",
        {n: (project / n).read_text() for n in PROTOCOL if (project / n).exists()},
    )
    store.put("inputs/prompts.json", {"naive": NAIVE})
    if endpoint_overrides:
        from .conditions import hosted_conditions

        conditions = store.get("inputs/conditions.json")
        conditions["hosted"] = hosted_conditions(
            store.get("inputs/model_registry.json"), endpoint_overrides
        )
        (root / "inputs/conditions.json").unlink()
        store.put("inputs/conditions.json", conditions)
    conditions = store.get("inputs/conditions.json")
    hosted = {c["id"]: c for c in conditions["hosted"]}
    local = {c["id"]: c for c in conditions["local"]}
    if any(m not in hosted and m not in local for m in models):
        raise IntegrityError("ceiling model absent from the frozen registry")
    profiles = sorted(profiles_from(store), key=lambda p: p.id)
    panels = {panel: [p.id for p in profiles if p.partition == panel] for panel in PANELS}
    if any(not ids for ids in panels.values()):
        raise IntegrityError("sealed panel missing from inputs")
    manifest = {
        "models": models,
        "conditions": {
            m: (
                {**hosted[m], "schema_enum": False, "structured_inventory": False}
                if m in hosted
                else {
                    **local[m],
                    "num_predict": output_tokens,
                    "timeout_seconds": 300,
                    "schema_enum": False,
                    "structured_inventory": False,
                }
            )
            for m in models
        },
        "output_tokens_local": output_tokens,
        "panels": panels,
        "prompt_sha256": digest(NAIVE),
        "endpoint_overrides": endpoint_overrides or {},
        "logical_calls": len(models) * sum(len(v) for v in panels.values()),
        "initial_step_estimate": {
            "ceiling": len(models) * sum(len(v) for v in panels.values()),
            "report": 1,
        },
        "plan_sha256": digest(store.get("plan.json")),
        "input_sha256": {n: digest(store.get(n)) for n in store.names("inputs/*.json")},
    }
    store.put("manifest.json", manifest)
    return manifest


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
    for model in manifest["models"]:
        condition = manifest["conditions"][model]
        campaign.check(condition)
        summaries[model] = {}
        for panel, ids in manifest["panels"].items():
            prefix = f"ceiling/{model}/{panel}"
            name = "panels/" + digest(prefix) + ".json"
            if store.exists(name):
                summaries[model][panel] = store.get(name)
                continue
            rows = []
            for identifier in ids:
                profile = by_id[identifier]
                if condition["provider"] == "ollama":
                    admitted, body = condition, executor_request(condition, NAIVE, profile)
                else:
                    admitted, body = ceiling_request(condition, NAIVE, profile)
                result = execution.call(
                    f"{prefix}/{identifier}",
                    admitted,
                    body,
                    role="executor",
                    allowed_ids={a.id for a in profile.advisories},
                )
                row = score(profile, result["value"], result["status"])
                store.put(
                    "evaluations/" + digest(f"{prefix}/{identifier}") + ".json",
                    {
                        "key": f"{prefix}/{identifier}",
                        "condition_id": model,
                        "prompt_sha256": digest(NAIVE),
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
                "model": model,
                "panel": panel,
                "temperature_supported": admitted.get("temperature_supported", True),
                "metrics": metrics(rows),
                "by_stratum": {
                    s: metrics([r for r in rows if r["stratum"] == s])
                    for s in ("easy", "medium", "hard")
                },
            }
            store.put(name, summary)
            summaries[model][panel] = summary
    after = execution.refresh_key()
    store.put(
        f"hosted-key-usage/{execution.session}-after.json",
        {"project_dedicated_key_total_usd": str(after), "timestamp_utc": utc_now()},
    )
    return {
        "status": "completed",
        "models": manifest["models"],
        "panels": summaries,
        "project_hosted_spend_usd": str(after),
        "costs": execution.accounts(),
        "note": (
            "Hosted models doing the task with the unchanged naive prompt; one pass; "
            "temperature 0 only where the pinned endpoint supports it (see temperature_supported)."
        ),
    }


def run(root: Path, env_file: Path) -> int:
    store = Store(root)
    with Store(root.parent).lock(), store.lock():
        plan, manifest = verify_runtime(store), store.get("manifest.json")
        if (
            plan["kind"] != "hosted_ceiling"
            or digest(plan) != manifest["plan_sha256"]
            or any(digest(store.get(n)) != h for n, h in manifest["input_sha256"].items())
        ):
            raise IntegrityError("ceiling inputs or runtime changed")
        if store.exists("reports/complete.json"):
            print("Ceiling run already complete; no new inference.", flush=True)
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
            progress.emit("report", "ceiling", "complete")
            print(f"Ceiling run complete. {root}", flush=True)
            return 0
        except KeyboardInterrupt:
            print("Interrupted; run resume.py to continue.", flush=True)
            return 130
        except (RunPaused, IntegrityError, OSError, ValueError) as exc:
            store.put(
                "pauses/" + uuid.uuid4().hex + ".json",
                {"reason": str(exc), "timestamp_utc": utc_now()},
            )
            print(f"CEILING PAUSED: {exc}. Resume with resume.py.", flush=True)
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
    parser.add_argument("--models", nargs="*", default=["anthropic/claude-opus-5", "z-ai/glm-5.3"])
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--max-cost-usd", default="25")
    parser.add_argument(
        "--endpoint", action="append", default=[], help="MODEL=TAG alternative pinned endpoint"
    )
    parser.add_argument("--output-tokens", type=int, default=2048, help="local output cap")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.action == "prepare":
        if any(p is None for p in (args.benchmark_dir, args.registry_dir, args.env_file)):
            parser.error("prepare requires benchmark, registry and env file")
        manifest = prepare_ceiling(
            root,
            args.benchmark_dir,
            args.registry_dir,
            args.env_file,
            models=args.models,
            max_seconds=args.max_hours * 3600,
            max_cost_usd=args.max_cost_usd,
            endpoint_overrides=dict(item.split("=", 1) for item in args.endpoint) or None,
            output_tokens=args.output_tokens,
        )
        print(f"Prepared ceiling run: {manifest['logical_calls']} hosted calls. {root}/resume.py")
        return 0
    return run(root, (args.env_file or Path(Store(root).get("plan.json")["env_file"])).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
