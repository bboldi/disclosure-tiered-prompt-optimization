"""Local-only measured recovery soak; repetitions are engineering evidence, not new samples."""

from __future__ import annotations

import argparse
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any

from ..runner import RunPaused, utc_now
from ..storage import IntegrityError, Store
from .campaign import Campaign, prepare, profiles_from, verify_runtime
from .conditions import NAIVE, selected_profiles
from .execution import Execution
from .progress import Progress
from .telemetry import Telemetry


def run(root: Path, env_file: Path, exercise: bool) -> int:
    store = Store(root)
    with Store(root.parent).lock(), store.lock():
        plan = verify_runtime(store)
        if store.exists("reports/complete.json"):
            print("Soak already complete; zero new inference.", flush=True)
            return 0
        condition = next(
            c
            for c in store.get("inputs/conditions.json")["local"]
            if c["id"] == plan["condition_id"]
        )
        profiles = selected_profiles(profiles_from(store), "pilot_development", 6)
        progress = Progress(store, {"soak": plan["max_attempts"], "report": 1})
        execution = Execution(
            store,
            env_file,
            max_seconds=plan["max_seconds"],
            max_cost_usd="1",
            max_attempts=plan["max_attempts"],
            study_root=root.parent,
            runtime=root / "runtime",
            progress=progress,
        )
        monitor = Telemetry(store, interval=1)
        execution.monitor = monitor
        campaign = Campaign(execution)

        def hook(boundary: str, key: str) -> None:
            if not exercise:
                return
            count = len(store.names("work/*/result.json"))
            if (
                boundary == "committed"
                and count == 3
                and not store.exists("injections/graceful.json")
            ):
                store.put(
                    "injections/graceful.json",
                    {"timestamp_utc": utc_now(), "boundary": boundary, "key": key},
                )
                raise KeyboardInterrupt
            if (
                boundary == "intent"
                and count == 6
                and not store.exists("injections/hard-intent.json")
            ):
                store.put(
                    "injections/hard-intent.json",
                    {"timestamp_utc": utc_now(), "boundary": boundary, "key": key},
                )
                os.kill(os.getpid(), signal.SIGKILL)
            if (
                boundary == "before_commit"
                and count == 9
                and not store.exists("injections/hard-response.json")
            ):
                store.put(
                    "injections/hard-response.json",
                    {"timestamp_utc": utc_now(), "boundary": boundary, "key": key},
                )
                os.kill(os.getpid(), signal.SIGKILL)

        execution.hook = hook
        prior_observed = 0.0
        for folder in (root / "sessions").glob("*"):
            if folder.name != execution.session:
                files = sorted(folder.glob("[0-9]*.json"))
                if files:
                    prior_observed += store.get(str(files[-1].relative_to(root)))["elapsed_seconds"]
        monitor.start()
        try:
            for index in range(plan["max_attempts"]):
                observed = prior_observed + time.perf_counter() - execution.started
                if observed >= plan["target_seconds"]:
                    report: dict[str, Any] = {
                        "status": "completed",
                        "observed_running_seconds": observed,
                        "accounted_seconds_including_unknown_bounds": plan["max_seconds"]
                        - execution.remaining_seconds(),
                        "completed_evaluations": len(store.names("work/*/result.json")),
                        "attempts": len(store.names("work/*/attempts/*/request.json")),
                        "purpose": "Repeated public pilot profiles; recovery/resource evidence only; no model selection or independent-sample claim.",
                        "costs": execution.accounts(),
                    }
                    store.put("reports/complete.json", report)
                    print(report, flush=True)
                    return 0
                campaign.evaluate(
                    f"soak/{index:05d}", condition, NAIVE, [profiles[index % len(profiles)]]
                )
            raise RunPaused("soak work ceiling reached before duration target")
        except KeyboardInterrupt:
            print("Graceful interruption; rerun resume.py to continue.", flush=True)
            return 130
        except (RunPaused, IntegrityError, OSError, ValueError) as exc:
            store.put(
                f"pauses/{uuid.uuid4().hex}.json", {"reason": str(exc), "timestamp_utc": utc_now()}
            )
            print(f"PAUSED: {exc}", flush=True)
            return 2
        finally:
            execution.checkpoint()
            monitor.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--registry-dir", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--exercise-recovery", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.benchmark_dir is None or args.registry_dir is None:
            parser.error("prepare requires benchmark and registry directories")
        prepare(
            args.run_dir,
            args.benchmark_dir,
            args.registry_dir,
            {
                "kind": "local_recovery_soak",
                "runner_module": "promptbench.live.soak",
                "condition_id": "granite4.2:30b/off",
                "target_seconds": 1800,
                "max_seconds": 2400,
                "max_attempts": 1800,
            },
        )
        print("Prepared frozen soak runtime. Start with this run's resume.py.")
        return 0
    return run(args.run_dir.resolve(), args.env_file.resolve(), args.exercise_recovery)


if __name__ == "__main__":
    raise SystemExit(main())
