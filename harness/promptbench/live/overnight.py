"""Run a frozen pilot and retain an automatic, explicitly partial-or-complete closeout."""

from __future__ import annotations

import argparse
import os
import signal
import uuid
from pathlib import Path
from typing import Any

from ..runner import utc_now
from ..storage import Store, atomic_write
from .pilot import run
from .process import invoke
from .reconstruct import reconstruct
from .transport import OPENROUTER, response_json


def key_snapshot(store: Store, session: str, point: str, env_file: Path) -> dict[str, Any]:
    attempt = f"operator/{session}/key-{point}"
    store.put(
        attempt + "/request.json",
        {
            "provider": "openrouter",
            "url": OPENROUTER + "/api/v1/key",
            "body": None,
            "timeout_seconds": 20,
        },
    )
    try:
        invoke(store, attempt, env_file, timeout=20, runtime=store.root / "runtime")
        data = response_json(store.get(attempt + "/response.json"))["data"]
        return {
            "available": True,
            "usage_usd": data.get("usage"),
            "label": data.get("label"),
            "timestamp_utc": utc_now(),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {
            "available": False,
            "usage_usd": None,
            "error": str(exc),
            "timestamp_utc": utc_now(),
        }


def supervise(root: Path, env_file: Path) -> int:
    store = Store(root)
    session = uuid.uuid4().hex
    with Store(root / "operator-supervisor").lock():
        store.put(
            f"operator/{session}/started.json",
            {
                "pid": os.getpid(),
                "started_utc": utc_now(),
                "purpose": "Authorized bounded pilot, then offline reconstruction. No automatic main-study launch.",
            },
        )
        before = key_snapshot(store, session, "before", env_file)
        error = None
        try:
            exit_code = run(root, env_file)
        except KeyboardInterrupt:
            exit_code, error = 130, "operator interruption"
        except Exception as exc:
            exit_code, error = 1, f"{type(exc).__name__}: {exc}"
        store.put(
            f"operator/{session}/runner-exit.json",
            {"exit_code": exit_code, "error": error, "timestamp_utc": utc_now()},
        )
        after = key_snapshot(store, session, "after", env_file)
        reconstruction = None
        reconstruction_error = None
        output = root / f"reconstruction-{session}"
        try:
            with Store(root.parent).lock(), store.lock():
                reconstruction = reconstruct(root, output)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reconstruction_error = str(exc)
        complete = (
            store.get("reports/complete.json") if store.exists("reports/complete.json") else None
        )
        closeout = {
            "exit_code": exit_code,
            "runner_error": error,
            "status": complete["status"] if complete else "paused_or_interrupted",
            "finished_utc": utc_now(),
            "key_before": before,
            "key_after": after,
            "reconstruction_directory": str(output) if reconstruction else None,
            "reconstruction_error": reconstruction_error,
            "report": complete,
        }
        store.put(f"operator/{session}/closeout.json", closeout)
        lines = [
            "# Overnight pilot report",
            "",
            f"Status: **{closeout['status']}**. Runner exit code: {exit_code}.",
            f"Recorded UTC: {closeout['finished_utc']}.",
            "",
        ]
        if reconstruction:
            lines += [
                f"- Committed work: {reconstruction['completed_work']}; physical attempts: {reconstruction['physical_attempts']}; unknown attempts: {reconstruction['unknown_attempts']}.",
                f"- Accounted running time: {reconstruction['running_seconds_accounted'] / 3600:.3f} hours.",
                f"- Reported inference cost: USD {reconstruction['costs']['reported_cost_usd']}; unresolved reservations: USD {reconstruction['costs']['reserved_unknown_usd']}.",
                f"- Raw-response reconstruction: passed; exports in `{output.name}/`.",
            ]
        else:
            lines.append(f"Reconstruction did not complete: {reconstruction_error}.")
        lines += [
            "",
            f"Dedicated-key usage after run: {after.get('usage_usd') if after['available'] else 'unavailable'} USD. Before-run snapshot: {before.get('usage_usd') if before['available'] else 'unavailable'} USD. These are provider snapshots; unresolved requests remain separate.",
            "",
        ]
        if error:
            lines += [f"Runner error: {error}.", ""]
        pauses = [store.get(name) for name in store.names("pauses/*.json")]
        if pauses:
            lines += ["Retained pause reasons:", ""] + [f"- {p['reason']}" for p in pauses] + [""]
        lines += [
            "Full selection, censoring and forecast decisions are in `reports/complete.json` if present, with stage evidence under `stages/`. A completed pilot may still be statistically inconclusive. No main study was launched.",
            "",
            "CPU/RAM and available resource samples are retained. Empty GPU measurements mean unavailable, not zero. Pi/OpenCode comparisons are unadmitted and are not results.",
            "",
            "To continue a paused run, use the exact frozen resume command in `OPERATOR.md`; existing results and cumulative limits are preserved.",
            "",
        ]
        atomic_write(root / "MORNING_REPORT.md", "\n".join(lines).encode(), immutable=False)
        print(f"Closeout saved: {root / 'MORNING_REPORT.md'}", flush=True)
        return exit_code if exit_code else (2 if reconstruction_error else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    return supervise(args.run_dir.resolve(), args.env_file.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
