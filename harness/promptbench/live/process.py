"""Hard request deadlines and process cleanup, independent of socket timeout behavior."""

from __future__ import annotations

import math
import os
import signal
import subprocess  # nosec B404 -- owned worker process, fixed argv and no shell
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from ..runner import utc_now
from ..storage import Store


def invoke(
    store: Store,
    attempt: str,
    env_file: Path,
    *,
    timeout: float,
    runtime: Path | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("worker deadline must be positive")
    argv = command or [
        sys.executable,
        "-m",
        "promptbench.live.worker",
        "--attempt-dir",
        str(store.root / attempt),
        "--env-file",
        str(env_file.resolve()),
        "--parent-pid",
        str(os.getpid()),
    ]
    cwd = runtime or Path(__file__).resolve().parents[2]
    environment = {"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
    store.put(
        attempt + "/worker-intent.json",
        {"command": argv, "cwd": str(cwd), "deadline_seconds": timeout, "started_utc": utc_now()},
    )
    start = time.perf_counter_ns()
    process = None
    status = "worker_exit"
    try:
        with (store.root / attempt / "worker.log").open("xb") as log:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )  # nosec B603 -- controller-owned argv
            try:
                code = process.wait(
                    timeout=max(0.001, timeout - (time.perf_counter_ns() - start) / 1e9)
                )
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
                status = "deadline"
            if store.exists(attempt + "/response.json"):
                status = "response"
    except BaseException:
        if process is not None and process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        raise
    result = {
        "status": status,
        "returncode": code,
        "finished_utc": utc_now(),
        "duration_ns": time.perf_counter_ns() - start,
    }
    store.put(attempt + "/worker-outcome.json", result)
    return result
