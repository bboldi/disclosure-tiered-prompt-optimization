"""One provider request in a killable process; never prints or journals credentials."""

from __future__ import annotations

import argparse
import ctypes
import os
import resource
import signal
import sys
from pathlib import Path

from ..storage import Store, digest
from .transport import Transport, read_key


def parent_lifetime(parent_pid: int) -> None:
    # This campaign targets the admitted Linux workstation. Kill an orphaned HTTP worker.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot bind worker lifetime to controller")
    if os.getppid() != parent_pid:
        raise RuntimeError("controller exited before worker initialization")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    parent_lifetime(args.parent_pid)
    store = Store(args.attempt_dir)
    request = store.get("request.json")
    authenticated = request["provider"] == "openrouter"
    transport = Transport(read_key(args.env_file) if authenticated else None)
    response = transport.request(
        request["url"],
        request["body"],
        authenticated=authenticated,
        timeout=request["timeout_seconds"],
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    response["worker"] = {
        "pid": os.getpid(),
        "python_version": sys.version,
        "source_sha256": digest(Path(__file__).read_text()),
        "cpu_user_seconds": usage.ru_utime,
        "cpu_system_seconds": usage.ru_stime,
        "peak_rss_bytes": usage.ru_maxrss * 1024,
    }
    store.put("response.json", response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
