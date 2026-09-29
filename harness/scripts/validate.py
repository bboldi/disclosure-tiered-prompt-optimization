"""Run offline validation and retain logs, raw test runs, and source provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess  # nosec B404 -- fixed local validation tools, no shell
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or code / "validation" / f"validation-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PROMPTBENCH_TEST_ARTIFACT_ROOT"] = str(output / "raw_test_runs")
    sources = [
        code / "pyproject.toml",
        code / "requirements-dev.lock",
        code / "fixtures/mini.json",
    ]
    for folder in ("promptbench", "tests", "scripts"):
        sources.extend(sorted((code / folder).rglob("*.py")))
    source_snapshot = {
        str(p.relative_to(code)): {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "text": p.read_text(),
        }
        for p in sources
    }
    (output / "source_snapshot.json").write_text(json.dumps(source_snapshot, indent=2) + "\n")
    tool_versions = {d.metadata["Name"]: d.version for d in distributions()}
    (output / "tool_versions.json").write_text(json.dumps(tool_versions, indent=2) + "\n")
    commands = [
        ("ruff", [sys.executable, "-m", "ruff", "check", "promptbench", "tests", "scripts"]),
        (
            "format",
            [sys.executable, "-m", "ruff", "format", "--check", "promptbench", "tests", "scripts"],
        ),
        (
            "mypy",
            [
                sys.executable,
                "-m",
                "mypy",
                "--config-file",
                "pyproject.toml",
                "promptbench",
                "scripts",
            ],
        ),
        ("bandit", [sys.executable, "-m", "bandit", "-r", "promptbench", "scripts", "-ll"]),
        ("tests", [sys.executable, "-m", "unittest", "discover", "-v"]),
    ]
    results: list[dict[str, Any]] = []
    for name, command in commands:
        start = time.perf_counter()
        started_utc = datetime.now(UTC).isoformat()
        # Retaining thousands of fsynced artifacts is slower than temporary-directory tests.
        timeout_seconds = 600 if name == "tests" else 60
        error = None
        with (output / f"{name}.log").open("w") as log:
            try:
                process = subprocess.run(  # nosec B603 -- fixed arguments, shell=False
                    command,
                    cwd=code,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                error = f"validation command exceeded {timeout_seconds} seconds"
                log.write(f"\n{error}\n")
        results.append(
            {
                "check": name,
                "command": command,
                "returncode": returncode,
                "started_utc": started_utc,
                "duration_seconds": time.perf_counter() - start,
                "timeout_seconds": timeout_seconds,
                "error": error,
                "log": f"{name}.log",
            }
        )
        (output / "check_progress.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{name}: {'PASS' if returncode == 0 else 'FAIL'}", flush=True)
    report = {
        "evidence_kind": "offline_engineering_validation",
        "phase": "release",
        "created_utc": stamp,
        "python": sys.version,
        "checks": results,
        "passed": all(r["returncode"] == 0 for r in results),
        "actual_api_cost_usd": "0",
        "raw_runs": "raw_test_runs",
        "limitations": [
            "Fake model responses and synthetic pricing; no live provider or GPU evidence.",
            "Corruption/failure tests intentionally leave damaged or incomplete test cases.",
            "Hard-kill tests include extra unknown attempts where responses were not durable.",
            "Fresh test artifacts are always retained; validation directories are never reused.",
        ],
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    conclusions = (
        "# Validation conclusions\n\n"
        + ("All validation checks passed." if report["passed"] else "Validation has failures.")
        + "\n\n"
        + "\n".join(f"- {r['check']}: exit {r['returncode']} ({r['log']})" for r in results)
        + "\n\nAll fake experiment artifacts are retained in raw_test_runs/. The hard-kill test "
        "compares an uninterrupted run with nine interrupted/resumed runs. Their committed "
        "prompts, feedback, decisions, predictions and metrics must match. Attempt histories "
        "may differ when a response was lost before persistence.\n\n"
        "Actual API expenditure is USD 0. These are engineering checks, not model performance "
        "results. See validation.json for scope and limitations.\n"
    )
    (output / "CONCLUSIONS.md").write_text(conclusions)
    print(f"Evidence: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
