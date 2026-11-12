#!/usr/bin/env python3
"""Revision verification gate: does the pinned stack reproduce the campaign's answers?

Compares a fresh naive-prompt run (promptbench.live.ceiling, local Executors) with the
campaign's sealed-panel rows of the same prompt, profile by profile. The gate passes only
if every test-panel answer (status and predicted set) is identical; the other panels are
reported but do not decide the gate.

    python scripts/verify_pinned_stack.py --run ../data/followups/naive-recheck-granite --data ../data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NAIVE_SHA = "0080638c0f58d5b18e6db10f0327e0c85125dd7c1ea1ae311fd5ad96c05b3078"


def answers(directory: Path, prefix: str) -> dict[tuple[str, str, str], tuple[str, tuple]]:
    """(executor, panel, profile) -> (status, sorted prediction) for one evaluation folder."""
    out = {}
    for f in directory.iterdir():
        env = json.loads(f.read_text())["payload"]
        parts = env["key"].split("/")
        if parts[0] != prefix or env["prompt_sha256"] != NAIVE_SHA:
            continue
        executor = "/".join(parts[1:-3] if prefix == "sealed" else parts[1:-2])
        panel = parts[-3] if prefix == "sealed" else parts[-2]
        row = env["row"]
        out[(executor, panel, row["profile_id"])] = (
            row["status"],
            tuple(sorted(row["prediction"] or [])),
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    fresh = answers(args.run / "evaluations", "ceiling")
    campaign = {}
    for section in ("campaign", "campaign-extension"):
        campaign.update(answers(args.data / section / "evaluations", "sealed"))
    report, gate = {}, True
    for key in sorted({(e, p) for e, p, _ in fresh}):
        ids = [k for k in fresh if k[:2] == key]
        known = [k for k in ids if k in campaign]
        same = sum(fresh[k] == campaign[k] for k in known)
        report[f"{key[0]} {key[1]}"] = {
            "fresh": len(ids),
            "compared": len(known),
            "identical": same,
        }
        if key[1] == "test" and (len(known) != len(ids) or same != len(known)):
            gate = False
    print(json.dumps({"gate_passed": gate, "panels": report}, indent=2))
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
