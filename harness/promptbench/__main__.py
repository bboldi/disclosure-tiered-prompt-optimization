"""Run, resume, or audit a phase-1 fixture campaign."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .audit import audit
from .experiment import Config, Experiment, read_fixture
from .runner import RunPaused
from .storage import IntegrityError, Store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="create or resume an offline fixture campaign")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--fixture", type=Path)
    run.add_argument("--tier", type=int, choices=[1, 2, 3])
    run.add_argument("--seed", type=int)
    run.add_argument("--iterations", type=int)
    run.add_argument("--stop-after-commits", type=int)
    check = sub.add_parser("audit", help="rebuild reports using saved evidence only")
    check.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    store = Store(args.run_dir)
    try:
        if args.command == "audit":
            result = audit(store)
        else:
            if store.exists("manifest.json"):
                config = Config(**store.get("manifest.json")["config"])
                fixture = (
                    read_fixture(args.fixture) if args.fixture else store.get("inputs/dataset.json")
                )
            else:
                if args.fixture is None:
                    parser.error("--fixture is required for a new run")
                config, fixture = Config(), read_fixture(args.fixture)
            values = asdict(config)
            for name in ("tier", "seed", "iterations"):
                if getattr(args, name) is not None:
                    values[name] = getattr(args, name)
            config = Config(**values)
            commits = 0

            def hook(point: str, key: str) -> None:
                nonlocal commits
                if point == "committed":
                    commits += 1
                    if args.stop_after_commits and commits >= args.stop_after_commits:
                        raise KeyboardInterrupt("requested dry-run checkpoint interruption")

            Experiment(args.run_dir, config, fixture, hook=hook).run()
            result = audit(store)
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt as exc:
        print(
            f"Interrupted; saved evidence retained. Resume with the same run directory. {exc}",
            file=sys.stderr,
        )
        return 130
    except (RunPaused, IntegrityError, ValueError, OSError) as exc:
        print(f"Run not complete: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
