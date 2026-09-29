"""Read-only live admission checks and bounded fixture smoke experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..experiment import read_fixture
from ..runner import RunPaused
from ..storage import IntegrityError, Store
from .preflight import run_preflight
from .smoke import Smoke
from .transport import Transport, read_key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--run-dir", type=Path, required=True)
    preflight.add_argument("--env-file", type=Path, default=Path(".env"))
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--run-dir", type=Path, required=True)
    smoke.add_argument("--registry-dir", type=Path, required=True)
    smoke.add_argument("--fixture", type=Path, default=Path("fixtures/mini.json"))
    smoke.add_argument("--env-file", type=Path, default=Path(".env"))
    smoke.add_argument("--stop-after-commits", type=int)
    smoke.add_argument("--max-cost-usd")
    prepare = sub.add_parser(
        "prepare-continuation", help="offline import under an explicitly revised budget"
    )
    prepare.add_argument("--parent-run", type=Path, required=True)
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--registry-dir", type=Path, required=True)
    prepare.add_argument("--fixture", type=Path, default=Path("fixtures/mini.json"))
    prepare.add_argument("--max-cost-usd", required=True)
    prepare.add_argument("--reason", required=True)
    for command in (smoke, prepare):
        command.add_argument(
            "--local-think",
            choices=("default", "on", "off"),
            help="freeze local reasoning mode; omitted on resume inherits the saved condition",
        )
    args = parser.parse_args()
    if args.command == "preflight":
        result = run_preflight(args.run_dir, Transport(read_key(args.env_file)))
        print(
            json.dumps(
                {
                    "evidence": str(args.run_dir),
                    "gpu_accessible": result["gpu_accessible"],
                    "local_models": len(result["local_models"]),
                    "hosted_models": len(result["hosted_models"]),
                    "inference_requests": 0,
                },
                indent=2,
            )
        )
    elif args.command in ("smoke", "prepare-continuation"):
        try:
            saved = Store(args.run_dir)
            cap = args.max_cost_usd or (
                saved.get("manifest.json")["max_cost_usd"] if saved.exists("manifest.json") else "1"
            )
            settings_source = (
                Store(args.parent_run) if args.command == "prepare-continuation" else saved
            )
            local_think = (
                {"default": None, "on": True, "off": False}[args.local_think]
                if args.local_think is not None
                else settings_source.get("manifest.json").get("local_think")
                if settings_source.exists("manifest.json")
                else None
            )
            controller = Smoke(
                args.run_dir,
                Store(args.registry_dir).get("model_registry.json"),
                read_fixture(args.fixture),
                Transport(read_key(args.env_file)) if args.command == "smoke" else Transport(),
                stop_after=getattr(args, "stop_after_commits", None),
                max_cost_usd=cap,
                local_think=local_think,
            )
            result = (
                controller.prepare_continuation(args.parent_run, args.reason)
                if args.command == "prepare-continuation"
                else controller.run()
            )
            print(json.dumps(result, indent=2))
        except KeyboardInterrupt:
            print(
                "Live smoke interrupted; resume using the same arguments without --stop-after-commits.",
                file=sys.stderr,
            )
            return 130
        except (RunPaused, IntegrityError, ValueError, OSError) as exc:
            print(f"Live smoke paused: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
