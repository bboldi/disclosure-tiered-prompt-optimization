"""Adopt a stopped local-only calibration without rewriting its original evidence."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from ..storage import IntegrityError, Store, atomic_write, digest


def scientific_functions(text: str) -> dict[str, str]:
    names = {"reference_request", "summarize_condition", "calibration_gate", "execute"}
    return {
        n.name: ast.dump(n)
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name in names
    }


def adopt_local_calibration(parent: Path, target: Path) -> dict[str, Any]:
    source, destination = Store(parent), Store(target)
    with source.lock():
        if source.exists("reports/complete.json") or not source.exists("operator-exit.json"):
            raise IntegrityError("only a stopped incomplete calibration can be adopted")
        old, new = source.get("manifest.json"), destination.get("manifest.json")
        fields = (
            "profile_ids",
            "profile_sha256",
            "local",
            "hosted",
            "gate",
            "local_calls",
            "hosted_calls",
            "total_calls",
        )
        if any(old[k] != new[k] for k in fields):
            raise IntegrityError("calibration scientific manifest changed")
        for name in source.names("inputs/*.json"):
            if name not in ("inputs/source.json", "inputs/protocol.json") and source.get(
                name
            ) != destination.get(name):
                raise IntegrityError("calibration input changed: " + name)
        old_code, new_code = source.get("inputs/source.json"), destination.get("inputs/source.json")
        if any(
            new_code.get(name) != text
            for name, text in old_code.items()
            if name not in ("live/calibrate.py", "live/taskbudget.py")
        ):
            raise IntegrityError("calibration request/scoring/runtime source changed")
        if scientific_functions(old_code["live/calibrate.py"]) != scientific_functions(
            new_code["live/calibrate.py"]
        ):
            raise IntegrityError("calibration scientific functions changed")
        for name in source.names("work/*/attempts/*/request.json"):
            if source.get(name)["provider"] != "ollama":
                raise IntegrityError(
                    "adoption requires local-only attempts; hosted billing must not duplicate"
                )
        hashes = {}
        for folder in ("work", "evaluations", "conditions"):
            for path in sorted((source.root / folder).rglob("*")):
                if not path.is_file():
                    continue
                name = str(path.relative_to(source.root))
                if path.suffix == ".json":
                    payload = source.get(name)
                    destination.put(name, payload)
                    hashes[name] = digest(payload)
                else:
                    atomic_write(destination.root / name, path.read_bytes())
        result = {
            "parent_root": str(source.root.resolve()),
            "parent_manifest_sha256": digest(old),
            "adopted_records": hashes,
            "adopted_commits": len(source.names("work/*/result.json")),
            "parent_clock_not_copied": True,
            "hosted_attempts_adopted": 0,
            "rule": "All local raw attempts and results retained; parent and child clocks charged separately. No parent files changed.",
        }
        destination.put("adoption.json", result)
        return result
