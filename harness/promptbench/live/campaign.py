"""Shared frozen-runtime preparation, bounded metadata checks and public-data evaluation."""

from __future__ import annotations

import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from ..benchmark.build import load
from ..benchmark.model import BenchmarkProfile, restore_advisory, restore_profile, score
from ..storage import IntegrityError, Store, atomic_write, digest
from .adapters import normalize
from .conditions import executor_request, hosted_conditions, local_conditions, save_registry
from .execution import Execution, StageExhausted
from .generation import generation_metadata
from .preflight import local_identity, sources
from .process import invoke
from .transport import OLLAMA, OPENROUTER, response_json


def prepare(root: Path, benchmark: Path, registry: Path, config: dict[str, Any]) -> dict[str, Any]:
    store = Store(root)
    with store.lock():
        source = sources()
        plan = {
            **config,
            "source_sha256": digest(source),
            "python_version": platform.python_version(),
            "benchmark_sha256": digest(Store(benchmark).get("manifest.json")),
            "source_quality": Store(benchmark)
            .get("manifest.json")
            .get("source_quality", {"status": "not_reviewed"}),
        }
        store.put("plan.json", plan)
        store.put("inputs/source.json", source)
        from ..config import config_text

        store.put("inputs/models_toml.json", {"text": config_text()})
        profiles = load(
            benchmark,
            set(config.get("input_partitions", ["pilot_development", "pilot_validation"])),
        )
        store.put("inputs/profiles.json", [p.record() for p in profiles])
        store.put(
            "inputs/advisories.json", {a.id: a.record() for p in profiles for a in p.advisories}
        )
        model_registry = save_registry(Store(registry), store)
        store.put(
            "inputs/conditions.json",
            {
                "local": local_conditions(model_registry),
                "hosted": hosted_conditions(model_registry),
            },
        )
        freeze_runtime(store.root, source)
        launcher = (
            "# Frozen runtime: run this file again after an interruption.\n"
            "import os, pathlib, sys\n"
            "root = pathlib.Path(__file__).resolve().parent\n"
            f"python = {sys.executable!r}\n"
            "os.chdir(root / 'runtime')\n"
            f"os.execv(python, [python, '-m', {config.get('runner_module', 'promptbench.live.pilot')!r}, 'run', '--run-dir', str(root), *sys.argv[1:]])\n"
        )
        atomic_write(store.root / "resume.py", launcher.encode())
        return plan


def freeze_runtime(root: Path, source: dict[str, str]) -> None:
    """Write a self-contained runtime: the package sources plus the model configuration.

    `promptbench.config` reads `models.toml` beside the package at import, so the frozen copy
    must carry its own; otherwise a run launched from `resume.py` fails before its first call.
    """
    from ..config import config_text

    for name, text in source.items():
        atomic_write(root / "runtime/promptbench" / name, text.encode())
    atomic_write(root / "runtime/models.toml", config_text().encode())


def profiles_from(store: Store) -> list[BenchmarkProfile]:
    advisories = {
        key: restore_advisory(row) for key, row in store.get("inputs/advisories.json").items()
    }
    return [restore_profile(row, advisories) for row in store.get("inputs/profiles.json")]


def verify_runtime(store: Store) -> dict[str, Any]:
    plan = store.get("plan.json")
    if (
        digest(sources()) != plan["source_sha256"]
        or platform.python_version() != plan["python_version"]
    ):
        raise IntegrityError(
            "runtime/Python changed; use this run's frozen resume.py and original interpreter"
        )
    if plan["kind"] == "public_selection_pilot":
        frozen = store.get("preparation.json")
        if frozen["plan_sha256"] != digest(plan) or any(
            digest(store.get(name)) != expected for name, expected in frozen["input_sha256"].items()
        ):
            raise IntegrityError("prepared pilot inputs changed")
    return dict(plan)


class Campaign:
    def __init__(self, execution: Execution):
        self.execution, self.store = execution, execution.store
        self.session = uuid.uuid4().hex
        self.checked: set[str] = set()
        self.last_local: str | None = None
        execution.key_reader = lambda: self.metadata(
            "key", OPENROUTER + "/api/v1/key", authenticated=True
        )

    def metadata(
        self,
        name: str,
        url: str,
        body: dict[str, Any] | None = None,
        *,
        authenticated: bool = False,
    ) -> dict[str, Any]:
        attempt = f"checks/{self.session}/{name}-{uuid.uuid4().hex}"
        remaining = self.execution.remaining_seconds() - 5
        if self.execution.stage_time_left is not None:
            remaining = min(remaining, self.execution.stage_time_left() - 5)
            if remaining < 1:
                raise StageExhausted("exploratory stage time allocation exhausted")
        if remaining < 1:
            from ..runner import RunPaused

            raise RunPaused("no time remains for provider identity check")
        timeout = min(60.0, remaining)  # catalog endpoint measured at 5-16 s under load
        self.store.put(
            attempt + "/request.json",
            {
                "url": url,
                "body": body,
                "provider": "openrouter" if authenticated else "metadata",
                "timeout_seconds": max(1, int(timeout)),
            },
        )
        self.execution.checkpoint(timeout)
        invoke(
            self.store,
            attempt,
            self.execution.env_file,
            timeout=timeout,
            runtime=self.execution.runtime,
        )
        self.execution.checkpoint()
        return response_json(self.store.get(attempt + "/response.json"))

    def check(self, condition: dict[str, Any]) -> None:
        identifier = condition["id"]
        if condition["provider"] == "ollama":
            model = condition["model"]
            if self.last_local and self.last_local != model:
                self.metadata(
                    "unload",
                    OLLAMA + "/api/generate",
                    {"model": self.last_local, "keep_alive": 0, "stream": False},
                )
                self.checked.discard(identifier)
            self.last_local = model
        if identifier in self.checked:
            return
        if condition["provider"] == "ollama":
            version = self.metadata("version", OLLAMA + "/api/version")["version"]
            tags = self.metadata("tags", OLLAMA + "/api/tags")["models"]
            tag = next((t for t in tags if t["name"] == condition["model"]), None)
            if tag is None:
                raise IntegrityError("frozen local model is no longer installed")
            details = self.metadata("show", OLLAMA + "/api/show", {"model": condition["model"]})
            if local_identity(tag, details, version) != condition["identity"]:
                raise IntegrityError("local model, template, parameters or Ollama version changed")
        else:
            model = condition["model"]
            catalog = self.metadata("catalog", OPENROUTER + "/api/v1/models")["data"]
            entry = next((m for m in catalog if m["id"] == model), None)
            if not entry or entry.get("canonical_slug") != condition["canonical_model"]:
                raise IntegrityError("hosted canonical model changed")
            endpoints = self.metadata(
                "endpoints", OPENROUTER + f"/api/v1/models/{model}/endpoints"
            )["data"]["endpoints"]
            admitted = condition["endpoint"]
            current = next((e for e in endpoints if e["tag"] == admitted["tag"]), None)
            if current is None or any(
                current.get(k) != admitted.get(k)
                for k in ("name", "context_length", "quantization", "provider_name")
            ):
                raise IntegrityError("pinned hosted endpoint identity changed")
            from .adapters import dollars

            if any(
                dollars(current["pricing"][k]) > dollars(admitted["pricing"][k])
                for k in ("prompt", "completion")
            ):
                raise IntegrityError("pinned hosted tariff increased; no silent price update")
        self.checked.add(identifier)

    def evaluate(
        self, prefix: str, condition: dict[str, Any], prompt: str, profiles: list[BenchmarkProfile]
    ) -> list[dict[str, Any]]:
        rows = []
        for profile in profiles:
            self.check(condition)
            key = f"{prefix}/{profile.id}"
            result = self.execution.call(
                key,
                condition,
                executor_request(condition, prompt, profile),
                role="executor",
                allowed_ids={a.id for a in profile.advisories},
            )
            row = score(profile, result["value"], result["status"])
            self.store.put(
                "evaluations/" + digest(key) + ".json",
                {
                    "key": key,
                    "condition_id": condition["id"],
                    "prompt_sha256": digest(prompt),
                    "row": row,
                },
            )
            rows.append(row)
        return rows

    def reconcile_generation(self, attempt: str, canonical_model: str) -> None:
        reply = normalize(self.store.get(attempt + "/response.json"), "openrouter")
        identifier = reply["generation_id"]
        if (
            not isinstance(identifier, str)
            or not identifier.startswith("gen-")
            or not all(c.isalnum() or c in "-_" for c in identifier)
        ):
            raise IntegrityError("hosted response lacks a usable generation identifier")
        name = "generations/" + identifier + ".json"
        if self.store.exists(name):
            return

        def wait(seconds: float) -> None:
            if self.execution.remaining_seconds() < seconds + 5:
                from ..runner import RunPaused

                raise RunPaused("no campaign time for generation metadata retry")
            self.execution.checkpoint(seconds + 5)
            if (
                self.execution.stage_time_left is not None
                and self.execution.stage_time_left() < seconds + 5
            ):
                raise StageExhausted("no stage time for generation metadata retry")
            time.sleep(seconds)

        try:
            data = generation_metadata(self.metadata, wait, identifier, canonical_model)
        except (ValueError, OSError) as exc:
            if isinstance(exc, IntegrityError):
                raise
            self.store.put(
                "generation-pending/" + uuid.uuid4().hex + ".json",
                {"id": identifier, "reason": str(exc), "canonical_model": canonical_model},
            )
            return
        if data.get("model") != canonical_model:
            raise IntegrityError("generation metadata differs from frozen hosted canonical version")
        self.store.put(name, {"id": identifier, "data": data, "canonical_model": canonical_model})
