"""A complete small optimization loop sharing one durable evaluator contract."""

from __future__ import annotations

import os
import platform
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .domain import (
    BASELINE,
    PARSER,
    SEMANTICS,
    TASK,
    Profile,
    feedback,
    load_profiles,
    metrics,
    score,
    strict_json,
)
from .providers import FakeProvider, Provider
from .runner import Hook, Runner, RunPaused, no_hook, utc_now
from .storage import IntegrityError, Store, digest


@dataclass(frozen=True)
class Config:
    tier: int = 2
    seed: int = 11
    iterations: int = 2
    candidates: int = 2
    batch_size: int = 3
    max_retries: int = 2
    max_attempts: int = 200
    faults: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if self.tier not in (1, 2, 3):
            raise ValueError("tier must be 1, 2, or 3")
        for name in ("iterations", "candidates", "batch_size", "max_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.max_retries <= 10:
            raise ValueError("invalid seed/retry limit")
        allowed = {
            "timeout_once",
            "timeout_always",
            "malformed_once",
            "malformed_always",
            "reasoning_only",
            "refusal",
            "auth",
        }
        if any(value not in allowed for value in self.faults.values()):
            raise ValueError("unknown injected fault")


def source_snapshot() -> dict[str, str]:
    return {path.name: path.read_text() for path in sorted(Path(__file__).parent.glob("*.py"))}


class Experiment:
    def __init__(
        self,
        root: Path,
        config: Config,
        fixture: dict[str, Any],
        *,
        provider: Provider | None = None,
        hook: Hook = no_hook,
    ):
        config.validate()
        self.config = config
        self.fixture = fixture
        self.profiles = load_profiles(fixture)
        self.store = Store(root)
        self.runner = Runner(
            self.store,
            provider or FakeProvider(),
            max_retries=config.max_retries,
            max_attempts=config.max_attempts,
            faults=config.faults,
            hook=hook,
        )
        self.required_work: set[str] = set()
        self.hook = hook

    def run(self) -> dict[str, Any]:
        with self.store.lock():
            self._initialize()
            self.runner.event("controller_started", pid=os.getpid())
            started = utc_now()
            try:
                result = self._search()
            except BaseException as exc:
                self.runner.event(
                    "controller_stopped",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    status="paused"
                    if isinstance(exc, (RunPaused, KeyboardInterrupt))
                    else "interrupted_or_failed",
                )
                raise
            self.runner.event("controller_finished", started_utc=started, finished_utc=utc_now())
            return result

    def _initialize(self) -> None:
        source = source_snapshot()
        manifest = {
            "protocol": "offline-phase1-v1",
            "semantics": SEMANTICS,
            "parser": PARSER,
            "provider": "deterministic-fake-v1",
            "python": platform.python_version(),
            "config": asdict(self.config),
            "dataset_sha256": digest(self.fixture),
            "source_sha256": digest(source),
            "usage": "synthetic_fixture_only",
            "actual_api_cost_usd": "0",
        }
        self.store.put("manifest.json", manifest)
        self.store.put("inputs/dataset.json", self.fixture)
        self.store.put("inputs/source.json", source)
        if not self.store.exists("environment.json"):
            self.store.put(
                "environment.json",
                {
                    "created_utc": utc_now(),
                    "python": sys.version,
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "hostname": platform.node(),
                },
            )
        elif self.store.get("environment.json")["hostname"] != platform.node():
            raise IntegrityError("cross-machine resume needs an explicit migration protocol")
        self.store.put("prompts/baseline.json", {"id": "baseline", "prompt": BASELINE})
        self.store.put("prompts/optimizer_task.json", {"text": TASK})

    def _call(self, key: str, request: dict[str, Any]) -> dict[str, Any]:
        self.required_work.add(digest({"key": key, "request": request}))
        return self.runner.execute(key, request)

    def _batch(self, iteration: int) -> list[Profile]:
        items = [p for p in self.profiles if p.partition == "optimization"]
        rng = random.Random(f"{self.config.seed}:{iteration}")  # deterministic scheduling only
        rng.shuffle(items)
        batch = items[: self.config.batch_size]
        self.store.put(
            f"schedules/{iteration:03d}.json",
            {
                "iteration": iteration,
                "profile_ids": [p.id for p in batch],
                "seed": self.config.seed,
            },
        )
        return batch

    def _evaluate(
        self,
        candidate: dict[str, Any],
        profiles: list[Profile],
        stage: str,
    ) -> list[dict[str, Any]]:
        rows = []
        for profile in profiles:
            request = {
                "role": "executor",
                "prompt": candidate["prompt"],
                "input": profile.executor_input(),
                "model": "deterministic-fake-v1",
                "settings": {"seed": self.config.seed, "mode": "fixture", "schema": PARSER},
            }
            key = f"evaluate/{stage}/{candidate['id']}/{profile.id}"
            result = self._call(key, request)
            row = score(profile, result["value"], result["status"])
            self.store.put(
                f"evaluations/{digest(key)}.json",
                {
                    "key": key,
                    "candidate_id": candidate["id"],
                    "stage": stage,
                    "work_id": digest({"key": key, "request": request}),
                    "score": row,
                },
            )
            rows.append(row)
        self.store.put(
            f"metrics/{digest([stage, candidate['id']])}.json",
            {
                "candidate_id": candidate["id"],
                "stage": stage,
                "metrics": metrics(rows),
            },
        )
        return rows

    @staticmethod
    def _utility(rows: list[dict[str, Any]]) -> float:
        return float(metrics(rows)["failure_aware_lower_bound"]["micro_f1"])

    def _search(self) -> dict[str, Any]:
        baseline = {"id": "baseline", "prompt": BASELINE}
        incumbent = baseline
        initial_rows = self._evaluate(baseline, self._batch(0), "baseline")
        history: list[dict[str, Any]] = [
            {
                "candidate_id": "baseline",
                "prompt": BASELINE,
                "batch": 0,
                "metrics": metrics(initial_rows),
            }
        ]
        for iteration in range(1, self.config.iterations + 1):
            batch = self._batch(iteration)
            current_rows = self._evaluate(incumbent, batch, f"iteration-{iteration}")
            view = feedback(current_rows, batch, self.config.tier)
            self.store.put(f"feedback/{iteration:03d}.json", view)
            contenders = [(incumbent, current_rows)]
            for slot in range(self.config.candidates):
                request = {
                    "role": "optimizer",
                    "model": "deterministic-fake-v1",
                    "task": TASK,
                    "feedback": view,
                    "history": history,
                    "iteration": iteration,
                    "candidate_slot": slot,
                    "seed": self.config.seed,
                }
                proposal = self._call(f"propose/{iteration}/{slot}", request)
                candidate = {"id": f"i{iteration}-c{slot}", "prompt": proposal["value"]}
                duplicate = next(
                    (item for item, _ in contenders if item["prompt"] == candidate["prompt"]), None
                )
                self.store.put(
                    f"prompts/{candidate['id']}.json",
                    {
                        **candidate,
                        "parent_id": incumbent["id"],
                        "prompt_sha256": digest(candidate["prompt"]),
                        "duplicate_of": duplicate["id"] if duplicate else None,
                    },
                )
                if duplicate is not None:
                    # A repeated proposal consumes its provider call, but no new evaluation.
                    continue
                rows = self._evaluate(candidate, batch, f"iteration-{iteration}")
                contenders.append((candidate, rows))
            # Stable tie order retains the incumbent; unlike-batch scores are never compared.
            incumbent, winning_rows = max(contenders, key=lambda pair: self._utility(pair[1]))
            self.store.put(
                f"decisions/iteration-{iteration:03d}.json",
                {
                    "selected": incumbent["id"],
                    "batch": [p.id for p in batch],
                    "contenders": [
                        {"id": item["id"], "metrics": metrics(rows)} for item, rows in contenders
                    ],
                    "criterion": "failure_aware_lower_bound.micro_f1; ties retain earlier contender",
                },
            )
            history.append(
                {
                    "candidate_id": incumbent["id"],
                    "prompt": incumbent["prompt"],
                    "batch": iteration,
                    "metrics": metrics(winning_rows),
                }
            )
            self.hook("iteration", str(iteration))
        validation = [p for p in self.profiles if p.partition == "validation"]
        shortlist = [baseline] if incumbent["id"] == "baseline" else [baseline, incumbent]
        validation_results = [
            (candidate, self._evaluate(candidate, validation, "validation"))
            for candidate in shortlist
        ]
        selected, selected_rows = max(validation_results, key=lambda pair: self._utility(pair[1]))
        selection = {
            "selected": selected,
            "validation_metrics": metrics(selected_rows),
            "shortlist": [c["id"] for c in shortlist],
            "criterion": "validation failure-aware micro-F1; ties prefer baseline",
        }
        self.store.put("decisions/selection.json", selection)
        self.hook("selection", selected["id"])
        tests = [p for p in self.profiles if p.partition == "test"]
        test_rows = self._evaluate(selected, tests, "test")
        result = {
            "status": "completed",
            "selected": selected,
            "test_metrics": metrics(test_rows),
            "required_work": sorted(self.required_work),
            "manifest_sha256": digest(self.store.get("manifest.json")),
            "actual_api_cost_usd": "0",
            "evidence_kind": "engineering_fixture_only",
        }
        # A completion lists every committed item, not merely a best-prompt filename.
        for work_id in result["required_work"]:
            if not self.store.exists(f"work/{work_id}/result.json"):
                raise IntegrityError("cannot complete with missing work")
        if set(self.store.names("work/*/result.json")) != {
            f"work/{work_id}/result.json" for work_id in self.required_work
        }:
            raise IntegrityError("unexpected committed work in run")
        self.store.put("completion.json", result)
        self.hook("completion", "run")
        return result


def read_fixture(path: Path) -> dict[str, Any]:
    data = strict_json(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("fixture must be an object")
    return dict(data)
