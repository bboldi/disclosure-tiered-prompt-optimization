"""Tier-restricted adaptation of the pinned GEPA engine; no raw reflection channel."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .. import domain
from ..benchmark.model import BenchmarkProfile, score
from ..storage import IntegrityError, Store, canonical, digest
from .conditions import NAIVE

GEPA_COMMIT = "15ee314f9c7d34ec153b809d401f42f55c4dcd76"
Bundle = tuple[str, ...]
Executor = Callable[[str, BenchmarkProfile], tuple[list[str] | None, str]]
Proposer = Callable[[dict[str, Any]], str]


class EvaluationBudgetExhausted(RuntimeError):
    """The next GEPA evaluation would exceed the frozen core-equivalent budget."""


def dependency_identity() -> dict[str, Any]:
    distribution = importlib.metadata.distribution("gepa")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != GEPA_COMMIT:
        raise IntegrityError("GEPA must be installed from the exact frozen commit")
    files = {
        str(p): digest(Path(str(distribution.locate_file(p))).read_text())
        for p in distribution.files or []
        if str(p).startswith("gepa/") and str(p).endswith(".py")
    }
    if not files:
        raise IntegrityError("GEPA source files are unavailable")
    return {"commit": GEPA_COMMIT, "version": distribution.version, "source_sha256": files}


class TierAdapter:
    """Each GEPA datum is one fixed-size profile bundle scored by pooled micro-F1."""

    def __init__(
        self,
        profiles: Sequence[BenchmarkProfile],
        executor: Executor,
        proposer: Proposer,
        *,
        tier: int,
        depth: int,
        candidates_per_round: int,
        batch_size: int,
        store: Store,
    ) -> None:
        if tier not in (1, 2, 3) or min(depth, candidates_per_round, batch_size) < 1:
            raise ValueError("invalid GEPA tier or matched budget")
        if not profiles or any(p.partition != "optimization" for p in profiles):
            raise IntegrityError("GEPA may access only the optimization partition")
        self.profiles = {p.id: p for p in profiles}
        if len(self.profiles) != len(profiles):
            raise IntegrityError("duplicate GEPA profile identifier")
        self.profile_hashes = {p.id: digest(p.record()) for p in profiles}
        self.executor, self.proposer, self.store = executor, proposer, store
        self.tier, self.batch_size = tier, batch_size
        self.batch_budget = 1 + depth * (candidates_per_round + 1)
        self.evaluations = 0
        self.profile_calls = 0
        self.fatal_error: Exception | None = None
        self._records: dict[str, dict[str, Any]] = {}
        self._reflections: dict[tuple[str, str], str] = {}
        self.dependency = dependency_identity()

    def _profiles(self, identifiers: Sequence[str]) -> list[BenchmarkProfile]:
        profiles = []
        for identifier in identifiers:
            if identifier not in self.profiles:
                raise IntegrityError("GEPA requested an unadmitted profile")
            p = self.profiles[identifier]
            if digest(p.record()) != self.profile_hashes[identifier]:
                raise IntegrityError("GEPA evaluator input changed")
            profiles.append(p)
        return profiles

    def evaluate(
        self, batch: list[Bundle], candidate: dict[str, str], capture_traces: bool = False
    ) -> Any:
        if self.fatal_error is not None:
            raise self.fatal_error
        if len(batch) != 1:
            raise IntegrityError("GEPA adapter requires one fixed-size bundle per evaluation")
        if self.evaluations >= self.batch_budget:
            raise EvaluationBudgetExhausted("exact candidate-evaluation budget consumed")
        prompt = domain.parse_prompt(canonical(candidate))
        identifiers = batch[0]
        if len(identifiers) != self.batch_size or len(set(identifiers)) != self.batch_size:
            raise IntegrityError("GEPA bundle violates frozen batch size")
        profiles = self._profiles(identifiers)
        rows = []
        try:
            for p in profiles:
                prediction, status = self.executor(prompt, p)
                self.profile_calls += 1
                if prediction is not None:
                    prediction = domain.parse_answer(
                        canonical({"applicable_cves": prediction}), {a.id for a in p.advisories}
                    )
                rows.append(score(p, prediction, status))
        except Exception as exc:
            self.fatal_error = exc
            raise
        self.evaluations += 1
        metric = domain.metrics(rows)["failure_aware_lower_bound"]["micro_f1"]
        record = {"candidate": dict(candidate), "profile_ids": list(identifiers), "rows": rows}
        handle = digest({"ordinal": self.evaluations, "record": record})
        self._records[handle] = record
        self.store.put(f"evaluations/{self.evaluations:04d}.json", {**record, "handle": handle})
        batch_type = importlib.import_module("gepa.core.adapter").EvaluationBatch
        return batch_type(
            outputs=[handle],
            scores=[metric],
            trajectories=[handle] if capture_traces else None,
            num_metric_calls=1,
        )

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: Any, components_to_update: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        if components_to_update != ["prompt"] or eval_batch.trajectories != eval_batch.outputs:
            raise IntegrityError("unrecognized GEPA reflection component or trace")
        if len(eval_batch.outputs) != 1 or eval_batch.outputs[0] not in self._records:
            raise IntegrityError("GEPA reflection must refer to a locally retained evaluation")
        record = self._records[eval_batch.outputs[0]]
        if record["candidate"] != candidate:
            raise IntegrityError("GEPA reflection candidate does not match its evaluation")
        profiles = self._profiles(record["profile_ids"])
        permitted = domain.feedback(record["rows"], profiles, self.tier)
        dataset = {"prompt": [{"feedback": permitted}]}
        self._reflections[(digest(candidate), digest(dataset))] = eval_batch.outputs[0]
        return dataset

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        if self.fatal_error is not None:
            raise self.fatal_error
        key = (digest(candidate), digest(reflective_dataset))
        if components_to_update != ["prompt"] or key not in self._reflections:
            raise IntegrityError("GEPA proposal cannot bypass domain.feedback")
        if self.evaluations >= self.batch_budget:
            raise EvaluationBudgetExhausted("no proposal after the final budgeted evaluation")
        record = self._records[self._reflections[key]]
        request = {
            "task": domain.TASK,
            "current_prompt": candidate["prompt"],
            "feedback": domain.feedback(
                record["rows"], self._profiles(record["profile_ids"]), self.tier
            ),
            "tier": self.tier,
            "feedback_version": "relational-abstractions-v2",
            "method": "gepa-tier-restricted",
        }
        self.store.put("disclosures/" + digest(request) + ".json", request)
        try:
            proposed = self.proposer(request)
            return {"prompt": domain.parse_prompt(canonical({"prompt": proposed}))}
        except Exception as exc:
            self.fatal_error = exc
            raise

    def on_error(self, event: dict[str, Any]) -> None:
        error = event["exception"]
        if not isinstance(error, EvaluationBudgetExhausted):
            self.fatal_error = error


def run_gepa(adapter: TierAdapter, schedule: list[Bundle], *, seed: int) -> dict[str, Any]:
    if not schedule or adapter.evaluations or adapter.store.exists("manifest.json"):
        raise IntegrityError("GEPA needs a fresh adapter, schedule and run identity")
    if any(
        len(b) != adapter.batch_size or any(i not in adapter.profiles for i in b) for b in schedule
    ):
        raise IntegrityError("GEPA schedule contains unadmitted data")
    adapter.store.put(
        "manifest.json",
        {
            "method": "gepa-tier-restricted",
            "tier": adapter.tier,
            "dependency": adapter.dependency,
            "profile_sha256": adapter.profile_hashes,
            "schedule": schedule,
            "internal_reference_bundle": schedule[0],
            "seed": seed,
            "candidate_evaluation_budget": adapter.batch_budget,
            "executor_profile_call_budget": adapter.batch_budget * adapter.batch_size,
            "task": domain.TASK,
            "start_prompt": NAIVE,
            "sealed_partitions_accessed": False,
        },
    )
    result = importlib.import_module("gepa").optimize(
        seed_candidate={"prompt": NAIVE},
        trainset=schedule,
        valset=[schedule[0]],
        adapter=adapter,
        reflection_minibatch_size=1,
        skip_perfect_score=False,
        max_metric_calls=adapter.batch_budget,
        stop_callbacks=lambda state: (
            adapter.evaluations >= adapter.batch_budget or adapter.fatal_error is not None
        ),
        candidate_selection_strategy="pareto",
        use_merge=False,
        cache_evaluation=False,
        track_best_outputs=False,
        raise_on_exception=False,
        seed=seed,
        run_dir=str(adapter.store.root / "engine"),
        callbacks=[adapter],
    )
    if adapter.fatal_error is not None:
        raise adapter.fatal_error
    if (
        adapter.evaluations != adapter.batch_budget
        or adapter.profile_calls != adapter.batch_budget * adapter.batch_size
    ):
        raise IntegrityError("GEPA did not consume the exact matched evaluation budget")
    if result.total_metric_calls != adapter.evaluations:
        raise IntegrityError("GEPA engine and immutable evaluator accounting disagree")
    summary = {
        "selected_candidate": result.best_candidate,
        "candidate_evaluations": adapter.evaluations,
        "executor_profile_calls": adapter.profile_calls,
        "engine_metric_calls": result.total_metric_calls,
        "internal_reference_scores": result.val_aggregate_scores,
        "note": "Internal reference is optimization data; common sealed selection is external to GEPA.",
    }
    adapter.store.put("result.json", summary)
    return summary
