from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from promptbench.benchmark.model import BenchmarkProfile
from promptbench.benchmark.oracle import Criterion, Installed, PublicAdvisory
from promptbench.live.analysis import analysis, forecast
from promptbench.live.campaign import Campaign
from promptbench.live.conditions import NAIVE, study_feedback_request
from promptbench.live.execution import Execution
from promptbench.live.reconstruct import reconstruct
from promptbench.live.study import (
    PANELS,
    Study,
    batch_schedule,
    build_manifest,
    prepare_continuation,
    record_disclosure,
    trajectory_calls,
)
from promptbench.runner import RunPaused
from promptbench.storage import IntegrityError, Store, digest

BETTER = "candidate"  # any proposed prompt containing this token answers correctly
SIZES = {
    "optimization": 24,
    "validation": 3,
    "test": 3,
    "temporal": 3,
    "product_heldout": 3,
}


def fixture_profiles() -> tuple[list[BenchmarkProfile], PublicAdvisory]:
    term = Criterion("vendor", "product", "1", None, None, False, False, "fixture", None)
    advisory = PublicAdvisory(
        "CVE-2099-1001", "fictional", "2024-01-01", "2024-01-01", (term,), "fixture"
    )
    profiles = []
    for partition, size in SIZES.items():
        for index in range(size):
            positive = index % 2 == 1
            profiles.append(
                BenchmarkProfile(
                    f"{partition}_{index:03d}",
                    partition,
                    "version " + ("1" if positive else "2"),
                    (Installed("vendor", "product", "1" if positive else "2"),),
                    (advisory,),
                    (advisory.id,) if positive else (),
                    ("easy", "medium", "hard")[(index // 2) % 3],
                    advisory.id,
                )
            )
    return profiles, advisory


def conditions(seed_supported: bool = True) -> dict[str, list[dict]]:
    local = [
        {
            "id": f"family{i}/off",
            "family": f"family{i}",
            "provider": "ollama",
            "url": "http://127.0.0.1:11434/api/chat",
            "model": f"family{i}",
            "expected_models": [f"family{i}"],
            "expected_provider": "local-ollama",
            "think": False,
            "num_ctx": 16384,
            "num_predict": 512,
        }
        for i in range(2)
    ]
    hosted = [
        {
            "id": model,
            "model": model,
            "canonical_model": model + "-dated",
            "provider": "openrouter",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "expected_models": [model, model + "-dated"],
            "expected_provider": "fixture-provider",
            "endpoint": {
                "tag": "fixture",
                "context_length": 1000,
                "pricing": {"prompt": ".000001", "completion": ".000002"},
                "supported_parameters": ["seed"] if seed_supported else [],
            },
        }
        for model in ("z-ai/glm-5.3", "deepseek/deepseek-v4-pro")
    ]
    return {"local": local, "hosted": hosted}


def design(**overrides):
    base = {
        "executors": ["family0/off", "family1/off"],
        "optimizer": "z-ai/glm-5.3",
        "reference_optimizer": "deepseek/deepseek-v4-pro",
        "tiers": [1, 2, 3],
        "repetitions": 2,
        "depth": 1,
        "candidates_per_round": 2,
        "batch_size": 2,
        "validation_size": 3,
        "panels": {"test": 3, "temporal": 3, "product_heldout": 3},
        "output_tokens": 512,
        "hosted_seed_base": 1000,
        "schedule_seed": 7,
        "gepa": None,
        "reference": {"executor": "family0/off", "tier": 2, "repetitions": 1},
    }
    base.update(overrides)
    return base


class StudyTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            self.root = Path(retained) / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            temp = tempfile.TemporaryDirectory()
            self.addCleanup(temp.cleanup)
            self.root = Path(temp.name)
        self.profiles, self.advisory = fixture_profiles()
        self.calls = 0
        self.tier_of_proposal: dict[str, int] = {}

    # ----- fixtures ---------------------------------------------------------------------
    def prepared_store(self, name: str = "study", **overrides) -> Store:
        store = Store(self.root / name)
        store.put("plan.json", {"kind": "main_study", "phase": "repilot", "max_seconds": 3600})
        store.put("inputs/profiles.json", [p.record() for p in self.profiles])
        store.put("inputs/advisories.json", {self.advisory.id: self.advisory.record()})
        cond = conditions()
        store.put("inputs/conditions.json", cond)
        store.put("inputs/study_conditions.json", cond)
        store.put("manifest.json", build_manifest(design(**overrides), self.profiles, cond))
        return store

    def worker(self, s, attempt, env_file, **kwargs):
        self.calls += 1
        request = s.get(attempt + "/request.json")
        if request["provider"] == "ollama":
            prompt = request["body"]["messages"][0]["content"]
            profile_text = json.loads(request["body"]["messages"][1]["content"])["system_profile"]
            truth = [self.advisory.id] if profile_text == "version 1" else []
            answer = truth if BETTER in prompt else []  # naive misses every positive
            body = {
                "model": request["body"]["model"],
                "done": True,
                "done_reason": "stop",
                "message": {"content": json.dumps({"applicable_cves": answer})},
                "eval_count": 5,
            }
        else:
            permitted = json.loads(request["body"]["messages"][0]["content"])
            text = f"{BETTER} tier {permitted['tier']} slot {permitted['candidate_slot']}"
            body = {
                "model": request["body"]["model"],
                "provider": "fixture-provider",
                "id": f"gen-fixture-{self.calls}",
                "choices": [
                    {"message": {"content": json.dumps({"prompt": text})}, "finish_reason": "stop"}
                ],
                "usage": {"cost": 0.001},
            }
        s.put(
            attempt + "/response.json",
            {
                "http_status": 200,
                "error": None,
                "headers": {},
                "body_text": json.dumps(body),
                "duration_ns": 2_000_000,
            },
        )

    def engine(self, store: Store, hook=None) -> Execution:
        engine = Execution(
            store,
            self.root / "no-key",
            max_seconds=3600,
            max_cost_usd="8",
            max_attempts=100000,
            study_root=self.root,
            worker=self.worker,
            **({"hook": hook} if hook else {}),
        )
        engine.key_reader = lambda: {"data": {"usage": 0.01}}
        return engine

    def run_study(self, store: Store, hook=None):
        with patch.object(Campaign, "check"), patch.object(Campaign, "reconcile_generation"):
            engine = self.engine(store, hook)
            study = Study(engine)  # Campaign installs a live key reader; replace it after
            engine.key_reader = lambda: {"data": {"usage": 0.01}}
            return study.execute()

    # ----- schedule and manifest --------------------------------------------------------
    def test_schedule_is_seeded_balanced_and_disjoint_within_a_repetition(self):
        schedules = batch_schedule(self.profiles, 3, 3, 8, seed=1)
        again = batch_schedule(self.profiles, 3, 3, 8, seed=1)
        self.assertEqual(schedules, again)
        self.assertNotEqual(schedules[0], schedules[1])
        by_id = {p.id: p for p in self.profiles}
        for rep in schedules:
            flat = [i for batch in rep for i in batch]
            self.assertEqual(len(flat), len(set(flat)))  # 24 profiles, 3x8 without replacement
            for batch in rep:
                self.assertTrue(all(by_id[i].partition == "optimization" for i in batch))
                positives = sum(bool(by_id[i].expected) for i in batch)
                self.assertEqual(positives, 4)  # balanced positives per batch
        cycled = batch_schedule(self.profiles, 1, 4, 8, seed=1)[0]
        self.assertEqual(len({i for b in cycled for i in b}), 24)  # wraps after exhaustion

    def test_manifest_rejects_unknown_conditions_and_bad_tiers(self):
        cond = conditions()
        with self.assertRaises(IntegrityError):
            build_manifest(design(executors=["ghost/off"]), self.profiles, cond)
        with self.assertRaises(IntegrityError):
            build_manifest(design(tiers=[0]), self.profiles, cond)
        with self.assertRaises(IntegrityError):
            build_manifest(design(validation_size=999), self.profiles, cond)
        with self.assertRaises(IntegrityError):
            build_manifest(design(reference_optimizer="nope"), self.profiles, cond)
        manifest = build_manifest(design(), self.profiles, cond)
        self.assertEqual(len(manifest["trajectories"]), 2 * 2 * 3 + 1)
        self.assertTrue(manifest["hosted_seed_supported"])
        self.assertEqual(
            trajectory_calls(design()),
            {"local_search": 2 * 1 * 3, "local_selection": 6, "local_sealed": 9, "hosted": 2},
        )

    def test_feedback_request_refuses_sealed_rows_and_has_no_search_label(self):
        rows = [{"profile_id": "test_000", "partition": "test", "categories": [], "prediction": []}]
        with self.assertRaises(ValueError):
            study_feedback_request(NAIVE, rows, [], tier=2, candidate=0)
        request = study_feedback_request(NAIVE, [], [], tier=1, candidate=1)
        self.assertNotIn("search_seed", request)
        self.assertNotIn("abstractions", request["feedback"])

    # ----- full fake campaign -----------------------------------------------------------
    def test_campaign_runs_all_tiers_shares_sealed_panels_and_resumes_without_inference(self):
        store = self.prepared_store()
        result = self.run_study(store)
        self.assertEqual(result["trajectories"], 13)
        self.assertEqual(result["prompt_changed"], 13)  # every arm found the better prompt
        before = self.calls
        repeated = self.run_study(store)
        self.assertEqual(self.calls, before)
        self.assertEqual(repeated["trajectories"], 13)
        # Disclosures never carry sealed partitions; tiers expose exactly their declared fields.
        sealed_ids = {p.id for p in self.profiles if p.partition != "optimization"}
        for name in store.names("disclosures/*.json"):
            record = store.get(name)
            permitted = json.loads(record["request"]["messages"][0]["content"])
            self.assertFalse(sealed_ids & set(json.dumps(permitted).split('"')))
            self.assertNotIn("search_seed", permitted)
            feedback = permitted["feedback"]
            self.assertEqual("abstractions" in feedback, record["tier"] >= 2)
            self.assertEqual("examples" in feedback, record["tier"] == 3)
            if record["hosted_seed_sent"] is not None:
                self.assertEqual(record["request"]["seed"], record["hosted_seed_sent"])
        # Repetitions 1 and 2 sent different hosted seeds; executors stayed at seed 11.
        seeds = {store.get(n)["hosted_seed_sent"] for n in store.names("disclosures/*.json")}
        self.assertEqual(seeds, {1001, 1002})
        for name in store.names("work/*/spec.json"):
            spec = store.get(name)
            if spec["role"] == "executor":
                self.assertEqual(
                    spec["body"]["options"],
                    {"temperature": 0, "seed": 11, "num_ctx": 16384, "num_predict": 512},
                )
        # Sealed panels: one evaluation per (executor, prompt text, panel), shared across arms.
        trajectories = [store.get(n) for n in store.names("trajectories/*.json")]
        selected = {(t["executor"], t["selected"]["prompt_sha256"]) for t in trajectories}
        self.assertEqual(len(store.names("sealed-panels/*.json")), len(selected) * len(PANELS))
        naive_selections = [
            n
            for n in store.names("selections/*.json")
            if store.get(n)["prompt_sha256"] == digest(NAIVE)
        ]
        self.assertEqual(len(naive_selections), 2)  # once per executor, not per trajectory
        for t in trajectories:
            self.assertEqual(len(t["trace"]), 1)
            self.assertEqual(
                t["sealed"]["test"]["metrics"]["failure_aware_lower_bound"]["micro_f1"], 1.0
            )
            self.assertEqual(set(t["sealed"]), set(PANELS))
        self.assertEqual({t["arm"] for t in trajectories}, {"main", "reference"})
        reference = next(t for t in trajectories if t["arm"] == "reference")
        self.assertEqual(reference["optimizer"], "deepseek/deepseek-v4-pro")
        # Analysis: paired contrast per executor with two repetitions.
        exported = store.get("reports/analysis.json")
        contrast = exported["primary_contrast_T2_minus_T3"]
        self.assertEqual(set(contrast), {"family0/off", "family1/off"})
        for cell in contrast.values():
            self.assertEqual(cell["pairs"], 2)
            self.assertEqual(cell["mean_delta"], 0.0)
            self.assertTrue(cell["retained_utility"])
        self.assertEqual(result["forecast"]["alternatives"][0]["families"], 2)
        audited = reconstruct(store.root, self.root / "reconstruction")
        self.assertEqual(audited["physical_attempts"], before)
        self.assertEqual(audited["unknown_attempts"], 0)

    def test_forfeited_proposal_keeps_incumbent_and_records_failure(self):
        store = self.prepared_store(repetitions=1, tiers=[2], reference=None)
        original = self.worker

        def worker(s, attempt, env_file, **kwargs):
            request = s.get(attempt + "/request.json")
            if request["provider"] == "openrouter":
                self.calls += 1
                s.put(
                    attempt + "/response.json",
                    {
                        "http_status": 200,
                        "error": None,
                        "headers": {},
                        "body_text": json.dumps(
                            {
                                "model": request["body"]["model"],
                                "provider": "fixture-provider",
                                "id": "gen-x",
                                "choices": [
                                    {"message": {"content": "not json"}, "finish_reason": "stop"}
                                ],
                                "usage": {"cost": 0.001},
                            }
                        ),
                        "duration_ns": 1,
                    },
                )
                return
            original(s, attempt, env_file, **kwargs)

        self.worker = worker
        result = self.run_study(store)
        self.assertEqual(result["prompt_changed"], 0)
        self.assertEqual(len(store.names("candidate-failures/*.json")), 2 * 1 * 2)
        for t in (store.get(n) for n in store.names("trajectories/*.json")):
            self.assertEqual(t["selected"]["prompt"], NAIVE)
            self.assertEqual(len(t["shortlist"]), 1)
            self.assertTrue(all(d["slot"] == -1 for d in t["trace"]))

    def test_hard_kills_at_every_boundary_recover_identical_committed_results(self):
        small = {"repetitions": 1, "reference": None, "tiers": [2, 3], "executors": ["family0/off"]}
        reference_store = self.prepared_store("reference", **small)
        reference = self.run_study(reference_store)
        reference_calls = self.calls
        expected = {
            store_key: {k: v for k, v in reference_store.get(store_key).items() if k != "timing"}
            for store_key in reference_store.names("trajectories/*.json")
        }
        boundaries = (
            "planned",
            "intent",
            "provider_returned",
            "response",
            "before_commit",
            "committed",
        )
        for index, boundary in enumerate(boundaries):
            store = self.prepared_store(f"kill-{boundary}", **small)
            self.calls = 0
            seen = {"count": 0}
            target = 2 + 3 * index  # different depth into the campaign for every boundary

            def stop(point, key, boundary=boundary, target=target, seen=seen):
                if point == boundary:
                    seen["count"] += 1
                    if seen["count"] == target:
                        raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                self.run_study(store, hook=stop)
            self.assertFalse(store.exists("reports/analysis.json"))
            resumed = self.run_study(store)
            self.assertEqual(resumed["trajectories"], reference["trajectories"])
            recovered = {
                k: {kk: vv for kk, vv in store.get(k).items() if kk != "timing"}
                for k in store.names("trajectories/*.json")
            }
            self.assertEqual(recovered, expected)
            lost = len(store.names("work/*/attempts/*/unknown.json"))
            self.assertLessEqual(self.calls, reference_calls + 1)
            self.assertLessEqual(lost, 1)
            self.assertEqual(
                len(store.names("work/*/result.json")),
                len(reference_store.names("work/*/result.json")),
            )

    def test_continuation_adopts_paused_journal_without_repeating_inference(self):
        small = {"repetitions": 1, "reference": None, "tiers": [2, 3], "executors": ["family0/off"]}
        reference_store = self.prepared_store("reference", **small)
        reference = self.run_study(reference_store)
        total = self.calls
        parent = self.prepared_store("parent", **small)
        self.calls = 0
        seen = {"count": 0}

        def pause(point, key):
            if point == "committed":
                seen["count"] += 1
                if seen["count"] == 5:
                    raise RunPaused("fixture time cap")

        with self.assertRaises(RunPaused):
            self.run_study(parent, hook=pause)
        parent.put("pauses/fixture.json", {"reason": "fixture time cap"})
        with self.assertRaises(IntegrityError):
            prepare_continuation(
                reference_store.root,
                self.root / "bad",
                max_seconds=60,
                max_cost_usd="1",
                reason="x",
            )
        child = self.root / "child"
        lineage = prepare_continuation(
            parent.root, child, max_seconds=3600, max_cost_usd="1", reason="fixture"
        )
        self.assertEqual(lineage["adopted"]["work"], len(parent.names("work/**/*.json")))
        store = Store(child)
        self.assertEqual(
            store.get("plan.json")["parent_plan_sha256"], digest(parent.get("plan.json"))
        )
        result = self.run_study(store)
        self.assertEqual(result["trajectories"], reference["trajectories"])
        self.assertEqual(self.calls, total)  # paused work plus continuation equals one clean run
        self.assertFalse(parent.exists("reports/complete.json"))  # parent untouched
        recovered = {
            k: {kk: vv for kk, vv in store.get(k).items() if kk != "timing"}
            for k in store.names("trajectories/*.json")
        }
        expected = {
            k: {kk: vv for kk, vv in reference_store.get(k).items() if kk != "timing"}
            for k in reference_store.names("trajectories/*.json")
        }
        self.assertEqual(recovered, expected)

    def test_deferred_generation_metadata_records_pending_without_lookups(self):
        store = self.prepared_store(
            repetitions=1, tiers=[2], reference=None, defer_generation_metadata=True
        )
        with patch.object(Campaign, "check"), patch.object(Campaign, "reconcile_generation") as r:
            engine = self.engine(store)
            study = Study(engine)
            engine.key_reader = lambda: {"data": {"usage": 0.01}}
            study.execute()
            r.assert_not_called()
        pending = [store.get(n) for n in store.names("generation-pending/*.json")]
        self.assertEqual(len(pending), 2 * 1 * 2)
        self.assertTrue(
            all("attempt" in p and p["canonical_model"].endswith("-dated") for p in pending)
        )

    def test_refreshed_runtime_continuation_records_changed_modules(self):
        small = {"repetitions": 1, "reference": None, "tiers": [2], "executors": ["family0/off"]}
        parent = self.prepared_store("parent-rt", **small)
        parent.put("inputs/source.json", {"live/study.py": "old text", "domain.py": "same"})
        parent.put("pauses/fixture.json", {"reason": "fixture"})
        (parent.root / "runtime/promptbench").mkdir(parents=True)
        with patch(
            "promptbench.live.preflight.sources",
            return_value={"live/study.py": "new text", "domain.py": "same"},
        ):
            lineage = prepare_continuation(
                parent.root,
                self.root / "child-rt",
                max_seconds=60,
                max_cost_usd="1",
                reason="fixture",
                refresh_runtime=True,
            )
        self.assertEqual(list(lineage["runtime_change"]["changed_modules"]), ["live/study.py"])
        child = Store(self.root / "child-rt")
        self.assertEqual(child.get("inputs/source.json")["live/study.py"], "new text")
        self.assertEqual(
            child.get("plan.json")["source_sha256"], digest(child.get("inputs/source.json"))
        )
        self.assertTrue(
            (child.root / "runtime/promptbench/live/study.py").read_text() == "new text"
        )

    def test_tariff_acceptance_updates_only_named_endpoint_and_records_change(self):
        small = {"repetitions": 1, "reference": None, "tiers": [2], "executors": ["family0/off"]}
        parent = self.prepared_store("parent-tariff", **small)
        parent.put("pauses/fixture.json", {"reason": "pinned hosted tariff increased"})
        # Two retained checks: the newer one has the lexicographically smaller name, so a
        # name-ordered choice would adopt the stale (lower) price.
        parent.put(
            "checks/s1/endpoints-x/response.json",
            {
                "finished_utc": "2026-09-16T21:20:00+00:00",
                "body_text": json.dumps(
                    {
                        "data": {
                            "endpoints": [
                                {
                                    "tag": "fixture",
                                    "pricing": {"prompt": ".000002", "completion": ".000004"},
                                }
                            ]
                        }
                    }
                ),
            },
        )
        parent.put(
            "checks/z9/endpoints-x/response.json",
            {
                "finished_utc": "2026-09-16T13:45:00+00:00",
                "body_text": json.dumps(
                    {
                        "data": {
                            "endpoints": [
                                {
                                    "tag": "fixture",
                                    "pricing": {"prompt": ".000001", "completion": ".000002"},
                                }
                            ]
                        }
                    }
                ),
            },
        )
        lineage = prepare_continuation(
            parent.root,
            self.root / "child-tariff",
            max_seconds=60,
            max_cost_usd="1",
            reason="fixture",
            accept_tariff="z-ai/glm-5.3",
        )
        self.assertEqual(lineage["tariff_change"]["old_pricing"]["prompt"], ".000001")
        self.assertEqual(lineage["tariff_change"]["new_pricing"]["prompt"], ".000002")
        child = Store(self.root / "child-tariff")
        hosted = {h["id"]: h for h in child.get("inputs/study_conditions.json")["hosted"]}
        self.assertEqual(hosted["z-ai/glm-5.3"]["endpoint"]["pricing"]["completion"], ".000004")
        self.assertEqual(
            hosted["deepseek/deepseek-v4-pro"]["endpoint"]["pricing"]["completion"], ".000002"
        )
        manifest = child.get("manifest.json")
        self.assertEqual(
            manifest["input_sha256"]["inputs/study_conditions.json"],
            digest(child.get("inputs/study_conditions.json")),
        )

    def test_disclosure_record_tolerates_only_an_accepted_price_ceiling_change(self):
        store = Store(self.root / "disclosure")
        first = {
            "key": "main/x/T2/R1/r0/proposal0",
            "request": {"model": "m", "messages": [], "provider": {"max_price": {"prompt": "1"}}},
            "sha256": "a",
            "tier": 2,
            "hosted_seed_sent": 1001,
        }
        record_disclosure(store, "disclosures/k.json", first)
        repriced = {
            **first,
            "request": {**first["request"], "provider": {"max_price": {"prompt": "2"}}},
            "sha256": "b",
        }
        record_disclosure(store, "disclosures/k.json", repriced)  # accepted tariff: no conflict
        self.assertEqual(
            store.get("disclosures/k.json")["request"]["provider"]["max_price"], {"prompt": "1"}
        )
        changed_view = {**first, "request": {**first["request"], "messages": [{"x": 1}]}}
        with self.assertRaises(IntegrityError):
            record_disclosure(store, "disclosures/k.json", changed_view)

    def test_extension_run_reuses_schedules_for_later_repetitions_only(self):
        cond = conditions()
        full = build_manifest(design(repetitions=8, reference=None), self.profiles, cond)
        ext = build_manifest(
            design(repetitions=8, first_repetition=6, reference=None, executors=["family1/off"]),
            self.profiles,
            cond,
        )
        self.assertEqual(sorted({t["repetition"] for t in ext["trajectories"]}), [6, 7, 8])
        self.assertEqual({t["executor"] for t in ext["trajectories"]}, {"family1/off"})
        for rep in ("6", "7", "8"):
            self.assertEqual(ext["schedules"][rep], full["schedules"][rep])
        with self.assertRaises(IntegrityError):
            build_manifest(design(repetitions=8, first_repetition=9), self.profiles, cond)

    def test_forecast_reports_measured_timings_and_fit(self):
        store = self.prepared_store(repetitions=1, tiers=[2], reference=None)
        self.run_study(store)
        report = forecast(store, store.get("manifest.json"))
        self.assertEqual(
            set(report["measured_local_wall_seconds_per_call"]), {"family0/off", "family1/off"}
        )
        self.assertGreater(report["accounted_running_seconds"], 0)
        self.assertEqual([a["depth"] for a in report["alternatives"]], [8, 6, 4])
        self.assertTrue(all(a["fits_72h"] for a in report["alternatives"]))
        exported = analysis(store)
        self.assertEqual(len(exported["trajectories"]), 2)
        self.assertEqual(exported["primary_contrast_T2_minus_T3"], {})


if __name__ == "__main__":
    unittest.main()
