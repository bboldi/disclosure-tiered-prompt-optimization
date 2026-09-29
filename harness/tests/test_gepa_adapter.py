import copy
import os
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

try:
    from gepa.core.adapter import EvaluationBatch
except ImportError:  # optional [gepa] extra not installed
    raise unittest.SkipTest("gepa extra not installed") from None

from promptbench import domain
from promptbench.live.conditions import NAIVE
from promptbench.live.gepa_adapter import (
    GEPA_COMMIT,
    EvaluationBudgetExhausted,
    TierAdapter,
    dependency_identity,
    run_gepa,
)
from promptbench.storage import IntegrityError, Store, canonical
from tests.test_feedback_relations import profile


class GEPAAdapterTests(unittest.TestCase):
    @contextmanager
    def artifact_directory(self, suffix="case"):
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            root = Path(retained) / self._testMethodName / suffix
            root.mkdir(parents=True, exist_ok=False)
            yield str(root)
        else:
            with tempfile.TemporaryDirectory() as directory:
                yield directory

    def test_exact_budget_with_real_pinned_engine_for_accepted_and_rejected_candidates(self):
        self.assertEqual(dependency_identity()["commit"], GEPA_COMMIT)
        for improve in (False, True):
            with self.subTest(improve=improve), self.artifact_directory(str(improve)) as directory:
                profiles = [profile(i, installed="701.702.703.701") for i in range(12)]
                calls, proposals = [], []

                def executor(prompt, p, *, calls=calls, improve=improve):
                    calls.append((prompt, p.id))
                    return (list(p.expected) if improve and prompt != NAIVE else []), "valid"

                def proposer(request, *, proposals=proposals):
                    proposals.append(copy.deepcopy(request))
                    return f"Instruction variant {len(proposals)}."

                adapter = TierAdapter(
                    profiles,
                    executor,
                    proposer,
                    tier=2,
                    depth=4,
                    candidates_per_round=2,
                    batch_size=2,
                    store=Store(Path(directory)),
                )
                schedule = [tuple(p.id for p in profiles[i : i + 2]) for i in range(0, 12, 2)]
                result = run_gepa(adapter, schedule, seed=1)
                self.assertEqual(result["candidate_evaluations"], 13)
                self.assertEqual(result["engine_metric_calls"], 13)
                self.assertEqual(result["executor_profile_calls"], 26)
                self.assertEqual(len(calls), 26)
                self.assertTrue(proposals)
                self.assertEqual(result["selected_candidate"]["prompt"] == NAIVE, not improve)
                self.assertTrue(all("search_seed" not in p for p in proposals))
                with self.assertRaises(EvaluationBudgetExhausted):
                    adapter.evaluate([schedule[0]], {"prompt": NAIVE})
                self.assertEqual(len(calls), 26)

    def test_serializer_is_unavoidable_and_raw_trace_injection_is_rejected(self):
        profiles = [profile(i) for i in range(6)]
        for tier in (1, 2, 3):
            with self.subTest(tier=tier), self.artifact_directory(str(tier)) as directory:
                sent = []

                def proposer(request, *, sent=sent):
                    sent.append(copy.deepcopy(request))
                    request["feedback"]["attacker"] = "PRIVATE_INJECTION"
                    return "A candidate instruction."

                adapter = TierAdapter(
                    profiles,
                    lambda _, p: ([p.advisories[0].id], "valid"),
                    proposer,
                    tier=tier,
                    depth=4,
                    candidates_per_round=2,
                    batch_size=6,
                    store=Store(Path(directory)),
                )
                candidate = {"prompt": NAIVE}
                result = adapter.evaluate([tuple(p.id for p in profiles)], candidate, True)
                with patch.object(domain, "feedback", wraps=domain.feedback) as serializer:
                    dataset = adapter.make_reflective_dataset(candidate, result, ["prompt"])
                    adapter.propose_new_texts(candidate, dataset, ["prompt"])
                    adapter.propose_new_texts(candidate, dataset, ["prompt"])
                    self.assertEqual(serializer.call_count, 3)
                self.assertEqual(sent[0], sent[1])
                text = canonical(sent[0])
                if tier < 3:
                    for secret in (
                        "vendor_canary",
                        "product_canary",
                        "701.702",
                        "CVE-2099",
                        "DESCRIPTION_CANARY",
                        "INVENTORY_CANARY",
                    ):
                        self.assertNotIn(secret, text)
                else:
                    self.assertEqual(len(sent[0]["feedback"]["examples"]), 4)
                forged = copy.deepcopy(dataset)
                forged["prompt"][0]["raw_inputs"] = "PRIVATE_INJECTION"
                with self.assertRaisesRegex(IntegrityError, "bypass"):
                    adapter.propose_new_texts(candidate, forged, ["prompt"])
                with self.assertRaises(IntegrityError):
                    adapter.make_reflective_dataset(
                        candidate,
                        EvaluationBatch(outputs=["forged"], scores=[1.0], trajectories=["forged"]),
                        ["prompt"],
                    )
                self.assertEqual(len(sent), 2)

    def test_mutated_inputs_and_sealed_partitions_are_rejected(self):
        p = profile()
        with self.artifact_directory() as directory:
            for partition in (
                "validation",
                "test",
                "temporal",
                "product_held_out",
                "pilot_development",
            ):
                with self.assertRaisesRegex(IntegrityError, "optimization"):
                    TierAdapter(
                        [replace(p, partition=partition)],
                        lambda *_: ([], "valid"),
                        lambda _: "instruction",
                        tier=2,
                        depth=1,
                        candidates_per_round=1,
                        batch_size=1,
                        store=Store(Path(directory)),
                    )
            adapter = TierAdapter(
                [p],
                lambda *_: ([], "valid"),
                lambda _: "instruction",
                tier=2,
                depth=1,
                candidates_per_round=1,
                batch_size=1,
                store=Store(Path(directory)),
            )
            p.presentation["components"][0]["alias_style"] = "repo"
            with self.assertRaisesRegex(IntegrityError, "input changed"):
                adapter.evaluate([(p.id,)], {"prompt": NAIVE})

    def test_unrelated_provider_failure_is_not_swallowed_by_budget_stop_policy(self):
        profiles = [profile(i) for i in range(2)]
        with self.artifact_directory() as directory:
            calls = []

            def fail(_):
                calls.append(1)
                raise RuntimeError("provider fixture failed")

            adapter = TierAdapter(
                profiles,
                lambda *_: ([], "valid"),
                fail,
                tier=2,
                depth=4,
                candidates_per_round=2,
                batch_size=2,
                store=Store(Path(directory)),
            )
            with self.assertRaisesRegex(RuntimeError, "provider fixture failed"):
                run_gepa(adapter, [tuple(p.id for p in profiles)], seed=1)
            self.assertLess(adapter.evaluations, adapter.batch_budget)
            self.assertEqual(len(calls), 1)
