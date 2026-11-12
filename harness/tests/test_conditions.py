from __future__ import annotations

import unittest
from dataclasses import replace

from promptbench.benchmark.model import BenchmarkProfile, score
from promptbench.benchmark.oracle import Installed, PublicAdvisory
from promptbench.live.conditions import (
    executor_request,
    feedback_request,
    local_conditions,
    optimizer_request,
    self_optimizer_condition,
    shuffle_abstractions,
    study_feedback_request,
)
from promptbench.storage import canonical


class ConditionTests(unittest.TestCase):
    def profile(self):
        advisory = PublicAdvisory(
            "CVE-2024-11111", "CANARY_DESCRIPTION", "2024-01-01", "2024-01-01", (), "CANARY_HASH"
        )
        return BenchmarkProfile(
            "CANARY_ID",
            "pilot_development",
            "CANARY_INVENTORY",
            (Installed("CANARY_VENDOR", "CANARY_PRODUCT", "1"),),
            (advisory,),
            (),
            "hard",
            advisory.id,
        )

    def test_complete_low_tier_request_excludes_private_channels(self):
        profile = self.profile()
        row = score(profile, [profile.advisories[0].id], "valid")
        for tier in (1, 2):
            request = feedback_request(
                "generic prompt", [row], [profile], tier=tier, seed=11, candidate=1
            )
            self.assertNotIn("CANARY", canonical(request))
            self.assertNotIn(profile.advisories[0].id, canonical(request))
        self.assertIn(
            "CANARY_INVENTORY",
            canonical(
                feedback_request("generic prompt", [row], [profile], tier=3, seed=11, candidate=1)
            ),
        )

    def test_executor_never_receives_labels_or_split_metadata(self):
        conditions = local_conditions(
            {
                "local_models": [
                    {
                        "requested_tag": "fixture:1",
                        "capabilities": ["thinking"],
                        "ollama_manifest_digest": "digest",
                    }
                ]
            }
        )
        self.assertEqual([c["think"] for c in conditions], [False, True])
        first = executor_request(conditions[0], "instruction", self.profile())
        second = executor_request(
            conditions[0],
            "instruction",
            replace(self.profile(), expected=("secret-label",), partition="secret-split"),
        )
        self.assertEqual(first, second)
        self.assertFalse(first["think"])
        self.assertFalse(first["truncate"])

    def test_hosted_tariff_and_endpoint_are_specific_to_condition(self):
        condition = {
            "canonical_model": "dated-model",
            "model": "request-model",
            "endpoint": {
                "tag": "fixture/fp8",
                "context_length": 1000,
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            },
        }
        admitted, body = optimizer_request(condition, {"task": "fixture"})
        self.assertEqual(admitted["reservation_usd"], "0.017384")
        self.assertEqual(body["provider"]["only"], ["fixture/fp8"])
        self.assertFalse(body["provider"]["allow_fallbacks"])
        self.assertTrue(body["provider"]["require_parameters"])
        self.assertEqual(body["model"], "request-model")
        self.assertNotIn("temperature", body)


class RevisionConditionTests(unittest.TestCase):
    """Round 1 revision: local self-optimization and the Shuffled Abstraction control."""

    def test_local_optimizer_request_is_an_ollama_chat_without_tariff(self):
        executor = {
            "id": "fixture:1/off",
            "provider": "ollama",
            "model": "fixture:1",
            "think": False,
            "num_ctx": 16384,
            "num_predict": 2048,
            "timeout_seconds": 300,
        }
        settings = {"think": "low", "num_ctx": 32768, "num_predict": 8192, "timeout_seconds": 900}
        optimizer = self_optimizer_condition(executor, settings)
        self.assertEqual(optimizer["id"], "fixture:1/off/self-optimizer")
        self.assertEqual(optimizer["model"], executor["model"])
        condition, body = optimizer_request(optimizer, {"tier": 3}, seed=1003)
        self.assertNotIn("reservation_usd", condition)
        self.assertEqual(body["think"], "low")
        self.assertEqual(body["options"], {"seed": 1003, "num_ctx": 32768, "num_predict": 8192})
        self.assertFalse(body["truncate"])
        self.assertEqual(body["format"]["required"], ["prompt"])
        self.assertEqual(len(body["messages"]), 1)

    def test_shuffle_keeps_count_and_field_multisets_and_is_seeded(self):
        records = [
            {"category": c, "version_position": v, "alias_style": a}
            for c, v, a in [
                ("version_excluded", "above", "repo"),
                ("missed_applicable", "on_inclusive", None),
                ("unrelated_product", None, "package_manager"),
                ("version_excluded", "below", "repo"),
            ]
        ]
        shuffled = shuffle_abstractions(records, 7)
        self.assertEqual(shuffled, shuffle_abstractions(records, 7))
        self.assertEqual(len(shuffled), len(records))
        for key in records[0]:
            self.assertEqual(
                sorted(map(str, (r[key] for r in shuffled))),
                sorted(map(str, (r[key] for r in records))),
            )
        self.assertNotEqual([shuffle_abstractions(records, s) for s in range(5)], [records] * 5)
        self.assertEqual(shuffle_abstractions([], 3), [])

    def test_shuffled_variant_is_refused_outside_tier_2(self):
        profile = ConditionTests().profile()
        profile = replace(profile, partition="optimization")
        row = score(profile, [], "valid")
        row["partition"] = "optimization"
        with self.assertRaises(ValueError):
            study_feedback_request("p", [row], [profile], tier=3, candidate=0, variant="shuffled")
        request = study_feedback_request(
            "p", [row], [profile], tier=2, candidate=0, variant="shuffled", shuffle_seed=5
        )
        self.assertEqual(request["feedback_version"], "relational-abstractions-v2-shuffled")
