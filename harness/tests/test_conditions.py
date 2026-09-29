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
