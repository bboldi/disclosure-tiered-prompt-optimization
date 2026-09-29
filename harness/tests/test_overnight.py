from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from promptbench.live.overnight import key_snapshot, supervise
from promptbench.storage import IntegrityError, Store


class OvernightTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            self.root = Path(retained) / self._testMethodName / "campaign"
        else:
            temp = tempfile.TemporaryDirectory()
            self.addCleanup(temp.cleanup)
            self.root = Path(temp.name) / "campaign"

    def test_success_retains_reconstruction_and_provider_key_snapshots(self):
        def runner(root, env):
            Store(root).put("reports/complete.json", {"status": "completed_exploratory_pilot"})
            return 0

        reconstruction = {
            "completed_work": 12,
            "physical_attempts": 13,
            "unknown_attempts": 1,
            "running_seconds_accounted": 600,
            "costs": {"reported_cost_usd": ".03", "reserved_unknown_usd": ".01"},
        }
        with (
            patch("promptbench.live.overnight.run", side_effect=runner),
            patch("promptbench.live.overnight.reconstruct", return_value=reconstruction),
            patch(
                "promptbench.live.overnight.key_snapshot",
                side_effect=[
                    {"available": True, "usage_usd": 1},
                    {"available": True, "usage_usd": 1.03},
                ],
            ),
        ):
            self.assertEqual(supervise(self.root, self.root / "unused.env"), 0)
        text = (self.root / "MORNING_REPORT.md").read_text()
        self.assertIn("completed_exploratory_pilot", text)
        self.assertIn("unknown attempts: 1", text)
        self.assertIn("unresolved reservations: USD .01", text)
        self.assertIn("1.03 USD", text)

    def test_breaking_failure_is_reported_without_claiming_completion_or_zero_cost(self):
        with (
            patch(
                "promptbench.live.overnight.run", side_effect=IntegrityError("fixture source drift")
            ),
            patch(
                "promptbench.live.overnight.reconstruct",
                side_effect=IntegrityError("fixture incomplete record"),
            ),
            patch(
                "promptbench.live.overnight.key_snapshot",
                return_value={"available": False, "usage_usd": None},
            ),
        ):
            self.assertEqual(supervise(self.root, self.root / "unused.env"), 1)
        text = (self.root / "MORNING_REPORT.md").read_text()
        self.assertIn("paused_or_interrupted", text)
        self.assertIn("fixture source drift", text)
        self.assertIn("unavailable USD", text)
        self.assertNotIn("reconstruction: passed", text)
        self.assertFalse(Store(self.root).exists("reports/complete.json"))

    def test_missing_key_response_is_unknown_and_keeps_dispatch_evidence(self):
        store = Store(self.root)
        with patch("promptbench.live.overnight.invoke", side_effect=OSError("fixture unreachable")):
            result = key_snapshot(store, "test", "after", self.root / "unused.env")
        self.assertFalse(result["available"])
        self.assertIsNone(result["usage_usd"])
        self.assertTrue(store.exists("operator/test/key-after/request.json"))
