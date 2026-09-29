"""Input-matched deterministic baseline: parsing, matching and blindness to labels."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from promptbench.baseline import (
    decide,
    normalize,
    parse_affected,
    parse_inventory,
    run,
    version_satisfies,
)
from promptbench.benchmark.model import BenchmarkProfile
from promptbench.benchmark.oracle import Criterion, Installed, PublicAdvisory
from promptbench.storage import Store

INVENTORY = """Public synthetic system inventory:
- phpgurukul-job-portal 1.0
- Ptc Vuforia Studio 9.8
- fortinet-fortianalyzer 7.0.0
- pingidentity/pingid_radius_pcv@3.0.0.0
- taismartfactory qplant sf 1.0.1
- something without a version
"""


def profile(text: str, components: tuple[Installed, ...], advisories, expected, presentation=None):
    return BenchmarkProfile(
        "p1",
        "test",
        text,
        components,
        advisories,
        expected,
        "easy",
        advisories[0].id,
        presentation or {},
    )


class BaselineParsingTests(unittest.TestCase):
    def test_every_inventory_alias_style_normalises_to_the_same_token(self):
        rows = parse_inventory(INVENTORY)
        self.assertEqual(
            [r["version"] for r in rows], ["1.0", "9.8", "7.0.0", "3.0.0.0", "1.0.1", None]
        )
        self.assertEqual(rows[0]["name"], normalize("phpgurukul/job_portal"))
        self.assertEqual(rows[1]["name"], normalize("ptc/vuforia_studio"))
        self.assertEqual(rows[3]["name"], normalize("pingidentity/pingid_radius_pcv"))
        self.assertEqual(rows[4]["name"], normalize("taismartfactory/qplant_sf"))

    def test_affected_clauses_with_or_terms_and_prose_template(self):
        terms = parse_affected(
            "fortinet/fortianalyzer: version >= 6.2.0 and <= 6.2.12 OR fortinet/fortianalyzer: version < 5"
        )
        assert terms is not None
        self.assertEqual(
            [t["conditions"] for t in terms], [[(">=", "6.2.0"), ("<=", "6.2.12")], [("<", "5")]]
        )
        prose = parse_affected("acme/widget: versions prior to 4.2 are affected")
        assert prose is not None
        self.assertEqual(prose[0]["conditions"], [("<", "4.2")])
        self.assertIsNone(parse_affected("acme/widget: all versions are vulnerable"))

    def test_version_comparison_uses_trailing_zero_equivalence_and_refuses_unsupported(self):
        self.assertTrue(version_satisfies("7.0", [("=", "7.0.0")]))
        self.assertTrue(version_satisfies("6.2.5", [(">=", "6.2.0"), ("<=", "6.2.12")]))
        self.assertFalse(version_satisfies("6.2.13", [(">=", "6.2.0"), ("<=", "6.2.12")]))
        self.assertIsNone(version_satisfies("2023-rc1", [("<", "4")]))


class BaselineDecisionTests(unittest.TestCase):
    def setUp(self):
        hit = Criterion("fortinet", "fortianalyzer", None, "7.0.0", "7.0.9", True, True, "f", None)
        miss = Criterion("ptc", "vuforia_studio", None, None, "9.8", False, False, "f", None)
        self.hit = PublicAdvisory("CVE-2023-0001", "x", "2024-01-01", "2024-01-01", (hit,), "f")
        self.miss = PublicAdvisory("CVE-2023-0002", "x", "2024-01-01", "2024-01-01", (miss,), "f")
        self.components = (
            Installed("fortinet", "fortianalyzer", "7.0.0"),
            Installed("ptc", "vuforia_studio", "9.8"),
        )

    def test_decides_from_rendered_text_only(self):
        p = profile(INVENTORY, self.components, (self.hit, self.miss), (self.hit.id,))
        predicted, counters = decide(p)
        self.assertEqual(predicted, [self.hit.id])
        self.assertEqual(counters["components_unparsed"], 1)
        self.assertEqual(counters["advisories_unparsed"], 0)
        # Labels flipped, text unchanged: the decision must not move.
        wrong = profile(INVENTORY, self.components, (self.hit, self.miss), (self.miss.id,))
        self.assertEqual(decide(wrong)[0], [self.hit.id])
        # Text changed, canonical components unchanged: the decision follows the text.
        blank = profile(
            "Public synthetic system inventory:\n", self.components, (self.hit, self.miss), ()
        )
        self.assertEqual(decide(blank)[0], [])

    def test_prose_presentation_is_read_from_the_rendered_advisory(self):
        prose = {
            "prose_advisories": {
                self.miss.id: {"text": "ptc/vuforia_studio: versions prior to 9.9 are affected"}
            }
        }
        p = profile(
            INVENTORY, self.components, (self.hit, self.miss), (self.hit.id, self.miss.id), prose
        )
        self.assertEqual(sorted(decide(p)[0]), [self.hit.id, self.miss.id])

    def test_run_writes_development_report_and_one_panel_per_sealed_partition(self):
        p = profile(INVENTORY, self.components, (self.hit, self.miss), (self.hit.id,))
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "bench"
            Store(bench).put("manifest.json", {"kind": "fixture"})
            with patch("promptbench.baseline.load", return_value=[p]):
                report = run(bench, Path(tmp) / "run")
            self.assertEqual(set(report["panels"]), {"test", "temporal", "product_heldout"})
            self.assertEqual(
                report["panels"]["test"]["metrics"]["failure_aware_lower_bound"]["micro_f1"], 1.0
            )
            self.assertEqual(report["development"]["partition"], "optimization")
            store = Store(Path(tmp) / "run")
            self.assertEqual(len(store.names("panels/*.json")), 3)
            self.assertEqual(len(store.names("evaluations/*.json")), 3)
            self.assertIn("rules", store.get("manifest.json"))
            self.assertEqual(
                json.loads(json.dumps(store.get("reports/complete.json")))["status"], "completed"
            )


if __name__ == "__main__":
    unittest.main()
