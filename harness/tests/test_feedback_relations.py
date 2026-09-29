import unittest
from dataclasses import replace
from pathlib import Path

from promptbench.benchmark.model import BenchmarkProfile, score
from promptbench.benchmark.oracle import Criterion, Installed, PublicAdvisory
from promptbench.domain import TASK, feedback
from promptbench.live.conditions import NAIVE, feedback_request
from promptbench.relations import ALIAS_STYLES, term_position
from promptbench.storage import canonical


def profile(index=0, *, installed="701.702.703.704", upper="701.702.703.703"):
    term = Criterion("vendor_canary", "product_canary", None, None, upper, True, False, "", None)
    a = PublicAdvisory(
        f"CVE-2099-{91000 + index}",
        "DESCRIPTION_CANARY_PRIVATE_ONLY_abcdefghijklmnopqrstuvwxyz",
        "2099-01-01",
        "2099-01-01",
        (term,),
        "fixture",
    )
    c = Installed(term.vendor, term.product, installed)
    return BenchmarkProfile(
        f"fixture_{index:02}",
        "optimization",
        "INVENTORY_CANARY_PRIVATE_ONLY",
        (c,),
        (a,),
        (a.id,) if a.applies((c,)) else (),
        "hard",
        a.id,
        {
            "components": [{"alias_style": "package_manager"}],
            "prose_advisories": {a.id: {"text": "PROSE_CANARY_PRIVATE_ONLY"}},
        },
    )


class FeedbackRelationsTests(unittest.TestCase):
    def test_bound_positions_inclusive_exclusive_exact_and_unbounded(self):
        term = profile().advisories[0].terms[0]
        for installed, inclusive, expected in (
            ("701.702.703.702", False, "below"),
            ("701.702.703.703", False, "on_exclusive"),
            ("701.702.703.703", True, "on_inclusive"),
            ("701.702.703.704", False, "above"),
        ):
            self.assertEqual(
                term_position(replace(term, upper_inclusive=inclusive), installed), expected
            )
        self.assertEqual(
            term_position(replace(term, exact=term.upper, upper=None), term.upper), "on_inclusive"
        )
        self.assertIsNone(term_position(replace(term, upper=None), "701.702"))
        interval = replace(term, lower="701.702.703.700", lower_inclusive=False)
        self.assertEqual(term_position(interval, "701.702.703.700"), "on_exclusive")
        self.assertEqual(term_position(interval, "701.702.703.704"), "above")

    def test_relations_differ_with_error_details_without_raw_identifiers(self):
        p = profile()
        row = score(p, [p.advisories[0].id], "valid")
        a = feedback([row], [p], 2)["abstractions"][0]
        self.assertEqual(
            a,
            {
                "category": "version_excluded",
                "product_installed": True,
                "version_position": "above",
                "bound_rendering": "prose",
                "version_segments": 4,
                "alias_style": "package_manager",
                "advisory_count": 1,
            },
        )
        for style in ALIAS_STYLES:
            styled = replace(p, presentation={"components": [{"alias_style": style}]})
            self.assertEqual(feedback([row], [styled], 2)["abstractions"][0]["alias_style"], style)
        absent = replace(p, components=(replace(p.components[0], product="absent_canary"),))
        b = feedback([score(absent, [p.advisories[0].id], "valid")], [absent], 2)["abstractions"][0]
        self.assertEqual(b["category"], "unrelated_product")
        self.assertFalse(b["product_installed"])
        self.assertIsNone(b["version_position"])
        self.assertIsNone(b["version_segments"])
        malicious = replace(p, presentation={"components": [{"alias_style": "SECRET_CANARY"}]})
        self.assertIsNone(feedback([row], [malicious], 2)["abstractions"][0]["alias_style"])

    def test_missed_applicable_uses_an_applying_branch_and_failed_output_is_unknown(self):
        p = profile(installed="701.702.703.701")
        wrong = replace(p.advisories[0].terms[0], exact="9.8.7", upper=None)
        p = replace(
            p, advisories=(replace(p.advisories[0], terms=(wrong, *p.advisories[0].terms)),)
        )
        relation = feedback([score(p, [], "valid")], [p], 2)["abstractions"][0]
        self.assertEqual(relation["category"], "missed_applicable")
        self.assertEqual(relation["version_position"], "below")
        relation = feedback([score(p, None, "transport_failure")], [p], 2)["abstractions"][0]
        self.assertEqual(relation["category"], "transport_failure")
        self.assertIsNone(relation["product_installed"])

    def test_every_fixture_profile_privacy_and_deterministic_caps(self):
        profiles = [profile(i) for i in range(16)]
        rows = [score(p, [p.advisories[0].id], "valid") for p in profiles]
        for tier in (1, 2):
            serialized = canonical(feedback(rows, profiles, tier)).casefold()
            for p in profiles:
                forbidden = [
                    p.text,
                    *(v for c in p.components for v in (c.vendor, c.product, c.version)),
                    *(a.id for a in p.advisories),
                ]
                for a in p.advisories:
                    forbidden += [a.description[i : i + 12] for i in range(len(a.description) - 11)]
                    forbidden += [
                        v
                        for t in a.terms
                        for v in (t.vendor, t.product, t.exact, t.lower, t.upper)
                        if v
                    ]
                for value in forbidden:
                    self.assertNotIn(value.casefold(), serialized)
        t2 = feedback(rows, profiles, 2)
        self.assertEqual(len(t2["abstractions"]), 12)
        self.assertEqual(t2, feedback(list(reversed(rows)), list(reversed(profiles)), 2))
        t3 = feedback(rows, profiles, 3)
        self.assertEqual(len(t3["examples"]), 4)
        for i, example in enumerate(t3["examples"]):
            self.assertEqual(set(example), {"input", "prediction", "expected"})
            self.assertEqual(example["input"], profiles[i].executor_input())
            self.assertEqual(example["prediction"], rows[i]["prediction"])
            self.assertEqual(example["expected"], rows[i]["expected"])
        for tier in (1, 2, 3):
            self.assertEqual(
                feedback_request(NAIVE, rows, profiles, tier=tier, seed=1, candidate=0)["feedback"],
                feedback(rows, profiles, tier),
            )

    def test_frozen_strings_no_archived_instruction_contamination(self):
        code = Path(__file__).resolve().parents[1]
        archive = code.parent / "_archive/code/prompts"
        sources = (
            [p for p in archive.rglob("*") if p.is_file()]
            if archive.is_dir()
            else sorted((code / "fixtures").glob("expert_prompt*.txt"))
        )
        fragments = {
            text[i : i + 20]
            for path in sources
            for text in [path.read_text()]
            for i in range(len(text) - 19)
        }
        schema_only = {' {"applicable_cves":', 'ly {"applicable_cves', 'y {"applicable_cves"'}
        for name, text in (("naive", NAIVE), ("task", TASK)):
            overlaps = {text[i : i + 20] for i in range(len(text) - 19)} & fragments
            self.assertEqual(overlaps, schema_only if name == "naive" else set())
            for phrase in (
                "for each advisory",
                "per advisory",
                "per-advisory",
                "scaffold",
                "step by step",
            ):
                self.assertNotIn(phrase, text.casefold())
            plan = code.parent / "docs/v2_addressing_limitations.md"
            if plan.is_file():  # the study plan freezes both strings; absent in the code release
                self.assertIn(text, plan.read_text())
        for hint in (
            "trailing zero",
            "trailing-zero",
            "OR across",
            "OR clause",
            "vendor/product",
            "numeric versions",
            "matching semantics",
        ):
            self.assertNotIn(hint.casefold(), TASK.casefold())
