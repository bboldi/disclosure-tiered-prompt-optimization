from __future__ import annotations

import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from promptbench.benchmark.build import (
    ALIAS_STYLES,
    audit_profiles,
    difficulty_statistics,
    family,
    generate_pool,
    inventory_name,
    prose_bound,
    release_branches,
    sample_version,
    version_choices,
)
from promptbench.benchmark.model import restore_profile
from promptbench.benchmark.oracle import Criterion, Installed
from promptbench.domain import Component, Profile
from promptbench.live.conditions import executor_request, local_conditions
from promptbench.storage import IntegrityError, Store
from tests import test_benchmark as fixtures


class DifficultyTests(unittest.TestCase):
    def test_weighted_versions_stay_in_existing_choices_and_preserve_default_sampling(self):
        choices = version_choices(
            Criterion("v", "p", None, "2.4", "2.4.9", True, False, "fixture", None)
        )
        rng, reference = random.Random(42), random.Random(42)
        self.assertEqual(
            [sample_version(choices, rng, None) for _ in range(40)],
            [reference.choice(choices) for _ in range(40)],
        )
        rng = random.Random(42)
        weighted = [sample_version(choices, rng, {"appended_1": 10000}) for _ in range(40)]
        self.assertTrue(all(v in choices for v in weighted))
        self.assertGreaterEqual(sum(v["type"] == "appended_1" for v in weighted), 39)
        for weight in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                sample_version(choices, rng, {"appended_1": weight})

    def test_calibrated_statistics_report_actual_configuration_and_roundtrip(self):
        _, pool = fixtures.ConstructionTests().make_pool()
        configuration = {
            "perturbation_fraction": {"easy": 0.4, "medium": 0.8, "hard": 1},
            "version_type_weights": {"appended_0": 4, "appended_1": 4},
            "alias_styles": ["package_manager"],
            "advisory_ranges": {"easy": [21, 23], "medium": [22, 25], "hard": [23, 25]},
            "prose_fraction": 0.6,
            "calibration_parent": "fixture-pilot-development-only",
        }
        profiles = generate_pool(pool, "pilot_development", 24, 42, difficulty=configuration)
        self.assertTrue(all(len(p.advisories) >= 21 for p in profiles))
        self.assertTrue(
            all(
                m["alias_style"] == "package_manager"
                for p in profiles
                for m in p.presentation["components"]
            )
        )
        stats = difficulty_statistics(profiles, configuration)
        self.assertEqual(stats["generator_configuration"], configuration)
        self.assertEqual(
            stats["perturbation_fraction_target"], configuration["perturbation_fraction"]
        )
        self.assertEqual(stats["prose_sampling_fraction_of_supported_advisories"], 0.6)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            store.put("statistics.json", stats)
            self.assertEqual(store.get("statistics.json"), stats)

    def test_statistics_survive_checksummed_json_roundtrip(self):
        _, pool = fixtures.ConstructionTests().make_pool()
        profiles = generate_pool(pool, "pilot_development", 24, 42)
        stats = difficulty_statistics(profiles)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            store.put("stats.json", stats)
            self.assertEqual(store.get("stats.json"), stats)

    def test_each_alias_style(self):
        component = Installed("nozominetworks", "cmc", "22.6.1")
        expected = [
            "cmc 22.6.1",
            "nozominetworks-cmc 22.6.1",
            "nozominetworks/cmc@22.6.1",
            "Nozomi Networks CMC 22.6.1",
            "nozominetworks cmc 22.6.1",
        ]
        self.assertEqual([inventory_name(component, s) for s in ALIAS_STYLES], expected)
        self.assertEqual(
            inventory_name(Installed("some_vendor", "some_product", "1"), "underscore_to_space"),
            "some vendor some product 1",
        )
        with self.assertRaises(ValueError):
            inventory_name(component, "invalid")

    def test_all_near_miss_types_and_numeric_lexicographic_trap(self):
        term = Criterion("v", "p", None, "2.4", "2.4.9", True, False, "fixture", None)
        choices = version_choices(term)
        for version, kind, bound in (
            ("2.4", "on_inclusive", "2.4"),
            ("2.4.9", "on_exclusive", "2.4.9"),
            ("2.4.8", "last_segment_minus_one", "2.4.9"),
            ("2.4.10", "last_segment_plus_one", "2.4.9"),
            ("2.4.0", "appended_0", "2.4"),
            ("2.4.1", "appended_1", "2.4"),
        ):
            self.assertIn({"version": version, "type": kind, "bound": bound}, choices)
        self.assertEqual(
            {c["type"] for c in choices},
            {
                "on_inclusive",
                "on_exclusive",
                "last_segment_minus_one",
                "last_segment_plus_one",
                "appended_0",
                "appended_1",
                "numeric_lexicographic_trap",
            },
        )
        self.assertIn(
            {"version": "2.4.10", "type": "numeric_lexicographic_trap", "bound": "2.4.9"}, choices
        )
        self.assertTrue(term.applies(Installed("v", "p", "2.4.0")))
        self.assertTrue(term.applies(Installed("v", "p", "2.4.1")))
        self.assertFalse(term.applies(Installed("v", "p", "2.4.9")))
        self.assertFalse(term.applies(Installed("v", "p", "2.4.10")))
        self.assertNotIn("-1", [c["version"] for c in version_choices(replace(term, lower="0"))])

    def test_default_input_no_components_and_enum_is_explicit_control(self):
        fixture = Profile(
            "fixture", "optimization", "inventory", (Component("v", "p", "1"),), (), ()
        )
        self.assertNotIn("components", fixture.executor_input())
        self.assertIn("components", fixture.executor_input(structured_inventory=True))
        _, pool = fixtures.ConstructionTests().make_pool()
        profile = generate_pool(pool, "pilot_development", 1, 42)[0]
        condition = local_conditions(
            {
                "local_models": [
                    {
                        "requested_tag": "fixture",
                        "capabilities": [],
                        "ollama_manifest_digest": "fixture",
                    }
                ]
            }
        )[0]
        self.assertFalse(condition["schema_enum"])
        request = executor_request(condition, "naive", profile)
        self.assertNotIn('"components"', request["messages"][1]["content"])
        self.assertNotIn("enum", request["format"]["properties"]["applicable_cves"]["items"])
        scaffold = executor_request(
            {**condition, "schema_enum": True, "structured_inventory": True}, "naive", profile
        )
        self.assertIn('"components"', scaffold["messages"][1]["content"])
        self.assertEqual(
            scaffold["format"]["properties"]["applicable_cves"]["items"]["enum"],
            [a.id for a in profile.advisories],
        )
        self.assertNotIn("presentation", request["messages"][1]["content"])
        self.assertEqual(restore_profile(profile.record(), {a.id: a for a in pool}), profile)

    def test_distractors_never_leak_positives_and_strata_balance(self):
        records, pool = fixtures.ConstructionTests().make_pool()
        profiles = generate_pool(pool, "pilot_development", 24, 42)
        for p in profiles:
            distractors = set(p.presentation["distractor_ids"])
            self.assertGreaterEqual(len(distractors) / len(p.advisories), 0.30)
            self.assertFalse(distractors & set(p.expected))
            self.assertTrue(
                all(
                    not any(a.matches_product(c) for c in p.components)
                    for a in p.advisories
                    if a.id in distractors
                )
            )
            if p.stratum == "hard":
                self.assertEqual(len(p.presentation["branch_pair"]), 2)
                pair = [a for a in p.advisories if a.id in p.presentation["branch_pair"]]
                self.assertTrue(
                    any(
                        release_branches(pair[0], k) != release_branches(pair[1], k)
                        for k in pair[0].products & pair[1].products
                    )
                )
        stats = difficulty_statistics(profiles)
        for row in stats["partition_strata"]:
            self.assertEqual(row["positive_profiles"] * 2, row["profiles"])
        broken = replace(profiles[0], text="incorrect inventory")
        with self.assertRaises(IntegrityError):
            audit_profiles(
                [broken],
                {r["id"]: r for r in records},
                {a.id: "pilot_development" for a in pool},
                {family(k): family(k) for a in pool for k in a.products},
            )

    def test_prose_requires_complete_agreeing_cna_text(self):
        _, pool = fixtures.ConstructionTests().make_pool()
        advisory = replace(
            pool[0],
            terms=(
                Criterion("vendor", "product", None, "0", "2.4.9", True, False, "fixture", None),
            ),
        )
        raw = {
            "containers": {
                "cna": {
                    "descriptions": [
                        {"lang": "en", "value": "Product versions prior to 2.4.9 are affected."}
                    ]
                }
            }
        }
        prose = prose_bound(advisory, raw)
        self.assertEqual(prose["text"], "vendor/product: versions prior to 2.4.9 are affected")
        for term in (
            replace(advisory.terms[0], lower="1"),
            replace(advisory.terms[0], upper_inclusive=True),
            replace(advisory.terms[0], upper="2.4.8"),
        ):
            self.assertIsNone(prose_bound(replace(advisory, terms=(term,)), raw))
        self.assertIsNone(prose_bound(replace(advisory, terms=advisory.terms * 2), raw))

    def test_unknown_cna_decisions_are_not_selected(self):
        from promptbench.benchmark.corroboration import status_for

        _, pool = fixtures.ConstructionTests().make_pool()
        records = {}
        for advisory in pool:
            term = advisory.terms[0]
            records[advisory.id] = {
                "containers": {
                    "cna": {
                        "affected": [
                            {
                                "vendor": term.vendor,
                                "product": term.product,
                                "defaultStatus": "unknown",
                                "versions": [
                                    {
                                        "version": term.lower,
                                        "lessThan": term.upper,
                                        "status": "affected",
                                    }
                                ],
                            }
                        ]
                    }
                }
            }
        profile = generate_pool(pool, "pilot_development", 1, 42, cna_records=records)[0]
        for advisory in profile.advisories:
            source = records[advisory.id]["containers"]["cna"]["affected"][0]
            for component in profile.components:
                if advisory.matches_product(component):
                    self.assertIn(status_for(source, component.version), ("affected", "unaffected"))
