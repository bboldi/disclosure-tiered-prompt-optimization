import copy
import unittest
from dataclasses import replace
from pathlib import Path

from promptbench.domain import (
    ABSTRACTIONS,
    Advisory,
    Component,
    ContractError,
    feedback,
    load_profiles,
    metrics,
    parse_answer,
    parse_prompt,
    score,
    version,
)
from promptbench.experiment import read_fixture
from promptbench.storage import canonical

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/mini.json"


class Contracts(unittest.TestCase):
    def test_strict_answer_contract(self) -> None:
        invalid = [
            "",
            "[]",
            "null",
            "true",
            "{}",
            '{"other":[]}',
            '{"applicable_cves":null}',
            '{"applicable_cves":{}}',
            '{"applicable_cves":["CVE-2099-9999"]}',
            '{"applicable_cves":[1]}',
            '{"applicable_cves":[],"reason":"ignored"}',
            '{"applicable_cves":[],"applicable_cves":[]}',
            '{"applicable_cves":["CVE-2099-1001","CVE-2099-1001"]}',
            chr(96) * 3 + 'json\n{"applicable_cves":[]}\n' + chr(96) * 3,
            'Answer: {"applicable_cves":[]}',
            '{"applicable_cves":[NaN]}',
            '{"applicable_cves":["CVE-2099-1001"',
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ContractError):
                parse_answer(text, {"CVE-2099-1001"})
        self.assertEqual(parse_answer('{"applicable_cves":[]}', set()), [])
        for finish in ("length", "refusal", "tool_calls", ""):
            with self.subTest(finish=finish), self.assertRaises(ContractError):
                parse_answer('{"applicable_cves":[]}', set(), finish)

    def test_prompt_shape(self) -> None:
        for text in (
            "[]",
            '"prompt"',
            '{"prompt":42}',
            '{"prompt":""}',
            '{"prompt":"a","x":0}',
            '{"prompt":"literal\nnewline"}',
        ):
            with self.subTest(text=text), self.assertRaises(ContractError):
                parse_prompt(text)
        self.assertEqual(parse_prompt('{"prompt":"line1\\nline2"}'), "line1\nline2")

    def test_numeric_fixture_semantics(self) -> None:
        self.assertEqual(version("5.0"), version("5.0.0"))
        self.assertGreater(version("1.10"), version("1.2"))
        for value in ("1.0rc1", "*", "-", "1..0", "", "v1.0", "-1", "１.０"):
            with self.subTest(value=value), self.assertRaises(ContractError):
                version(value)
        component = Component("v", "p", "2.0")
        lower = Advisory("CVE-2099-0001", "generic", "v", "p", start="2", start_inclusive=False)
        upper = Advisory("CVE-2099-0002", "generic", "v", "p", end="2")
        self.assertFalse(lower.applies(component))
        self.assertFalse(upper.applies(component))
        self.assertTrue(replace(upper, end_inclusive=True).applies(component))
        self.assertFalse(replace(upper, vendor="other", end_inclusive=True).applies(component))

    def test_fixture_labels_and_leakage(self) -> None:
        data = read_fixture(FIXTURE)
        self.assertEqual(len(load_profiles(data)), 9)
        changed = copy.deepcopy(data)
        changed["profiles"][0]["expected"] = []
        with self.assertRaisesRegex(ContractError, "labels disagree"):
            load_profiles(changed)
        changed = copy.deepcopy(data)
        changed["profiles"][3]["advisories"][0]["id"] = "CVE-2099-1001"
        with self.assertRaisesRegex(ContractError, "overlaps"):
            load_profiles(changed)
        changed = copy.deepcopy(data)
        changed["profiles"][0]["advisories"][0]["start"] = "10"
        with self.assertRaisesRegex(ContractError, "inverted"):
            load_profiles(changed)

    def test_affected_only_classifier_and_tiers(self) -> None:
        profiles = load_profiles(read_fixture(FIXTURE))[:3]
        rows = [score(p, [a.id for a in p.advisories], "valid") for p in profiles]
        counts = feedback(rows, profiles, 1)["error_counts"]
        self.assertEqual(counts["version_excluded"], 2)
        self.assertEqual(counts["unrelated_product"], 2)
        for tier in (1, 2):
            text = canonical(feedback(rows, profiles, tier))
            for value in ("CANARY_PRIVATE", "Acme", "Bravo", "Cobalt", "CVE-2099", "1.2"):
                self.assertNotIn(value, text)
        self.assertIn("CANARY_PRIVATE", canonical(feedback(rows, profiles, 3)))
        abstractions = feedback(rows, profiles, 2)["abstractions"]
        self.assertTrue(all(item["category"] in ABSTRACTIONS for item in abstractions))
        self.assertEqual(len(abstractions), 4)

    def test_failed_answers_are_not_valid_empty_predictions(self) -> None:
        profile = load_profiles(read_fixture(FIXTURE))[0]
        failed = score(profile, None, "transport_failure")
        empty = score(profile, [], "valid")
        self.assertFalse(failed["valid"])
        self.assertIsNone(failed["prediction"])
        self.assertTrue(empty["valid"])
        self.assertEqual(empty["prediction"], [])
        self.assertEqual(metrics([failed])["coverage"], 0)
        self.assertEqual(metrics([empty])["coverage"], 1)
        self.assertEqual(failed["fp"], 1)
        self.assertEqual(failed["fn"], 1)

    def test_micro_f1_uses_counts_not_batch_averages(self) -> None:
        rows = [
            {"tp": 9, "fp": 0, "fn": 0, "valid": True},
            {"tp": 0, "fp": 0, "fn": 1, "valid": True},
        ]
        self.assertAlmostEqual(metrics(rows)["valid_answers"]["micro_f1"], 18 / 19)
