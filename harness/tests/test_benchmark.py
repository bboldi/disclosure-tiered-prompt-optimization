from __future__ import annotations

import copy
import gzip
import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from promptbench.benchmark.build import assign_pools, audit_profiles, family, generate_pool
from promptbench.benchmark.model import score
from promptbench.benchmark.oracle import Installed, Unsupported, numeric, parse_advisory
from promptbench.benchmark.reference import check, compare
from promptbench.benchmark.sources import acquire, parse_meta, verify_feed
from promptbench.storage import IntegrityError, Store


class SourceTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            self.root = Path(retained) / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            temp = tempfile.TemporaryDirectory()
            self.addCleanup(temp.cleanup)
            self.root = Path(temp.name)
        self.raw = b'{"vulnerabilities": []}'
        self.compressed = gzip.compress(self.raw, mtime=0)
        self.meta = (
            f"lastModifiedDate:2026-09-12T00:00:00Z\nsize:{len(self.raw)}\n"
            f"gzSize:{len(self.compressed)}\nsha256:{hashlib.sha256(self.raw).hexdigest()}\n"
        ).encode()

    def response(self, body):
        response = io.BytesIO(body)
        response.status = 200
        response.headers = {"content-type": "application/octet-stream"}
        response.url = "https://nvd.nist.gov/fixture"
        return response

    def test_source_resume_verifies_existing_bytes_without_network(self):
        opener = Mock()
        opener.open.side_effect = [self.response(self.meta), self.response(self.compressed)]
        with patch(
            "promptbench.benchmark.sources.urllib.request.build_opener", return_value=opener
        ):
            first = acquire(self.root, (2023,))
        self.assertEqual(opener.open.call_count, 2)
        opener.reset_mock(side_effect=True)
        with patch(
            "promptbench.benchmark.sources.urllib.request.build_opener", return_value=opener
        ):
            self.assertEqual(first, acquire(self.root, (2023,)))
        opener.open.assert_not_called()
        self.assertEqual(first["2023"]["uncompressed_sha256"], hashlib.sha256(self.raw).hexdigest())

    def test_feed_verification_rejects_hash_size_and_metadata_corruption(self):
        path = self.root / "fixture.gz"
        path.write_bytes(self.compressed)
        expected = parse_meta(self.meta)
        verify_feed(path, expected)
        for altered in (
            {**expected, "uncompressed_sha256": "0" * 64},
            {**expected, "compressed_bytes": len(self.compressed) + 1},
            {**expected, "uncompressed_bytes": len(self.raw) - 1},
        ):
            with self.assertRaises(IntegrityError):
                verify_feed(path, altered)
        with self.assertRaises(IntegrityError):
            parse_meta(self.meta + b"size:12\n")

    def test_failed_source_download_preserves_partial_and_has_no_receipt(self):
        opener = Mock()
        opener.open.side_effect = [self.response(self.meta), self.response(b"short")]
        with patch(
            "promptbench.benchmark.sources.urllib.request.build_opener", return_value=opener
        ):
            with self.assertRaises(IntegrityError):
                acquire(self.root, (2023,))
        store = Store(self.root)
        self.assertFalse(store.exists("sources/2023/receipt.json"))
        self.assertEqual(len(store.names("sources/2023/downloads/*/*.partial")), 1)
        outcome = store.get(store.names("sources/2023/downloads/*/outcome.json")[0])
        self.assertEqual(outcome["status"], "failed")


def nvd_record(matches=None):
    if matches is None:
        matches = [
            {
                "vulnerable": True,
                "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                "versionStartIncluding": "1.2",
                "versionEndExcluding": "1.10",
            }
        ]
    return {
        "id": "CVE-2099-1001",
        "vulnStatus": "Analyzed",
        "published": "2024-01-01T00:00:00.000",
        "lastModified": "2026-09-12T00:00:00.000",
        "descriptions": [{"lang": "en", "value": "Fictional oracle engineering example."}],
        "configurations": [{"nodes": [{"operator": "OR", "negate": False, "cpeMatch": matches}]}],
    }


class OracleTests(unittest.TestCase):
    def retain(self, value):
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            Store(Path(retained) / self._testMethodName).put("oracle-evidence.json", value)

    def test_hand_labeled_boundaries_product_identity_and_trailing_zeros(self):
        cases = [
            ("vendor", "product", "1.1", False),
            ("vendor", "product", "1.2", True),
            ("vendor", "product", "1.2.0", True),
            ("vendor", "product", "1.9", True),
            ("vendor", "product", "1.10", False),
            ("vendor", "product", "2", False),
            ("VENDOR", "PRODUCT", "1.3", True),
            ("other", "product", "1.3", False),
            ("vendor", "product_extra", "1.3", False),
        ]
        raw = nvd_record()
        advisory = parse_advisory(raw)
        observations = []
        for vendor, product, version, expected in cases:
            component = Installed(vendor, product, version)
            observed = advisory.applies((component,))
            reference = check(raw, [{"vendor": vendor, "product": product, "version": version}])
            self.assertEqual(observed, expected)
            self.assertEqual(reference, expected)
            observations.append(
                {
                    "case": [vendor, product, version],
                    "expected": expected,
                    "production": observed,
                    "reference": reference,
                }
            )
        self.assertFalse(advisory.applies((Installed("vendor", "product", "1.3", part="o"),)))
        self.retain({"source": raw, "hand_labeled_cases": observations})

    def test_exact_singleton_or_ranges_and_escaped_literal_names(self):
        exact = {"vulnerable": True, "criteria": r"cpe:2.3:a:vendor:prod\:uct:2.0:*:*:*:*:*:*:*"}
        second = {
            "vulnerable": True,
            "criteria": "cpe:2.3:a:vendor:other:*:*:*:*:*:*:*:*",
            "versionStartIncluding": "3",
            "versionEndIncluding": "3.0",
        }
        raw = nvd_record([exact, second])
        advisory = parse_advisory(raw)
        cases = [
            ("prod:uct", "2", True),
            ("prod:uct", "2.1", False),
            ("other", "3", True),
            ("other", "3.1", False),
        ]
        for product, version, expected in cases:
            self.assertEqual(advisory.applies((Installed("vendor", product, version),)), expected)
            self.assertEqual(
                check(raw, [{"vendor": "vendor", "product": product, "version": version}]), expected
            )
        self.retain({"source": raw, "hand_labeled_cases": cases})

    def test_unsupported_conditions_never_become_negative_labels(self):
        mutations = [
            ("nonapplication_cpe", {"criteria": "cpe:2.3:o:vendor:product:*:*:*:*:*:*:*:*"}),
            ("nonvulnerable_or_prerequisite", {"vulnerable": False}),
            ("nonliteral_product", {"criteria": "cpe:2.3:a:vendor:prod*:*:*:*:*:*:*:*:*"}),
            ("unsupported_cpe_escape", {"criteria": r"cpe:2.3:a:vendor:product:\*:*:*:*:*:*:*:*"}),
            (
                "additional_cpe_constraints",
                {"criteria": "cpe:2.3:a:vendor:product:*:beta:*:*:*:*:*:*"},
            ),
            ("unsupported_numeric_version", {"versionEndExcluding": "1.10rc1"}),
            ("unsupported_numeric_version", {"versionEndExcluding": None}),
            ("unsupported_numeric_version", {"versionEndExcluding": "01.10"}),
            ("conflicting_range_endpoints", {"versionEndIncluding": "2.0"}),
            ("empty_or_inverted_range", {"versionEndExcluding": "1.2"}),
            ("empty_or_inverted_range", {"versionEndExcluding": "1.1"}),
            ("exact_plus_range", {"criteria": "cpe:2.3:a:vendor:product:1.2:*:*:*:*:*:*:*"}),
        ]
        for reason, mutation in mutations:
            raw = nvd_record()
            raw["configurations"][0]["nodes"][0]["cpeMatch"][0].update(mutation)
            with self.assertRaisesRegex(Unsupported, reason):
                parse_advisory(raw)
        for level in ("configuration", "node"):
            for mutation in ({"operator": "AND"}, {"negate": True}):
                raw = nvd_record()
                target = raw["configurations"][0]
                if level == "node":
                    target = target["nodes"][0]
                target.update(mutation)
                with self.assertRaises(Unsupported):
                    parse_advisory(raw)
        self.retain(
            {
                "unsupported_match_cases": mutations,
                "logical_cases": ["AND configuration", "AND node", "negation"],
            }
        )

    def test_numeric_comparators_and_raw_oracle_agree_on_endpoint_grid(self):
        versions = ["0", "0.0", "1", "1.0", "1.2", "1.2.0", "1.9", "1.10", "2", "10"]
        checked = 0
        for left in versions:
            for right in versions:
                self.assertEqual(
                    compare(left, right),
                    (numeric(left) > numeric(right)) - (numeric(left) < numeric(right)),
                )
        for lower_inclusive in (False, True):
            for upper_inclusive in (False, True):
                raw = copy.deepcopy(nvd_record())
                match = raw["configurations"][0]["nodes"][0]["cpeMatch"][0]
                del match["versionStartIncluding"]
                del match["versionEndExcluding"]
                match["versionStartIncluding" if lower_inclusive else "versionStartExcluding"] = (
                    "1.2"
                )
                match["versionEndIncluding" if upper_inclusive else "versionEndExcluding"] = "1.10"
                advisory = parse_advisory(raw)
                for version in versions:
                    self.assertEqual(
                        advisory.applies((Installed("vendor", "product", version),)),
                        check(
                            raw, [{"vendor": "vendor", "product": "product", "version": version}]
                        ),
                    )
                    checked += 1
        self.retain({"version_pairs": 100, "range_component_decisions": checked})


class ConstructionTests(unittest.TestCase):
    def make_pool(self, count=96):
        records = []
        for index in range(count):
            raw = nvd_record()
            raw["id"] = f"CVE-2023-{1000 + index}"
            match = raw["configurations"][0]["nodes"][0]["cpeMatch"][0]
            match["criteria"] = f"cpe:2.3:a:vendor:product_{index % 12}:*:*:*:*:*:*:*:*"
            match["versionStartIncluding"] = str((index // 12) % 3 + 1)
            match["versionEndExcluding"] = str((index // 12) % 3 + 2)
            records.append(raw)
        return records, [parse_advisory(raw) for raw in records]

    def test_generation_is_repeatable_balanced_and_raw_oracle_checked(self):
        records, pool = self.make_pool()
        profiles = generate_pool(pool, "pilot_development", 12, 42)
        again = generate_pool(pool, "pilot_development", 12, 42)
        self.assertEqual(profiles, again)
        self.assertEqual(sum(bool(p.expected) for p in profiles), 6)
        self.assertTrue(all(15 <= len(p.advisories) <= 25 for p in profiles))
        self.assertTrue(all(6 <= len(p.components) <= 10 for p in profiles))
        assignments = {a.id: "pilot_development" for a in pool}
        groups = {family(p): family(p) for a in pool for p in a.products}
        audit = audit_profiles(profiles, {r["id"]: r for r in records}, assignments, groups)
        self.assertTrue(audit["passed"])
        for profile in profiles:
            visible = profile.executor_input()
            self.assertEqual(set(visible), {"system_profile", "advisories"})
            self.assertTrue(
                all(set(a) == {"id", "description", "affected"} for a in visible["advisories"])
            )
            scored = score(profile, list(profile.expected), "valid")
            self.assertEqual((scored["fp"], scored["fn"]), (0, 0))
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            Store(Path(retained) / self._testMethodName).put(
                "construction.json", {"audit": audit, "profiles": [p.record() for p in profiles]}
            )

    def test_temporal_assignment_uses_publication_not_cve_identifier_year(self):
        records, _ = self.make_pool(100)
        for index, raw in enumerate(records):
            raw["published"] = "2025-02-01T00:00:00.000"
            raw["configurations"][0]["nodes"][0]["cpeMatch"][0]["criteria"] = (
                f"cpe:2.3:a:vendor:distinct_{index}:*:*:*:*:*:*:*:*"
            )
        pool = [parse_advisory(raw) for raw in records]
        assignments, _ = assign_pools(pool, 42)
        self.assertIn("temporal", assignments.values())
        self.assertIn("product_heldout", assignments.values())
        self.assertTrue(set(assignments.values()) <= {"temporal", "product_heldout"})

    def test_negative_advisory_cannot_cross_partition(self):
        records, pool = self.make_pool()
        profile = next(p for p in generate_pool(pool, "pilot_development", 6, 7) if not p.expected)
        assignments = {a.id: "pilot_development" for a in pool}
        assignments[profile.advisories[0].id] = "test"
        groups = {family(p): family(p) for a in pool for p in a.products}
        with self.assertRaisesRegex(IntegrityError, "crosses its assigned partition"):
            audit_profiles([profile], {r["id"]: r for r in records}, assignments, groups)

    def test_connected_and_punctuation_variant_products_share_a_group(self):
        records, _ = self.make_pool()
        match = records[0]["configurations"][0]["nodes"][0]["cpeMatch"][0]
        linked = copy.deepcopy(match)
        linked["criteria"] = "cpe:2.3:a:vendor:second_product:*:*:*:*:*:*:*:*"
        records[0]["configurations"][0]["nodes"][0]["cpeMatch"].append(linked)
        pool = [parse_advisory(raw) for raw in records]
        _, groups = assign_pools(pool, 42)
        self.assertEqual(
            groups[family(("vendor", "product_0"))], groups[family(("vendor", "second_product"))]
        )
        self.assertEqual(family(("vendor", "second_product")), family(("Vendor", "second-product")))
