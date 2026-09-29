from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from promptbench.benchmark.admission import approved, audit_cna, screen
from promptbench.benchmark.build import build, write_jsonl
from promptbench.benchmark.cna_sources import record_url
from promptbench.benchmark.corroboration import TEXT_SCOPE_POLICY
from promptbench.benchmark.model import BenchmarkProfile
from promptbench.benchmark.oracle import Installed, parse_advisory
from promptbench.storage import IntegrityError, Store, digest
from tests.test_benchmark import nvd_record
from tests.test_corroboration import cna_record


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            self.root = Path(retained) / self._testMethodName
        else:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name)
        self.root.mkdir(parents=True, exist_ok=True)

    def fixture(self, textual_constraint=False):
        candidates, cna, parent = [
            Store(self.root / name) for name in ("candidates", "cna", "parent")
        ]
        records = [nvd_record(), nvd_record()]
        records[1]["id"] = "CVE-2099-1002"
        candidate_hash = write_jsonl(
            candidates.root / "advisories.jsonl", [parse_advisory(r).record() for r in records]
        )
        candidate_manifest = {"advisories_sha256": candidate_hash}
        candidates.put("manifest.json", candidate_manifest)
        partitions = {
            "cve_assignments": {r["id"]: "pilot_development" for r in records},
            "product_family_groups": {"vendor/product": "vendor/product"},
        }
        parent.put("partitions.json", partitions)
        parent.put("manifest.json", {"partition_sha256": digest(partitions)})
        cna.put(
            "acquisition-report.json",
            {
                "commit": "a" * 40,
                "candidate_manifest_sha256": digest(candidate_manifest),
                "unavailable": [],
            },
        )
        for index, record in enumerate(records):
            raw = cna_record()
            if textual_constraint and index == 0:
                raw["containers"]["cna"]["descriptions"] = [
                    {"lang": "en", "value": "This vulnerability affects only 64 bit ARM systems."}
                ]
            raw["cveMetadata"]["cveId"] = record["id"]
            if index == 1:
                raw["containers"]["cna"]["affected"][0].pop("defaultStatus")
            body = json.dumps(raw)
            cna.put(
                f"records/{record['id']}.json",
                {
                    "url": record_url("a" * 40, record["id"]),
                    "body_text": body,
                    "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "error": None,
                },
            )
        destination = self.root / "admission"
        manifest = screen(destination, candidates.root, cna.root, parent.root)
        review = {
            "text_scope_policy": TEXT_SCOPE_POLICY,
            "status": "admitted_after_source_review",
            "admission_manifest_sha256": digest(manifest),
            "quarantine": {},
            "records": {
                records[0]["id"]: {
                    "status": "corroborated",
                    "rationale": "Hand-labeled fictional fixture: vendor/product and [1.2,1.10) agree; explicit unaffected default.",
                }
            },
        }
        return destination, manifest, review, partitions

    def test_screen_excludes_unknown_defaults_and_preserves_original_splits(self):
        destination, manifest, review, partitions = self.fixture()
        self.assertEqual(manifest["candidate_count"], 2)
        self.assertEqual(manifest["corroborated_count"], 1)
        self.assertEqual(manifest["reason_counts"]["cna_unknown_or_nonnegative_default"], 1)
        identifiers, restored, _ = approved(destination, review)
        self.assertEqual(identifiers, {"CVE-2099-1001"})
        self.assertEqual(restored, partitions)

    def test_missing_conflicting_or_unbound_reviews_cannot_admit_scientific_data(self):
        destination, _, review, _ = self.fixture()
        variants = []
        missing = copy.deepcopy(review)
        missing["records"] = {}
        variants.append(missing)
        variants.append({**review, "admission_manifest_sha256": "0" * 64})
        variants.append({**review, "quarantine": {"CVE-2099-1001": "source conflict"}})
        for variant in variants:
            with self.assertRaises(IntegrityError):
                approved(destination, variant)
        with self.assertRaisesRegex(IntegrityError, "complete corroboration"):
            build(self.root / "invalid", self.root / "unused", review=review)

    def test_admission_detects_changed_cohort_bytes_even_with_valid_review(self):
        destination, _, review, _ = self.fixture()
        with (destination / "advisories.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(IntegrityError, "artifact changed"):
            approved(destination, review)

    def test_textual_constraints_cannot_bypass_review_quarantine(self):
        destination, _, review, _ = self.fixture(textual_constraint=True)
        with self.assertRaisesRegex(IntegrityError, "textual prerequisite"):
            approved(destination, review)
        review["quarantine"]["CVE-2099-1001"] = "64 bit ARM is outside the inventory scope."
        review["records"]["CVE-2099-1001"]["status"] = "quarantined"
        self.assertEqual(approved(destination, review)[0], set())

    def test_cna_decision_audit_checks_positives_negatives_and_changed_labels(self):
        advisory = parse_advisory(nvd_record())
        raw = {advisory.id: cna_record()}
        positive = BenchmarkProfile(
            "positive",
            "pilot_development",
            "fixture",
            (Installed("vendor", "product", "1.2"),),
            (advisory,),
            (advisory.id,),
            "easy",
            advisory.id,
        )
        negative = replace(
            positive,
            id="negative",
            components=(Installed("vendor", "product", "1.10"),),
            expected=(),
        )
        unrelated = replace(
            negative, id="unrelated", components=(Installed("vendor", "different", "1.2"),)
        )
        result = audit_cna([positive, negative, unrelated], raw)
        self.assertTrue(result["passed"])
        self.assertEqual(result["checked_decisions"], 3)
        wrong = audit_cna([replace(negative, expected=(advisory.id,))], raw)
        self.assertFalse(wrong["passed"])
        self.assertEqual(wrong["disagreements"][0]["statuses"], ["unaffected"])
