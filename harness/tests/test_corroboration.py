from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from promptbench.benchmark.build import write_jsonl
from promptbench.benchmark.cna_sources import record_url
from promptbench.benchmark.corroboration import corroborate, probes, status_for, text_scope_flags
from promptbench.benchmark.oracle import Unsupported, numeric, parse_advisory
from promptbench.storage import read_jsonl
from tests.test_benchmark import nvd_record


def cna_record():
    return {
        "dataType": "CVE_RECORD",
        "cveMetadata": {"cveId": "CVE-2099-1001", "state": "PUBLISHED"},
        "containers": {
            "cna": {
                "affected": [
                    {
                        "vendor": "Vendor",
                        "product": "Product",
                        "defaultStatus": "unaffected",
                        "versions": [
                            {
                                "version": "1.2",
                                "lessThan": "1.10",
                                "versionType": "semver",
                                "status": "affected",
                            }
                        ],
                    }
                ]
            }
        },
    }


class CorroborationTests(unittest.TestCase):
    def test_textual_environment_cues_are_visible_when_structured_fields_omit_them(self):
        raw = cna_record()
        raw["containers"]["cna"]["descriptions"] = [
            {
                "lang": "en",
                "value": "AES decryption on 64 bit ARM; Python bindings enabled in configuration.",
            }
        ]
        advisory = parse_advisory(nvd_record())
        self.assertEqual(corroborate(advisory, raw)["status"], "corroborated")
        self.assertEqual(
            text_scope_flags(advisory, raw),
            ["64 bit", "arm", "bindings", "configuration", "enabled"],
        )
        raw["containers"]["cna"]["descriptions"] = [
            {"lang": "en", "value": "Authenticated users can submit SQL input."}
        ]
        self.assertEqual(text_scope_flags(advisory, raw), [])

    def test_jsonl_preserves_unicode_line_separators_inside_public_descriptions(self):
        rows = [
            {"id": "one", "description": "first\u2028second\u0085third\u2029last"},
            {"id": "two"},
        ]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "advisories.jsonl"
            write_jsonl(path, rows)
            self.assertEqual(read_jsonl(path), rows)

    def test_explicit_boundaries_and_default_unaffected_are_corroborated(self):
        raw = cna_record()
        result = corroborate(parse_advisory(nvd_record()), raw)
        self.assertEqual(result["status"], "corroborated")
        statuses = {r["version"]: r["status"] for r in result["boundary_and_cell_checks"]}
        self.assertEqual(statuses["1.2"], "affected")
        self.assertEqual(statuses["1.10"], "unaffected")
        self.assertEqual(statuses["0"], "unaffected")
        self.assertEqual(status_for(raw["containers"]["cna"]["affected"][0], "1.2.0"), "affected")

    def test_unknown_or_omitted_default_never_becomes_negative(self):
        for mode in ("unknown", None):
            raw = cna_record()
            product = raw["containers"]["cna"]["affected"][0]
            if mode is None:
                del product["defaultStatus"]
            else:
                product["defaultStatus"] = mode
            self.assertEqual(status_for(product, "2"), "unknown")
            with self.assertRaisesRegex(Unsupported, "default"):
                corroborate(parse_advisory(nvd_record()), raw)

    def test_interval_cells_find_disagreement_between_documented_endpoints(self):
        source = nvd_record()
        source["configurations"][0]["nodes"][0]["cpeMatch"][0]["versionEndExcluding"] = "1.3"
        raw = cna_record()
        entry = raw["containers"]["cna"]["affected"][0]["versions"][0]
        del entry["lessThan"]
        entry["lessThanOrEqual"] = "1.2"
        with self.assertRaisesRegex(Unsupported, "predicate_disagreement"):
            corroborate(parse_advisory(source), raw)
        points = [
            numeric(v)
            for v in probes({numeric("1"), numeric("1.0.0.1"), numeric("1.1"), numeric("2")})
        ]
        for lower, upper in [(numeric("1"), numeric("1.0.0.1")), (numeric("1.1"), numeric("2"))]:
            self.assertTrue(any(lower < p < upper for p in points))

    def test_product_constraints_overlaps_and_status_changes_are_rejected(self):
        alterations = [
            lambda p: p.update(product="Different"),
            lambda p: p.update(platforms=["Windows"]),
            lambda p: p["versions"].append(copy.deepcopy(p["versions"][0])),
            lambda p: p["versions"][0].update(changes=[{"at": "1.3", "status": "unaffected"}]),
            lambda p: p.update(cpes=["cpe:2.3:a:vendor:product:1.2:*:*:*:*:*:*:*"]),
        ]
        for change in alterations:
            raw = cna_record()
            change(raw["containers"]["cna"]["affected"][0])
            with self.assertRaises(Unsupported):
                corroborate(parse_advisory(nvd_record()), raw)

    def test_source_urls_pin_revision_and_preserve_cve_directory_padding(self):
        self.assertIn("/2024/0xxx/CVE-2024-0937.json", record_url("a" * 40, "CVE-2024-0937"))
        for revision, identifier in [("main", "CVE-2024-1234"), ("a" * 40, "../secret")]:
            with self.assertRaises(ValueError):
                record_url(revision, identifier)
