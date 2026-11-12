#!/usr/bin/env python3
"""Post-hoc: matcher-decided cells of the Real-Record Panel whose label has one of the three
defects found by the label audit. The labels stay as they are in the
paper; this list feeds only the rescoring sensitivity of g34_analysis.py --corrections.

- wrong_product: positive only through a name of another vendor's different product (the
  pairs below were read by hand);
- adp_over_cna: positive only through an ADP row, while a related CNA row says not affected,
  or lists only exact versions under a default of unaffected that the matcher cannot compare;
- unsplit_list: negative, but the component's words are one item of a comma-separated
  product list of the same vendor, and its version is affected.

Run with the early-signals interpreter (as build_real_record_panel.py):

    python scripts/g34_label_corrections.py --early-signals <early-signals repository> \
        --panel ../data/real-record-panel --out <dir>/g34_label_corrections.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

from build_real_record_panel import INGESTION_MAX_ID

WRONG = {  # (component product, record product prefix)
    ("Cisco Smart Software Manager On-Prem", "Smart Manager"),
    ("Web Server", "Red Hat JBoss Web Server"),
    ("DICOM Viewer Pro", "DICOM Viewer"),
    ("DICOM Viewer Pro", "dicom_viewer"),
    ("Solution Builder", "ShopEngine"),
}
WRONG_VENDORS = {"DICOM Viewer Pro": {"microdicom"}}  # Santesoft's own entries are the product
AUDITED = {  # reference errors adjudicated in the audit: the rules must find them
    ("real_07_1", "CVE-2024-0566"): "wrong_product",
    ("real_02_2", "CVE-2024-6077"): "adp_over_cna",
    ("real_09_5", "CVE-2025-65078"): "unsplit_list",
}
NOT_DEFECTS = {
    ("real_07_2", "CVE-2023-4398"),
    ("real_07_4", "CVE-2023-4398"),
    ("real_07_4", "CVE-2024-11667"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--early-signals", "--panel", "--out"):
        ap.add_argument(name, type=Path, required=True)
    args = ap.parse_args()
    app = args.early_signals.resolve() / "app"
    sys.path.insert(0, str(app / "src"))
    from earlysignals.profiles import _VENDOR_NOISE, NameIndex, _tokens
    from earlysignals.versions import affects

    truth = sqlite3.connect(f"file:{app / 'data/truth.sqlite'}?mode=ro", uri=True)
    ingestion = sqlite3.connect(f"file:{app / 'data/ingestion.sqlite'}?mode=ro", uri=True)
    names = NameIndex(truth)
    profiles = {
        p["id"]: p for p in map(json.loads, (args.panel / "profiles.jsonl").open(encoding="utf-8"))
    }

    def vendor_words(v):
        return frozenset(_tokens(v or "")) - _VENDOR_NOISE

    def is_wrong(component, vendor, product):
        return any(component == c and (product or "").startswith(p) for c, p in WRONG) and (
            component not in WRONG_VENDORS or (vendor or "").lower() in WRONG_VENDORS[component]
        )

    def cna_denies(verdict, status, versions):
        """The CNA row excludes the version: decided so, or exact versions only, default unaffected."""
        if verdict is False:
            return True
        vs = json.loads(versions) if versions else []
        return (
            verdict is None
            and status == "unaffected"
            and bool(vs)
            and all(
                x.get("status") == "affected"
                and "lessThan" not in x
                and "lessThanOrEqual" not in x
                and str(x.get("version", "")).strip().lower()
                not in ("", "n/a", "unspecified", "0", "*")
                # "4.16 through 5.37" is a prose range, not an exact version
                and not re.search(
                    r"through|thru|before|prior|\bto\b|[<>≤]", str(x.get("version", ""))
                )
                for x in vs
            )
        )

    def cna_rows(cve, rows):
        """How many leading rows are the CNA's, from the stored record with exactly these rows."""
        for (content,) in ingestion.execute(
            "SELECT content FROM raw_item WHERE source_id = 'cvelist' AND external_id = ?"
            " AND id <= ? ORDER BY id DESC",
            (cve, INGESTION_MAX_ID),
        ):
            doc = json.loads(content)
            listed = (doc["containers"].get("cna") or {}).get("affected") or []
            entries = listed + [
                e for a in doc["containers"].get("adp") or [] for e in a.get("affected") or []
            ]
            if len([e for e in entries if isinstance(e, dict)]) == len(rows):
                return len([e for e in listed if isinstance(e, dict)])
        raise SystemExit(f"{cve}: no stored record with these rows")

    out = []
    for r in csv.DictReader((args.panel / "cells.csv").open(newline="")):
        if r["stratum"] != "matcher_decided":
            continue
        sub, cve, label = r["subprofile_id"], r["cve_id"], int(r["matcher_label"])
        rows = truth.execute(
            "SELECT vendor, product, default_status, versions_json FROM cve_product"
            " WHERE cve_id = ? ORDER BY position",
            (cve,),
        ).fetchall()
        pattern, evidence = None, ""
        if label:
            cna = cna_rows(cve, rows)
            hits = []  # (is CNA, verdict, wrong product, component, row)
            for c in profiles[sub]["components"]:
                keys = {
                    ((v or "").lower(), p.lower())
                    for v, p in names.related(c["vendor"], c["product"])
                }
                for i, (v, p, s, vj) in enumerate(rows):
                    if ((v or "").lower(), (p or "").lower()) in keys:
                        verdict = affects(json.loads(vj) if vj else None, s, c["version"])
                        hits.append(
                            (i < cna, verdict, is_wrong(c["product"], v, p), c["product"], p, s, vj)
                        )
            right = [h for h in hits if h[1] is True and not h[2]]
            if not right:
                pattern = "wrong_product"
                evidence = "; ".join(sorted({f"{h[3]} ~ {h[4]}" for h in hits if h[1] is True}))
            elif not any(h[0] for h in right) and any(
                h[0] and not h[2] and cna_denies(h[1], h[5], h[6]) for h in hits
            ):
                pattern = "adp_over_cna"
                evidence = "; ".join(sorted({f"{h[3]} ~ {h[4]}" for h in right}))
        else:
            for c in profiles[sub]["components"]:
                want = set(_tokens(c["product"])) - {"firmware"}
                for v, p, s, vj in rows:
                    items = [set(_tokens(x)) for x in re.split(r",", p or "")]
                    if (
                        len(items) > 1
                        and want in items
                        and vendor_words(v) & vendor_words(c["vendor"])
                        and affects(json.loads(vj) if vj else None, s, c["version"]) is True
                    ):
                        pattern, evidence = "unsplit_list", f"{c['product']} {c['version']}"
        if pattern:
            out.append(
                {
                    "subprofile_id": sub,
                    "cve_id": cve,
                    "kind": r["kind"],
                    "matcher_label": label,
                    "corrected_label": 1 - label,
                    "pattern": pattern,
                    "evidence": evidence,
                }
            )
    found = {(x["subprofile_id"], x["cve_id"]): x["pattern"] for x in out}
    for cell, pattern in AUDITED.items():
        assert found.get(cell) == pattern, (cell, found.get(cell))
    assert not NOT_DEFECTS & found.keys()
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for p in sorted({x["pattern"] for x in out}):
        print(p, sum(x["pattern"] == p for x in out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
