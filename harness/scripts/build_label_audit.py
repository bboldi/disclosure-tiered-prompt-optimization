#!/usr/bin/env python3
"""Blind label audit (round 1 revision, prespecified): a stratified sample of benchmark
decisions (sample A) and of the Real-Record Panel's matcher-decided cells (sample B), shown
as installed inventory plus the CVE record as published, without the label.

Writes the sheet and manifest to --out (the guide is hand-written beside them) and the answer key to --key-out, which must lie
outside the repository until the labels are frozen; the manifest records the key's hash.

    python scripts/build_label_audit.py --data ../data --panel ../data/real-record-panel \
        --cells ../data/real-record-panel/cells.csv --out <dir> --key-out <dir outside the repository>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

from build_real_record_panel import description, render_affected

SEED = 20261030
# sample A: benchmark decisions whose CVE concerns an installed product, uniform per stratum
A_STRATA = {"applicable": 30, "not_affected": 30}
# sample B: Real-Record Panel cells decided by the source study's matcher
B_STRATA = {"positives": 18, "near_misses": 18, "off_inventory": 24}
SHEET_COLS = ["row", "cve_id", "installed_components", "description", "affected"]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def inventory(components: list[dict]) -> str:
    return "; ".join(f"{c['vendor']} {c['product']} {c['version']}" for c in components)


def draw(rng: random.Random, pool: list, n: int) -> list:
    """A simple random sample of n items, in random order."""
    return rng.sample(sorted(pool, key=lambda x: json.dumps(x, sort_keys=True)), n)


def content_sha256(rows: list[dict]) -> str:
    """Hash of the columns the annotator must not change, in row order."""
    return hashlib.sha256(
        json.dumps([[r[k] for k in SHEET_COLS] for r in rows], ensure_ascii=False).encode()
    ).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--data", "--panel", "--cells", "--out", "--key-out"):
        ap.add_argument(name, type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(SEED)
    rows = []

    bench = args.data / "benchmark"
    profiles = {p["id"]: p for p in jsonl(bench / "profiles.jsonl")}
    cna = {d["cveMetadata"]["cveId"]: d for d in jsonl(bench / "selected-source-cna.jsonl")}
    decisions = json.load((bench / "cna-decision-audit.json").open())["payload"]["decisions"]
    related = [d for d in decisions if set(d["component_statuses"]) - {"unrelated_product"}]
    population_a = {
        "applicable": [d for d in related if d["applicable"]],
        "not_affected": [d for d in related if not d["applicable"]],
    }
    for stratum, n in A_STRATA.items():
        for d in draw(rng, population_a[stratum], n):
            doc = cna[d["cve_id"]]
            rows.append(
                {
                    "sample": "A",
                    "stratum": stratum,
                    "source": d["profile_id"],
                    "cve_id": d["cve_id"],
                    "installed_components": inventory(profiles[d["profile_id"]]["components"]),
                    "description": description(doc),
                    "affected": render_affected(doc),
                    "reference_label": "yes" if d["applicable"] else "no",
                }
            )

    panel = {p["id"]: p for p in jsonl(args.panel / "profiles.jsonl")}
    texts = {a["id"]: a["description"] for a in jsonl(args.panel / "advisories.jsonl")}
    cells = [c for c in csv.DictReader(args.cells.open()) if c["stratum"] == "matcher_decided"]
    for stratum, n in B_STRATA.items():
        for c in draw(rng, [c for c in cells if c["kind"] == stratum], n):
            p = panel[c["subprofile_id"]]
            rows.append(
                {
                    "sample": "B",
                    "stratum": stratum,
                    "source": c["subprofile_id"],
                    "cve_id": c["cve_id"],
                    "installed_components": inventory(p["components"]),
                    "description": texts[c["cve_id"]],
                    "affected": p["presentation"]["prose_advisories"][c["cve_id"]]["text"],
                    "reference_label": "yes" if c["matcher_label"] == "1" else "no",
                }
            )

    rng.shuffle(rows)
    for i, r in enumerate(rows, 1):
        r["row"] = f"R{i:03d}"
    args.out.mkdir(parents=True, exist_ok=True)
    args.key_out.mkdir(parents=True, exist_ok=True)
    with (args.out / "audit_sheet.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, SHEET_COLS + ["label", "note"])
        w.writeheader()
        w.writerows({k: r[k] for k in SHEET_COLS} | {"label": "", "note": ""} for r in rows)
    key_path = args.key_out / "audit_key.csv"
    with key_path.open("w", newline="") as f:
        w = csv.DictWriter(f, ["row", "sample", "stratum", "source", "cve_id", "reference_label"])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in rows)
    manifest = {
        "adr": "docs/adr/0043-round1-revision-experiments.md (label audit)",
        "seed": SEED,
        "sample_a": A_STRATA,
        "population_a": {k: len(v) for k, v in population_a.items()}
        | {"distinct_cves": {k: len({d["cve_id"] for d in v}) for k, v in population_a.items()}},
        "sample_b": B_STRATA,
        "population_b": {k: sum(c["kind"] == k for c in cells) for k in B_STRATA},
        "rows": len(rows),
        "key_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest(),
        "sheet_content_sha256": content_sha256(rows),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
