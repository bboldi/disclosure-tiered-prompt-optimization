#!/usr/bin/env python3
"""Label audit analysis (round 1 revision), fixed before any labelling.

Per sample and stratum, over all sampled rows: the share of definite answers (yes/no) that
disagree with the reference label, the share of unsure answers, and the disagreement among
definite answers only. Per sample, population-weighted versions of the first two with
conservative intervals (stratum Clopper-Pearson limits at Bonferroni-adjusted level,
weighted), and the sensitivity range of disagreement if every unsure answer agreed or
disagreed. With --second, the three-category agreement table and Cohen's kappa between the
two annotators, per sample.

    python scripts/label_audit_analysis.py --sheet ../data/real-record-panel/label-audit/audit_sheet.csv \
        --key ../data/real-record-panel/label-audit/audit_key.csv \
        --manifest ../data/real-record-panel/label-audit/manifest.json --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

from build_label_audit import SHEET_COLS, content_sha256
from scipy.stats import beta

ANSWERS = ("yes", "no", "unsure")


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return [float(lo), float(hi)]


def labels(path: Path, manifest: dict) -> dict[str, str]:
    """The sheet's answers, after checking it is the committed sheet with answers added."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        cells = list(reader)
    if header != SHEET_COLS + ["label", "note"]:
        sys.exit(f"{path}: columns must be exactly {SHEET_COLS + ['label', 'note']}, got {header}")
    if any(len(c) != len(header) for c in cells):
        sys.exit(f"{path}: a row has missing or surplus cells")
    rows = [dict(zip(header, c, strict=True)) for c in cells]
    ids = [r["row"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit(f"{path}: duplicate row ids")
    if content_sha256(rows) != manifest["sheet_content_sha256"]:
        sys.exit(f"{path}: rows, order or record text differ from the committed sheet")
    out = {r["row"]: r["label"].strip().lower() for r in rows}
    bad = sorted(k for k, v in out.items() if v not in ANSWERS)
    if bad:
        sys.exit(f"{path}: rows without a yes/no/unsure label: {bad[:10]}")
    return out


def weighted(parts: list[tuple[float, int, int]], alpha: float) -> dict:
    """Population-weighted proportion over strata (weight, count, rows), with the weighted
    stratum limits at level alpha / number of strata."""
    total = sum(w for w, _, _ in parts)
    a = alpha / len(parts)
    limits = [clopper_pearson(k, n, a) for _, k, n in parts]
    return {
        "estimate": sum(w * k / n for w, k, n in parts) / total,
        "ci": [
            sum(w * lim[i] for (w, _, _), lim in zip(parts, limits, strict=True)) / total
            for i in (0, 1)
        ],
    }


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pe = sum(sum(a == v for a, _ in pairs) / n * sum(b == v for _, b in pairs) / n for v in ANSWERS)
    return (po - pe) / (1 - pe) if pe < 1 else None


def clean(x):
    """NaN as null, so the JSON is standard."""
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--sheet", "--key", "--manifest", "--out"):
        ap.add_argument(name, type=Path, required=True)
    ap.add_argument("--second", type=Path, help="a second annotator's copy of the sheet")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if hashlib.sha256(args.key.read_bytes()).hexdigest() != manifest["key_sha256"]:
        sys.exit("the key does not match the hash committed with the sheet")
    key = {r["row"]: r for r in csv.DictReader(args.key.open(newline=""))}
    first = labels(args.sheet, manifest)
    assert first.keys() == key.keys()
    population = {"A": manifest["population_a"], "B": manifest["population_b"]}

    strata, samples = [], []
    for sample in ("A", "B"):
        rows = [r for r in key.values() if r["sample"] == sample]
        parts_d, parts_u, parts_du = [], [], []
        for s in sorted({r["stratum"] for r in rows}):
            mine = [r for r in rows if r["stratum"] == s]
            unsure = [r for r in mine if first[r["row"]] == "unsure"]
            wrong = [r for r in mine if first[r["row"]] not in ("unsure", r["reference_label"])]
            n, u, d = len(mine), len(unsure), len(wrong)
            strata.append(
                {
                    "sample": sample,
                    "stratum": s,
                    "rows": n,
                    "disagree": d,
                    "unsure": u,
                    "disagree_share": d / n,
                    "disagree_ci": clopper_pearson(d, n),
                    "unsure_share": u / n,
                    "disagree_among_definite": d / (n - u) if n > u else float("nan"),
                    "disagreeing_rows": [r["row"] for r in wrong],
                    "unsure_rows": [r["row"] for r in unsure],
                }
            )
            w = population[sample][s]
            parts_d.append((w, d, n))
            parts_u.append((w, u, n))
            parts_du.append((w, d + u, n))
        dis, uns, worst = (weighted(p, 0.05) for p in (parts_d, parts_u, parts_du))
        definite = sum(w * (n - u) / n for (w, u, n) in parts_u)
        samples.append(
            {
                "sample": sample,
                "disagree": dis,
                "unsure": uns,
                "disagree_if_unsure_disagreed": worst,
                "disagree_among_definite": (
                    sum(w * d / n for w, d, n in parts_d) / definite if definite else float("nan")
                ),
                "population": population[sample],
            }
        )

    out = {
        "strata": strata,
        "samples": samples,
        "intervals": "Clopper-Pearson per stratum; weighted limits at 0.05 / strata per sample",
    }
    if args.second:
        second = labels(args.second, manifest)
        out["second_annotator"] = {}
        for sample in ("A", "B"):
            pairs = [(first[k], second[k]) for k, r in key.items() if r["sample"] == sample]
            out["second_annotator"][sample] = {
                "table": {f"{a}/{b}": c for (a, b), c in sorted(Counter(pairs).items())},
                "agreement": sum(a == b for a, b in pairs) / len(pairs),
                "kappa": kappa(pairs),
            }
    out = clean(out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "label_audit.json").write_text(json.dumps(out, indent=1) + "\n")

    def pct(x):
        return "n/a" if x is None else f"{x:.3f}"

    lines = [
        "| Sample | Stratum | Rows | Disagree | Unsure | Disagree share [95% CI] |",
        "|---|---|---|---|---|---|",
    ] + [
        f"| {s['sample']} | {s['stratum']} | {s['rows']} | {s['disagree']} | {s['unsure']} "
        f"| {pct(s['disagree_share'])} [{pct(s['disagree_ci'][0])}, {pct(s['disagree_ci'][1])}] |"
        for s in out["strata"]
    ]
    lines += [
        "",
        "| Sample | Weighted disagreement [CI] | Weighted unsure [CI] | If unsure disagreed [CI] |",
        "|---|---|---|---|",
    ] + [
        f"| {s['sample']} "
        + " ".join(
            f"| {pct(s[k]['estimate'])} [{pct(s[k]['ci'][0])}, {pct(s[k]['ci'][1])}]"
            for k in ("disagree", "unsure", "disagree_if_unsure_disagreed")
        )
        + " |"
        for s in out["samples"]
    ]
    (args.out / "label_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.second:
        print(json.dumps(out["second_annotator"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
