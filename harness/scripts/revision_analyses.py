#!/usr/bin/env python3
"""Round 1 revision analyses on the released data (no model calls).

G2.7  dependence-aware intervals for the paired tier contrasts on the sealed
      test panel. The prespecified interval resamples repetitions only; here the
      same statistic is also resampled over profiles and CVEs (pigeonhole
      bootstrap), because profiles share CVEs and every trajectory is scored on
      the same 192 profiles.
G2.8  supplementary metrics per tier: macro-F1 over CVEs, F2, precision,
G2.10 recall, valid-answer F1 and coverage.
G2.9  the primary and secondary contrasts as expected cost per profile for a
      range of FN:FP cost ratios, with the same intervals as G2.7.
G2.5  structural features of the selected (changed) prompts per tier: length,
      rule items, and how often boundary, alias and uncertainty language occurs.

    python scripts/revision_analyses.py --data ../data --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from make_tables import RESAMPLES, bootstrap_ci

EXECUTORS = {"granite4.2:30b/off": "Granite", "qwen3.8:27b/off": "Qwen"}
MAIN_OPTIMIZER = "z-ai/glm-5.3"
CONTRASTS = (  # (a, b, status): delta = score(tier a) - score(tier b)
    (2, 3, "primary"),
    (1, 2, "secondary"),
    (1, 3, "exploratory"),
)
COST_RATIOS = (1, 2, 5, 10, 20)  # cost of a missed applicable CVE, in false positives
SEED = 20261028


def load(data: Path):
    profiles = [
        json.loads(line)
        for line in (data / "benchmark" / "profiles.jsonl").read_text().splitlines()
        if line.strip()
    ]
    test = [p for p in profiles if p["partition"] == "test"]
    trajectories = [
        r
        for r in csv.DictReader((data / "exports" / "trajectories.csv").open())
        if r["arm"] == "main"
        and r["optimizer"] == MAIN_OPTIMIZER
        and r["release_section"] in ("campaign", "campaign-extension")
    ]
    # sealed test rows from the raw envelopes of both campaign sections
    rows = {}
    for section in ("campaign", "campaign-extension"):
        for f in (data / section / "evaluations").iterdir():
            env = json.loads(f.read_text())["payload"]
            parts = env["key"].split("/")
            if parts[0] != "sealed" or parts[-3] != "test":
                continue
            executor = "/".join(parts[1:-3])
            rows.setdefault((executor, env["prompt_sha256"], env["row"]["profile_id"]), env["row"])
    return test, trajectories, rows


def cell_matrices(test, cves, rows, executor, sha):
    """Per (profile, CVE) TP/FP/FN under the failure-aware rule P := C \\ G if invalid."""
    index = {c: k for k, c in enumerate(cves)}
    tp, fp, fn = (np.zeros((len(test), len(cves))) for _ in range(3))
    valid = np.zeros(len(test), dtype=bool)
    for i, p in enumerate(test):
        r = rows[(executor, sha, p["id"])]
        cand, gold = set(p["advisories"]), set(p["expected"])
        valid[i] = bool(r["valid"])
        pred = set(r["prediction"]) if valid[i] else cand - gold
        for c in pred & gold:
            tp[i, index[c]] = 1
        for c in pred - gold:
            fp[i, index[c]] = 1
        for c in gold - pred:
            fn[i, index[c]] = 1
        # the released per-row counts must be reproduced exactly
        assert (tp[i].sum(), fp[i].sum(), fn[i].sum()) == (
            int(r["tp"]),
            int(r["fp"]),
            int(r["fn"]),
        ), (executor, sha, p["id"])
    return tp, fp, fn, valid


def f1(tp, fp, fn, n=None):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0.0)


def cells(data: Path):
    """Per Executor: selected prompt per (tier, repetition) and stacked cell matrices."""
    test, trajectories, rows = load(data)
    cves = sorted({c for p in test for c in p["advisories"]})
    out = {}
    for executor, name in EXECUTORS.items():
        mine = [t for t in trajectories if t["executor"] == executor]
        sel = {(int(t["tier"]), int(t["repetition"])): t["selected_prompt_sha256"] for t in mine}
        naive = {t["selected_prompt_sha256"] for t in mine if t["prompt_changed"] == "0"}
        assert len(naive) == 1, naive
        shas = sorted(set(sel.values()))
        mats = [cell_matrices(test, cves, rows, executor, s) for s in shas]
        out[name] = {
            "sel": sel,
            "naive": naive.pop(),
            "pos": {s: k for k, s in enumerate(shas)},
            "TP": np.stack([m[0] for m in mats]),  # S x N x K
            "FP": np.stack([m[1] for m in mats]),
            "FN": np.stack([m[2] for m in mats]),
            "valid": np.stack([m[3] for m in mats]),  # S x N
        }
    return out


def scopes(sel):
    reps = sorted({r for _, r in sel})
    out = [("original", [r for r in reps if r <= 5])]
    return out + [("extended", reps)] if len(reps) > 5 else out


def resampled(c, ia, ib, score) -> dict:
    """Percentile intervals of the mean paired delta of `score` under two item schemes."""
    rng = np.random.default_rng(SEED)
    TP, FP, FN = c["TP"], c["FP"], c["FN"]
    S, N, K = TP.shape
    R = len(ia)
    draws = {"items_ci": [], "repetitions_items_ci": []}
    for _ in range(RESAMPLES):
        wp = np.bincount(rng.integers(0, N, N), minlength=N).astype(float)
        wc = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)
        s = score(
            np.einsum("snk,n,k->s", TP, wp, wc),
            np.einsum("snk,n,k->s", FP, wp, wc),
            np.einsum("snk,n,k->s", FN, wp, wc),
            wp.sum(),
        )
        d = s[ia] - s[ib]
        draws["items_ci"].append(d.mean())
        draws["repetitions_items_ci"].append(d[rng.integers(0, R, R)].mean())
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()
    }


def paired(c, a, b, reps, score) -> dict:
    N = c["TP"].shape[1]
    point = score(c["TP"].sum((1, 2)), c["FP"].sum((1, 2)), c["FN"].sum((1, 2)), N)
    ia = np.array([c["pos"][c["sel"][(a, r)]] for r in reps])
    ib = np.array([c["pos"][c["sel"][(b, r)]] for r in reps])
    deltas = point[ia] - point[ib]
    lo, hi = bootstrap_ci([float(x) for x in deltas])
    res = {"mean_delta": float(deltas.mean()), "repetitions_ci": [lo, hi]}
    return res | resampled(c, ia, ib, score)


def contrasts(data_cells) -> list[dict]:
    out = []
    for name, c in data_cells.items():
        for scope, reps in scopes(c["sel"]):
            for a, b, status in CONTRASTS:
                out.append(
                    {
                        "executor": name,
                        "contrast": f"Tier {a} - Tier {b}",
                        "status": status,
                        "scope": f"{scope}, n={len(reps)}",
                    }
                    | paired(c, a, b, reps, f1)
                )
    return out


def costs(data_cells) -> list[dict]:
    """Expected cost per profile in false-positive units: (k * FN + FP) / N."""
    out = []
    for name, c in data_cells.items():
        for scope, reps in scopes(c["sel"]):
            for a, b, status in CONTRASTS[:2]:
                for k in COST_RATIOS:
                    cost = lambda tp, fp, fn, n, k=k: (k * fn + fp) / n  # noqa: E731
                    out.append(
                        {
                            "executor": name,
                            "contrast": f"Tier {a} - Tier {b}",
                            "status": status,
                            "scope": f"{scope}, n={len(reps)}",
                            "fn_fp_cost_ratio": k,
                        }
                        | paired(c, a, b, reps, cost)
                    )
    return out


def metrics(data_cells) -> list[dict]:
    """Per tier, mean over repetitions of each selected prompt's test-panel metrics."""
    out = []
    for name, c in data_cells.items():
        reps = sorted({r for _, r in c["sel"]})
        arms = [(f"Tier {t}", [c["sel"][(t, r)] for r in reps]) for t in (1, 2, 3)]
        arms.append(("Naive prompt", [c["naive"]]))
        for arm, shas in arms:
            per = []
            for s in shas:
                k = c["pos"][s]
                tp_c, fp_c, fn_c = (m[k].sum(0) for m in (c["TP"], c["FP"], c["FN"]))
                tp, fp, fn = tp_c.sum(), fp_c.sum(), fn_c.sum()
                v = c["valid"][k]
                vt, vf, vn = (m[k][v].sum() for m in (c["TP"], c["FP"], c["FN"]))
                gold = (tp_c + fn_c) > 0  # CVEs applicable to at least one test profile
                per.append(
                    {
                        "micro_f1": float(f1(tp, fp, fn)),
                        "macro_f1_cves": float(f1(tp_c, fp_c, fn_c)[gold].mean()),
                        "f2": float(5 * tp / (5 * tp + 4 * fn + fp)),
                        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                        "recall": float(tp / (tp + fn)),
                        "valid_answer_f1": float(f1(vt, vf, vn)),
                        "coverage": float(v.mean()),
                    }
                )
            out.append(
                {"executor": name, "arm": arm, "runs": len(shas)}
                | {m: float(np.mean([p[m] for p in per])) for m in per[0]}
            )
    return out


# keyword families, counted per prompt; crude by design and reported as such
FEATURES = {
    "boundary_terms": r"inclusive|exclusive|endpoint|boundar|equal",
    "alias_terms": r"alias|normali[sz]|prefix|suffix|package.?manager|distro",
    "uncertainty_include": r"[^.]*\b(ambiguous|unsure|uncertain|in doubt)\b[^.]*\binclu[^.]*\.",
    "uncertainty_exclude": r"[^.]*\b(ambiguous|unsure|uncertain|in doubt)\b[^.]*\b(exclude|omit)[^.]*\.",
}


def prompt_features(data: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (data / "exports" / "selected_prompts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        if (
            r["arm"] != "main"
            or r["release_section"] not in ("campaign", "campaign-extension")
            or r["optimizer"] != MAIN_OPTIMIZER
            or not int(r["prompt_changed"])
            or r["executor"] not in EXECUTORS
        ):
            continue
        p = r["selected_prompt"]
        f = {
            "chars": len(p),
            "rule_items": sum(
                1 for line in p.splitlines() if re.match(r"\s*([-*•]|\d+[.)]|Step \d+:)\s", line)
            ),
        }
        f |= {k: len(re.findall(pat, p, re.I)) for k, pat in FEATURES.items()}
        groups.setdefault((EXECUTORS[r["executor"]], int(r["tier"])), []).append(f)
    return [
        {"executor": e, "tier": t, "changed_prompts": len(v)}
        | {k: float(np.mean([x[k] for x in v])) for k in v[0]}
        for (e, t), v in sorted(groups.items())
    ]


def write(out: Path, results: dict[str, list[dict]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fmt = lambda x: f"{x:+.4f}"  # noqa: E731
    ci = lambda v: f"[{fmt(v[0])}, {fmt(v[1])}]"  # noqa: E731
    head = (
        "| Mean Δ | Repetitions (prespecified) | Profiles × CVEs | Repetitions × profiles × CVEs |"
    )
    tables = {
        "g27_contrast_intervals": (
            "| Executor | Contrast | Status | Scope " + head,
            lambda r: f"| {r['executor']} | {r['contrast']} | {r['status']} | {r['scope']} ",
        ),
        "g29_cost_sensitivity": (
            "| Executor | Contrast | Scope | FN:FP " + head,
            lambda r: (
                f"| {r['executor']} | {r['contrast']} | {r['scope']} | {r['fn_fp_cost_ratio']}:1 "
            ),
        ),
    }
    for name, (header, lead) in tables.items():
        lines = [header, "|" + "---|" * (header.count("|") - 1)]
        for r in results[name]:
            lines.append(
                lead(r)
                + f"| {fmt(r['mean_delta'])} | {ci(r['repetitions_ci'])} | {ci(r['items_ci'])} | "
                f"{ci(r['repetitions_items_ci'])} |"
            )
        (out / f"{name}.md").write_text("\n".join(lines) + "\n")
    cols = ["micro_f1", "macro_f1_cves", "f2", "precision", "recall", "valid_answer_f1", "coverage"]
    lines = [
        "| Executor | Arm | Runs | " + " | ".join(cols) + " |",
        "|---|---|---|" + "---|" * len(cols),
    ]
    for r in results["g28_metrics"]:
        lines.append(
            f"| {r['executor']} | {r['arm']} | {r['runs']} | "
            + " | ".join(f"{r[k]:.3f}" for k in cols)
            + " |"
        )
    (out / "g28_metrics.md").write_text("\n".join(lines) + "\n")
    cols = ["chars", "rule_items", *FEATURES]
    lines = [
        "| Executor | Tier | Changed prompts | " + " | ".join(cols) + " |",
        "|---|---|---|" + "---|" * len(cols),
    ]
    for r in results["g25_prompt_features"]:
        lines.append(
            f"| {r['executor']} | {r['tier']} | {r['changed_prompts']} | "
            + " | ".join(f"{r[k]:.1f}" for k in cols)
            + " |"
        )
    (out / "g25_prompt_features.md").write_text("\n".join(lines) + "\n")
    for name, rows in results.items():
        (out / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data_cells = cells(args.data)
    results = {
        "g27_contrast_intervals": contrasts(data_cells),
        "g28_metrics": metrics(data_cells),
        "g29_cost_sensitivity": costs(data_cells),
        "g25_prompt_features": prompt_features(args.data),
    }
    write(args.out, results)
    for name in results:
        print((args.out / f"{name}.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
