#!/usr/bin/env python3
"""Analysis of the Real-Record Panel run (descriptive).

Two strata, scored per prompt pass:
- matcher-decided cells (22 per sub-profile): failure-aware micro-F1, precision, recall, F2
  and coverage; an invalid output predicts every non-gold candidate and misses every gold one;
- the hand-labelled undecided cell (1 per sub-profile): accuracy against the yes/no labels
  frozen before the run ("unsure" excluded and reported); an invalid output is wrong.
Tier means (over the main arm's repetitions; an unchanged selection scores as naive) are
contrasted with naive and with the output scaffold under a two-stage bootstrap: sub-profiles,
then cells within each drawn sub-profile. The rule baseline is run here (it is deterministic).

    python scripts/g34_analysis.py --run ../data/followups/real-record-panel \
        ../data/followups/real-record-panel-hosted --panel ../data/real-record-panel \
        --labels ../data/real-record-panel/labels/labelling_sheet.csv --data ../data --out <dir> \
        [--corrections ../data/real-record-panel/label-audit/g34_label_corrections.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from promptbench.baseline import decide  # noqa: E402
from promptbench.live.campaign import profiles_from  # noqa: E402
from promptbench.live.conditions import NAIVE  # noqa: E402
from promptbench.storage import Store, digest  # noqa: E402

EXECUTORS = {"granite4.2:30b/off": "Granite", "qwen3.8:27b/off": "Qwen"}
MAIN_OPTIMIZER = "z-ai/glm-5.3"
SEED, RESAMPLES = 20261029, 2000


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, nargs="+", required=True, help="run dirs on one panel")
    for name in ("--panel", "--labels", "--data", "--out"):
        ap.add_argument(name, type=Path, required=True)
    ap.add_argument(
        "--corrections",
        type=Path,
        help="post-hoc label sensitivity: rescore with g34_label_corrections.py's cells flipped",
    )
    args = ap.parse_args()
    stores = [Store(r) for r in args.run]
    manifests = [s.get("manifest.json") for s in stores]
    inputs = ("inputs/profiles.json", "inputs/advisories.json")
    if len({tuple(m["input_sha256"][n] for n in inputs) for m in manifests}) != 1:
        raise SystemExit("runs were prepared on different panels")
    conditions = {k: v for m in manifests for k, v in m["conditions"].items()}
    complete = {
        str(r): s.exists("reports/complete.json") for r, s in zip(args.run, stores, strict=True)
    }
    store_of: dict[tuple[str, str], Store] = {}
    # per work item token counts, derived by export_release.py from the unreleased work/
    work = {
        (item := json.loads(line))["key"]: item
        for r in args.run
        for line in (r / "work-summary.jsonl").read_text().splitlines()
    }

    # cells: decided (gold 0/1) in a fixed order; undecided with its hand label
    decided, hand = [], {}
    corrected = {}
    if args.corrections:
        corrected = {
            (r["subprofile_id"], r["cve_id"]): int(r["corrected_label"])
            for r in csv.DictReader(args.corrections.open(newline=""))
        }
    for r in csv.DictReader((args.panel / "cells.csv").open()):
        if r["stratum"] == "matcher_decided":
            key = (r["subprofile_id"], r["cve_id"])
            decided.append((*key, corrected.get(key, int(r["matcher_label"]))))
    for r in csv.DictReader(args.labels.open()):
        hand[r["subprofile_id"]] = (r["cve_id"], r["label"].strip().lower())
    assert len(hand) == 48 and {v[1] for v in hand.values()} <= {"yes", "no", "unsure"}
    assert corrected.keys() <= {(s, c) for s, c, _ in decided}
    gold = np.array([g for _, _, g in decided], dtype=float)
    subs = sorted({s for s, _, _ in decided})
    sub_of = np.array([subs.index(s) for s, _, _ in decided])

    # rows per pass (condition, prompt sha) -> profile -> row
    passes: dict[tuple[str, str], dict[str, dict]] = {}
    for store in stores:
        for name in store.names("evaluations/*.json"):
            env = store.get(name)
            key = ("/".join(env["key"].split("/")[1:-2]), env["prompt_sha256"])
            if store_of.setdefault(key, store) is not store:
                raise SystemExit(f"pass evaluated in two runs: {key}")
            passes.setdefault(key, {})[env["row"]["profile_id"]] = env["row"]
    incomplete = [k for k, v in passes.items() if len(v) != len(subs)]
    passes = {k: v for k, v in passes.items() if len(v) == len(subs)}
    # a pass planned in one run may be completed in another (the hosted pass, 2026-10-29)
    planned = {
        (r["condition"], digest(m["prompts"][r["prompt"]])) for m in manifests for r in m["runs"]
    }
    missing = sorted(planned - passes.keys())

    def cells(predicted: dict[str, set[str] | None]):
        """TP/FP/FN per decided cell and hand-stratum answers, from per-profile predictions."""
        pred = np.array(
            [(1 - g) if predicted[s] is None else float(c in predicted[s]) for (s, c, g) in decided]
        )
        tp, fp, fn = pred * gold, pred * (1 - gold), (1 - pred) * gold
        answers = {
            s: None if predicted[s] is None else ("yes" if hand[s][0] in predicted[s] else "no")
            for s in subs
        }
        return tp, fp, fn, answers

    def summary(tp, fp, fn, answers, valid):
        TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
        labelled = [s for s in subs if hand[s][1] != "unsure"]
        correct = sum(answers[s] == hand[s][1] for s in labelled)
        return {
            "decided_f1": float(f1(TP, FP, FN)),
            "decided_precision": float(TP / (TP + FP)) if TP + FP else 0.0,
            "decided_recall": float(TP / (TP + FN)) if TP + FN else 0.0,
            "decided_f2": float(5 * TP / (5 * TP + 4 * FN + FP)) if TP + FN + FP else 0.0,
            "coverage": valid / len(subs),
            "hand_accuracy": correct / len(labelled),
            "hand_correct": int(correct),
            "hand_correct_yes": sum(answers[s] == "yes" for s in labelled if hand[s][1] == "yes"),
            "hand_correct_no": sum(answers[s] == "no" for s in labelled if hand[s][1] == "no"),
            "hand_labelled": len(labelled),
            "hand_invalid": sum(answers[s] is None for s in labelled),
            "unsure_answered_yes": sum(answers[s] == "yes" for s in subs if hand[s][1] == "unsure"),
        }

    def audit(condition, sha, rows):
        """Every request: input tokens + output cap against num_ctx, and non-valid statuses.

        The runner checks only each pass's first request; with
        truncate=false an overflow is rejected, never truncated, so this full check suffices.
        """
        settings = conditions[condition]
        tokens = [work[f"real_record/{condition}/{sha}/{s}"]["input_tokens"] for s in subs]
        known = [t for t in tokens if t is not None]
        statuses = [rows[s]["status"] for s in subs]
        out = {
            "max_input_tokens": max(known) if known else None,
            "missing_token_counts": len(tokens) - len(known),
            "non_valid_statuses": {x: statuses.count(x) for x in set(statuses) if x != "valid"},
        }
        if settings["provider"] == "ollama":
            out["num_ctx"], out["num_predict"] = settings["num_ctx"], settings["num_predict"]
            out["fits"] = (
                len(known) == len(tokens)  # a rejected request has no count: not shown to fit
                and max(known) + settings["num_predict"] <= settings["num_ctx"]
            )
        return out

    results, mats, context = {}, {}, {}
    for (condition, sha), rows in sorted(passes.items()):
        predicted = {s: set(rows[s]["prediction"]) if rows[s]["valid"] else None for s in subs}
        tp, fp, fn, answers = cells(predicted)
        mats[(condition, sha)] = (tp, fp, fn)
        valid = sum(rows[s]["valid"] for s in subs)
        results[(condition, sha)] = summary(tp, fp, fn, answers, valid)
        context[(condition, sha)] = audit(condition, sha, rows)

    # rule baseline: deterministic, on the same rendered inputs
    profiles = {p.id: p for p in profiles_from(stores[0])}
    coverage = {}
    predicted = {}
    for s in subs:
        ids, counters = decide(profiles[s])
        predicted[s] = set(ids)
        for k, v in counters.items():
            coverage[k] = coverage.get(k, 0) + v
    tp, fp, fn, answers = cells(predicted)
    baseline = summary(tp, fp, fn, answers, len(subs)) | {"parse_coverage": coverage}

    # main-arm selections per (section, tier, repetition)
    naive_sha, scaffold_sha = digest(NAIVE), digest(manifests[0]["prompts"]["output_scaffold"])
    selections: dict[str, dict[str, list[str]]] = {}
    for r in csv.DictReader((args.data / "exports" / "trajectories.csv").open()):
        if (
            r["arm"] == "main"
            and r["optimizer"] == MAIN_OPTIMIZER
            and r["release_section"] in ("campaign", "campaign-extension")
            and r["executor"] in EXECUTORS
        ):
            selections.setdefault(r["executor"], {}).setdefault(r["tier"], []).append(
                r["selected_prompt_sha256"]
            )

    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(RESAMPLES):
        w = np.zeros(len(decided))
        for k in rng.integers(0, len(subs), len(subs)):
            # each draw of a sub-profile gets its own resample of its cells
            idx = np.flatnonzero(sub_of == k)
            w[idx] += np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx))
        draws.append(w)
    W = np.array(draws)  # resamples x cells

    # sensitivity: sub-profiles share their source system profile (real_NN_k), so also
    # resample the ten systems, then sub-profiles within each drawn system, then cells
    system_of = [s.rsplit("_", 1)[0] for s in subs]
    systems = sorted(set(system_of))
    members = [[k for k, x in enumerate(system_of) if x == y] for y in systems]
    rng_sys = np.random.default_rng(SEED + 1)
    draws = []
    for _ in range(RESAMPLES):
        w = np.zeros(len(decided))
        for g in rng_sys.integers(0, len(systems), len(systems)):
            for k in rng_sys.choice(members[g], len(members[g])):
                idx = np.flatnonzero(sub_of == k)
                w[idx] += np.bincount(rng_sys.integers(0, len(idx), len(idx)), minlength=len(idx))
        draws.append(w)
    W_sys = np.array(draws)

    def boot_f1(key, W=W):
        tp, fp, fn = mats[key]
        return f1(W @ tp, W @ fp, W @ fn)

    def ci(d):
        return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

    arms, contrasts = [], []
    for executor, name in EXECUTORS.items():
        naive_key, scaffold_key = (
            (executor, naive_sha),
            (executor + "/output_scaffold", scaffold_sha),
        )
        if naive_key not in mats or scaffold_key not in mats:
            continue
        rows = {
            "naive": (
                results[naive_key]["decided_f1"],
                boot_f1(naive_key),
                boot_f1(naive_key, W_sys),
            ),
            "output_scaffold": (
                results[scaffold_key]["decided_f1"],
                boot_f1(scaffold_key),
                boot_f1(scaffold_key, W_sys),
            ),
        }
        tier_shas = selections[executor]
        tiers = {}
        for tier, shas in sorted(tier_shas.items()):
            keys = [(executor, s) for s in shas]
            if any(k not in mats for k in keys):
                continue
            point = float(np.mean([results[k]["decided_f1"] for k in keys]))
            boot = np.mean([boot_f1(k) for k in keys], axis=0)
            boot_sys = np.mean([boot_f1(k, W_sys) for k in keys], axis=0)
            hand_acc = float(np.mean([results[k]["hand_accuracy"] for k in keys]))
            means = {
                m: float(np.mean([results[k][m] for k in keys]))
                for m in (
                    "decided_precision",
                    "decided_recall",
                    "hand_correct_yes",
                    "hand_correct_no",
                )
            }
            tiers[f"Tier {tier}"] = (point, boot, boot_sys)
            arms.append(
                {
                    "executor": name,
                    "arm": f"Tier {tier}",
                    "repetitions": len(keys),
                    "changed": sum(s != naive_sha for s in shas),
                    "decided_f1_mean": point,
                    "hand_accuracy_mean": hand_acc,
                }
                | {m + "_mean": v for m, v in means.items()}
            )
        pairs = [
            (label, arm, ref)
            for label, arm in tiers.items()
            for ref in ("naive", "output_scaffold")
        ]
        pairs.append(("output_scaffold", rows["output_scaffold"], "naive"))
        for label, (point, boot, boot_sys), ref in pairs:
            contrasts.append(
                {
                    "executor": name,
                    "contrast": f"{label} - {ref}",
                    "a": point,
                    "b": rows[ref][0],
                    "delta": point - rows[ref][0],
                    "ci": ci(boot - rows[ref][1]),
                    "ci_systems": ci(boot_sys - rows[ref][2]),
                }
            )

    def label(condition, sha):
        if sha == naive_sha:
            return "naive"
        if sha == scaffold_sha:
            return "output_scaffold"
        return "selected " + sha[:12]

    table = [
        {"condition": c, "prompt": label(c, s), "prompt_sha256": s} | m
        for (c, s), m in results.items()
    ] + [{"condition": "rule-baseline", "prompt": "-", "prompt_sha256": ""} | baseline]
    out = {
        "runs": [str(r) for r in args.run],
        "run_complete": complete,
        "planned_passes_missing": [list(k) for k in missing],
        "incomplete_passes": [list(k) for k in incomplete],
        "decided_cells": len(decided),
        "positives": int(gold.sum()),
        "hand_labels": {v: sum(x[1] == v for x in hand.values()) for v in ("yes", "no", "unsure")},
        "passes": table,
        "tier_means": arms,
        "contrasts": contrasts,
        "context_audit": [
            {"condition": c, "prompt_sha256": h} | a for (c, h), a in sorted(context.items())
        ],
        "context_all_fit": all(a.get("fits", True) for a in context.values()),
        "bootstrap": f"two-stage (sub-profiles, then decided cells within), B={RESAMPLES}, seed {SEED}, percentile 95%",
        "bootstrap_systems": f"three-stage sensitivity (source systems, sub-profiles within, then cells), B={RESAMPLES}, seed {SEED + 1}",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "g34_results.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = [
        "| Condition | Prompt | F1 | P | R | F2 | Coverage | Hand acc. | Unsure→yes |",
        "|---|---|---|---|---|---|---|---|---|",
    ] + [
        f"| {r['condition']} | {r['prompt']} | {r['decided_f1']:.3f} | {r['decided_precision']:.3f} "
        f"| {r['decided_recall']:.3f} | {r['decided_f2']:.3f} | {r['coverage']:.3f} "
        f"| {r['hand_correct']}/{r['hand_labelled']} | {r['unsure_answered_yes']} |"
        for r in table
    ]
    lines += [
        "",
        "| Executor | Arm | Reps | Changed | Mean F1 | Mean hand acc. |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {a['executor']} | {a['arm']} | {a['repetitions']} | {a['changed']} "
        f"| {a['decided_f1_mean']:.3f} | {a['hand_accuracy_mean']:.3f} |"
        for a in arms
    ]
    lines += [
        "",
        "| Executor | Contrast | A | B | Δ | 95% CI | 95% CI, systems |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {c['executor']} | {c['contrast']} | {c['a']:.3f} | {c['b']:.3f} | {c['delta']:+.3f} "
        f"| [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | [{c['ci_systems'][0]:+.3f}, {c['ci_systems'][1]:+.3f}] |"
        for c in contrasts
    ]
    (args.out / "g34_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if missing:
        print(f"\nNOT COMPLETE: {len(missing)} planned passes missing or partial", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
