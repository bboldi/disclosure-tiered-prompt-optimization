#!/usr/bin/env python3
"""Analyses of the round 1 revision runs (local self-optimization,
Shuffled Abstraction) against the campaign.

Each contrast is the mean paired delta of failure-aware micro-F1 on the sealed test
panel, paired by repetition, with the prespecified repetition bootstrap, a paired-t
interval and the repetitions x profiles x CVEs bootstrap of revision_analyses.py.

    python scripts/rev1_runs_analysis.py --data ../data --g33 ../data/followups/local-self-optimization \
        --g39-shuffled ../data/followups/shuffled-abstraction \
        --g39-tier2 ../data/followups/shuffled-abstraction-plain-tier2 --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from make_tables import bootstrap_ci
from revision_analyses import EXECUTORS, MAIN_OPTIMIZER, cell_matrices, f1, load, resampled

MARGIN = -0.02
T975_DF4 = 2.7764451051977987  # t quantile, 4 degrees of freedom (n = 5 pairs)
REPS = [1, 2, 3, 4, 5]


def run_rows(run: Path, rows: dict) -> list[dict]:
    """Add the run's sealed test rows to `rows`; return its trajectories."""
    for f in (run / "evaluations").iterdir():
        env = json.loads(f.read_text())["payload"]
        parts = env["key"].split("/")
        if parts[0] != "sealed" or parts[-3] != "test":
            continue
        key = ("/".join(parts[1:-3]), env["prompt_sha256"], env["row"]["profile_id"])
        if key in rows:  # a prompt already scored in the campaign must score identically
            assert (rows[key]["prediction"], rows[key]["valid"]) == (
                env["row"]["prediction"],
                env["row"]["valid"],
            ), key
        rows.setdefault(key, env["row"])
    return json.loads((run / "reports" / "analysis.json").read_text())["payload"]["trajectories"]


def timing(run: Path) -> dict:
    """Worker seconds per trajectory (main phase) and per run phase, and run GPU energy."""
    traj: dict[str, dict] = {}
    phases: dict[str, float] = {}
    # work-summary.jsonl: per committed work item, derived by export_release.py from work/
    for line in (run / "work-summary.jsonl").read_text().splitlines():
        item = json.loads(line)
        key, secs, load = item["key"], item["worker_seconds"], item["model_load_seconds"]
        parts = key.split("/")
        phases[parts[0]] = phases.get(parts[0], 0.0) + secs
        if parts[0] == "main":
            t = traj.setdefault(
                "/".join(parts[1:5]),
                dict.fromkeys(
                    ("optimizer_s", "executor_s", "model_load_s", "optimizer_calls"), 0.0
                ),
            )
            role = "optimizer_s" if parts[-1].startswith("proposal") else "executor_s"
            t[role] += secs
            t["model_load_s"] += load
            t["optimizer_calls"] += role == "optimizer_s"
    summary = json.loads(next((run / "telemetry").glob("*/summary.json")).read_text())["payload"]
    return {
        "trajectories": traj,
        "phase_worker_seconds": phases,
        "gpu_energy_joules": summary["gpu_energy_joules_observed"],
        "gpu_covered_seconds": summary["gpu_energy_covered_seconds"],
    }


def stack(test, cves, rows, executor, sel):
    shas = sorted(set(sel.values()))
    mats = [cell_matrices(test, cves, rows, executor, s) for s in shas]
    return {
        "sel": sel,
        "pos": {s: k for k, s in enumerate(shas)},
        "TP": np.stack([m[0] for m in mats]),
        "FP": np.stack([m[1] for m in mats]),
        "FN": np.stack([m[2] for m in mats]),
    }


def score(tp, fp, fn, n=None):
    return f1(tp, fp, fn)


def contrast(c, a, b, label, status) -> dict:
    point = score(c["TP"].sum((1, 2)), c["FP"].sum((1, 2)), c["FN"].sum((1, 2)))
    ia = np.array([c["pos"][c["sel"][(a, r)]] for r in REPS])
    ib = np.array([c["pos"][c["sel"][(b, r)]] for r in REPS])
    d = [float(x) for x in point[ia] - point[ib]]
    half = T975_DF4 * statistics.stdev(d) / len(d) ** 0.5
    out = {
        "contrast": label,
        "status": status,
        "a_mean": float(point[ia].mean()),
        "b_mean": float(point[ib].mean()),
        "mean_delta": statistics.mean(d),
        "deltas": d,
        "repetitions_ci": list(bootstrap_ci(d)),
        "paired_t_ci": [statistics.mean(d) - half, statistics.mean(d) + half],
    }
    return out | resampled(c, ia, ib, score)


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--data", "--g33", "--g39-shuffled", "--g39-tier2", "--out"):
        ap.add_argument(name, type=Path, required=True)
    args = ap.parse_args()
    test, trajectories, rows = load(args.data)
    cves = sorted({c for p in test for c in p["advisories"]})
    new = {
        "local": run_rows(args.g33, rows),
        "shuffled": run_rows(args.g39_shuffled, rows),
        "tier2_new": run_rows(args.g39_tier2, rows),
    }
    results, trajectory_rows, scaffold_rows = [], [], []
    # RQ4 output scaffold: one fixed prompt under its own schema and a 4,096-token cap
    scaffold = {
        r["Executor"] + "/off": float(r["F1"])
        for r in csv.DictReader((args.data / "tables" / "table_scaffolding_ablation.csv").open())
        if r["Fixed prompt variant"] == "off/output scaffold"
    }
    for executor, name in EXECUTORS.items():
        sel = {
            (int(t["tier"]), int(t["repetition"])): t["selected_prompt_sha256"]
            for t in trajectories
            if t["executor"] == executor and t["optimizer"] == MAIN_OPTIMIZER
        }
        naive = {
            t["selected_prompt_sha256"]
            for t in trajectories
            if t["executor"] == executor and t["prompt_changed"] == "0"
        }.pop()
        for r in REPS:
            sel[("naive", r)] = naive
        for arm, trs in new.items():
            for t in trs:
                if t["executor"] == executor:
                    sel[(arm, t["repetition"])] = t["selected_prompt_sha256"]
                    trajectory_rows.append(
                        {"executor": name, "arm": arm}
                        | {
                            k: t[k]
                            for k in (
                                "repetition",
                                "optimizer",
                                "prompt_changed",
                                "selected_prompt_sha256",
                                "validation_f1",
                                "test_f1",
                                "test_recall",
                                "test_precision",
                                "test_coverage",
                                "temporal_f1",
                                "product_heldout_f1",
                            )
                        }
                    )
        c = stack(test, cves, rows, executor, sel)
        plan = [
            ("local", 3, "Local T3 - GLM T3", "primary"),
            ("local", "naive", "Local T3 - naive", "descriptive"),
            (3, "naive", "GLM T3 - naive", "descriptive"),
        ]
        if name == "Granite":
            plan += [
                ("shuffled", "tier2_new", "Shuffled - Tier 2 (new endpoint)", "primary"),
                ("shuffled", 1, "Shuffled - campaign Tier 1", "descriptive, endpoint differs"),
                ("shuffled", 3, "Shuffled - campaign Tier 3", "descriptive, endpoint differs"),
                (
                    "tier2_new",
                    2,
                    "Tier 2 new endpoint - campaign Tier 2",
                    "not prespecified, endpoint check",
                ),
                ("shuffled", "naive", "Shuffled - naive", "not prespecified"),
                ("tier2_new", "naive", "Tier 2 new endpoint - naive", "not prespecified"),
            ]
        for a, b, label, status in plan:
            results.append({"executor": name} | contrast(c, a, b, label, status))
        point = score(c["TP"].sum((1, 2)), c["FP"].sum((1, 2)), c["FN"].sum((1, 2)))
        for arm in ["local", 3] + (["shuffled", "tier2_new"] if name == "Granite" else []):
            mean = float(np.mean([point[c["pos"][sel[(arm, r)]]] for r in REPS]))
            scaffold_rows.append(
                {"executor": name, "arm": str(arm), "mean_f1": mean}
                | {"scaffold_f1": scaffold[executor], "difference": mean - scaffold[executor]}
            )
    for r in results:
        if r["status"] == "primary" and r["contrast"].startswith("Local T3"):
            r["retention_established"] = r["repetitions_ci"][0] > MARGIN
        if r["contrast"].startswith("Shuffled - Tier 2"):
            r["equivalence_within_0.02"] = (
                MARGIN < r["repetitions_ci"][0] and r["repetitions_ci"][1] < -MARGIN
            )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "g33_g39_contrasts.json").write_text(json.dumps(results, indent=1) + "\n")
    (args.out / "g33_g39_trajectories.json").write_text(
        json.dumps(trajectory_rows, indent=1) + "\n"
    )
    fmt = lambda v: f"[{v[0]:+.4f}, {v[1]:+.4f}]"  # noqa: E731
    lines = [
        "| Executor | Contrast | Status | A | B | Mean Δ | Rep. bootstrap | Paired t | Reps × profiles × CVEs |",
        "|---|---|---|---|---|---|---|---|---|",
    ] + [
        f"| {r['executor']} | {r['contrast']} | {r['status']} | {r['a_mean']:.4f} | {r['b_mean']:.4f} "
        f"| {r['mean_delta']:+.4f} | {fmt(r['repetitions_ci'])} | {fmt(r['paired_t_ci'])} "
        f"| {fmt(r['repetitions_items_ci'])} |"
        for r in results
    ]
    (args.out / "g33_g39_contrasts.md").write_text("\n".join(lines) + "\n")
    (args.out / "g33_g39_scaffold.json").write_text(json.dumps(scaffold_rows, indent=1) + "\n")
    runs = {
        "Local self-optimization": args.g33,
        "Shuffled": args.g39_shuffled,
        "Tier 2 (new endpoint)": args.g39_tier2,
    }
    (args.out / "g33_g39_timing.json").write_text(
        json.dumps({k: timing(v) for k, v in runs.items()}, indent=1) + "\n"
    )
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
