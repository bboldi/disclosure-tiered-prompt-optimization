#!/usr/bin/env python3
"""Figure 2 (revision): paired tier differences per repetition, with the repetition-bootstrap
interval and the dependence-aware interval (repetitions x profiles x CVEs, from
revision_analyses.py) side by side. Qwen and Granite on the five prespecified repetitions;
Granite's eight-repetition extension in its own panel.

    python scripts/fig_paired_deltas.py --data ../data \
        --intervals <revision_analyses.py out>/g27_contrast_intervals.json --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_tables import CONTRASTS, MARGIN, main_trajectories, read_csv, save_figure

NAMES = {"T2 − T3": "Tier 2 - Tier 3", "T1 − T2": "Tier 1 - Tier 2", "T1 − T3": "Tier 1 - Tier 3"}
PANELS = (  # executor id, label, scope in the interval file, campaign sections
    ("qwen3.8:27b/off", "Qwen", "original, n=5", ("campaign",)),
    ("granite4.2:30b/off", "Granite", "original, n=5", ("campaign",)),
    ("granite4.2:30b/off", "Granite", "extended, n=8", ("campaign", "campaign-extension")),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--data", "--intervals", "--out"):
        ap.add_argument(name, type=Path, required=True)
    args = ap.parse_args()
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans"],
            "axes.unicode_minus": False,
            "font.size": 8,
        }
    )
    import matplotlib.pyplot as plt

    main_rows = main_trajectories(read_csv(args.data / "exports" / "trajectories.csv"))
    intervals = {
        (r["executor"], r["contrast"], r["scope"]): r
        for r in json.loads(args.intervals.read_text())
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.1), sharey=True)
    for ax, letter, (ex, name, scope, sections) in zip(axes, "ABC", PANELS, strict=True):
        by = {
            (int(t["tier"]), int(t["repetition"])): float(t["test_f1"])
            for t in main_rows
            if t["executor"] == ex and t["release_section"] in sections
        }
        reps = sorted({r for _, r in by})
        assert f"n={len(reps)}" in scope, (name, scope, reps)
        for i, (left, right, label) in enumerate(CONTRASTS):
            deltas = [by[(left, r)] - by[(right, r)] for r in reps]
            c = intervals[(name, NAMES[label], scope)]
            assert abs(sum(deltas) / len(deltas) - c["mean_delta"]) < 1e-9, (name, label)
            x = [i - 0.18 + (j - (len(deltas) - 1) / 2) * 0.05 for j in range(len(deltas))]
            ax.scatter(x, deltas, s=14, color="#888", zorder=3)
            for dx, ci, style in (
                (0.08, c["repetitions_ci"], "-"),
                (0.22, c["repetitions_items_ci"], ":"),
            ):
                ax.errorbar(
                    [i + dx],
                    [c["mean_delta"]],
                    yerr=[[c["mean_delta"] - ci[0]], [ci[1] - c["mean_delta"]]],
                    fmt="o",
                    ms=3,
                    color="black",
                    capsize=3,
                    lw=1.1,
                    ls="none",
                    zorder=4,
                )
                if style == ":":
                    ax.lines[-1].set_linestyle(":")  # caps keep their style; the bar is dotted
                    ax.collections[-1].set_linestyle(":")
        ax.axhline(0, color="#aaa", lw=0.8)
        ax.axhline(MARGIN, color="#c33", lw=0.8, ls="--")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["T2 − T3", "T1 − T2", "T1 − T3"])
        title = f"{name}, {len(reps)} repetitions" + (" (extension)" if len(reps) > 5 else "")
        ax.set_title(title, fontsize=8)
        ax.text(
            0,
            1.03,
            letter,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    axes[0].set_ylabel(r"$\Delta$ sealed-test micro-F1")
    fig.tight_layout()
    save_figure(fig, args.out, "fig_paired_deltas")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
