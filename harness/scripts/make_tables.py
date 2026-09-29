"""Regenerate every table and figure of the paper from the data release.

Reads the flat exports written by `export_release.py` plus the calibration and benchmark
records, and writes Markdown and CSV tables, plus PNG and PDF figures. Nothing here contacts a model
or a provider. Every number quoted in the manuscript must come from this script's output.

Usage:
  python scripts/make_tables.py --data ../data --out ../data/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

NAIVE_SHA = "0080638c0f58d5b18e6db10f0327e0c85125dd7c1ea1ae311fd5ad96c05b3078"
TIERS = (1, 2, 3)
PANELS = ("test", "temporal", "product_heldout")
PANEL_LABEL = {"test": "Test", "temporal": "Temporal", "product_heldout": "Product held-out"}
EXECUTOR_LABEL = {"qwen3.8:27b/off": "Qwen 3.8 27B", "granite4.2:30b/off": "Granite 4.2 30B"}
MARGIN = -0.02
RESAMPLES = 2000
CAMPAIGN_SECTIONS = ("campaign", "campaign-extension")
MODEL_LABEL = {
    "anthropic/claude-opus-5": "Claude Opus 5 (hosted)",
    "z-ai/glm-5.3": "GLM-5.3 (hosted)",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro (hosted)",
    "rule-baseline": "Rule baseline (deterministic)",
    "qwen3.8:27b/off": "Qwen 3.8 27B, naive prompt",
    "granite4.2:30b/off": "Granite 4.2 30B, naive prompt",
}
CONTRASTS = ((2, 3, "T2 − T3"), (1, 2, "T1 − T2"), (1, 3, "T1 − T3"))
# two-sided 97.5% Student t quantiles for n-1 degrees of freedom, n = 2..12
T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
}


def paired_t_ci(values: list[float]) -> tuple[float, float]:
    """Sensitivity interval: mean ± t(0.975, n−1) · sd/√n."""
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    half = T975[n - 1] * statistics.stdev(values) / n**0.5
    return (statistics.mean(values) - half, statistics.mean(values) + half)


def main_trajectories(traj: list[dict[str, str]]) -> list[dict[str, str]]:
    return [t for t in traj if t["release_section"] in CAMPAIGN_SECTIONS and t["arm"] == "main"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(x: str | float | None, digits: int = 3) -> str:
    if x is None or x == "":
        return ""
    return f"{float(x):.{digits}f}"


def payload(path: Path) -> Any:
    data = json.loads(path.read_text())
    return (
        data["payload"] if isinstance(data, dict) and set(data) == {"payload", "sha256"} else data
    )


def write_table(
    out: Path, name: str, header: list[str], rows: list[list[Any]], caption: str
) -> None:
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    (out / f"{name}.md").write_text("\n".join(md) + "\n")
    with (out / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def bootstrap_ci(values: list[float], seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)  # noqa: S311 - reproducible resampling
    means = sorted(statistics.mean(rng.choice(values) for _ in values) for _ in range(RESAMPLES))
    return means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]


def pooled(rows: list[dict[str, str]]) -> dict[str, float]:
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    return {
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "coverage": sum(int(r["valid"]) for r in rows) / len(rows) if rows else 0.0,
        "n": len(rows),
    }


# --------------------------------------------------------------------------- tables


def table_calibration(data: Path, out: Path) -> None:
    report = payload(data / "calibration" / "reports" / "complete.json")
    rows = []
    for cid, cell in sorted(report["local"].items()):
        naive, expert = cell["naive"], cell["historical_expert"]
        verdict = report["verdict"]["condition_exclusion_reasons"].get(cid, [])
        rows.append(
            [
                cid,
                f(naive["metrics"]["failure_aware_lower_bound"]["micro_f1"]),
                f(naive["metrics"]["failure_aware_lower_bound"]["recall"]),
                f(expert["metrics"]["failure_aware_lower_bound"]["micro_f1"]),
                f(min(naive["metrics"]["coverage"], expert["metrics"]["coverage"])),
                f(naive["nonzero_batch_fraction"], 2),
                f(naive["latency_p90_seconds"], 1),
                "promoted"
                if cid in report["verdict"]["promoted"]
                else "; ".join(verdict) or "eligible",
            ]
        )
    hosted = report["hosted"]["metrics"]["failure_aware_lower_bound"]
    rows.append(
        [
            "DeepSeek V4 Pro (hosted reference)",
            f(hosted["micro_f1"]),
            f(hosted["recall"]),
            "",
            f(report["hosted"]["metrics"]["coverage"]),
            "",
            "",
            "decidability reference",
        ]
    )
    write_table(
        out,
        "table_calibration",
        [
            "Condition",
            "Naive F1",
            "Naive recall",
            "Expert F1",
            "Coverage",
            "Nonzero T2 batches",
            "p90 s",
            "Gate outcome",
        ],
        rows,
        "Calibration gate on benchmark 07, 48 pilot-development profiles, failure-aware micro-F1.",
    )


def table_primary(traj: list[dict[str, str]], out: Path) -> dict[tuple[str, ...], Any]:
    """Prespecified contrasts on the original repetitions, then on all repetitions if extended.

    Returns the per-contrast summaries for every (executor, contrast) pair over ALL repetitions
    (used by the figures); the original-five rows are also written so both are on record.
    """
    main = main_trajectories(traj)
    result: dict[tuple[str, ...], Any] = {}
    rows = []
    for ex in sorted({t["executor"] for t in main}):
        by = {(int(t["tier"]), int(t["repetition"])): t for t in main if t["executor"] == ex}
        original = sorted({r for (_, r) in by if by[(1, r)]["release_section"] == "campaign"})
        every = sorted({r for (_, r) in by})
        scopes = [("original", original)] + ([("all", every)] if every != original else [])
        for scope, reps in scopes:
            for left, right, label in CONTRASTS:
                deltas = [
                    float(by[(left, r)]["test_f1"]) - float(by[(right, r)]["test_f1"])
                    for r in reps
                    if (left, r) in by and (right, r) in by
                ]
                rdeltas = [
                    float(by[(left, r)]["test_recall"]) - float(by[(right, r)]["test_recall"])
                    for r in reps
                    if (left, r) in by and (right, r) in by
                ]
                lo, hi = bootstrap_ci(deltas)
                tlo, thi = paired_t_ci(deltas)
                summary = {
                    "scope": scope,
                    "repetitions": len(deltas),
                    "deltas": deltas,
                    "mean": statistics.mean(deltas),
                    "sd": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                    "ci": (lo, hi),
                    "t_ci": (tlo, thi),
                    "recall_mean": statistics.mean(rdeltas),
                }
                result[(ex, label, scope)] = summary
                result[(ex, label)] = summary  # the widest scope wins for the figures
                kind = {"T2 − T3": "prespecified", "T1 − T2": "secondary"}.get(label, "exploratory")
                rows.append(
                    [
                        EXECUTOR_LABEL.get(ex, ex),
                        label,
                        kind,
                        f"{len(deltas)} ({scope})",
                        ", ".join(f"{d:+.3f}" for d in deltas),
                        f"{statistics.mean(deltas):+.4f}",
                        f"{summary['sd']:.4f}",
                        f"[{lo:+.4f}, {hi:+.4f}]",
                        f"[{tlo:+.4f}, {thi:+.4f}]",
                        f"{statistics.mean(rdeltas):+.3f}",
                        ("yes" if lo > MARGIN else "no") if label == "T2 − T3" else "",
                    ]
                )
    write_table(
        out,
        "table_primary_contrast",
        [
            "Executor",
            "Contrast",
            "Status",
            "Repetitions (scope)",
            "Per-repetition deltas",
            "Mean",
            "Sample SD",
            "95% percentile bootstrap (primary)",
            "95% paired t (sensitivity)",
            "Recall delta",
            "Lower bound above −0.02",
        ],
        rows,
        "Paired tier contrasts on the sealed test panel, failure-aware micro-F1. The percentile "
        "bootstrap over repetitions is the prespecified analysis; the paired t interval is a "
        "sensitivity analysis. 'original' = the five campaign repetitions; 'all' adds the "
        "prospectively planned extension repetitions.",
    )
    return result


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def table_tier_means(traj: list[dict[str, str]], panels: list[dict[str, str]], out: Path) -> None:
    main = main_trajectories(traj)
    rows = []
    for ex in sorted({t["executor"] for t in main}):
        for tier in TIERS:
            cell = [t for t in main if t["executor"] == ex and int(t["tier"]) == tier]
            f1 = [float(t["test_f1"]) for t in cell]
            rec = [float(t["test_recall"]) for t in cell]
            rows.append(
                [
                    EXECUTOR_LABEL.get(ex, ex),
                    f"Tier {tier}",
                    len(cell),
                    sum(int(t["prompt_changed"]) for t in cell),
                    f"{statistics.mean(f1):.3f} ± {_sd(f1):.3f}",
                    f(statistics.mean(rec)),
                    f(statistics.mean(float(t["test_valid_f1"]) for t in cell)),
                    f(statistics.mean(float(t["test_coverage"]) for t in cell)),
                    f(statistics.mean(float(t["temporal_f1"]) for t in cell)),
                    f(statistics.mean(float(t["product_heldout_f1"]) for t in cell)),
                ]
            )
        # naive row from the dedicated local run, else from shared sealed panels
        naive = [
            p
            for p in panels
            if p["kind"] == "ceiling"
            and p["model_or_executor"] == ex
            and p["release_section"].endswith("naive-local")
        ]
        if not naive:
            naive = [
                p
                for p in panels
                if p["kind"] == "sealed"
                and p["model_or_executor"] == ex.split("/")[0]
                and p["prompt_sha256"] == NAIVE_SHA
            ]
        byp = {p["panel"]: p for p in naive}
        if byp:
            rows.append(
                [
                    EXECUTOR_LABEL.get(ex, ex),
                    "Naive prompt",
                    "",
                    "",
                    f(byp["test"]["f1"]),
                    f(byp["test"]["recall"]),
                    f(byp["test"]["valid_f1"]),
                    f(byp["test"]["coverage"]),
                    f(byp["temporal"]["f1"]) if "temporal" in byp else "",
                    f(byp["product_heldout"]["f1"]) if "product_heldout" in byp else "",
                ]
            )
    for other in ("gepa", "reference"):
        cell = [t for t in traj if t["release_section"] == "campaign" and t["arm"] == other]
        if cell:
            f1 = [float(t["test_f1"]) for t in cell]
            rows.append(
                [
                    EXECUTOR_LABEL.get(cell[0]["executor"], cell[0]["executor"]),
                    {
                        "gepa": "GEPA, Tier 2",
                        "reference": f"{MODEL_LABEL.get(cell[0]['optimizer'], cell[0]['optimizer']).replace(' (hosted)', '')}, Tier 2",
                    }[other],
                    len(cell),
                    sum(int(t["prompt_changed"]) for t in cell),
                    f"{statistics.mean(f1):.3f} ± {_sd(f1):.3f}",
                    f(statistics.mean(float(t["test_recall"]) for t in cell)),
                    f(statistics.mean(float(t["test_valid_f1"]) for t in cell)),
                    f(statistics.mean(float(t["test_coverage"]) for t in cell)),
                    f(statistics.mean(float(t["temporal_f1"]) for t in cell)),
                    f(statistics.mean(float(t["product_heldout_f1"]) for t in cell)),
                ]
            )
    write_table(
        out,
        "table_tier_means",
        [
            "Executor",
            "Arm",
            "Trajectories",
            "Prompt changed",
            "Test F1 (mean ± sample SD)",
            "Test recall",
            "Valid-answer F1",
            "Coverage",
            "Temporal F1",
            "Held-out F1",
        ],
        rows,
        "Per-tier sealed-panel means across all repetitions (extension included where present), "
        "with naive and method references. Valid-answer F1 scores only parsed answers; coverage "
        "is the fraction of profiles with a parsed answer.",
    )


def table_portability(traj: list[dict[str, str]], out: Path) -> None:
    rows = []
    groups = {
        "GLM-5.3 (main loop)": [
            t
            for t in traj
            if t["release_section"] == "campaign"
            and t["arm"] == "main"
            and int(t["tier"]) == 2
            and int(t["repetition"]) <= 3
        ],
        "GLM-5.3 via GEPA": [
            t for t in traj if t["release_section"] == "campaign" and t["arm"] == "gepa"
        ],
        "DeepSeek V4 Pro": [
            t for t in traj if t["release_section"] == "campaign" and t["arm"] == "reference"
        ],
        "Claude Opus 5": [t for t in traj if t["release_section"] == "followups/opus-optimizer"],
    }
    for label, cell in groups.items():
        for ex in sorted({t["executor"] for t in cell}):
            sub = [t for t in cell if t["executor"] == ex]
            rows.append(
                [
                    label,
                    EXECUTOR_LABEL.get(ex, ex),
                    len(sub),
                    sum(int(t["prompt_changed"]) for t in sub),
                    ", ".join(
                        f(t["test_f1"]) for t in sorted(sub, key=lambda t: int(t["repetition"]))
                    ),
                    f(statistics.mean(float(t["test_f1"]) for t in sub)),
                    f(statistics.mean(float(t["test_recall"]) for t in sub)),
                ]
            )
    write_table(
        out,
        "table_optimizer_portability",
        [
            "Optimizer",
            "Executor",
            "Trajectories",
            "Prompt changed",
            "Test F1 per repetition",
            "Mean F1",
            "Mean recall",
        ],
        rows,
        "Tier 2 optimizer portability on the sealed test panel, repetitions 1 to 3, matched budgets.",
    )


def table_ceiling(panels: list[dict[str, str]], out: Path) -> None:
    rows = []
    ceiling = [p for p in panels if p["kind"] == "ceiling"]
    for model in sorted({(p["release_section"], p["model_or_executor"]) for p in ceiling}):
        sub = {
            p["panel"]: p
            for p in ceiling
            if (p["release_section"], p["model_or_executor"]) == model
        }
        first = sub[next(iter(sub))]
        rows.append(
            [
                MODEL_LABEL.get(model[1], model[1]),
                model[0].split("/")[-1],
                *[
                    f"{f(sub[pn]['f1'])} / {f(sub[pn]['recall'])}" if pn in sub else "—"
                    for pn in PANELS
                ],
                ", ".join(f(sub[pn]["coverage"]) for pn in PANELS if pn in sub),
                {"True": "yes", "False": "no", "deterministic": "n/a (deterministic)"}.get(
                    first["temperature_supported"], first["temperature_supported"]
                ),
            ]
        )
    write_table(
        out,
        "table_ceiling",
        [
            "Model",
            "Run",
            "Test F1 / recall",
            "Temporal F1 / recall",
            "Held-out F1 / recall",
            "Coverage",
            "Temperature 0",
        ],
        rows,
        "Reference rows on the sealed panels: hosted and local models with the unchanged naive "
        "prompt (one pass), and the input-matched deterministic rule baseline that parses the "
        "same rendered text.",
    )


def table_ablation(data: Path, out: Path) -> None:
    report = payload(data / "followups" / "scaffolding-ablation" / "reports" / "complete.json")
    rows = []
    for key, v in sorted(report["conditions"].items()):
        m = v["metrics"]["failure_aware_lower_bound"]
        ot = v["usage_and_latency"]["output_tokens"]
        d = v["paired_difference_from_naive"]
        _, ex, variant = key.split("/", 2) if key.count("/") >= 2 else ("", key, "")
        rows.append(
            [
                EXECUTOR_LABEL.get(ex, ex),
                variant.replace("_", " "),
                f(m["micro_f1"]),
                f(m["recall"]),
                f(m["precision"]),
                f(v["metrics"]["coverage"]),
                f(v["metrics"]["valid_answers"]["micro_f1"]),
                round(ot["mean_known"] or 0),
                f"{(d['output_tokens']['mean_known'] or 0):+.0f}",
                f"{(d['latency_seconds']['mean_known'] or 0):+.2f}",
            ]
        )
    write_table(
        out,
        "table_scaffolding_ablation",
        [
            "Executor",
            "Fixed prompt variant",
            "F1",
            "Recall",
            "Precision",
            "Coverage",
            "Valid-answer F1",
            "Mean output tokens",
            "Δ tokens vs naive",
            "Δ latency s",
        ],
        rows,
        "Fixed-prompt scaffolding ablation on the 192 sealed test profiles, 4,096-token output cap, no optimization.",
    )


def table_trajectories(traj: list[dict[str, str]], out: Path) -> None:
    rows = []
    for t in sorted(
        traj,
        key=lambda t: (
            t["release_section"],
            t["arm"],
            t["executor"],
            int(t["tier"]),
            int(t["repetition"]),
        ),
    ):
        rows.append(
            [
                t["release_section"].split("/")[-1],
                t["arm"],
                EXECUTOR_LABEL.get(t["executor"], t["executor"]),
                t["optimizer"],
                t["tier"],
                t["repetition"],
                "yes" if t["prompt_changed"] == "1" else "no",
                f(t["validation_f1"]),
                f(t["test_f1"]),
                f(t["test_recall"]),
                f(t["test_coverage"]),
                f(t["temporal_f1"]),
                f(t["product_heldout_f1"]),
                f"{f(t['test_easy_f1'], 2)}/{f(t['test_medium_f1'], 2)}/{f(t['test_hard_f1'], 2)}",
            ]
        )
    write_table(
        out,
        "table_trajectories",
        [
            "Run",
            "Arm",
            "Executor",
            "Optimizer",
            "Tier",
            "Rep",
            "Changed",
            "Validation F1",
            "Test F1",
            "Test recall",
            "Coverage",
            "Temporal F1",
            "Held-out F1",
            "Easy/medium/hard",
        ],
        rows,
        "Every optimization trajectory with its selected prompt's sealed-panel scores (appendix).",
    )


def table_cost(data: Path, out: Path) -> None:
    rows = []
    sections = [("campaign", data / "campaign")]
    if (data / "campaign-extension").is_dir():
        sections.append(("campaign-extension", data / "campaign-extension"))
    sections += [
        (f"followups/{p.name}", p) for p in sorted((data / "followups").iterdir()) if p.is_dir()
    ]
    lineage_seconds_of: dict[str, float] = {}
    for label, root in sections:
        total = 0.0
        for parent in sorted((root / "lineage").glob("*/RESOURCES.json")):
            total += float(json.loads(parent.read_text())["running_seconds_this_directory"])
        lineage_seconds_of[label] = total
    for label, root in sections:
        prov_path = root / "PROVENANCE.json"
        if not prov_path.is_file():
            continue
        r = json.loads(prov_path.read_text())["resources"]
        seconds = r["running_seconds_this_directory"] + lineage_seconds_of.get(label, 0.0)
        rows.append(
            [
                label.split("/")[-1],
                r["committed_results"],
                r["local_attempts"],
                r["hosted_attempts"],
                f"{seconds / 3600:.2f}",
                f"{r['hosted_reported_cost_usd']:.2f}",
                r["hosted_attempts_without_reported_cost"],
            ]
        )
    write_table(
        out,
        "table_cost_and_time",
        [
            "Run",
            "Committed results",
            "Local calls",
            "Hosted calls",
            "Running hours (continuation lineage included)",
            "Hosted cost USD (reported)",
            "Hosted calls without reported cost",
        ],
        rows,
        "Time and hosted cost per run, from the journal totals shipped with the release.",
    )


def table_budget(data: Path, traj: list[dict[str, str]], out: Path) -> None:
    """Search and evaluation budget per arm, from the campaign manifest and GEPA result files."""
    design = payload(data / "campaign" / "manifest.json")["design"]
    depth, cands, batch = design["depth"], design["candidates_per_round"], design["batch_size"]
    proposals = depth * cands * batch
    incumbents = depth * batch
    validation = design["validation_size"]
    panel_total = sum(design["panels"].values())
    gepa_calls: list[int] = []
    for result in sorted((data / "campaign" / "gepa-runs").glob("*/*/result.json")):
        r = payload(result)
        gepa_calls.append(int(r["executor_profile_calls"]))
    rows = [
        [
            "Main (GLM-5.3, Tiers 1–3)",
            f"{depth} rounds × {cands} candidates × {batch} profiles",
            proposals,
            incumbents,
            proposals + incumbents,
            f"shortlist × {validation}",
            panel_total,
        ],
        [
            "Reference optimizer (DeepSeek V4 Pro, Tier 2)",
            f"{depth} rounds × {cands} candidates × {batch} profiles",
            proposals,
            incumbents,
            proposals + incumbents,
            f"shortlist × {validation}",
            panel_total,
        ],
    ]
    if gepa_calls:
        rows.append(
            [
                "GEPA (Tier 2)",
                "engine-controlled; capped by executor profile calls",
                "—",
                "—",
                str(min(gepa_calls))
                if min(gepa_calls) == max(gepa_calls)
                else f"{min(gepa_calls)}–{max(gepa_calls)}",
                f"shortlist × {validation}",
                panel_total,
            ]
        )
    main = main_trajectories(traj)
    n_main = len(main)
    write_table(
        out,
        "table_budget",
        [
            "Arm",
            "Search schedule",
            "Proposal evaluations",
            "Incumbent re-evaluations",
            "Search evaluations per trajectory",
            "Selection (validation)",
            "Sealed evaluations per selected prompt",
        ],
        rows,
        f"Executor calls per trajectory by arm. Main and reference arms plan "
        f"{proposals + incumbents} search evaluations per trajectory ({n_main} main trajectories "
        f"in total); a candidate slot whose proposal failed is forfeited and recorded in the "
        f"trajectory's decisions, so some trajectories spent fewer. GEPA runs report their own "
        f"executor call counts. Selection and sealed "
        f"evaluations are shared across arms through the sealed-panel cache.",
    )


def table_partitions(data: Path, out: Path) -> None:
    stats = payload(data / "benchmark" / "dataset-statistics.json")
    per: dict[str, dict[str, float]] = {}
    for cell in stats["partition_strata"]:
        agg = per.setdefault(
            cell["partition"],
            {
                "profiles": 0,
                "positive_profiles": 0,
                "candidate_decisions": 0,
                "positive_decisions": 0,
                "prose_advisories": 0,
            },
        )
        for key in agg:
            agg[key] += cell[key]
    appearances: dict[str, dict[str, int]] = {}
    with (data / "benchmark" / "profiles.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            counts = appearances.setdefault(record["partition"], {})
            for cve in record["advisories"]:
                counts[cve] = counts.get(cve, 0) + 1
    order = [
        "optimization",
        "validation",
        "test",
        "temporal",
        "product_heldout",
        "pilot_development",
        "pilot_validation",
    ]
    rows = []
    for partition in order:
        if partition not in per:
            continue
        a = per[partition]
        rows.append(
            [
                partition,
                int(a["profiles"]),
                int(a["positive_profiles"]),
                int(a["candidate_decisions"]),
                int(a["positive_decisions"]),
                f"{a['positive_decisions'] / a['candidate_decisions']:.3f}",
                f"{a['candidate_decisions'] / a['profiles']:.1f}",
                len(appearances.get(partition, {})),
                f"{a['candidate_decisions'] / max(len(appearances.get(partition, {})), 1):.1f}",
                f"{a['prose_advisories'] / a['candidate_decisions']:.2f}",
            ]
        )
    write_table(
        out,
        "table_partitions",
        [
            "Partition",
            "Profiles",
            "Profiles with ≥1 applicable CVE",
            "Candidate decisions",
            "Applicable decisions",
            "Decision-level prevalence",
            "Advisories per profile",
            "Distinct CVEs appearing",
            "Mean appearances per CVE",
            "Prose-rendered fraction",
        ],
        rows,
        "Benchmark 07 composition. Each profile pairs one synthetic inventory with a candidate "
        "advisory list; a decision is one (profile, advisory) pair. CVEs are assigned to exactly "
        "one partition and reused across that partition's profiles as anchors or distractors.",
    )


def table_chronology(data: Path, out: Path) -> None:
    """Run chronology from the timestamps embedded in the released run names."""
    index = json.loads((data / "INDEX.json").read_text())
    events: list[tuple[str, str, str]] = []

    def stamp(name: str) -> str:
        m = re.search(r"(\d{8})T(\d{2})(\d{2})", name)
        if m:
            d, hh, mm = m.groups()
            return f"{d[:4]}-{d[4:6]}-{d[6:]} {hh}:{mm} UTC"
        m = re.search(r"(\d{8})-\d{2}", name)
        if m:
            d = m.group(1)
            return f"{d[:4]}-{d[4:6]}-{d[6:]}"
        return ""

    labels = {
        "benchmark": "Benchmark 07 built (admitted after source review)",
        "calibration": "Calibration gate on benchmark 07",
        "followups/scaffolding-ablation": "Fixed-prompt scaffolding control (before the campaign)",
        "campaign": "Phase 3 campaign, final continuation (complete journal)",
        "campaign-extension": "Repetition extension, final continuation (complete journal)",
        "followups/ceiling-opus-and-glm-reka": "Hosted reference rows: Claude Opus 5, GLM-5.3 (Reka)",
        "followups/ceiling-glm-akashml": "Hosted reference row: GLM-5.3 (AkashML) after WAF block",
        "followups/opus-optimizer": "Claude Opus 5 as optimizer, both executors, Tier 2",
        "followups/naive-local": "Local naive-prompt rows on all sealed panels",
        "followups/rule-baseline": "Input-matched deterministic rule baseline",
    }
    for section, info in index["sections"].items():
        if section not in labels:
            continue
        run = info.get("source_run") or ""
        events.append((stamp(run), labels[section], run))
    for parent in sorted((data / "campaign" / "lineage").iterdir()):
        if parent.is_dir():
            events.append((stamp(parent.name), "Campaign lineage segment", parent.name))
    events.sort()
    write_table(
        out,
        "table_chronology",
        ["Start (from run identifier)", "Event", "Released run"],
        [list(e) for e in events],
        "Chronology of the released runs. Start times are the UTC timestamps embedded in each "
        "run identifier; the campaign's continuations are listed as lineage segments.",
    )


def save_figure(fig: Any, out: Path, stem: str, panels: bool = False) -> None:
    """Publication exports: final-size labels, vector PDF and opaque 900-dpi PNG."""
    import matplotlib as mpl
    from matplotlib.text import Text
    from PIL import Image

    for label in fig.findobj(Text):
        label.set_fontsize(max(8, label.get_fontsize()))
    if panels:
        for letter, ax in zip("AB", fig.axes[:2], strict=False):
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
    fig.savefig(out / f"{stem}.png", dpi=900, facecolor="white", transparent=False)
    with Image.open(out / f"{stem}.png") as image:
        rgb = image.convert("RGB")
    rgb.save(out / f"{stem}.png", dpi=(900, 900))
    # Embed the fonts as TrueType subsets; production rejects unembedded core fonts.
    with mpl.rc_context({"pdf.fonttype": 42}):
        fig.savefig(out / f"{stem}.pdf", dpi=900, facecolor="white", transparent=False)


def figure_architecture(out: Path) -> list[str]:
    """Main-loop data flow; the caption supplies the full configuration."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
    except ImportError:
        return []
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.97)
    ax.set(xlim=(0, 100), ylim=(0, 64))
    ax.axis("off")
    ink = "#333333"

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.3,rounding_size=1",
                facecolor=color,
                edgecolor=ink,
                linewidth=0.8,
            )
        )
        ax.text(
            x + w / 2, y + h - 3, title, ha="center", va="center", fontsize=8, fontweight="bold"
        )
        ax.text(x + w / 2, y + (h - 5) / 2, body, ha="center", va="center", fontsize=8)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, linewidth=0.8, color=ink
            )
        )

    ax.add_patch(
        Rectangle(
            (0.5, 1), 74.5, 53, fill=False, edgecolor="#777777", linewidth=0.8, linestyle="--"
        )
    )
    ax.text(2, 52, "Operator-controlled host", fontsize=8, va="center")
    local = "#e8f0f8"
    box(2, 30, 21, 18, "Optimization", "240 profiles\n8 × 24 per run\nPaired schedules", "#f2f2f2")
    box(
        27, 30, 21, 18, "Local executor", "Qwen or Granite\nPrompt under test\nTemperature 0", local
    )
    box(
        52, 30, 21, 18, "Evaluator", "Scores answers\nSerializes feedback\nat Tier 1, 2 or 3", local
    )
    box(79, 30, 20, 18, "Hosted optimizer", "Main loop: GLM\n2 candidates\nper round", "#fbe9e4")
    for x0, x1 in [(23.3, 26.7), (48.3, 51.7), (73.3, 78.7)]:
        arrow(x0, 39, x1, 39)
    ax.plot([89, 89, 37.5], [48.5, 59, 59], color=ink, linewidth=0.8)
    arrow(37.5, 59, 37.5, 48.5)
    ax.text(63, 62, "Candidate prompts: score on the same batch", ha="center", fontsize=8)
    ax.text(87.5, 22, "Request:\nfeedback, task,\ncurrent prompt", ha="center", fontsize=8)
    box(27, 3, 21, 17, "Selection", "Final vs naive\n96 validation\nTies keep naive", local)
    box(52, 3, 21, 17, "Sealed panels", "Test: 192\nTemporal: 96\nHeld-out: 96", local)
    arrow(37.5, 29.5, 37.5, 20.5)
    ax.text(39, 25, "After 8 rounds", fontsize=8, va="center")
    arrow(48.3, 11.5, 51.7, 11.5)
    ax.text(13, 15, "Validation and\nsealed scores\nstay local", ha="center", fontsize=8)
    save_figure(fig, out, "fig_study_design")
    plt.close(fig)
    return ["fig_study_design.png", "fig_study_design.pdf"]


# --------------------------------------------------------------------------- figures


def figures(
    traj: list[dict[str, str]],
    panels: list[dict[str, str]],
    primary: dict[tuple[str, ...], Any],
    out: Path,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib missing; figures skipped"]
    made = []
    # Figure 2: paired per-repetition deltas
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3), sharey=True)
    for ax, ex in zip(axes, ["qwen3.8:27b/off", "granite4.2:30b/off"], strict=False):
        for i, label in enumerate([c[2] for c in CONTRASTS]):
            c = primary.get((ex, label))
            if not c:
                continue
            n = len(c["deltas"])
            x = [i + (j - (n - 1) / 2) * 0.06 for j in range(n)]
            ax.scatter(x, c["deltas"], s=18, color=["#444", "#888", "#1f77b4"][i], zorder=3)
            ax.errorbar(
                [i],
                [c["mean"]],
                yerr=[[c["mean"] - c["ci"][0]], [c["ci"][1] - c["mean"]]],
                fmt="_",
                color="black",
                capsize=4,
                lw=1.2,
                zorder=4,
            )
        ax.axhline(0, color="#aaa", lw=0.8)
        ax.axhline(MARGIN, color="#c33", lw=0.8, ls="--")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Tier 2 - Tier 3", "Tier 1 - Tier 2", "Tier 1 - Tier 3"], fontsize=8)
        n_rep = len(primary[(ex, "T2 − T3")]["deltas"]) if (ex, "T2 − T3") in primary else 0
        ax.set_title(f"{EXECUTOR_LABEL.get(ex, ex)} ({n_rep} repetitions)", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel(r"$\Delta$ sealed-test micro-F1", fontsize=8)
    fig.tight_layout()
    save_figure(fig, out, "fig_paired_deltas", panels=True)
    plt.close(fig)
    made.extend(["fig_paired_deltas.png", "fig_paired_deltas.pdf"])
    # Figure 3: ceiling ladder
    ladder: list[tuple[str, float, float]] = []
    for p in panels:
        if p["kind"] == "ceiling" and p["panel"] == "test":
            ladder.append(
                (
                    MODEL_LABEL.get(p["model_or_executor"], p["model_or_executor"]),
                    float(p["f1"]),
                    float(p["recall"]),
                )
            )
    main = main_trajectories(traj)
    for ex in sorted({t["executor"] for t in main}):
        best = max((t for t in main if t["executor"] == ex), key=lambda t: float(t["test_f1"]))
        ladder.append(
            (
                f"{EXECUTOR_LABEL.get(ex, ex)} best optimized",
                float(best["test_f1"]),
                float(best["test_recall"]),
            )
        )
    ladder.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(ladder) + 1.0))
    y = range(len(ladder))
    ax.barh(list(y), [r[1] for r in ladder], color="#555", height=0.5, label="micro-F1")
    ax.scatter([r[2] for r in ladder], list(y), color="#c33", s=16, zorder=3, label="recall")
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in ladder], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Sealed test panel", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_ceiling_ladder.png", dpi=300)
    plt.close(fig)
    made.append("fig_ceiling_ladder.png")
    return made


def _strata_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in ("easy", "medium", "hard"):
        sub = [r for r in rows if r.get("stratum") == s]
        tp = sum(int(r["tp"]) for r in sub)
        fp = sum(int(r["fp"]) for r in sub)
        fn = sum(int(r["fn"]) for r in sub)
        out[s] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return out


def more_figures(
    data: Path, traj: list[dict[str, str]], panels: list[dict[str, str]], out: Path
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made: list[str] = []
    tier_color = {1: "#7a7a7a", 2: "#1f77b4", 3: "#d62728"}

    # --- Figure: disclosure volume per tier ------------------------------------------
    sizes: dict[int, list[int]] = {1: [], 2: [], 3: []}
    disclosures = data / "exports" / "disclosures.jsonl"
    if disclosures.is_file():
        for line in disclosures.read_text().splitlines():
            d = json.loads(line)
            if d["release_section"] in CAMPAIGN_SECTIONS and d["tier"] in sizes:
                sizes[d["tier"]].append(
                    len(json.dumps(d["optimizer_view"]["feedback"], ensure_ascii=False).encode())
                )
        if all(sizes.values()):
            fig, ax = plt.subplots(figsize=(4.2, 3.0))
            ax.boxplot(
                [sizes[t] for t in (1, 2, 3)],
                tick_labels=["Tier 1", "Tier 2", "Tier 3"],
                showfliers=True,
                widths=0.5,
            )
            ax.set_yscale("log")
            ax.set_ylabel("Feedback payload per proposal (bytes, log)", fontsize=8)
            ax.tick_params(labelsize=8)
            for i, tier in enumerate((1, 2, 3), start=1):
                ax.text(
                    i,
                    max(sizes[tier]) * 1.15,
                    f"median {statistics.median(sizes[tier]):,.0f}",
                    ha="center",
                    fontsize=8,
                )
            fig.tight_layout()
            fig.savefig(out / "fig_disclosure_volume.png", dpi=300)
            plt.close(fig)
            made.append("fig_disclosure_volume.png")
            (out / "disclosure_volume.json").write_text(
                json.dumps(
                    {
                        str(t): {
                            "n": len(v),
                            "median_bytes": statistics.median(v),
                            "max_bytes": max(v),
                            "min_bytes": min(v),
                        }
                        for t, v in sizes.items()
                    },
                    indent=2,
                )
                + "\n"
            )

    # --- Figure: search dynamics and validation-vs-test ------------------------------
    decisions = read_csv(data / "exports" / "decisions.csv")
    main = main_trajectories(traj)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3))
    ax = axes[0]
    for ex, ls in (("qwen3.8:27b/off", "-"), ("granite4.2:30b/off", ":")):
        groups: dict[tuple[str, str], list[tuple[int, float]]] = {}
        for d in decisions:
            if (
                d["release_section"] in CAMPAIGN_SECTIONS
                and d["arm"] == "main"
                and d["executor"] == ex
            ):
                groups.setdefault((d["tier"], d["repetition"]), []).append(
                    (int(d["round"]), float(d["batch_f1"]))
                )
        for (tier_key, _), pts in groups.items():
            pts.sort()
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                ls,
                color=tier_color[int(tier_key)],
                lw=0.8,
                alpha=0.7,
            )
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("Winning batch micro-F1 (24 profiles)", fontsize=8)
    ax.set_title("Search dynamics", fontsize=8)
    ax.tick_params(labelsize=8)
    ax = axes[1]
    for t in main:
        ax.scatter(
            float(t["validation_f1"]),
            float(t["test_f1"]),
            s=18,
            color=tier_color[int(t["tier"])],
            marker="o" if t["executor"].startswith("qwen") else "s",
            alpha=0.85,
        )
    naive = {
        p["model_or_executor"]: p
        for p in panels
        if p["kind"] == "ceiling"
        and p["release_section"].endswith("naive-local")
        and p["panel"] == "test"
    }
    for ex, mk in (("qwen3.8:27b/off", "o"), ("granite4.2:30b/off", "s")):
        if ex in naive:
            # naive validation score: any main trajectory that kept naive has it as validation_f1
            any_t = next(t for t in main if t["executor"] == ex)
            vx = float(any_t.get("naive_validation_f1") or any_t["validation_f1"])
            ax.scatter(
                [vx],
                [float(naive[ex]["f1"])],
                s=60,
                facecolors="none",
                edgecolors="black",
                marker=mk,
                zorder=4,
            )
    lo, hi = 0.25, 0.85
    ax.plot([lo, hi], [lo, hi], color="#bbb", lw=0.8)
    ax.set_xlabel("Selection validation micro-F1 (96 profiles)", fontsize=8)
    ax.set_ylabel("Sealed test micro-F1 (192 profiles)", fontsize=8)
    ax.set_title("Selected prompts", fontsize=8)
    ax.tick_params(labelsize=8)
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([], [], color=tier_color[tier], marker="o", ls="", label=f"Tier {tier}")
            for tier in (1, 2, 3)
        ],
        fontsize=8,
        loc="lower right",
    )
    fig.tight_layout()
    save_figure(fig, out, "fig_search_dynamics", panels=True)
    plt.close(fig)
    made.extend(["fig_search_dynamics.png", "fig_search_dynamics.pdf"])

    # --- Figure: precision-recall operating points ------------------------------------
    ablation_path = data / "followups" / "scaffolding-ablation" / "reports" / "complete.json"
    ablation = payload(ablation_path)["conditions"] if ablation_path.is_file() else {}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4), sharex=True, sharey=True)
    for ax, ex in zip(axes, ("qwen3.8:27b/off", "granite4.2:30b/off"), strict=False):
        for t in main:
            if t["executor"] == ex:
                ax.scatter(
                    float(t["test_recall"]),
                    float(t["test_precision"]),
                    s=18,
                    color=tier_color[int(t["tier"])],
                    alpha=0.85,
                )
        if ex in naive:
            ax.scatter(
                [float(naive[ex]["recall"])],
                [float(naive[ex]["precision"])],
                s=70,
                facecolors="none",
                edgecolors="black",
                label="naive",
            )
        key = f"scaffolding/{ex}/output_scaffold"
        if key in ablation:
            m = ablation[key]["metrics"]["failure_aware_lower_bound"]
            ax.scatter(
                [m["recall"]],
                [m["precision"]],
                s=70,
                marker="*",
                color="#2ca02c",
                label="fixed scaffold",
            )
        for p in panels:
            if (
                p["kind"] == "ceiling"
                and p["panel"] == "test"
                and not p["release_section"].endswith("naive-local")
            ):
                lab = p["model_or_executor"].split("/")[-1]
                ax.scatter(
                    [float(p["recall"])], [float(p["precision"])], s=40, marker="^", color="#9467bd"
                )
                ax.annotate(
                    lab,
                    (float(p["recall"]), float(p["precision"])),
                    fontsize=8,
                    xytext=(-4, -9),
                    textcoords="offset points",
                    ha="right",
                )
        ax.set_title(EXECUTOR_LABEL.get(ex, ex), fontsize=9)
        ax.set_xlabel("Recall", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0.2, 1.02)
        ax.set_ylim(0.1, 1.02)
    axes[0].set_ylabel("Precision", fontsize=8)
    axes[0].legend(
        handles=[
            Line2D([], [], color=tier_color[tier], marker="o", ls="", label=f"Tier {tier}")
            for tier in (1, 2, 3)
        ]
        + [
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markerfacecolor="none",
                markeredgecolor="black",
                label="naive",
            ),
            Line2D([], [], marker="*", ls="", color="#2ca02c", label="fixed scaffold"),
            Line2D([], [], marker="^", ls="", color="#9467bd", label="reference"),
        ],
        fontsize=8,
        loc="lower left",
    )
    fig.tight_layout()
    save_figure(fig, out, "fig_precision_recall", panels=True)
    plt.close(fig)
    made.extend(["fig_precision_recall.png", "fig_precision_recall.pdf"])

    # --- Figure: per-stratum heatmap --------------------------------------------------
    strata = ("easy", "medium", "hard")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5))
    for ax, ex in zip(axes, ("qwen3.8:27b/off", "granite4.2:30b/off"), strict=False):
        rows: list[tuple[str, list[float]]] = []
        if ex in naive:
            rows.append(("Naive", [float(naive[ex][f"{s}_f1"]) for s in strata]))
        for tier in TIERS:
            cell = [t for t in main if t["executor"] == ex and int(t["tier"]) == tier]
            rows.append(
                (
                    f"Tier {tier} (mean)",
                    [statistics.mean(float(t[f"test_{s}_f1"]) for t in cell) for s in strata],
                )
            )
        key = f"scaffolding/{ex}/output_scaffold"
        if key in ablation:
            sv = _strata_rows(ablation[key]["rows"])
            rows.append(("Fixed scaffold", [sv[s] for s in strata]))
        for p in panels:
            if (
                p["kind"] == "ceiling"
                and p["panel"] == "test"
                and not p["release_section"].endswith("naive-local")
                and p["release_section"] != "followups/ceiling-opus-and-glm-reka"
                or (
                    p["kind"] == "ceiling"
                    and p["panel"] == "test"
                    and "opus" in p["model_or_executor"]
                )
            ):
                rows.append(
                    (p["model_or_executor"].split("/")[-1], [float(p[f"{s}_f1"]) for s in strata])
                )
        grid = [r[1] for r in rows]
        im = ax.imshow(grid, vmin=0.2, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels([s.capitalize() for s in strata], fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=8)
        for i, r in enumerate(grid):
            for j, v in enumerate(r):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v < 0.6 else "black",
                )
        ax.set_title(EXECUTOR_LABEL.get(ex, ex), fontsize=9)
    axes[1].set_yticklabels([])
    fig.subplots_adjust(left=0.20, right=0.94, bottom=0.12, top=0.87, wspace=0.08)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02).set_label("Sealed test micro-F1", fontsize=8)
    save_figure(fig, out, "fig_strata_heatmap", panels=True)
    plt.close(fig)
    made.extend(["fig_strata_heatmap.png", "fig_strata_heatmap.pdf"])

    # --- Figure: calibration history ----------------------------------------------------
    history: list[tuple[str, dict[str, float]]] = []
    hist_root = data / "calibration" / "history"
    for name in ("pilot-benchmark-03", "calibration-benchmark-06"):
        d = hist_root / name
        if (d / "stages" / "P1" / "report.json").is_file():
            r = payload(d / "stages" / "P1" / "report.json")
            history.append(
                (
                    "03",
                    {
                        s["condition_id"]: s["baselines"]["naive"]["failure_aware_lower_bound"][
                            "micro_f1"
                        ]
                        for s in r["summaries"]
                    },
                )
            )
        elif (d / "reports" / "complete.json").is_file():
            r = payload(d / "reports" / "complete.json")
            history.append(
                (
                    "06",
                    {
                        k: v["naive"]["metrics"]["failure_aware_lower_bound"]["micro_f1"]
                        for k, v in r["local"].items()
                    },
                )
            )
    r7 = payload(data / "calibration" / "reports" / "complete.json")
    history.append(
        (
            "07",
            {
                k: v["naive"]["metrics"]["failure_aware_lower_bound"]["micro_f1"]
                for k, v in r7["local"].items()
            },
        )
    )
    conds = sorted({c for _, h in history for c in h})
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax.axhspan(0.30, 0.70, color="#e8f4e8", zorder=0)
    xs = list(range(len(history)))
    for c in conds:
        ys = [h.get(c, float("nan")) for _, h in history]
        ax.plot(xs, ys, marker="o", lw=1, ms=4, label=c)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"benchmark {v}" for v, _ in history], fontsize=8)
    ax.set_ylabel("Naive-prompt micro-F1", fontsize=8)
    ax.set_ylim(0.2, 1.05)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig_calibration_history.png", dpi=300)
    plt.close(fig)
    made.append("fig_calibration_history.png")
    return made


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data = args.data.resolve()
    out = (args.out or data / "tables").resolve()
    out.mkdir(parents=True, exist_ok=True)
    traj = read_csv(data / "exports" / "trajectories.csv")
    panels = read_csv(data / "exports" / "panels.csv")
    table_calibration(data, out)
    primary = table_primary(traj, out)
    table_tier_means(traj, panels, out)
    table_portability(traj, out)
    table_ceiling(panels, out)
    if (data / "followups" / "scaffolding-ablation").is_dir():
        table_ablation(data, out)
    table_trajectories(traj, out)
    table_cost(data, out)
    table_budget(data, traj, out)
    table_partitions(data, out)
    table_chronology(data, out)
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans"],
            "axes.unicode_minus": False,
            "font.size": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    made = (
        figure_architecture(out)
        + figures(traj, panels, primary, out)
        + more_figures(data, traj, panels, out)
    )
    summary = {
        "tables": sorted(p.name for p in out.glob("table_*.md")),
        "figures": made,
        "primary_contrast": {
            f"{k[0]} {k[1]} {k[2]}": {
                "repetitions": v["repetitions"],
                "mean": v["mean"],
                "sd": v["sd"],
                "ci95_bootstrap": v["ci"],
                "ci95_paired_t": v["t_ci"],
            }
            for k, v in primary.items()
            if len(k) == 3
        },
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
