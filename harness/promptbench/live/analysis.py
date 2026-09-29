"""Offline analysis export for a study run: per-trajectory panels, paired tier contrast, forecast.

Units of analysis are trajectories (repetitions), never calls. Intervals are percentile
bootstraps: over repetitions for the primary contrast, and over sealed-test profiles as a
sensitivity view. Nothing here dispatches inference.
"""

from __future__ import annotations

import random
import statistics
from typing import Any

from ..domain import metrics
from ..storage import Store, digest

PRIMARY_PANEL = "test"
MARGIN = -0.02
RESAMPLES = 2000
SEED = 0


def _f1(rows: list[dict[str, Any]], failure_aware: bool = True) -> float:
    summary = metrics(rows)
    key = "failure_aware_lower_bound" if failure_aware else "valid_answers"
    return float(summary[key]["micro_f1"])


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(q * len(ordered))))]


def panel_rows(store: Store, executor: str, panel: str, prompt_sha: str) -> list[dict[str, Any]]:
    prefix = f"sealed/{executor}/{panel}/{prompt_sha}"
    rows = []
    for name in store.names("evaluations/*.json"):
        record = store.get(name)
        if record["key"].startswith(prefix + "/"):
            rows.append(record["row"])
    return sorted(rows, key=lambda r: r["profile_id"])


def trajectory_table(store: Store) -> list[dict[str, Any]]:
    table = []
    for name in sorted(store.names("trajectories/*.json")):
        t = store.get(name)
        row = {
            "arm": t["arm"],
            "executor": t["executor"],
            "optimizer": t["optimizer"],
            "tier": t["tier"],
            "repetition": t["repetition"],
            "prompt_changed": t["prompt_changed"],
            "selected_prompt_sha256": t["selected"]["prompt_sha256"],
            "validation_f1": t["selected"]["metrics"]["failure_aware_lower_bound"]["micro_f1"],
            "validation_recall": t["selected"]["metrics"]["failure_aware_lower_bound"]["recall"],
            "hosted_seed_sent": t.get("hosted_seed_sent"),
        }
        for panel, summary in t["sealed"].items():
            fa, va = (
                summary["metrics"]["failure_aware_lower_bound"],
                summary["metrics"]["valid_answers"],
            )
            row[f"{panel}_f1"] = fa["micro_f1"]
            row[f"{panel}_recall"] = fa["recall"]
            row[f"{panel}_precision"] = fa["precision"]
            row[f"{panel}_valid_f1"] = va["micro_f1"]
            row[f"{panel}_coverage"] = summary["metrics"]["coverage"]
            for stratum, s in summary["by_stratum"].items():
                row[f"{panel}_{stratum}_f1"] = s["failure_aware_lower_bound"]["micro_f1"]
        table.append(row)
    return table


def paired_contrast(
    store: Store, table: list[dict[str, Any]], *, left: int = 2, right: int = 3
) -> dict[str, Any]:
    """delta = F1(tier left) - F1(tier right) per (executor, repetition) on the primary panel."""
    rng = random.Random(SEED)  # noqa: S311 - reproducible resampling, not security
    result: dict[str, Any] = {}
    main = [r for r in table if r["arm"] == "main"]
    for executor in sorted({r["executor"] for r in main}):
        pairs = []
        for rep in sorted({r["repetition"] for r in main if r["executor"] == executor}):
            cell = {
                r["tier"]: r for r in main if r["executor"] == executor and r["repetition"] == rep
            }
            if left in cell and right in cell:
                pairs.append((cell[left], cell[right]))
        if not pairs:
            continue
        deltas = [a[f"{PRIMARY_PANEL}_f1"] - b[f"{PRIMARY_PANEL}_f1"] for a, b in pairs]
        recall_deltas = [
            a[f"{PRIMARY_PANEL}_recall"] - b[f"{PRIMARY_PANEL}_recall"] for a, b in pairs
        ]
        rep_boot = []
        for _ in range(RESAMPLES):
            sample = [rng.choice(deltas) for _ in deltas]
            rep_boot.append(statistics.mean(sample))
        # Sensitivity: resample sealed-test profiles jointly for both prompts of every pair.
        profile_boot = []
        pair_rows = [
            (
                panel_rows(store, executor, PRIMARY_PANEL, a["selected_prompt_sha256"]),
                panel_rows(store, executor, PRIMARY_PANEL, b["selected_prompt_sha256"]),
            )
            for a, b in pairs
        ]
        if all(len(x) == len(y) and x for x, y in pair_rows):
            n = len(pair_rows[0][0])
            for _ in range(RESAMPLES):
                idx = [rng.randrange(n) for _ in range(n)]
                profile_boot.append(
                    statistics.mean(
                        _f1([x[i] for i in idx]) - _f1([y[i] for i in idx]) for x, y in pair_rows
                    )
                )
        lower = _percentile(rep_boot, 0.025) if len(deltas) > 1 else None
        result[executor] = {
            "pairs": len(pairs),
            "deltas": deltas,
            "recall_deltas": recall_deltas,
            "mean_delta": statistics.mean(deltas),
            "mean_recall_delta": statistics.mean(recall_deltas),
            "repetition_bootstrap_ci95": [lower, _percentile(rep_boot, 0.975)]
            if len(deltas) > 1
            else None,
            "profile_bootstrap_ci95": [
                _percentile(profile_boot, 0.025),
                _percentile(profile_boot, 0.975),
            ]
            if profile_boot
            else None,
            "retained_utility": (lower is not None and lower > MARGIN),
            "note": (
                "Retained utility requires the repetition-bootstrap lower bound above -0.02 per "
                "executor; fewer than two pairs cannot establish it. Profile bootstrap is a "
                "sensitivity view that treats prompts as fixed."
            ),
        }
    return result


def analysis(store: Store) -> dict[str, Any]:
    table = trajectory_table(store)
    by_cell: dict[str, Any] = {}
    for row in table:
        cell = by_cell.setdefault(
            f"{row['arm']}/{row['executor']}/T{row['tier']}",
            {"repetitions": 0, "prompt_changed": 0, "test_f1": [], "test_recall": []},
        )
        cell["repetitions"] += 1
        cell["prompt_changed"] += int(row["prompt_changed"])
        cell["test_f1"].append(row.get(f"{PRIMARY_PANEL}_f1"))
        cell["test_recall"].append(row.get(f"{PRIMARY_PANEL}_recall"))
    for cell in by_cell.values():
        known = [v for v in cell["test_f1"] if v is not None]
        cell["mean_test_f1"] = statistics.mean(known) if known else None
        cell["sd_test_f1"] = statistics.pstdev(known) if len(known) > 1 else None
    return {
        "trajectories": table,
        "cells": by_cell,
        "primary_contrast_T2_minus_T3": paired_contrast(store, table, left=2, right=3),
        "secondary_contrast_T1_minus_T2": paired_contrast(store, table, left=1, right=2),
        "margin": MARGIN,
        "resamples": RESAMPLES,
        "table_sha256": digest(table),
    }


def forecast(store: Store, manifest: dict[str, Any]) -> dict[str, Any]:
    """Main-study hours from this run's measured per-call timings; serial hosted calls."""
    local: dict[str, list[float]] = {}
    hosted: list[float] = []
    for name in store.names("work/*/spec.json"):
        spec = store.get(name)
        folder = name.rsplit("/", 1)[0]
        for response in store.names(folder + "/attempts/*/response.json"):
            seconds = store.get(response)["duration_ns"] / 1e9
            if spec["role"] == "optimizer":
                hosted.append(seconds)
            else:
                local.setdefault(spec["condition"]["id"], []).append(seconds)
    if not local or not hosted:
        return {"status": "insufficient_measurements"}
    local_mean = {k: statistics.mean(v) for k, v in local.items()}
    hosted_mean = statistics.mean(hosted)
    # Wall clock per committed call from the run's own session clocks; includes journaling,
    # checkpoints, identity checks and hosted waits. This is what forecasts must use.
    accounted = 0.0
    for session in (store.root / "sessions").glob("*"):
        names = sorted(session.glob("[0-9]*.json"))
        if names:
            last = store.get(str(names[-1].relative_to(store.root)))
            accounted += last["elapsed_seconds"]
    committed = len(store.names("work/*/result.json"))
    hosted_seconds = sum(hosted)
    local_calls = max(1, committed - len(hosted))
    wall_local = max((accounted - hosted_seconds) / local_calls, max(local_mean.values()))
    local_mean = {k: wall_local for k in local_mean}
    design = manifest["design"]
    alternatives = []
    for families in (2, 3):
        executors = list(design["executors"])[:families]
        if len(executors) < families:
            continue
        for depth in (8, 6, 4):
            batch, slots, reps = design["batch_size"], design["candidates_per_round"], 5
            per_local = (
                batch * depth * (slots + 1)
                + 2 * design["validation_size"]
                + sum(design["panels"].values())
            )
            per_hosted = depth * slots
            trajectories = reps * len(design["tiers"])
            local_seconds = sum(
                trajectories * per_local * local_mean.get(e, max(local_mean.values()))
                for e in executors
            )
            hosted_seconds = trajectories * families * per_hosted * hosted_mean
            total = 1.15 * (local_seconds + hosted_seconds) + 3600
            alternatives.append(
                {
                    "families": families,
                    "executors": executors,
                    "depth": depth,
                    "repetitions": reps,
                    "local_calls_upper_bound": trajectories * per_local * families,
                    "hosted_calls": trajectories * per_hosted * families,
                    "local_hours": local_seconds / 3600,
                    "hosted_hours_serial": hosted_seconds / 3600,
                    "conservative_hours": total / 3600,
                    "fits_48h": total <= 48 * 3600,
                    "fits_72h": total <= 72 * 3600,
                }
            )
    return {
        "measured_local_wall_seconds_per_call": local_mean,
        "accounted_running_seconds": accounted,
        "measured_hosted_mean_seconds": hosted_mean,
        "alternatives": alternatives,
        "excluded_from_forecast": "executed scaffolding controls; GEPA and reference arms; Pi contingency",
        "note": "Local time is measured wall clock per committed call (not response time), 1.15 multiplier plus one hour for swaps and closeout; hosted calls serial.",
    }
