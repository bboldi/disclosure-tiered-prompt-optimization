"""Predeclared, testable pilot allocation and promotion rules."""

from __future__ import annotations

import math
from typing import Any


def utility(summary: dict[str, Any]) -> float:
    return float(summary["metrics"]["failure_aware_lower_bound"]["micro_f1"])


def pareto_and_promote(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        s
        for s in summaries
        if s["metrics"]["coverage"] >= 0.95 and s["latency_p90_seconds"] is not None
    ]
    frontier = [
        s
        for s in eligible
        if not any(
            utility(t) >= utility(s)
            and t["latency_p90_seconds"] <= s["latency_p90_seconds"]
            and (utility(t) > utility(s) or t["latency_p90_seconds"] < s["latency_p90_seconds"])
            for t in eligible
        )
    ]
    # No unapproved practical-equivalence or recall threshold: exact F1, then recall,
    # then latency. Recall is retained prominently. Promotion is exploratory.
    ranked = sorted(
        eligible,
        key=lambda s: (
            -utility(s),
            -s["metrics"]["failure_aware_lower_bound"]["recall"],
            s["latency_p90_seconds"],
            s["condition_id"],
        ),
    )
    chosen: list[dict[str, Any]] = []
    for summary in ranked:
        if summary["family"] not in {s["family"] for s in chosen}:
            chosen.append(summary)
        if len(chosen) == 2:
            break
    return {
        "frontier": [s["condition_id"] for s in frontier],
        "selected": [s["condition_id"] for s in chosen],
        "ranked": [s["condition_id"] for s in ranked],
        "status": "selected" if len(chosen) == 2 else "inconclusive_insufficient_families",
        "rule": "coverage >= .95; exact failure-aware micro-F1, recall, p90 latency, stable ID; two distinct families; frontier also reported",
    }


def allocate(
    remaining_seconds: float,
    executor_p90: list[float],
    hosted_p90: float,
    *,
    available_usd: float,
    mean_hosted_cost: float,
) -> dict[str, Any]:
    if (
        len(executor_p90) != 2
        or any(
            not math.isfinite(x) or x <= 0 for x in [remaining_seconds, hosted_p90, *executor_p90]
        )
        or available_usd < 0
        or mean_hosted_cost < 0
    ):
        raise ValueError("allocation needs two measured Executors and finite timing/cost")
    reserve = 1800.0  # P3/P4, reconciliation and model loads; protected before P2.
    tried = []
    for depth in (3, 2, 1):
        for batch, validation in ((12, 24), (6, 12)):
            calls_per_trajectory = (1 + 3 * depth) * batch + 2 * validation
            hosted_calls = 8 * 2 * depth
            seconds = (
                1.25 * (4 * calls_per_trajectory * sum(executor_p90) + hosted_calls * hosted_p90)
                + 300
            )
            cost = hosted_calls * mean_hosted_cost * 2 + 0.25
            choice = {
                "iterations": depth,
                "batch_size": batch,
                "validation_size": validation,
                "seeds": [11, 29],
                "candidate_slots": 2,
                "executor_calls": 8 * calls_per_trajectory,
                "hosted_calls": hosted_calls,
                "conservative_seconds": seconds,
                "forecast_cost_usd": cost,
            }
            tried.append(choice)
            if seconds + reserve <= remaining_seconds and cost <= available_usd:
                return {
                    "status": "admitted",
                    "selected": choice,
                    "alternatives": tried,
                    "reserved_p3_p4_seconds": reserve,
                    "method": "observed p90 serial request wall time, 25% slack, 300s overhead; cloud forecast 2x observed mean, separate hard reservations",
                }
    return {
        "status": "inconclusive_minimum_coverage_does_not_fit",
        "selected": None,
        "alternatives": tried,
        "reserved_p3_p4_seconds": reserve,
    }


def main_forecast(
    executor_p90: list[float], hosted_p90: float, mean_hosted_cost: float, remaining_usd: float
) -> dict[str, Any]:
    alternatives = []
    for depth in (8, 6, 4):
        per_run = (1 + 3 * depth) * 24 + 2 * 96 + 192
        local_seconds = 15 * per_run * sum(executor_p90)
        proposals = 30 * 2 * depth
        # Higher-priority baseline/scaffolding/challenge panels still require exact work
        # manifests. This reserve is explicit and never represented as those executed arms.
        seconds = 1.3 * (local_seconds + proposals * hosted_p90) + 6 * 3600
        cost = proposals * mean_hosted_cost * 2 + 5
        alternatives.append(
            {
                "iterations": depth,
                "seeds_per_tier": 5,
                "families": 2,
                "tiers": 3,
                "core_executor_calls": 30 * per_run,
                "core_optimizer_calls": proposals,
                "conservative_hours_including_6h_control_reserve": seconds / 3600,
                "forecast_usd_including_5usd_control_reserve": cost,
                "fits_72h_and_funds": seconds <= 72 * 3600 and cost <= remaining_usd,
            }
        )
    return {
        "alternatives": alternatives,
        "status": "planning_forecast_only_exact_control_manifest_and_precision_analysis_required",
        "target_hours": 48,
        "hard_hours": 72,
    }
