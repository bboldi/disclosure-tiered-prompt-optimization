"""Research profile records with local labels and a separate provider-visible view."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..domain import render_executor_input
from .oracle import Criterion, Installed, PublicAdvisory


@dataclass(frozen=True)
class BenchmarkProfile:
    id: str
    partition: str
    text: str
    components: tuple[Installed, ...]
    advisories: tuple[PublicAdvisory, ...]
    expected: tuple[str, ...]
    stratum: str
    anchor_cve: str
    presentation: dict[str, Any] = field(default_factory=dict)

    def executor_input(self, *, structured_inventory: bool = False) -> dict[str, Any]:
        prose = self.presentation.get("prose_advisories", {})
        advisories = [
            {**a.executor_input(), **({"affected": prose[a.id]["text"]} if a.id in prose else {})}
            for a in self.advisories
        ]
        return render_executor_input(
            self.text,
            [asdict(c) for c in self.components],
            advisories,
            structured_inventory=structured_inventory,
        )

    def record(self) -> dict[str, Any]:
        return {**asdict(self), "advisories": [a.id for a in self.advisories]}


def restore_advisory(row: dict[str, Any]) -> PublicAdvisory:
    return PublicAdvisory(**{**row, "terms": tuple(Criterion(**t) for t in row["terms"])})


def restore_profile(row: dict[str, Any], advisories: dict[str, PublicAdvisory]) -> BenchmarkProfile:
    return BenchmarkProfile(
        **{
            **row,
            "components": tuple(Installed(**c) for c in row["components"]),
            "advisories": tuple(advisories[a] for a in row["advisories"]),
            "expected": tuple(row["expected"]),
        }
    )


def score(profile: BenchmarkProfile, prediction: list[str] | None, status: str) -> dict[str, Any]:
    truth = set(profile.expected)
    allowed = {a.id for a in profile.advisories}
    categories = []
    if prediction is None:
        tp, fp, fn = 0, len(allowed - truth), len(truth)
        categories = [
            "transport_failure"
            if status in ("transport_failure", "request_rejected")
            else "invalid_output"
        ]
    else:
        predicted = set(prediction)
        tp, fp, fn = len(predicted & truth), len(predicted - truth), len(truth - predicted)
        for advisory in profile.advisories:
            if advisory.id in predicted - truth:
                categories.append(
                    "version_excluded"
                    if any(advisory.matches_product(c) for c in profile.components)
                    else "unrelated_product"
                )
            elif advisory.id in truth - predicted:
                categories.append("missed_applicable")
    return {
        "profile_id": profile.id,
        "partition": profile.partition,
        "stratum": profile.stratum,
        "status": status,
        "prediction": prediction,
        "expected": sorted(truth),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "valid": prediction is not None,
        "categories": categories,
    }
