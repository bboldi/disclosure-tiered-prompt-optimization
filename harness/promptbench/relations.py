"""Local per-error relations: construct a bounded vocabulary, never redact raw text."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .benchmark.model import BenchmarkProfile
from .benchmark.oracle import Criterion, Installed, numeric
from .domain import Advisory, Component, Profile, version

ALIAS_STYLES = frozenset(
    {"vendor_stripped", "package_manager", "repo", "title_case", "underscore_to_space"}
)


def position(
    installed: tuple[int, ...], bounds: list[tuple[tuple[int, ...], bool, str]]
) -> str | None:
    """Prefer an excluding endpoint, then equality, then the first (lower) endpoint."""
    for bound, inclusive, side in bounds:
        if installed == bound:
            return "on_inclusive" if inclusive else "on_exclusive"
        if (side == "lower" and installed < bound) or (side == "upper" and installed > bound):
            return "below" if installed < bound else "above"
    if bounds:
        return "below" if installed < bounds[0][0] else "above"
    return None


def term_position(term: Advisory | Criterion, installed: str) -> str | None:
    compare = version if isinstance(term, Advisory) else numeric
    if term.exact is not None:
        return position(compare(installed), [(compare(term.exact), True, "exact")])
    lower = term.start if isinstance(term, Advisory) else term.lower
    upper = term.end if isinstance(term, Advisory) else term.upper
    lower_inc = term.start_inclusive if isinstance(term, Advisory) else term.lower_inclusive
    upper_inc = term.end_inclusive if isinstance(term, Advisory) else term.upper_inclusive
    bounds = []
    if lower is not None:
        bounds.append((compare(lower), lower_inc, "lower"))
    if upper is not None:
        bounds.append((compare(upper), upper_inc, "upper"))
    return position(compare(installed), bounds)


def error_relation(
    profile: Profile | BenchmarkProfile, advisory_id: str | None, category: str
) -> dict[str, Any]:
    relation: dict[str, Any] = {
        "category": category,
        "product_installed": None,
        "version_position": None,
        "bound_rendering": None,
        "version_segments": None,
        "alias_style": None,
        "advisory_count": len(profile.advisories),
    }
    if advisory_id is None:
        return relation
    pairs: list[tuple[int, Component | Installed, Advisory | Criterion, bool]] = []
    if isinstance(profile, Profile):
        advisory = next(a for a in profile.advisories if a.id == advisory_id)
        for index, component in enumerate(profile.components):
            if advisory.matches_product(component):
                pairs.append((index, component, advisory, advisory.applies(component)))
        relation["bound_rendering"] = "structured"
    else:
        public = next(a for a in profile.advisories if a.id == advisory_id)
        for index, installed in enumerate(profile.components):
            for term in public.terms:
                if term.matches_product(installed):
                    pairs.append((index, installed, term, term.applies(installed)))
        relation["bound_rendering"] = (
            "prose"
            if advisory_id in profile.presentation.get("prose_advisories", {})
            else "structured"
        )
    relation["product_installed"] = bool(pairs)
    if pairs:
        chosen = next((p for p in pairs if p[3]), pairs[0])
        index, component_value, clause, _ = chosen
        relation["version_position"] = term_position(clause, component_value.version)
        relation["version_segments"] = len(component_value.version.split("."))
        if isinstance(profile, BenchmarkProfile):
            styles = profile.presentation.get("components", [])
            style = styles[index].get("alias_style") if index < len(styles) else None
            relation["alias_style"] = (
                style if isinstance(style, str) and style in ALIAS_STYLES else None
            )
    return relation


def abstractions(
    rows: list[dict[str, Any]], profiles: Sequence[Profile | BenchmarkProfile]
) -> list[dict[str, Any]]:
    by_id = {p.id: p for p in profiles}
    result = []
    for row in sorted(rows, key=lambda r: r["profile_id"]):
        profile = by_id[row["profile_id"]]
        if row["prediction"] is None:
            category = (
                "transport_failure"
                if "transport_failure" in row["categories"]
                else "invalid_output"
            )
            result.append(error_relation(profile, None, category))
        else:
            predicted, expected = set(row["prediction"]), set(profile.expected)
            for advisory_id in sorted(predicted ^ expected):
                category = "missed_applicable" if advisory_id in expected else "version_excluded"
                relation = error_relation(profile, advisory_id, category)
                if category == "version_excluded" and not relation["product_installed"]:
                    relation["category"] = "unrelated_product"
                result.append(relation)
        if len(result) >= 12:
            break
    return result[:12]
