"""Strict fixture semantics, response contracts, metrics, and tier feedback."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .benchmark.model import BenchmarkProfile

SEMANTICS = "synthetic-numeric-or-v1"
PARSER = "strict-json-v1"
BASELINE = "Return applicable_cves as JSON. Consider the candidate advisories."
TASK = (
    "Write an instruction for matching a system inventory to a supplied list of CVE advisories. "
    "Input is JSON containing system_profile and advisories. Each advisory supplies its ID, "
    "description, and affected information. The answer must be a JSON object whose "
    "applicable_cves array contains unique IDs from that supplied list. Return your instruction "
    "as a JSON object with one nonempty prompt string of at most 8000 characters."
)
ABSTRACTIONS = {
    "unrelated_product": "A predicted advisory does not match an installed vendor/product.",
    "version_excluded": "An installed product matches but its version is outside the affected range.",
    "missed_applicable": "An applicable advisory was omitted.",
    "invalid_output": "The response violates the required output contract.",
    "transport_failure": "An evaluation has no usable provider response.",
}


class ContractError(ValueError):
    """Input is outside the explicit benchmark or response contract."""


def strict_json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ContractError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ContractError("response is not strict JSON") from exc


def version(value: str) -> tuple[int, ...]:
    """Only synthetic dotted nonnegative integers; not a universal CPE comparator."""
    if not isinstance(value, str) or not re.fullmatch(r"\d+(?:\.\d+)*", value, flags=re.ASCII):
        raise ContractError(f"unsupported synthetic version: {value!r}")
    parts = [int(item) for item in value.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


@dataclass(frozen=True)
class Component:
    vendor: str
    product: str
    version: str


@dataclass(frozen=True)
class Advisory:
    id: str
    description: str
    vendor: str
    product: str
    exact: str | None = None
    start: str | None = None
    end: str | None = None
    start_inclusive: bool = True
    end_inclusive: bool = False

    def matches_product(self, component: Component) -> bool:
        return (self.vendor, self.product) == (component.vendor, component.product)

    def applies(self, component: Component) -> bool:
        if not self.matches_product(component):
            return False
        installed = version(component.version)
        if self.exact is not None:
            return installed == version(self.exact)
        if self.start is not None:
            lower = version(self.start)
            if installed < lower or (installed == lower and not self.start_inclusive):
                return False
        if self.end is not None:
            upper = version(self.end)
            if installed > upper or (installed == upper and not self.end_inclusive):
                return False
        return True


@dataclass(frozen=True)
class Profile:
    id: str
    partition: str
    text: str
    components: tuple[Component, ...]
    advisories: tuple[Advisory, ...]
    expected: tuple[str, ...]

    def executor_input(self, *, structured_inventory: bool = False) -> dict[str, Any]:
        # Labels and partition metadata deliberately never enter the provider request.
        return render_executor_input(
            self.text,
            [asdict(c) for c in self.components],
            [asdict(a) for a in self.advisories],
            structured_inventory=structured_inventory,
        )


def render_executor_input(
    text: str,
    components: list[dict[str, Any]],
    advisories: list[dict[str, Any]],
    *,
    structured_inventory: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {"system_profile": text, "advisories": advisories}
    if structured_inventory:
        result["components"] = components
    return result


def load_profiles(data: dict[str, Any]) -> list[Profile]:
    if data.get("semantics") != SEMANTICS:
        raise ContractError("fixture semantics version mismatch")
    profiles: list[Profile] = []
    identifiers: set[str] = set()
    cve_partitions: dict[str, str] = {}
    for row in data["profiles"]:
        try:
            profile = Profile(
                id=row["id"],
                partition=row["partition"],
                text=row["text"],
                components=tuple(Component(**item) for item in row["components"]),
                advisories=tuple(Advisory(**item) for item in row["advisories"]),
                expected=tuple(row["expected"]),
            )
        except (KeyError, TypeError) as exc:
            raise ContractError("invalid profile fixture shape") from exc
        if not re.fullmatch(r"[a-z0-9_-]+", profile.id) or profile.id in identifiers:
            raise ContractError("invalid or duplicate profile ID")
        identifiers.add(profile.id)
        if profile.partition not in {"optimization", "validation", "test"}:
            raise ContractError("unknown partition")
        if not profile.components or not profile.advisories:
            raise ContractError("empty profile inputs")
        for component in profile.components:
            version(component.version)
        cves = [a.id for a in profile.advisories]
        if len(cves) != len(set(cves)):
            raise ContractError("duplicate candidate advisory")
        for advisory in profile.advisories:
            if not re.fullmatch(r"CVE-\d{4}-\d{4,}", advisory.id):
                raise ContractError("invalid advisory ID")
            previous = cve_partitions.setdefault(advisory.id, profile.partition)
            if previous != profile.partition:
                raise ContractError("advisory overlaps partitions")
            if advisory.exact is not None and (
                advisory.start is not None or advisory.end is not None
            ):
                raise ContractError("exact and ranged versions cannot be combined")
            for bound in (advisory.exact, advisory.start, advisory.end):
                if bound is not None:
                    version(bound)
            if not isinstance(advisory.start_inclusive, bool) or not isinstance(
                advisory.end_inclusive, bool
            ):
                raise ContractError("range inclusivity must be boolean")
            if advisory.start is not None and advisory.end is not None:
                if version(advisory.start) > version(advisory.end):
                    raise ContractError("inverted range")
        if len(profile.expected) != len(set(profile.expected)):
            raise ContractError("duplicate expected label")
        computed = {
            a.id for a in profile.advisories if any(a.applies(c) for c in profile.components)
        }
        if computed != set(profile.expected):
            raise ContractError(f"independent fixture labels disagree: {profile.id}")
        profiles.append(profile)
    if {p.partition for p in profiles} != {"optimization", "validation", "test"}:
        raise ContractError("all three partitions are required")
    return profiles


def parse_answer(
    text: str, allowed: set[str], finish_reason: str = "stop", *, scaffold: bool = False
) -> list[str]:
    if finish_reason != "stop":
        raise ContractError(f"incomplete or refused response: {finish_reason}")
    data = strict_json(text)
    expected_keys = {"applicable_cves", "advisory_decisions"} if scaffold else {"applicable_cves"}
    if not isinstance(data, dict) or set(data) != expected_keys:
        raise ContractError("expected only applicable_cves")
    values = data["applicable_cves"]
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ContractError("applicable_cves must be an array of strings")
    if len(values) != len(set(values)):
        raise ContractError("duplicate prediction")
    if not set(values) <= allowed:
        raise ContractError("prediction includes unknown advisory")
    if scaffold:
        decisions = data["advisory_decisions"]
        if not isinstance(decisions, list) or any(
            not isinstance(d, dict)
            or set(d) != {"id", "applicable"}
            or not isinstance(d["id"], str)
            or type(d["applicable"]) is not bool
            for d in decisions
        ):
            raise ContractError("invalid per-advisory decision shape")
        identifiers = [d["id"] for d in decisions]
        if len(identifiers) != len(set(identifiers)) or set(identifiers) != allowed:
            raise ContractError("scaffold must cover every supplied advisory exactly once")
        if {d["id"] for d in decisions if d["applicable"]} != set(values):
            raise ContractError("scaffold and final prediction disagree")
    return sorted(values)


def parse_prompt(text: str, finish_reason: str = "stop") -> str:
    if finish_reason != "stop":
        raise ContractError("optimizer response incomplete")
    data = strict_json(text)
    if not isinstance(data, dict) or set(data) != {"prompt"}:
        raise ContractError("expected exactly one prompt field")
    prompt = data["prompt"]
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ContractError("prompt must contain 1–8000 characters")
    return prompt


def score(profile: Profile, prediction: list[str] | None, status: str) -> dict[str, Any]:
    truth = set(profile.expected)
    allowed = {a.id for a in profile.advisories}
    if prediction is None:
        # Conservative all-scheduled bound: each unresolved decision is wrong.
        return {
            "profile_id": profile.id,
            "partition": profile.partition,
            "status": status,
            "prediction": None,
            "expected": sorted(truth),
            "tp": 0,
            "fp": len(allowed - truth),
            "fn": len(truth),
            "valid": False,
            "categories": [
                "transport_failure" if status == "transport_failure" else "invalid_output"
            ],
        }
    predicted = set(prediction)
    categories = []
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
        "status": status,
        "prediction": prediction,
        "expected": sorted(truth),
        "tp": len(predicted & truth),
        "fp": len(predicted - truth),
        "fn": len(truth - predicted),
        "valid": True,
        "categories": categories,
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def pooled(items: list[dict[str, Any]]) -> dict[str, Any]:
        tp, fp, fn = (sum(row[key] for row in items) for key in ("tp", "fp", "fn"))
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        }

    valid = [r for r in rows if r["valid"]]
    return {
        "scheduled": len(rows),
        "valid": len(valid),
        "coverage": len(valid) / len(rows) if rows else 0.0,
        "valid_answers": pooled(valid),
        "failure_aware_lower_bound": pooled(rows),
    }


def feedback(
    rows: list[dict[str, Any]],
    profiles: Sequence[Profile | BenchmarkProfile],
    tier: int,
) -> dict[str, Any]:
    if tier not in (1, 2, 3):
        raise ContractError("unknown disclosure tier")
    counts = {key: 0 for key in ABSTRACTIONS}
    for row in rows:
        for category in row["categories"]:
            counts[category] += 1
    output: dict[str, Any] = {"metrics": metrics(rows), "error_counts": counts}
    if tier >= 2:
        from .relations import abstractions

        output["abstractions"] = abstractions(rows, profiles)
    if tier == 3:
        by_id = {p.id: p for p in profiles}
        output["examples"] = [
            {
                "input": by_id[row["profile_id"]].executor_input(),
                "prediction": row["prediction"],
                "expected": row["expected"],
            }
            for row in sorted(rows, key=lambda row: row["profile_id"])
            if row["categories"]
        ][:4]
    return output
