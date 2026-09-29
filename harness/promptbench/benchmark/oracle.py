"""Conservative NVD OR/application projection; not universal CPE/version conformance."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from ..storage import digest

SEMANTICS = "nvd-or-application-numeric-v1"


class Unsupported(ValueError):
    """A source condition has no label under this contract."""


def numeric(value: str) -> tuple[int, ...]:
    if (
        not isinstance(value, str)
        or len(value) > 100
        or not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))*", value)
    ):
        raise Unsupported("unsupported_numeric_version")
    parts = tuple(int(item) for item in value.split("."))
    while len(parts) > 1 and parts[-1] == 0:
        parts = parts[:-1]
    return parts


def split_cpe(criteria: str) -> tuple[str, ...]:
    if not isinstance(criteria, str) or len(criteria) > 2048:
        raise Unsupported("invalid_cpe")
    fields, buffer, escaped = [], [], False
    for char in criteria:
        if escaped:
            if not char.isascii() or char.isalnum() or char.isspace() or char in "*?":
                raise Unsupported("unsupported_cpe_escape")
            buffer.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ":":
            fields.append("".join(buffer))
            buffer = []
        elif char.isascii() and (char.isalnum() or char in "_.-*?"):
            buffer.append(char)
        else:
            raise Unsupported("invalid_cpe_character")
    fields.append("".join(buffer))
    if escaped or len(fields) != 13 or fields[:2] != ["cpe", "2.3"]:
        raise Unsupported("invalid_cpe")
    return tuple(fields[2:])


@dataclass(frozen=True)
class Installed:
    vendor: str
    product: str
    version: str
    part: str = "a"


@dataclass(frozen=True)
class Criterion:
    vendor: str
    product: str
    exact: str | None
    lower: str | None
    upper: str | None
    lower_inclusive: bool
    upper_inclusive: bool
    criteria: str
    match_criteria_id: str | None

    @property
    def product_key(self) -> tuple[str, str]:
        return self.vendor, self.product

    def matches_product(self, component: Installed) -> bool:
        return component.part == "a" and self.product_key == (
            component.vendor.casefold(),
            component.product.casefold(),
        )

    def applies(self, component: Installed) -> bool:
        if not self.matches_product(component):
            return False
        value = numeric(component.version)
        if self.exact is not None:
            return value == numeric(self.exact)
        if self.lower is not None:
            lower = numeric(self.lower)
            if value < lower or (value == lower and not self.lower_inclusive):
                return False
        if self.upper is not None:
            upper = numeric(self.upper)
            if value > upper or (value == upper and not self.upper_inclusive):
                return False
        return True

    def render(self) -> str:
        if self.exact is not None:
            bounds = "version = " + self.exact
        else:
            pieces = []
            if self.lower is not None:
                pieces.append((">= " if self.lower_inclusive else "> ") + self.lower)
            if self.upper is not None:
                pieces.append(("<= " if self.upper_inclusive else "< ") + self.upper)
            bounds = "version " + " and ".join(pieces) if pieces else "all numeric versions"
        return f"{self.vendor}/{self.product}: {bounds}"


def parse_criterion(match: dict[str, Any]) -> Criterion:
    if match.get("vulnerable") is not True:
        raise Unsupported("nonvulnerable_or_prerequisite")
    criteria = match.get("criteria")
    if not isinstance(criteria, str):
        raise Unsupported("invalid_cpe")
    fields = split_cpe(criteria)
    part, vendor, product, exact, *extra = fields
    if part != "a":
        raise Unsupported("nonapplication_cpe")
    if any(not item or item == "-" or "*" in item or "?" in item for item in (vendor, product)):
        raise Unsupported("nonliteral_product")
    if extra != ["*"] * 7:
        raise Unsupported("additional_cpe_constraints")
    lower_values = [
        match[k] for k in ("versionStartIncluding", "versionStartExcluding") if k in match
    ]
    upper_values = [match[k] for k in ("versionEndIncluding", "versionEndExcluding") if k in match]
    if len(lower_values) > 1 or len(upper_values) > 1:
        raise Unsupported("conflicting_range_endpoints")
    for bound in lower_values + upper_values:
        numeric(bound)
    lower, upper = next(iter(lower_values), None), next(iter(upper_values), None)
    if exact != "*" and (lower is not None or upper is not None):
        raise Unsupported("exact_plus_range")
    for value in (None if exact == "*" else exact, lower, upper):
        if value is not None:
            numeric(value)
    lower_inclusive = "versionStartIncluding" in match
    upper_inclusive = "versionEndIncluding" in match
    if lower is not None and upper is not None:
        if numeric(lower) > numeric(upper) or (
            numeric(lower) == numeric(upper) and not (lower_inclusive and upper_inclusive)
        ):
            raise Unsupported("empty_or_inverted_range")
    return Criterion(
        vendor.casefold(),
        product.casefold(),
        None if exact == "*" else exact,
        lower,
        upper,
        lower_inclusive,
        upper_inclusive,
        match["criteria"],
        match.get("matchCriteriaId"),
    )


@dataclass(frozen=True)
class PublicAdvisory:
    id: str
    description: str
    published: str
    last_modified: str
    terms: tuple[Criterion, ...]
    source_sha256: str

    @property
    def products(self) -> set[tuple[str, str]]:
        return {term.product_key for term in self.terms}

    def applies(self, components: tuple[Installed, ...]) -> bool:
        return any(term.applies(component) for term in self.terms for component in components)

    def matches_product(self, component: Installed) -> bool:
        return any(term.matches_product(component) for term in self.terms)

    def executor_input(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "affected": " OR ".join(term.render() for term in self.terms),
        }

    def record(self) -> dict[str, Any]:
        return asdict(self)


def parse_advisory(raw: dict[str, Any]) -> PublicAdvisory:
    if not isinstance(raw.get("id"), str) or not re.fullmatch(r"CVE-[0-9]{4}-[0-9]{4,}", raw["id"]):
        raise Unsupported("invalid_cve_id")
    if raw.get("vulnStatus") in {"Rejected", "Deferred"}:
        raise Unsupported("rejected_or_deferred")
    description = next(
        (d["value"] for d in raw.get("descriptions", []) if d.get("lang") == "en"), ""
    )
    if not isinstance(description, str) or not description.strip():
        raise Unsupported("missing_english_description")
    if len(description) > 4000:
        raise Unsupported("description_over_4000_characters")
    configurations = raw.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise Unsupported("missing_configurations")
    terms: list[Criterion] = []
    for config in configurations:
        if "children" in config or "cpeMatch" in config:
            raise Unsupported("unsupported_configuration_layout")
        nodes = config.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise Unsupported("missing_nodes")
        if config.get("negate", False) is not False or config.get("operator", "OR") != "OR":
            raise Unsupported("non_or_or_negated_configuration")
        if "operator" not in config and len(nodes) != 1:
            raise Unsupported("implicit_operator_multiple_nodes")
        for node in nodes:
            if node.get("operator") != "OR" or node.get("negate", False) is not False:
                raise Unsupported("non_or_or_negated_node")
            if "children" in node or "nodes" in node:
                raise Unsupported("nested_nodes")
            matches = node.get("cpeMatch")
            if not isinstance(matches, list) or not matches:
                raise Unsupported("missing_cpe_matches")
            terms.extend(parse_criterion(match) for match in matches)
    if len(terms) > 16:
        raise Unsupported("more_than_16_or_terms")
    if not raw.get("published") or not raw.get("lastModified"):
        raise Unsupported("missing_source_dates")
    return PublicAdvisory(
        raw["id"], description, raw["published"], raw["lastModified"], tuple(terms), digest(raw)
    )
