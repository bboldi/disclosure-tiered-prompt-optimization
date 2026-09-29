"""Conservative CNA admission with explicit negative status and complete interval checks."""

from __future__ import annotations

import re
from typing import Any

from .oracle import Installed, PublicAdvisory, Unsupported, numeric, split_cpe
from .reference import compare

CONTRACT = "nvd-cna-numeric-agreement-v1"
TEXT_SCOPE_POLICY = "conservative-text-prerequisite-cues-v1"
TEXT_SCOPE_CUES = re.compile(
    r"\b(?:windows|linux|macos|mac os|android|ios|aarch64|arm(?:64)?|x86|x64|amd64|"
    r"(?:32|64)[ -]bit|platforms?|architectures?|bindings?|plugins?|modules?|"
    r"configuration|configured|enabled|disabled|optional|settings?|compiled|"
    r"fips|non[ -]default|kernel|drivers?|requires?|prerequisites?)\b",
    re.IGNORECASE,
)
VENDOR_ALIASES = {
    "apachesoftwarefoundation": "apache",
    "oraclecorporation": "oracle",
    "sickag": "sick",
}
# These product aliases were checked against the retained initial source review.
PRODUCT_ALIASES = {
    ("apache", "apachesuperset"): "superset",
    ("apache", "apachecloudstack"): "cloudstack",
    ("apache", "apacheairflowapachehiveprovider"): "apacheairflowprovidersapachehive",
}


def folded(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def text_scope_flags(advisory: PublicAdvisory, raw: dict[str, Any]) -> list[str]:
    """Conservative exclusions, not a claim that every cue is an actual prerequisite."""
    descriptions = [advisory.description] + [
        d["value"]
        for d in raw["containers"]["cna"].get("descriptions", [])
        if d.get("lang") == "en" and isinstance(d.get("value"), str)
    ]
    return sorted(
        {
            match.group().casefold()
            for text in descriptions
            for match in TEXT_SCOPE_CUES.finditer(text)
        }
    )


def identity(product: dict[str, Any], advisory: PublicAdvisory) -> set[tuple[str, str]]:
    cpes = product.get("cpes") or []
    if cpes:
        pairs = set()
        for cpe in cpes:
            part, vendor, name, cpe_version, *extra = split_cpe(cpe)
            if (
                part != "a"
                or cpe_version != "*"
                or any(x != "*" for x in extra)
                or any(c in vendor + name for c in "*?")
            ):
                raise Unsupported("cna_cpe_constraints")
            pairs.add((vendor.casefold(), name.casefold()))
        if not pairs <= advisory.products:
            raise Unsupported("cna_cpe_product_mismatch")
        return pairs
    raw_vendor, raw_name = product.get("vendor"), product.get("product")
    if not isinstance(raw_vendor, str) or not isinstance(raw_name, str):
        raise Unsupported("cna_missing_product_identity")
    vendor_key = VENDOR_ALIASES.get(folded(raw_vendor), folded(raw_vendor))
    name_key = PRODUCT_ALIASES.get((vendor_key, folded(raw_name)), folded(raw_name))
    matches = {
        (v, p) for v, p in advisory.products if folded(v) == vendor_key and folded(p) == name_key
    }
    if len(matches) != 1:
        raise Unsupported("cna_product_identity_not_confirmed")
    return matches


def entry_matches(entry: dict[str, Any], version: str) -> bool:
    start = entry["version"]
    if "lessThan" not in entry and "lessThanOrEqual" not in entry:
        return compare(version, start) == 0
    upper = str(entry.get("lessThan", entry.get("lessThanOrEqual")))
    return compare(version, start) >= 0 and (
        upper == "*"
        or compare(version, upper) < 0
        or ("lessThanOrEqual" in entry and compare(version, upper) == 0)
    )


def status_for(product: dict[str, Any], version: str) -> str:
    """CNA raw-status evaluation, separate from the NVD Criterion implementation."""
    matching = [e for e in product["versions"] if entry_matches(e, version)]
    if len(matching) > 1:
        raise Unsupported("cna_overlapping_version_entries")
    return str(matching[0]["status"]) if matching else str(product.get("defaultStatus", "unknown"))


def probes(bounds: set[tuple[int, ...]]) -> list[str]:
    """One point in every constant-truth cell, including exact boundaries and unbounded tail."""
    ordered = sorted(bounds | {(0,)})
    points = set(ordered)
    for lower, upper in zip(ordered, ordered[1:], strict=False):
        between = lower + (0,) * max(0, len(upper) - len(lower)) + (1,)
        if not lower < between < upper:
            raise ValueError("failed to construct an interior numeric interval witness")
        points.add(between)
    points.add(ordered[-1] + (1,))
    return [".".join(map(str, point)) for point in sorted(points)]


def corroborate(advisory: PublicAdvisory, raw: dict[str, Any]) -> dict[str, Any]:
    if (
        raw.get("dataType") != "CVE_RECORD"
        or raw.get("cveMetadata", {}).get("cveId") != advisory.id
        or raw.get("cveMetadata", {}).get("state") != "PUBLISHED"
    ):
        raise Unsupported("cna_identity_or_publication_state")
    cna = raw["containers"]["cna"]
    affected = cna.get("affected")
    if not isinstance(affected, list) or not affected:
        raise Unsupported("cna_missing_affected")
    mapped: dict[tuple[str, str], dict[str, Any]] = {}
    for product in affected:
        # The initial audit demonstrated that omitted/unknown defaults cannot support
        # arbitrary version negatives. No source status is silently imputed here.
        if product.get("defaultStatus", "unknown") != "unaffected":
            raise Unsupported("cna_unknown_or_nonnegative_default")
        if any(product.get(k) for k in ("platforms", "modules", "programFiles", "programRoutines")):
            raise Unsupported("cna_additional_platform_or_component_constraints")
        entries = product.get("versions")
        if not isinstance(entries, list) or not entries or len(entries) > 32:
            raise Unsupported("cna_missing_or_excessive_versions")
        for entry in entries:
            if entry.get("status") != "affected" or entry.get("changes"):
                raise Unsupported("cna_explicit_status_override_or_unknown")
            numeric(entry["version"])
            if "lessThan" in entry and "lessThanOrEqual" in entry:
                raise Unsupported("cna_conflicting_upper_bounds")
            if "lessThan" in entry or "lessThanOrEqual" in entry:
                if entry.get("versionType") not in ("semver", "custom"):
                    raise Unsupported("cna_unsupported_version_type")
                upper = entry.get("lessThan", entry.get("lessThanOrEqual"))
                if upper != "*":
                    numeric(upper)
                    comparison = compare(entry["version"], upper)
                    if comparison > 0 or (comparison == 0 and "lessThan" in entry):
                        raise Unsupported("cna_empty_or_inverted_range")
        for pair in identity(product, advisory):
            if pair in mapped:
                raise Unsupported("cna_duplicate_product_scopes")
            mapped[pair] = product
    if set(mapped) != advisory.products:
        raise Unsupported("cna_incomplete_product_coverage")
    checked = []
    for (vendor, name), product in sorted(mapped.items()):
        bounds = {
            numeric(value)
            for term in advisory.terms
            if term.product_key == (vendor, name)
            for value in (term.exact, term.lower, term.upper)
            if value is not None
        }
        bounds.update(
            numeric(value)
            for entry in product["versions"]
            for value in (entry["version"], entry.get("lessThan"), entry.get("lessThanOrEqual"))
            if value not in (None, "*")
        )
        for version in probes(bounds):
            status = status_for(product, version)
            applicable = advisory.applies((Installed(vendor, name, version),))
            if status not in ("affected", "unaffected") or applicable != (status == "affected"):
                raise Unsupported("nvd_cna_numeric_predicate_disagreement")
            checked.append(
                {"vendor": vendor, "product": name, "version": version, "status": status}
            )
    return {
        "cve_id": advisory.id,
        "contract": CONTRACT,
        "status": "corroborated",
        "products": [list(p) for p in sorted(mapped)],
        "boundary_and_cell_checks": checked,
        "cna_provider": cna.get("providerMetadata"),
        "scope": "Agreement of complete explicit affected/unaffected numeric predicates; no unknown defaults, platform/module requirements or full ecosystem-version conformance.",
    }
