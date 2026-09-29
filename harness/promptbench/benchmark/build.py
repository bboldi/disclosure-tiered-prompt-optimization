"""Deterministic, pre-partitioned public-source profiles with a separate label check."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..live.preflight import sources
from ..runner import utc_now
from ..storage import IntegrityError, Store, atomic_write, canonical, digest, read_jsonl
from .model import BenchmarkProfile, restore_advisory, restore_profile
from .oracle import (
    SEMANTICS,
    Criterion,
    Installed,
    PublicAdvisory,
    Unsupported,
    numeric,
    parse_advisory,
)
from .reference import check
from .sources import verify_feed

COUNTS = {
    "pilot_development": 72,
    "pilot_validation": 48,
    "optimization": 240,
    "validation": 96,
    "test": 192,
    "temporal": 96,
    "product_heldout": 96,
}
STRATA = {"easy": 17, "medium": 21, "hard": 25}
SEED = 20260912


def family(product: tuple[str, str]) -> str:
    return "/".join(re.sub("[^a-z0-9]", "", part.casefold()) for part in product)


def publication_year(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).year


def assign_pools(
    advisories: list[PublicAdvisory], seed: int
) -> tuple[dict[str, str], dict[str, str]]:
    parents: dict[str, str] = {}

    def root(name: str) -> str:
        parents.setdefault(name, name)
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    for advisory in advisories:
        names = sorted(family(p) for p in advisory.products)
        for name in names[1:]:
            left, right = root(names[0]), root(name)
            parents[max(left, right)] = min(left, right)
        root(names[0])
    groups = {name: root(name) for name in sorted(parents)}
    assignments = {}
    for advisory in advisories:
        group = groups[family(min(advisory.products))]
        if int(digest({"group": group, "seed": seed}), 16) % 10 == 0:
            partition = "product_heldout"
        elif publication_year(advisory.published) == 2025:
            partition = "temporal"
        else:
            bucket = int(digest({"cve": advisory.id, "seed": seed}), 16) % 100
            partition = next(
                name
                for limit, name in (
                    (10, "pilot_development"),
                    (20, "pilot_validation"),
                    (60, "optimization"),
                    (75, "validation"),
                    (100, "test"),
                )
                if bucket < limit
            )
        assignments[advisory.id] = partition
    return assignments, groups


ALIAS_STYLES = ("vendor_stripped", "package_manager", "repo", "title_case", "underscore_to_space")
PERTURBATION_FRACTION = {"easy": 0.25, "medium": 0.65, "hard": 0.9}


def inventory_name(component: Installed, style: str) -> str:
    vendor, product = component.vendor, component.product
    if style == "vendor_stripped":
        return f"{product} {component.version}"
    if style == "package_manager":
        return f"{vendor.replace('_', '-')}-{product.replace('_', '-')} {component.version}"
    if style == "repo":
        return f"{vendor}/{product}@{component.version}"
    if style == "title_case":
        vendor = {"nozominetworks": "Nozomi Networks"}.get(vendor, vendor.replace("_", " ").title())
        product = product.upper() if len(product) <= 4 else product.replace("_", " ").title()
        return f"{vendor} {product} {component.version}"
    if style == "underscore_to_space":
        return f"{vendor.replace('_', ' ')} {product.replace('_', ' ')} {component.version}"
    raise ValueError("unknown inventory alias style")


def version_choices(term: Criterion) -> list[dict[str, str]]:
    """Synthetic versions anchored to real bounds; no new source predicates."""
    choices = []
    bounds = [
        (term.exact, True),
        (term.lower, term.lower_inclusive),
        (term.upper, term.upper_inclusive),
    ]
    for bound, inclusive in bounds:
        if bound is None:
            continue
        choices.append(
            {
                "version": bound,
                "type": "on_inclusive" if inclusive else "on_exclusive",
                "bound": bound,
            }
        )
        parts = [int(n) for n in bound.split(".")]
        for delta in (-1, 1):
            if parts[-1] + delta >= 0:
                value = ".".join(map(str, parts[:-1] + [parts[-1] + delta]))
                choices.append(
                    {
                        "version": value,
                        "type": "last_segment_minus_one" if delta < 0 else "last_segment_plus_one",
                        "bound": bound,
                    }
                )
        for suffix in ("0", "1"):
            choices.append(
                {"version": bound + "." + suffix, "type": "appended_" + suffix, "bound": bound}
            )
        for end in (10, 2, 100, 9, 11):
            value = ".".join(map(str, parts[:-1] + [end]))
            if (numeric(value) < numeric(bound)) != (value < bound) and numeric(value) != numeric(
                bound
            ):
                choices.append(
                    {"version": value, "type": "numeric_lexicographic_trap", "bound": bound}
                )
                break
    return choices


def sample_version(
    possible: list[dict[str, str]], rng: random.Random, weights: dict[str, float] | None
) -> dict[str, str]:
    """Weight existing perturbations without inventing versions or changing labels."""
    if weights is None:
        return rng.choice(possible)
    if any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("version-type weights must be finite and strictly positive")
    return rng.choices(possible, weights=[weights.get(v["type"], 1) for v in possible], k=1)[0]


def prose_bound(advisory: PublicAdvisory, raw: dict[str, Any]) -> dict[str, str] | None:
    """Only translate a complete single interval supported by a precise CNA phrase."""
    if len(advisory.terms) != 1:
        return None
    term = advisory.terms[0]
    # A 'before X' phrase completely describes this predicate only with no
    # positive lower bound. Multi-branch or partial textual claims stay structured.
    if term.exact is not None or term.upper is None or term.upper_inclusive:
        return None
    if term.lower is not None and (numeric(term.lower) != (0,) or not term.lower_inclusive):
        return None
    for description in raw["containers"]["cna"].get("descriptions", []):
        if description.get("lang") != "en":
            continue
        text = description["value"]
        pattern = (
            r"\b(?:before|prior to|earlier than|below)\s+(?:version\s+)?v?"
            + re.escape(term.upper)
            + r"(?![\d.])"
        )
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if re.search(
                r"\b(?:not|unaffected|fixed)\b",
                text[max(0, match.start() - 24) : match.start()],
                re.IGNORECASE,
            ):
                continue
            return {
                "text": f"{term.vendor}/{term.product}: versions prior to {term.upper} are affected",
                "source_phrase": match.group(),
                "source_sha256": digest(raw),
            }
    return None


def predicate_signature(advisory: PublicAdvisory, product: tuple[str, str]) -> str:
    return digest(
        [
            (t.exact, t.lower, t.upper, t.lower_inclusive, t.upper_inclusive)
            for t in advisory.terms
            if t.product_key == product
        ]
    )


def release_branches(advisory: PublicAdvisory, product: tuple[str, str]) -> set[tuple[int, int]]:
    """Operational branch: major/minor prefix of nonzero affected endpoints."""
    result = set()
    for term in advisory.terms:
        if term.product_key != product:
            continue
        for bound in (term.exact, term.lower, term.upper):
            if bound is not None and numeric(bound) != (0,):
                parts = numeric(bound) + (0,)
                result.add((parts[0], parts[1]))
    return result


def generate_pool(
    pool: list[PublicAdvisory],
    partition: str,
    count: int,
    seed: int,
    *,
    cna_records: dict[str, dict[str, Any]] | None = None,
    difficulty: dict[str, Any] | None = None,
) -> list[BenchmarkProfile]:
    rng = random.Random(f"{seed}:{partition}")
    pool = sorted(pool, key=lambda a: a.id)
    config = difficulty or {}
    by_product: dict[tuple[str, str], list[PublicAdvisory]] = defaultdict(list)
    options: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for advisory in pool:
        for product in sorted(advisory.products):
            by_product[product].append(advisory)
        for term in advisory.terms:
            options[term.product_key].extend(
                {**v, "source_cve": advisory.id} for v in version_choices(term)
            )
    products = sorted(k for k in by_product if options[k])
    paired = [
        (key, a, b)
        for key in products
        for i, a in enumerate(by_product[key])
        for b in by_product[key][i + 1 :]
        if predicate_signature(a, key) != predicate_signature(b, key)
        and release_branches(a, key)
        and release_branches(b, key)
        and release_branches(a, key) != release_branches(b, key)
    ]
    if len(pool) < 15 or len(products) < 7 or not paired:
        raise IntegrityError(f"insufficient products/advisories/branch pairs in {partition}")
    mappings = {}
    prose = {}
    if cna_records is not None:
        from .corroboration import identity

        for advisory in pool:
            mappings[advisory.id] = {
                pair: item
                for item in cna_records[advisory.id]["containers"]["cna"]["affected"]
                for pair in identity(item, advisory)
            }
            value = prose_bound(advisory, cna_records[advisory.id])
            if value:
                prose[advisory.id] = value

    def known(advisory: PublicAdvisory, components: tuple[Installed, ...]) -> bool:
        if cna_records is None:
            return True  # Engineering fixtures only; live build supplies raw CNA records.
        from .corroboration import status_for

        for component in components:
            product = mappings[advisory.id].get((component.vendor, component.product))
            if product is not None:
                status = status_for(product, component.version)
                if status not in ("affected", "unaffected") or (
                    status == "affected"
                ) != advisory.applies((component,)):
                    return False
        return True

    profiles = []
    seen = set()
    for index in range(count):
        stratum = tuple(STRATA)[index % 3]
        positive = index % 2 == 0
        for _ in range(2000):
            if stratum == "hard":
                key, anchor, branch = rng.choice(paired)
                required = [anchor, branch]
            else:
                key = rng.choice(products)
                anchor = rng.choice(by_product[key])
                required = [anchor]
            n_components = rng.randint(6, min(10, len(products) - 1))
            chosen_products = [key] + rng.sample(
                [p for p in products if p != key], n_components - 1
            )
            components_list, metadata = [], []
            for product in chosen_products:
                possible = options[product]
                if not positive:
                    possible = [
                        v
                        for v in possible
                        if not any(
                            a.applies((Installed(*product, v["version"]),))
                            for a in by_product[product]
                        )
                    ]
                elif product == key:
                    possible = [
                        v for v in possible if anchor.applies((Installed(*product, v["version"]),))
                    ]
                if not possible:
                    break
                perturbed = (
                    rng.random()
                    < config.get("perturbation_fraction", PERTURBATION_FRACTION)[stratum]
                )
                preferred = [v for v in possible if (not v["type"].startswith("on_")) == perturbed]
                choice = sample_version(
                    preferred or possible, rng, config.get("version_type_weights")
                )
                component = Installed(*product, choice["version"])
                styles = config.get("alias_styles", ALIAS_STYLES)
                style = rng.choice(styles)
                components_list.append(component)
                metadata.append({**choice, "alias_style": style})
            if len(components_list) != n_components:
                continue
            components = tuple(components_list)
            available = [
                a for a in pool if known(a, components) and (positive or not a.applies(components))
            ]
            if any(a not in available for a in required):
                continue
            distractors = [
                a for a in available if not any(a.matches_product(c) for c in components)
            ]
            related = [a for a in available if a not in distractors and a not in required]
            lo, hi = config.get(
                "advisory_ranges", {"easy": (15, 17), "medium": (18, 21), "hard": (21, 25)}
            )[stratum]
            if min(hi, len(available)) < lo:
                continue
            n_advisories = rng.randint(lo, min(hi, len(available)))
            minimum_distractors = math.ceil(n_advisories * 0.30)
            if len(distractors) < minimum_distractors:
                continue
            preferred_related = n_advisories - math.ceil(
                n_advisories * (0.55 if stratum == "easy" else 0.4 if stratum == "medium" else 0.3)
            )
            related_count = min(len(related), max(0, preferred_related - len(required)))
            selected = required + rng.sample(related, related_count)
            if n_advisories - len(selected) > len(distractors):
                continue
            selected += rng.sample(distractors, n_advisories - len(selected))
            rng.shuffle(selected)
            expected = tuple(sorted(a.id for a in selected if a.applies(components)))
            if bool(expected) != positive:
                continue
            signature = digest(
                {
                    "components": [asdict(c) for c in components],
                    "cves": sorted(a.id for a in selected),
                }
            )
            if signature in seen:
                continue
            seen.add(signature)
            presentation = {
                "components": metadata,
                "prose_advisories": {
                    a.id: prose[a.id]
                    for a in selected
                    if a.id in prose and rng.random() < config.get("prose_fraction", 0.30)
                },
                "distractor_ids": sorted(a.id for a in selected if a in distractors),
                "branch_pair": [a.id for a in required] if stratum == "hard" else [],
            }
            profiles.append(
                BenchmarkProfile(
                    f"{partition}_{index:04d}",
                    partition,
                    "Public synthetic system inventory:\n"
                    + "\n".join(
                        "- " + inventory_name(c, m["alias_style"])
                        for c, m in zip(components, metadata, strict=True)
                    ),
                    components,
                    tuple(selected),
                    expected,
                    stratum,
                    anchor.id,
                    presentation,
                )
            )
            break
        else:
            raise IntegrityError(f"cannot construct declared {stratum} prevalence in {partition}")
    return profiles


def audit_profiles(
    profiles: list[BenchmarkProfile],
    raw: dict[str, dict[str, Any]],
    assignments: dict[str, str],
    groups: dict[str, str],
) -> dict[str, Any]:
    cve_pools: dict[str, set[str]] = defaultdict(set)
    product_pools: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    disagreements = []
    identities = set()
    for profile in profiles:
        if profile.id in identities:
            raise IntegrityError("duplicate generated profile")
        identities.add(profile.id)
        if len(profile.advisories) != len({a.id for a in profile.advisories}):
            raise IntegrityError("duplicate candidate in generated profile")
        if profile.presentation:
            rendered = "Public synthetic system inventory:\n" + "\n".join(
                "- " + inventory_name(c, m["alias_style"])
                for c, m in zip(profile.components, profile.presentation["components"], strict=True)
            )
            if profile.text != rendered:
                raise IntegrityError("rendered profile differs from structured label evidence")
            distractors = [
                a.id
                for a in profile.advisories
                if not any(a.matches_product(c) for c in profile.components)
            ]
            if (
                not 6 <= len(profile.components) <= 10
                or not 15 <= len(profile.advisories) <= 25
                or len(distractors) / len(profile.advisories) < 0.30
                or sorted(distractors) != profile.presentation["distractor_ids"]
            ):
                raise IntegrityError("difficulty profile contract failed")
            if profile.stratum == "hard":
                pair = [
                    a for a in profile.advisories if a.id in profile.presentation["branch_pair"]
                ]
                installed_products = {(c.vendor, c.product) for c in profile.components}
                if len(pair) != 2 or not any(
                    release_branches(pair[0], product) != release_branches(pair[1], product)
                    for product in pair[0].products & pair[1].products & installed_products
                ):
                    raise IntegrityError("hard profile lacks distinct installed-product branches")
        checked = []
        for advisory in profile.advisories:
            if assignments[advisory.id] != profile.partition:
                raise IntegrityError("candidate CVE crosses its assigned partition")
            if digest(raw[advisory.id]) != advisory.source_sha256:
                raise IntegrityError("normalized advisory differs from source identity")
            if parse_advisory(raw[advisory.id]) != advisory:
                raise IntegrityError("normalized advisory no longer reproduces from source")
            cve_pools[advisory.id].add(profile.partition)
            for product in advisory.products:
                product_pools[groups[family(product)]].add(profile.partition)
            reference = check(raw[advisory.id], [asdict(c) for c in profile.components])
            production = advisory.applies(profile.components)
            counts["checked_decisions"] += 1
            if reference:
                checked.append(advisory.id)
            if reference != production:
                disagreements.append(
                    {
                        "profile_id": profile.id,
                        "cve_id": advisory.id,
                        "production": production,
                        "reference": reference,
                    }
                )
            year = publication_year(advisory.published)
            if profile.partition == "temporal" and year != 2025:
                raise IntegrityError("temporal panel contains a pre-2025 CVE")
            if profile.partition not in ("temporal", "product_heldout") and year not in (
                2023,
                2024,
            ):
                raise IntegrityError("2025 publication leaked into development or IID pool")
        for component in profile.components:
            product_pools[groups[family((component.vendor, component.product))]].add(
                profile.partition
            )
        if sorted(checked) != sorted(profile.expected):
            disagreements.append(
                {"profile_id": profile.id, "stored": profile.expected, "reference": checked}
            )
        counts["profiles"] += 1
        counts["positive_profiles"] += bool(profile.expected)
    overlapping_cves = {k: sorted(v) for k, v in cve_pools.items() if len(v) > 1}
    heldout_overlap = {
        k: sorted(v) for k, v in product_pools.items() if "product_heldout" in v and len(v) > 1
    }
    return {
        "counts": dict(counts),
        "cve_overlap": overlapping_cves,
        "product_heldout_overlap": heldout_overlap,
        "oracle_disagreements": disagreements,
        "temporal_product_overlap_groups": sum(
            "temporal" in p and bool(p - {"temporal", "product_heldout"})
            for p in product_pools.values()
        ),
        "passed": not (overlapping_cves or heldout_overlap or disagreements),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = "".join(canonical(row) + "\n" for row in rows).encode()
    atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def difficulty_statistics(
    profiles: list[BenchmarkProfile], difficulty: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = json.loads(canonical(difficulty or {}))
    rows = []
    for partition in sorted({p.partition for p in profiles}):
        for stratum in STRATA:
            selected = [p for p in profiles if p.partition == partition and p.stratum == stratum]
            if not selected:
                continue
            decisions = sum(len(p.advisories) for p in selected)
            rows.append(
                {
                    "partition": partition,
                    "stratum": stratum,
                    "profiles": len(selected),
                    "positive_profiles": sum(bool(p.expected) for p in selected),
                    "positive_decisions": sum(len(p.expected) for p in selected),
                    "candidate_decisions": decisions,
                    "components_per_profile": dict(
                        Counter(str(len(p.components)) for p in selected)
                    ),
                    "advisories_per_profile": dict(
                        Counter(str(len(p.advisories)) for p in selected)
                    ),
                    "alias_style_counts": dict(
                        Counter(
                            m["alias_style"] for p in selected for m in p.presentation["components"]
                        )
                    ),
                    "near_miss_type_counts": dict(
                        Counter(m["type"] for p in selected for m in p.presentation["components"])
                    ),
                    "distractor_ratio": sum(len(p.presentation["distractor_ids"]) for p in selected)
                    / decisions,
                    "prose_advisories": sum(
                        len(p.presentation["prose_advisories"]) for p in selected
                    ),
                    "profiles_with_branch_pair": sum(
                        bool(p.presentation["branch_pair"]) for p in selected
                    ),
                    "branch_pair_counts": dict(
                        Counter(
                            ",".join(p.presentation["branch_pair"])
                            for p in selected
                            if p.presentation["branch_pair"]
                        )
                    ),
                }
            )
    return {
        "partition_strata": rows,
        "generator_configuration": config,
        "perturbation_fraction_target": config.get("perturbation_fraction", PERTURBATION_FRACTION),
        "prose_sampling_fraction_of_supported_advisories": config.get("prose_fraction", 0.30),
        "note": "Seeded sampling; realized mixes are reported, not forced by changing source labels.",
    }


def build(
    root: Path,
    source_root: Path,
    *,
    counts: dict[str, int] | None = None,
    seed: int = SEED,
    review: dict[str, Any] | None = None,
    admission_root: Path | None = None,
    difficulty: dict[str, Any] | None = None,
) -> dict[str, Any]:
    counts = dict(COUNTS if counts is None else counts)
    if set(counts) != set(COUNTS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("all declared partitions require positive profile counts")
    store, source = Store(root), Store(source_root)
    review = review or {"status": "not_reviewed", "quarantine": {}}
    admission_manifest: dict[str, Any] | None = None
    admission_ids: set[str] | None = None
    preserved_partitions: dict[str, Any] | None = None
    if admission_root is not None:
        from .admission import approved

        admission_ids, preserved_partitions, admission_manifest = approved(admission_root, review)
    elif review.get("status") == "admitted_after_source_review":
        raise IntegrityError("scientific admission requires complete corroboration evidence")
    quarantine = review.get("quarantine", {})
    if not isinstance(quarantine, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not v.strip()
        for k, v in quarantine.items()
    ):
        raise ValueError("source review quarantine must map CVE identifiers to reasons")
    with store.lock():
        store.put(
            "plan.json",
            {
                "semantics": SEMANTICS,
                "seed": seed,
                "counts": counts,
                "strata_advisory_counts": STRATA,
                "difficulty": difficulty or {},
                "source_sha256": digest(sources()),
                "python_version": sys.version,
                "source_review_sha256": digest(review),
                "admission_manifest_sha256": digest(admission_manifest)
                if admission_manifest
                else None,
            },
        )
        store.put("inputs/source.json", sources())
        store.put("inputs/source-review.json", review)
        all_advisories = []
        raw_cves = {}
        source_receipts = {}
        exclusions = []
        summaries = {}
        observed_ids = set()
        for year in (2023, 2024, 2025):
            receipt = source.get(f"sources/{year}/receipt.json")
            path = source.root / receipt["feed_path"]
            verify_feed(path, receipt)
            source_receipts[str(year)] = receipt
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                feed = json.load(stream)
            if (
                feed.get("format") != "NVD_CVE"
                or feed.get("version") != "2.0"
                or feed["totalResults"] != len(feed["vulnerabilities"])
            ):
                raise IntegrityError("unexpected annual feed schema/count")
            summary: Counter[str] = Counter()
            for entry in feed["vulnerabilities"]:
                raw = entry["cve"]
                cve_id = raw.get("id")
                if cve_id in observed_ids:
                    raise IntegrityError("duplicate CVE across source feeds")
                observed_ids.add(cve_id)
                try:
                    if cve_id in quarantine:
                        raise Unsupported("source_review_quarantine")
                    advisory = parse_advisory(raw)
                    if publication_year(advisory.published) not in (2023, 2024, 2025):
                        raise Unsupported("publication_outside_2023_2025")
                except (Unsupported, ValueError, KeyError, TypeError) as exc:
                    reason = str(exc) if isinstance(exc, Unsupported) else "malformed_source_record"
                    summary[reason] += 1
                    exclusions.append(
                        {
                            "cve_id": cve_id,
                            "source_year": year,
                            "reason": reason,
                            "published": raw.get("published"),
                            "last_modified": raw.get("lastModified"),
                            "source_sha256": digest(raw),
                            "review_reason": quarantine.get(cve_id),
                        }
                    )
                    continue
                summary["eligible"] += 1
                all_advisories.append(advisory)
                raw_cves[advisory.id] = raw
            summaries[str(year)] = {
                "counts": dict(summary),
                "total": len(feed["vulnerabilities"]),
                "feed_timestamp": feed["timestamp"],
            }
            print(
                f"[benchmark sources {year - 2022}/3] {summary['eligible']} eligible CVEs",
                flush=True,
            )
        if set(quarantine) - observed_ids:
            raise IntegrityError("source review names CVEs absent from the source snapshots")
        if admission_ids is not None and preserved_partitions is not None:
            if admission_ids - {a.id for a in all_advisories}:
                raise IntegrityError("admitted CNA cohort differs from the frozen NVD source")
            all_advisories = [a for a in all_advisories if a.id in admission_ids]
            assignments = preserved_partitions["cve_assignments"]
            groups = preserved_partitions["product_family_groups"]
            store.put("inputs/corroboration-manifest.json", admission_manifest)
        else:
            assignments, groups = assign_pools(all_advisories, seed)
        store.put(
            "partitions.json", {"cve_assignments": assignments, "product_family_groups": groups}
        )
        write_jsonl(store.root / "exclusions.jsonl", exclusions)
        cna_records = None
        if admission_manifest is not None:
            from .admission import source_record

            cna_store = Store(Path(admission_manifest["cna_root"]))
            cna_records = {
                a.id: source_record(cna_store, admission_manifest["cna_revision"], a.id)[0]
                for a in all_advisories
            }
        profiles = []
        for index, (partition, count) in enumerate(counts.items()):
            generated = generate_pool(
                [a for a in all_advisories if assignments[a.id] == partition],
                partition,
                count,
                seed,
                cna_records=cna_records,
                difficulty=difficulty,
            )
            profiles.extend(generated)
            print(
                f"[benchmark {100 * (index + 1) / len(counts):.1f}%] {partition}: {len(generated)} profiles",
                flush=True,
            )
        audit = audit_profiles(profiles, raw_cves, assignments, groups)
        store.put("oracle-and-split-audit.json", audit)
        if not audit["passed"]:
            raise IntegrityError("independent label or split audit failed; disagreements retained")
        selected = sorted({a.id for p in profiles for a in p.advisories})
        by_id = {a.id: a for a in all_advisories}
        cna_audit = None
        cna_artifacts = {}
        if admission_manifest is not None:
            from .admission import audit_cna, source_record

            cna_store = Store(Path(admission_manifest["cna_root"]))
            cna_records = {
                i: source_record(cna_store, admission_manifest["cna_revision"], i)[0]
                for i in selected
            }
            cna_audit = audit_cna(profiles, cna_records)
            store.put("cna-decision-audit.json", cna_audit)
            if not cna_audit["passed"]:
                raise IntegrityError("generated applicability disagrees with CNA status")
            cna_artifacts["selected-source-cna.jsonl"] = write_jsonl(
                store.root / "selected-source-cna.jsonl", [cna_records[i] for i in selected]
            )
        artifacts = {
            **cna_artifacts,
            "profiles.jsonl": write_jsonl(
                store.root / "profiles.jsonl", [p.record() for p in profiles]
            ),
            "advisories.jsonl": write_jsonl(
                store.root / "advisories.jsonl", [by_id[i].record() for i in selected]
            ),
            "selected-source-cves.jsonl": write_jsonl(
                store.root / "selected-source-cves.jsonl", [raw_cves[i] for i in selected]
            ),
        }
        statistics = difficulty_statistics(profiles, difficulty)
        store.put("dataset-statistics.json", statistics)
        store.get("dataset-statistics.json")  # Verify JSON round-trip before admitting the build.
        manifest = {
            "semantics": SEMANTICS,
            "seed": seed,
            "counts": counts,
            "source_feeds": source_receipts,
            "source_root": str(source.root),
            "source_sha256": digest(sources()),
            "artifact_sha256": artifacts,
            "partition_sha256": digest(store.get("partitions.json")),
            "audit_sha256": digest(audit),
            "statistics_sha256": digest(statistics),
            "source_summary": summaries,
            "selected_unique_cves": len(selected),
            "construction": "balanced aliased inventories, perturbed numeric bounds, 6–10 components and 15–25 advisories with >=30% absent-product distractors; no item selection by model scores",
            "difficulty_calibration_parent": (difficulty or {}).get("calibration_parent"),
            "scope": "OR-only application numeric projection; canonical/co-occurrence product groups; no general CPE conformance or independent human label review",
            "source_quality": {
                "status": review.get("status", "not_reviewed"),
                "review_sha256": digest(review),
                "quarantined_cves": sorted(quarantine),
                "corroboration_manifest_sha256": digest(admission_manifest)
                if admission_manifest
                else None,
                "corroboration_root": str(admission_root.resolve()) if admission_root else None,
                "corroborated_eligible_cves": len(all_advisories) if admission_manifest else None,
                "cna_revision": admission_manifest["cna_revision"] if admission_manifest else None,
                "cna_decision_audit_sha256": digest(cna_audit) if cna_audit else None,
            },
        }
        store.put("manifest.json", manifest)
        atomic_write(
            store.root / "DATA_CARD.md",
            (
                "# Benchmark difficulty rebuild\n\n"
                f"Manifest SHA-256: `{digest(manifest)}`. {len(profiles)} profiles; "
                f"{len(selected)} selected CVEs from the unchanged admitted pool.\n\n"
                "The CNA-corroborated eligible cohort, frozen CVE/product/temporal assignments, "
                "production oracle and independent reference are unchanged. This remains public "
                "synthetic inventory-scoped potential applicability, not exploitability.\n\n"
                "Each inventory has 6–10 canonical components rendered with seeded inventory aliases. "
                "The default request contains no structured inventory; schema enum defaults off. "
                "Installed versions use inclusive/exclusive bounds, ±1 final segment, appended .0/.1, "
                "and numeric/lexicographic traps. Every selected decision is independently checked "
                "against raw NVD and raw CNA status, including perturbed versions.\n\n"
                "Each profile has 15–25 advisories and at least 30% absent-product distractors. "
                "Positive/negative profiles alternate within each stratum. Hard profiles include "
                "two advisories for an installed product with distinct major/minor endpoint branch sets. "
                "This branch rule and realized pair multiplicities are explicit; some partitions have few qualifying pairs. "
                f"A seeded {(difficulty or {}).get('prose_fraction', 0.30):.0%} of eligible single-interval advisories use prose; eligibility requires "
                "a matching complete CNA before/prior-to phrase, with no positive lower bound. "
                "Unsupported or conflicting prose is never substituted.\n\n"
                "Per-component alias/version provenance and per-advisory prose provenance are hidden "
                "in profiles.jsonl presentation metadata. Realized counts, distractor ratios and sizes "
                "are in dataset-statistics.json; oracle-and-split-audit.json and cna-decision-audit.json "
                "retain every check. Source selection bias, public-data contamination and synthetic "
                "external-validity limitations remain. "
                + (
                    "Global difficulty parameters were adjusted using pilot-development calibration only; "
                    "no validation, test, temporal or product-held-out outcomes guided the build. "
                    "The parent calibration and exact settings are in the manifest and statistics.\n"
                    if (difficulty or {}).get("calibration_parent")
                    else "No model scores informed this build.\n"
                )
            ).encode(),
        )
        store.put(
            "completion.json",
            {"manifest_sha256": digest(manifest), "profiles": len(profiles), "audit_passed": True},
        )
        if not store.exists("acquisition-context.json"):
            store.put(
                "acquisition-context.json",
                {
                    "completed_utc": utc_now(),
                    "source_dates": "Per-feed metadata timestamps; CVE publication/modification retained independently.",
                },
            )
        return manifest


def load(root: Path, partitions: set[str] | None = None) -> list[BenchmarkProfile]:
    store = Store(root)
    manifest = store.get("manifest.json")
    if manifest["semantics"] != SEMANTICS or store.get("completion.json")[
        "manifest_sha256"
    ] != digest(manifest):
        raise IntegrityError("incomplete or unsupported benchmark")
    if (
        "statistics_sha256" in manifest
        and digest(store.get("dataset-statistics.json")) != manifest["statistics_sha256"]
    ):
        raise IntegrityError("benchmark statistics changed")
    for name, expected in manifest["artifact_sha256"].items():
        if hashlib.sha256((store.root / name).read_bytes()).hexdigest() != expected:
            raise IntegrityError("benchmark artifact changed: " + name)
    advisories = {
        row["id"]: restore_advisory(row) for row in read_jsonl(store.root / "advisories.jsonl")
    }
    profiles = [
        restore_profile(row, advisories) for row in read_jsonl(store.root / "profiles.jsonl")
    ]
    return [p for p in profiles if partitions is None or p.partition in partitions]
