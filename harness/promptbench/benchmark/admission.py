"""Screen every eligible public CVE before profile generation; retain source exclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..live.preflight import sources
from ..storage import IntegrityError, Store, digest, read_jsonl
from .build import SEED, publication_year, write_jsonl
from .cna_sources import record_url
from .corroboration import (
    CONTRACT,
    TEXT_SCOPE_POLICY,
    corroborate,
    identity,
    status_for,
    text_scope_flags,
)
from .model import BenchmarkProfile, restore_advisory
from .oracle import PublicAdvisory, Unsupported


def source_record(cna: Store, revision: str, identifier: str) -> tuple[dict[str, Any], str]:
    receipt = cna.get(f"records/{identifier}.json")
    if (
        receipt["url"] != record_url(revision, identifier)
        or hashlib.sha256(receipt["body_text"].encode()).hexdigest() != receipt["body_sha256"]
        or receipt["error"] is not None
    ):
        raise IntegrityError("CNA source receipt no longer matches its pinned bytes")
    return json.loads(receipt["body_text"]), receipt["body_sha256"]


def shape(advisory: PublicAdvisory) -> str:
    if len(advisory.terms) > 1:
        return "multi_term_or"
    return "exact" if advisory.terms[0].exact is not None else "single_range"


def screen(root: Path, candidate_root: Path, cna_root: Path, parent_root: Path) -> dict[str, Any]:
    store, candidates, cna, parent = map(Store, (root, candidate_root, cna_root, parent_root))
    with store.lock():
        candidate_manifest = candidates.get("manifest.json")
        acquisition = cna.get("acquisition-report.json")
        if acquisition["candidate_manifest_sha256"] != digest(candidate_manifest):
            raise IntegrityError("CNA acquisition belongs to another candidate population")
        candidate_path = candidates.root / "advisories.jsonl"
        if (
            hashlib.sha256(candidate_path.read_bytes()).hexdigest()
            != candidate_manifest["advisories_sha256"]
        ):
            raise IntegrityError("candidate population bytes changed")
        partitions = parent.get("partitions.json")
        if digest(partitions) != parent.get("manifest.json")["partition_sha256"]:
            raise IntegrityError("parent split evidence changed")
        advisories = [restore_advisory(r) for r in read_jsonl(candidate_path)]
        if {a.id for a in advisories} != set(partitions["cve_assignments"]):
            raise IntegrityError("candidate population differs from preassigned parent splits")
        store.put(
            "plan.json",
            {
                "contract": CONTRACT,
                "seed": SEED,
                "source_sha256": digest(sources()),
                "candidate_manifest_sha256": digest(candidate_manifest),
                "acquisition_report_sha256": digest(acquisition),
                "parent_manifest_sha256": digest(parent.get("manifest.json")),
            },
        )
        store.put("inputs/source.json", sources())
        store.put("inputs/candidate-manifest.json", candidate_manifest)
        store.put("inputs/acquisition-report.json", acquisition)
        store.put("partitions.json", partitions)
        outcomes, admitted = [], []
        reasons: Counter[str] = Counter()
        cells: dict[str, list[PublicAdvisory]] = defaultdict(list)
        unavailable = set(acquisition["unavailable"])
        for index, advisory in enumerate(sorted(advisories, key=lambda a: a.id)):
            cna_sha = None
            try:
                if advisory.id in unavailable:
                    raise Unsupported("cna_source_unavailable")
                raw, cna_sha = source_record(cna, acquisition["commit"], advisory.id)
                evidence = corroborate(advisory, raw)
            except (Unsupported, ValueError, TypeError, KeyError) as exc:
                if isinstance(exc, IntegrityError):
                    raise
                reason = (
                    str(exc) if isinstance(exc, Unsupported) else "cna_malformed_supported_fields"
                )
                evidence = {"status": "excluded", "reason": reason}
            else:
                reason = "corroborated"
                admitted.append(advisory)
                cells[f"{publication_year(advisory.published)}/{shape(advisory)}"].append(advisory)
            reasons[reason] += 1
            outcomes.append(
                {
                    "cve_id": advisory.id,
                    "nvd_source_sha256": advisory.source_sha256,
                    "cna_body_sha256": cna_sha,
                    "partition": partitions["cve_assignments"][advisory.id],
                    "publication_year": publication_year(advisory.published),
                    **evidence,
                }
            )
            if index % 1000 == 0 or index + 1 == len(advisories):
                print(
                    f"[corroboration {100 * (index + 1) / len(advisories):.1f}%] {index + 1}/{len(advisories)}; admitted {len(admitted)}",
                    flush=True,
                )
        selection = {
            "method": "Two lowest SHA-256 values per available publication-year × predicate-shape cell, seed 20260912; selected before any model results. Empty cells retained.",
            "cells": {
                f"{year}/{kind}": {
                    "available": len(cells[f"{year}/{kind}"]),
                    "selected": [
                        a.id
                        for a in sorted(
                            cells[f"{year}/{kind}"],
                            key=lambda a: digest(f"{SEED}/corroboration-review/{a.id}"),
                        )[:2]
                    ],
                }
                for year in (2023, 2024, 2025)
                for kind in ("exact", "single_range", "multi_term_or")
            },
        }
        store.put("review-selection.json", selection)
        artifacts = {
            "outcomes.jsonl": write_jsonl(store.root / "outcomes.jsonl", outcomes),
            "advisories.jsonl": write_jsonl(
                store.root / "advisories.jsonl", [a.record() for a in admitted]
            ),
        }
        manifest = {
            "contract": CONTRACT,
            "status": "awaiting_textual_source_review",
            "candidate_count": len(advisories),
            "corroborated_count": len(admitted),
            "reason_counts": dict(reasons),
            "cna_revision": acquisition["commit"],
            "cna_root": str(cna.root),
            "candidate_root": str(candidates.root),
            "parent_root": str(parent.root),
            "parent_manifest_sha256": digest(parent.get("manifest.json")),
            "partition_sha256": digest(partitions),
            "review_selection_sha256": digest(selection),
            "artifact_sha256": artifacts,
            "partition_counts": dict(
                Counter(partitions["cve_assignments"][a.id] for a in admitted)
            ),
            "limitation": "Conservative numeric applicability projection; CNA and NVD are separate source representations, potentially sharing upstream information. No independent human review or exploitability claim.",
        }
        store.put("manifest.json", manifest)
        return manifest


def approved(root: Path, review: dict[str, Any]) -> tuple[set[str], dict[str, Any], dict[str, Any]]:
    """Validate the frozen source-review gate before allowing scientific generation."""
    store = Store(root)
    manifest = store.get("manifest.json")
    if manifest["contract"] != CONTRACT or review.get("admission_manifest_sha256") != digest(
        manifest
    ):
        raise IntegrityError("source review does not bind the corroboration manifest")
    for name, expected in manifest["artifact_sha256"].items():
        if hashlib.sha256((store.root / name).read_bytes()).hexdigest() != expected:
            raise IntegrityError("corroboration artifact changed")
    selection = store.get("review-selection.json")
    if digest(selection) != manifest["review_selection_sha256"]:
        raise IntegrityError("source-review selection changed")
    selected = {
        identifier for cell in selection["cells"].values() for identifier in cell["selected"]
    }
    reviews = review.get("records", {})
    quarantine = review.get("quarantine", {})
    if review.get("status") != "admitted_after_source_review" or not selected <= set(reviews):
        raise IntegrityError("selected textual source reviews remain incomplete")
    for identifier in selected:
        item = reviews[identifier]
        if not item.get("rationale") or item.get("status") not in ("corroborated", "quarantined"):
            raise IntegrityError("source review lacks an explicit disposition and rationale")
        if (item["status"] == "quarantined") != (identifier in quarantine):
            raise IntegrityError("source review and quarantine disagree")
    if review.get("text_scope_policy") != TEXT_SCOPE_POLICY:
        raise IntegrityError("source review lacks the declared textual prerequisite screen")
    cohort = [restore_advisory(a) for a in read_jsonl(store.root / "advisories.jsonl")]
    outcomes = {r["cve_id"]: r for r in read_jsonl(store.root / "outcomes.jsonl")}
    cna = Store(Path(manifest["cna_root"]))
    for advisory in cohort:
        raw, checksum = source_record(cna, manifest["cna_revision"], advisory.id)
        if checksum != outcomes[advisory.id]["cna_body_sha256"]:
            raise IntegrityError("CNA record differs from screened source bytes")
        flags = text_scope_flags(advisory, raw)
        if flags and advisory.id not in quarantine:
            raise IntegrityError("unresolved textual prerequisite cues in admitted cohort")
    ids = {a.id for a in cohort} - set(quarantine)
    partitions = store.get("partitions.json")
    if digest(partitions) != manifest["partition_sha256"]:
        raise IntegrityError("corroboration split evidence changed")
    return ids, partitions, manifest


def audit_cna(profiles: list[BenchmarkProfile], raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
    advisories = {a.id: a for profile in profiles for a in profile.advisories}
    mappings = {}
    for identifier, advisory in advisories.items():
        corroborate(advisory, raw[identifier])
        mappings[identifier] = {
            pair: product
            for product in raw[identifier]["containers"]["cna"]["affected"]
            for pair in identity(product, advisory)
        }
    decisions, disagreements = [], []
    for profile in profiles:
        for advisory in profile.advisories:
            statuses = []
            for component in profile.components:
                product = mappings[advisory.id].get((component.vendor, component.product))
                statuses.append(
                    "unrelated_product"
                    if product is None
                    else status_for(product, component.version)
                )
            applicable = "affected" in statuses
            if any(
                s not in ("affected", "unaffected", "unrelated_product") for s in statuses
            ) or applicable != (advisory.id in profile.expected):
                disagreements.append(
                    {"profile_id": profile.id, "cve_id": advisory.id, "statuses": statuses}
                )
            decisions.append(
                {
                    "profile_id": profile.id,
                    "cve_id": advisory.id,
                    "component_statuses": statuses,
                    "applicable": applicable,
                }
            )
    return {
        "passed": not disagreements,
        "checked_decisions": len(decisions),
        "disagreements": disagreements,
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "candidate-dir", "cna-dir", "parent-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            screen(args.run_dir, args.candidate_dir, args.cna_dir, args.parent_dir), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
