"""Acquire, build and independently recheck the supported public benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..storage import IntegrityError, Store, digest, read_jsonl
from .build import SEED, audit_profiles, build, load
from .sources import acquire


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "build", "audit"):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "build":
            command.add_argument("--source-dir", type=Path, required=True)
            command.add_argument("--seed", type=int, default=SEED)
            command.add_argument("--review-file", type=Path)
            command.add_argument("--admission-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            result = acquire(args.run_dir)
            print(json.dumps({"source_years": sorted(result), "verified": True}))
        elif args.command == "build":
            review = (
                Store(args.review_file.parent).get(args.review_file.name)
                if args.review_file
                else None
            )
            result = build(
                args.run_dir,
                args.source_dir,
                seed=args.seed,
                review=review,
                admission_root=args.admission_dir,
            )
            print(
                json.dumps(
                    {
                        "counts": result["counts"],
                        "selected_cves": result["selected_unique_cves"],
                        "manifest_sha256": digest(result),
                    },
                    indent=2,
                )
            )
        else:
            store = Store(args.run_dir)
            profiles = load(args.run_dir)
            raw = {c["id"]: c for c in read_jsonl(store.root / "selected-source-cves.jsonl")}
            partitions = store.get("partitions.json")
            if digest(partitions) != store.get("manifest.json")["partition_sha256"]:
                raise IntegrityError("partition evidence changed")
            result = audit_profiles(
                profiles, raw, partitions["cve_assignments"], partitions["product_family_groups"]
            )
            if digest(result) != store.get("manifest.json")["audit_sha256"] or not result["passed"]:
                raise IntegrityError("benchmark audit no longer matches committed evidence")
            quality = store.get("manifest.json").get("source_quality", {})
            if quality.get("status") == "admitted_after_source_review":
                from .admission import audit_cna

                raw_cna = {
                    r["cveMetadata"]["cveId"]: r
                    for r in read_jsonl(store.root / "selected-source-cna.jsonl")
                }
                cna_audit = audit_cna(profiles, raw_cna)
                if (
                    not cna_audit["passed"]
                    or digest(cna_audit) != quality["cna_decision_audit_sha256"]
                ):
                    raise IntegrityError("CNA source audit no longer reproduces")
                result["cna_checked_decisions"] = cna_audit["checked_decisions"]
                result["cna_audit_passed"] = True
            print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        print(
            "Benchmark interrupted; rerun with the same paths and unchanged source.",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, KeyError) as exc:
        print(f"Benchmark paused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
