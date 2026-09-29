"""Assemble the public data release from run directories.

Copies the scientific records (never raw requests/responses, checks, telemetry samples or the
frozen runtime) into a release tree and writes flat CSV/JSONL exports next to them. Each
release section records the source run identity and the manifest hash it was taken from.

Usage:
  python scripts/export_release.py --out release/data \
      --benchmark <benchmark dir> --admission <admission dir> --candidates <candidates dir> \
      --feeds <benchmark-01/sources dir> --cna <cna acquisition dir> \
      --calibration <calibration dir> --campaign <final continuation dir> \
      --lineage <parent dir> [--lineage <parent dir> ...] \
      --followup name=<dir> [--followup name=<dir> ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

RECORD_FOLDERS = (
    "trajectories",
    "decisions",
    "disclosures",
    "evaluations",
    "selections",
    "sealed-panels",
    "panels",
    "conditions",
    "reports",
    "generations",
    "reconciliations",
    "hosted-key-usage",
    "candidate-failures",
    "exports",
    "pauses",
)
ROOT_FILES = (
    "manifest.json",
    "plan.json",
    "parent-lineage.json",
    "SCAFFOLDING_REPORT.md",
    "scientific-integrity-audit.json",
    "completed-replay-audit.json",
    "final-integrity-audit.json",
    "development.json",
)
INPUT_FILES = (
    "profiles.json",
    "advisories.json",
    "prompts.json",
    "conditions.json",
    "study_conditions.json",
    "model_registry.json",
    "calibration.json",
    "models_toml.json",
)


def payload(path: Path) -> Any:
    data = json.loads(path.read_text())
    return (
        data["payload"] if isinstance(data, dict) and set(data) == {"payload", "sha256"} else data
    )


def resource_summary(src: Path) -> dict[str, Any]:
    """Derived usage figures from the run's private journal: counts, seconds and hosted cost.

    The raw requests and responses stay unpublished; only these totals travel with the release.
    """
    hosted_attempts = local_attempts = 0
    cost = 0.0
    unknown = 0
    for request in (src / "work").glob("*/attempts/*/request.json"):
        record = payload(request)
        response = request.with_name("response.json")
        if record.get("provider") == "openrouter":
            hosted_attempts += 1
            if response.is_file():
                try:
                    usage = json.loads(payload(response)["body_text"]).get("usage", {})
                    cost += float(usage.get("cost") or 0.0)
                except (ValueError, TypeError, KeyError):
                    unknown += 1
            else:
                unknown += 1
        else:
            local_attempts += 1
    seconds = 0.0
    for session in (src / "sessions").glob("*"):
        names = sorted(session.glob("[0-9]*.json"))
        if names:
            last = payload(names[-1])
            seconds += float(last.get("elapsed_seconds", 0.0))
    committed = sum(1 for _ in (src / "work").glob("*/result.json"))
    return {
        "committed_results": committed,
        "local_attempts": local_attempts,
        "hosted_attempts": hosted_attempts,
        "hosted_attempts_without_reported_cost": unknown,
        "hosted_reported_cost_usd": round(cost, 6),
        "running_seconds_this_directory": round(seconds, 3),
    }


def copy_records(src: Path, dst: Path, note: str) -> dict[str, Any]:
    dst.mkdir(parents=True, exist_ok=True)
    copied: dict[str, int] = {}
    for folder in RECORD_FOLDERS:
        if (src / folder).is_dir():
            shutil.copytree(src / folder, dst / folder, dirs_exist_ok=True)
            copied[folder] = sum(1 for p in (dst / folder).rglob("*") if p.is_file())
    for name in ROOT_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    if (src / "inputs").is_dir():
        (dst / "inputs").mkdir(exist_ok=True)
        for name in INPUT_FILES:
            if (src / "inputs" / name).is_file():
                shutil.copy2(src / "inputs" / name, dst / "inputs" / name)
    if (src / "gepa-runs").is_dir():
        for run in (src / "gepa-runs").glob("*/*"):
            target = dst / "gepa-runs" / run.parent.name / run.name
            target.mkdir(parents=True, exist_ok=True)
            for item in ("manifest.json", "result.json"):
                if (run / item).is_file():
                    shutil.copy2(run / item, target / item)
            for folder in ("evaluations", "disclosures"):
                if (run / folder).is_dir():
                    shutil.copytree(run / folder, target / folder, dirs_exist_ok=True)
    summary_path = src / "telemetry"
    if summary_path.is_dir():
        for s in summary_path.glob("*/summary.json"):
            (dst / "telemetry" / s.parent.name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, dst / "telemetry" / s.parent.name / "summary.json")
    resources = resource_summary(src)
    manifest_hash = None
    if (src / "manifest.json").is_file():
        raw = json.loads((src / "manifest.json").read_text())
        manifest_hash = raw.get("sha256") if isinstance(raw, dict) else None
    provenance = {
        "source_run": src.name,
        "manifest_sha256": manifest_hash,
        "copied": copied,
        "resources": resources,
        "excluded": [
            "work/",
            "checks/",
            "telemetry/*/samples.jsonl",
            "runtime/",
            "sessions/",
            "progress/",
            "logical-keys/",
            "operator*",
        ],
        "note": note,
    }
    (dst / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def copy_lineage(parent: Path, target: Path) -> None:
    """Plans, pauses, operator notes and journal totals of an earlier continuation directory."""
    target.mkdir(parents=True, exist_ok=True)
    for name in ("plan.json", "manifest.json", "parent-lineage.json"):
        if (parent / name).is_file():
            shutil.copy2(parent / name, target / name)
    if (parent / "pauses").is_dir():
        shutil.copytree(parent / "pauses", target / "pauses", dirs_exist_ok=True)
    if (parent / "operator").is_dir():
        for log in (parent / "operator").glob("*.log"):
            shutil.copy2(log, target / log.name)
        for md in (parent / "operator").glob("*.md"):
            shutil.copy2(md, target / md.name)
    (target / "RESOURCES.json").write_text(json.dumps(resource_summary(parent), indent=2) + "\n")


def split_key(key: str) -> tuple[str, str, str, str, str, str]:
    """Split an evaluation key into (arm, executor, tier, repetition, stage, panel).

    Local executor identifiers contain a slash (``qwen3.8:27b/off``) and hosted model
    identifiers contain one too (``z-ai/glm-5.3``), so fields are located by their
    position relative to the fixed markers, never by counting from the left.
    """
    parts = key.split("/")
    arm = parts[0]
    executor = tier = repetition = stage = panel = ""
    if arm in ("main", "gepa", "reference"):
        i = next(
            n
            for n in range(1, len(parts) - 1)
            if parts[n][:1] == "T" and parts[n][1:].isdigit() and parts[n + 1][:1] == "R"
        )
        executor = "/".join(parts[1:i])
        tier, repetition = parts[i][1:], parts[i + 1][1:]
        stage = "/".join(parts[i + 2 : -1])
    elif arm == "sealed":
        executor, panel = "/".join(parts[1:-3]), parts[-3]
    elif arm == "selection":
        executor, panel = "/".join(parts[1:-2]), "validation"
    elif arm == "ceiling":
        executor, panel = "/".join(parts[1:-2]), parts[-2]
    elif arm in ("calibration", "scaffolding"):
        executor, stage = "/".join(parts[1:-2]), parts[-2]
    return arm, executor, tier, repetition, stage, panel


def export_evaluations(runs: dict[str, Path], out: Path) -> int:
    fields = [
        "release_section",
        "key",
        "arm",
        "executor",
        "tier",
        "repetition",
        "stage",
        "panel",
        "condition_id",
        "prompt_sha256",
        "profile_id",
        "partition",
        "stratum",
        "status",
        "valid",
        "tp",
        "fp",
        "fn",
        "prediction",
        "expected",
        "categories",
    ]
    count = 0
    with (out / "evaluations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for section, root in runs.items():
            for path in sorted((root / "evaluations").glob("*.json")):
                record = payload(path)
                row = record["row"]
                arm, executor, tier, repetition, stage, panel = split_key(record["key"])
                writer.writerow(
                    {
                        "release_section": section,
                        "key": record["key"],
                        "arm": arm,
                        "executor": executor,
                        "tier": tier,
                        "repetition": repetition,
                        "stage": stage,
                        "panel": panel or row.get("partition", ""),
                        "condition_id": record.get("condition_id", ""),
                        "prompt_sha256": record.get("prompt_sha256", ""),
                        "profile_id": row["profile_id"],
                        "partition": row.get("partition", ""),
                        "stratum": row.get("stratum", ""),
                        "status": row["status"],
                        "valid": int(bool(row["valid"])),
                        "tp": row["tp"],
                        "fp": row["fp"],
                        "fn": row["fn"],
                        "prediction": json.dumps(row["prediction"]),
                        "expected": json.dumps(row["expected"]),
                        "categories": json.dumps(row.get("categories", [])),
                    }
                )
                count += 1
    return count


def export_trajectories(runs: dict[str, Path], out: Path) -> int:
    fields = [
        "release_section",
        "arm",
        "executor",
        "optimizer",
        "tier",
        "repetition",
        "prompt_changed",
        "hosted_seed_sent",
        "selected_prompt_sha256",
        "validation_f1",
        "validation_recall",
        "naive_validation_f1",
        "test_f1",
        "test_recall",
        "test_precision",
        "test_coverage",
        "test_valid_f1",
        "temporal_f1",
        "temporal_recall",
        "product_heldout_f1",
        "product_heldout_recall",
        "test_easy_f1",
        "test_medium_f1",
        "test_hard_f1",
        "rounds",
        "final_prompt",
    ]
    count = 0
    with (
        (out / "trajectories.csv").open("w", newline="") as handle,
        (out / "selected_prompts.jsonl").open("w") as prompts,
    ):
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for section, root in runs.items():
            for path in sorted((root / "trajectories").glob("*.json")):
                t = payload(path)
                sealed = t["sealed"]
                row: dict[str, Any] = {
                    "release_section": section,
                    "arm": t["arm"],
                    "executor": t["executor"],
                    "optimizer": t["optimizer"],
                    "tier": t["tier"],
                    "repetition": t["repetition"],
                    "prompt_changed": int(t["prompt_changed"]),
                    "hosted_seed_sent": t.get("hosted_seed_sent"),
                    "selected_prompt_sha256": t["selected"]["prompt_sha256"],
                    "validation_f1": t["selected"]["metrics"]["failure_aware_lower_bound"][
                        "micro_f1"
                    ],
                    "validation_recall": t["selected"]["metrics"]["failure_aware_lower_bound"][
                        "recall"
                    ],
                    "naive_validation_f1": t["shortlist"][0]["metrics"][
                        "failure_aware_lower_bound"
                    ]["micro_f1"],
                    "rounds": len(t.get("trace", [])),
                    "final_prompt": t.get("final_prompt", ""),
                }
                for panel in ("test", "temporal", "product_heldout"):
                    m = sealed[panel]["metrics"]["failure_aware_lower_bound"]
                    row[f"{panel}_f1"], row[f"{panel}_recall"] = m["micro_f1"], m["recall"]
                row["test_precision"] = sealed["test"]["metrics"]["failure_aware_lower_bound"][
                    "precision"
                ]
                row["test_coverage"] = sealed["test"]["metrics"]["coverage"]
                row["test_valid_f1"] = sealed["test"]["metrics"]["valid_answers"]["micro_f1"]
                for s in ("easy", "medium", "hard"):
                    row[f"test_{s}_f1"] = sealed["test"]["by_stratum"][s][
                        "failure_aware_lower_bound"
                    ]["micro_f1"]
                writer.writerow(row)
                prompts.write(
                    json.dumps(
                        {
                            "release_section": section,
                            "arm": t["arm"],
                            "executor": t["executor"],
                            "optimizer": t["optimizer"],
                            "tier": t["tier"],
                            "repetition": t["repetition"],
                            "selected_prompt": t["selected"]["prompt"],
                            "prompt_changed": t["prompt_changed"],
                        }
                    )
                    + "\n"
                )
                count += 1
    return count


def export_decisions(runs: dict[str, Path], out: Path) -> int:
    fields = [
        "release_section",
        "arm",
        "executor",
        "tier",
        "repetition",
        "round",
        "slot",
        "prompt_sha256",
        "batch_f1",
        "compared_prompt_hashes",
        "batch",
    ]
    count = 0
    with (out / "decisions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for section, root in runs.items():
            if not (root / "trajectories").is_dir():
                continue
            for tpath in sorted((root / "trajectories").glob("*.json")):
                t = payload(tpath)
                for d in t.get("trace", []):
                    writer.writerow(
                        {
                            "release_section": section,
                            "arm": t["arm"],
                            "executor": t["executor"],
                            "tier": t["tier"],
                            "repetition": t["repetition"],
                            "round": d["round"],
                            "slot": d["slot"],
                            "prompt_sha256": d["prompt_sha256"],
                            "batch_f1": d["metrics"]["failure_aware_lower_bound"]["micro_f1"],
                            "compared_prompt_hashes": json.dumps(d["compared_prompt_hashes"]),
                            "batch": json.dumps(d.get("batch", [])),
                        }
                    )
                    count += 1
    return count


def export_disclosures(runs: dict[str, Path], out: Path) -> int:
    count = 0
    with (out / "disclosures.jsonl").open("w") as handle:
        for section, root in runs.items():
            for path in sorted((root / "disclosures").glob("*.json")):
                d = payload(path)
                request = d["request"]
                permitted = json.loads(request["messages"][0]["content"])
                handle.write(
                    json.dumps(
                        {
                            "release_section": section,
                            "key": d["key"],
                            "tier": d["tier"],
                            "hosted_seed_sent": d.get("hosted_seed_sent"),
                            "model": request["model"],
                            "optimizer_view": permitted,
                        }
                    )
                    + "\n"
                )
                count += 1
    return count


def export_panels(runs: dict[str, Path], out: Path) -> int:
    fields = [
        "release_section",
        "kind",
        "prefix",
        "model_or_executor",
        "panel",
        "prompt_sha256",
        "f1",
        "recall",
        "precision",
        "coverage",
        "valid_f1",
        "easy_f1",
        "medium_f1",
        "hard_f1",
        "temperature_supported",
    ]
    count = 0
    with (out / "panels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for section, root in runs.items():
            for kind, folder in (("sealed", "sealed-panels"), ("ceiling", "panels")):
                for path in sorted((root / folder).glob("*.json")):
                    s = payload(path)
                    m = s["metrics"]["failure_aware_lower_bound"]
                    writer.writerow(
                        {
                            "release_section": section,
                            "kind": kind,
                            "prefix": s["prefix"],
                            "model_or_executor": s.get("model") or s["prefix"].split("/")[1],
                            "panel": s["panel"],
                            "prompt_sha256": s.get("prompt_sha256", ""),
                            "f1": m["micro_f1"],
                            "recall": m["recall"],
                            "precision": m["precision"],
                            "coverage": s["metrics"]["coverage"],
                            "valid_f1": s["metrics"]["valid_answers"]["micro_f1"],
                            "easy_f1": s["by_stratum"]["easy"]["failure_aware_lower_bound"][
                                "micro_f1"
                            ],
                            "medium_f1": s["by_stratum"]["medium"]["failure_aware_lower_bound"][
                                "micro_f1"
                            ],
                            "hard_f1": s["by_stratum"]["hard"]["failure_aware_lower_bound"][
                                "micro_f1"
                            ],
                            "temperature_supported": s.get("temperature_supported", ""),
                        }
                    )
                    count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--feeds", type=Path, required=True)
    parser.add_argument("--cna", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, action="append", default=[])
    parser.add_argument("--followup", action="append", default=[], help="name=<run dir>")
    parser.add_argument(
        "--extension",
        type=Path,
        help="prospectively planned repetition-extension run, released as campaign-extension",
    )
    parser.add_argument(
        "--extension-lineage",
        type=Path,
        action="append",
        default=[],
        help="earlier directory of the extension's continuation chain (plans, pauses, notes)",
    )
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        help="name=<run dir>: earlier calibration or pilot whose summary report is kept for the calibration history",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {"sections": {}}

    # 1. Benchmark: complete directory (labels, audits, data card), it is the dataset.
    shutil.copytree(args.benchmark, out / "benchmark", dirs_exist_ok=True)
    index["sections"]["benchmark"] = {"source_run": args.benchmark.name}

    # 2. Sources: recipe and outcomes, no raw snapshots.
    sources = out / "sources"
    sources.mkdir(exist_ok=True)
    for name in (
        "outcomes.jsonl",
        "plan.json",
        "manifest.json",
        "partitions.json",
        "source-review.json",
        "SOURCE_REVIEW.md",
        "admission-closeout.json",
        "textual-scope-and-vendor-audit.json",
        "retained-review-selection.json",
    ):
        if (args.admission / name).is_file():
            shutil.copy2(args.admission / name, sources / ("admission-" + name))
    if (args.candidates / "REPRODUCE.md").is_file():
        shutil.copy2(args.candidates / "REPRODUCE.md", sources / "candidates-REPRODUCE.md")
    for year in sorted(p.name for p in args.feeds.iterdir() if p.is_dir()):
        for name in ("metadata.json", "receipt.json"):
            if (args.feeds / year / name).is_file():
                shutil.copy2(args.feeds / year / name, sources / f"nvd-{year}-{name}")
    for name in ("repository-revision.json", "acquisition-plan.json", "acquisition-report.json"):
        if (args.cna / name).is_file():
            shutil.copy2(args.cna / name, sources / ("cna-" + name))
    index["sections"]["sources"] = {
        "admission": args.admission.name,
        "candidates": args.candidates.name,
        "cna": args.cna.name,
    }

    # 3. Calibration, 4. campaign, 5. follow-ups: scientific records only.
    runs: dict[str, Path] = {}
    index["sections"]["calibration"] = copy_records(
        args.calibration,
        out / "calibration",
        "calibration gate screen on pilot-development profiles",
    )
    runs["calibration"] = out / "calibration"
    index["sections"]["campaign"] = copy_records(
        args.campaign,
        out / "campaign",
        "final continuation holding the complete adopted journal of the Phase 3 campaign",
    )
    runs["campaign"] = out / "campaign"
    for parent in args.lineage:
        copy_lineage(parent, out / "campaign" / "lineage" / parent.name)
    if args.extension:
        index["sections"]["campaign-extension"] = copy_records(
            args.extension,
            out / "campaign-extension",
            "prospectively planned extension: further paired repetitions with the same schedule "
            "generator, analysed beside the original campaign",
        )
        runs["campaign-extension"] = out / "campaign-extension"
        for parent in args.extension_lineage:
            copy_lineage(parent, out / "campaign-extension" / "lineage" / parent.name)
    for item in args.followup:
        name, path = item.split("=", 1)
        index["sections"][f"followups/{name}"] = copy_records(
            Path(path), out / "followups" / name, f"follow-up run: {name}"
        )
        runs[f"followups/{name}"] = out / "followups" / name

    # 5b. Calibration history: summary reports of earlier benchmark versions only.
    for item in args.history:
        name, path = item.split("=", 1)
        src = Path(path)
        target = out / "calibration" / "history" / name
        target.mkdir(parents=True, exist_ok=True)
        for rel in ("reports/complete.json", "manifest.json", "plan.json", "stages/P1/report.json"):
            if (src / rel).is_file():
                (target / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / rel, target / rel)
        index["sections"][f"calibration/history/{name}"] = {"source_run": src.name}

    # 6. Flat exports.
    exports = out / "exports"
    exports.mkdir(exist_ok=True)
    index["exports"] = {
        "evaluations.csv": export_evaluations(runs, exports),
        "trajectories.csv": export_trajectories(runs, exports),
        "decisions.csv": export_decisions(runs, exports),
        "disclosures.jsonl": export_disclosures(runs, exports),
        "panels.csv": export_panels(runs, exports),
    }
    (out / "INDEX.json").write_text(json.dumps(index, indent=2) + "\n")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(json.dumps({"out": str(out), "bytes": total, "exports": index["exports"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
