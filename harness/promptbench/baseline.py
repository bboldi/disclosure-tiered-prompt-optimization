"""Input-matched deterministic baseline: name normalisation plus version-range parsing.

The baseline sees exactly what the Executor sees, the rendered `system_profile` text and each
advisory's `affected` string, and nothing else. It never reads canonical component records,
labels, or presentation metadata. Rules were fixed on the optimization partition and are then
evaluated once per sealed panel; a development report records parse coverage there.

Usage:
  python -m promptbench.baseline --benchmark-dir <benchmark> --run-dir <new dir>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .benchmark.build import load
from .benchmark.model import BenchmarkProfile, score
from .benchmark.oracle import Unsupported, numeric
from .domain import metrics
from .runner import utc_now
from .storage import Store, digest

BASELINE_ID = "rule-baseline"
PANELS = ("test", "temporal", "product_heldout")
_LINE = re.compile(r"^- (?P<name>.+?)\s+v?(?P<version>\d[\w.\-]*)$")
_REPO = re.compile(r"^(?P<name>[^@\s]+)@v?(?P<version>\d[\w.\-]*)$")
_CLAUSE = re.compile(r"^(?P<vp>[^:]+):\s*version\s+(?P<expr>.+)$")
_PROSE = re.compile(r"^(?P<vp>[^:]+):\s*versions prior to\s+v?(?P<bound>\S+)\s+are affected$")
_COND = re.compile(r"(?P<op>>=|<=|=|<|>)\s*v?(?P<value>[\w.\-]+)")


def normalize(name: str) -> str:
    """Collapse the inventory rendering styles onto one comparable token."""
    return re.sub(r"[\s\-_/@.,]+", "", name.casefold())


def parse_inventory(text: str) -> list[dict[str, Any]]:
    components = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        match = _REPO.match(line[2:].strip()) or _LINE.match(line)
        if not match:
            components.append({"raw": line, "name": None, "version": None})
            continue
        components.append(
            {"raw": line, "name": normalize(match["name"]), "version": match["version"]}
        )
    return components


def parse_affected(affected: str) -> list[dict[str, Any]] | None:
    """Return one record per OR-term: normalised vendor+product, product, and conditions."""
    terms = []
    for part in re.split(r"\s+OR\s+", affected.strip()):
        prose = _PROSE.match(part)
        if prose:
            vp, conditions = prose["vp"], [("<", prose["bound"])]
        else:
            clause = _CLAUSE.match(part)
            if not clause:
                return None
            vp = clause["vp"]
            conditions = [(m["op"], m["value"]) for m in _COND.finditer(clause["expr"])]
            if not conditions:
                return None
        vendor, _, product = vp.strip().partition("/")
        terms.append(
            {
                "full": normalize(vendor + product),
                "product": normalize(product or vendor),
                "conditions": conditions,
            }
        )
    return terms


def version_satisfies(installed: str, conditions: list[tuple[str, str]]) -> bool | None:
    try:
        v = numeric(installed)
        bounds = [(op, numeric(value)) for op, value in conditions]
    except Unsupported:
        return None
    for op, b in bounds:
        ok = {"=": v == b, "<": v < b, "<=": v <= b, ">": v > b, ">=": v >= b}[op]
        if not ok:
            return False
    return True


def decide(profile: BenchmarkProfile) -> tuple[list[str], dict[str, int]]:
    """Applicable advisory IDs from the rendered input only, plus parse-coverage counters."""
    view = profile.executor_input()
    components = parse_inventory(view["system_profile"])
    counters = {
        "components": len(components),
        "components_unparsed": sum(1 for c in components if c["name"] is None),
        "advisories": len(view["advisories"]),
        "advisories_unparsed": 0,
        "version_unsupported": 0,
    }
    predicted = []
    for advisory in view["advisories"]:
        terms = parse_affected(advisory["affected"])
        if terms is None:
            counters["advisories_unparsed"] += 1
            continue
        hit = False
        for term in terms:
            for component in components:
                if component["name"] is None:
                    continue
                if component["name"] not in (term["full"], term["product"]):
                    continue
                verdict = version_satisfies(component["version"], term["conditions"])
                if verdict is None:
                    counters["version_unsupported"] += 1
                elif verdict:
                    hit = True
        if hit:
            predicted.append(advisory["id"])
    return predicted, counters


def evaluate_panel(store: Store, profiles: list[BenchmarkProfile], panel: str) -> dict[str, Any]:
    rows, coverage = (
        [],
        {
            "components": 0,
            "components_unparsed": 0,
            "advisories": 0,
            "advisories_unparsed": 0,
            "version_unsupported": 0,
        },
    )
    prefix = f"ceiling/{BASELINE_ID}/{panel}"
    for profile in sorted(profiles, key=lambda p: p.id):
        predicted, counters = decide(profile)
        for key in coverage:
            coverage[key] += counters[key]
        row = score(profile, predicted, "valid")
        store.put(
            "evaluations/" + digest(f"{prefix}/{profile.id}") + ".json",
            {
                "key": f"{prefix}/{profile.id}",
                "condition_id": BASELINE_ID,
                "prompt_sha256": "",
                "row": row,
            },
        )
        rows.append(row)
    summary = {
        "prefix": prefix,
        "model": BASELINE_ID,
        "panel": panel,
        "temperature_supported": "deterministic",
        "metrics": metrics(rows),
        "by_stratum": {
            s: metrics([r for r in rows if r["stratum"] == s]) for s in ("easy", "medium", "hard")
        },
        "parse_coverage": coverage,
    }
    store.put("panels/" + digest(prefix) + ".json", summary)
    return summary


def run(benchmark: Path, root: Path) -> dict[str, Any]:
    store = Store(root)
    development = load(benchmark, {"optimization"})
    dev_rows, dev_cov = (
        [],
        {
            "components": 0,
            "components_unparsed": 0,
            "advisories": 0,
            "advisories_unparsed": 0,
            "version_unsupported": 0,
        },
    )
    for profile in development:
        predicted, counters = decide(profile)
        for key in dev_cov:
            dev_cov[key] += counters[key]
        dev_rows.append(score(profile, predicted, "valid"))
    store.put(
        "development.json",
        {
            "partition": "optimization",
            "profiles": len(development),
            "metrics": metrics(dev_rows),
            "parse_coverage": dev_cov,
            "note": "Rules were fixed by inspection of parse coverage on this partition only.",
        },
    )
    panels = {}
    for panel in PANELS:
        profiles = load(benchmark, {panel})
        panels[panel] = evaluate_panel(store, profiles, panel)
    manifest = {
        "kind": "rule_baseline",
        "benchmark_sha256": digest(Store(benchmark).get("manifest.json")),
        "rules": {
            "normalisation": "casefold; drop whitespace, '-', '_', '/', '@', '.', ','",
            "match": "normalised inventory name equals normalised vendor+product or product",
            "version": "dotted numeric compare with trailing-zero equivalence; unsupported strings never match",
            "prose": "'versions prior to X are affected' read as '< X'",
        },
        "timestamp_utc": utc_now(),
    }
    store.put("manifest.json", manifest)
    report = {
        "status": "completed",
        "baseline": BASELINE_ID,
        "panels": panels,
        "development": store.get("development.json"),
    }
    store.put("reports/complete.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.benchmark_dir.resolve(), args.run_dir.resolve())
    print(
        json.dumps(
            {
                panel: {
                    "f1": round(s["metrics"]["failure_aware_lower_bound"]["micro_f1"], 4),
                    "recall": round(s["metrics"]["failure_aware_lower_bound"]["recall"], 4),
                    "precision": round(s["metrics"]["failure_aware_lower_bound"]["precision"], 4),
                    "parse_coverage": s["parse_coverage"],
                }
                for panel, s in report["panels"].items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
