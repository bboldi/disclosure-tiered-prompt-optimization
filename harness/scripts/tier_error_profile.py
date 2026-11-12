#!/usr/bin/env python3
"""Where the Granite tiers lose on the sealed test panel (revision, R2.4; descriptive).

Groups: all, the generator's bound-type tag of the component, structured or prose bound,
and "inclusive bound, equal" (an installed component exactly on an inclusive bound of the
advisory's own terms, exact affected versions included, found from the terms, not the tag).

For every main-arm selected prompt of the campaign and the extension, over valid answers:
the share of applicable advisories missed, by the benchmark's bound type and by structured
or prose bound, and the share of non-applicable advisories wrongly included. Invalid answers
are counted separately (the failure-aware metric charges them as worst cases).

    python scripts/tier_error_profile.py --data ../data --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

EXECUTOR = "granite4.2:30b/off"


def numeric(version: str) -> tuple[int, ...]:
    """Dotted version with trailing zeros dropped, as the label matcher compares them."""
    parts = [int(x) for x in version.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def on_inclusive_bound(profile: dict, terms: list[dict]) -> bool:
    """An installed component of the advisory's product sits exactly on an inclusive bound; an
    exact affected version is a range whose two bounds are both inclusive."""
    for c in profile["components"]:
        for term in terms:
            if (term["vendor"], term["product"]) != (c["vendor"], c["product"]):
                continue
            if term["exact"] and numeric(term["exact"]) == numeric(c["version"]):
                return True
            for side in ("lower", "upper"):
                if (
                    term[f"{side}_inclusive"]
                    and term[side]
                    and numeric(term[side]) == numeric(c["version"])
                ):
                    return True
    return False


SECTIONS = ("campaign", "campaign-extension")


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("--data", "--out"):
        ap.add_argument(name, type=Path, required=True)
    args = ap.parse_args()
    profiles = {}
    for line in (args.data / "benchmark" / "profiles.jsonl").open():
        p = json.loads(line)
        if p["partition"] == "test":
            profiles[p["id"]] = p
    # (section, selected prompt) -> [(tier, repetition)]; identical prompts share one evaluation
    terms = {
        a["id"]: a["terms"]
        for a in map(json.loads, (args.data / "benchmark" / "advisories.jsonl").open())
    }
    runs = defaultdict(list)
    for r in csv.DictReader((args.data / "exports" / "trajectories.csv").open(newline="")):
        if r["arm"] == "main" and r["executor"] == EXECUTOR and r["release_section"] in SECTIONS:
            runs[(r["release_section"], r["selected_prompt_sha256"])].append(
                (int(r["tier"]), int(r["repetition"]))
            )
    missed, positives, included, negatives, invalid, answers = (Counter() for _ in range(6))
    seen = set()
    for r in csv.DictReader((args.data / "exports" / "evaluations.csv").open(newline="")):
        if not (r["arm"] == "sealed" and r["partition"] == "test" and r["executor"] == EXECUTOR):
            continue
        key = (r["release_section"], r["prompt_sha256"], r["profile_id"])
        if (key[0], key[1]) not in runs or key in seen:
            continue
        seen.add(key)
        p = profiles[r["profile_id"]]
        kind = {c["source_cve"]: c["type"] for c in p["presentation"]["components"]}
        prose = set(p["presentation"]["prose_advisories"])
        for tier, _ in runs[(r["release_section"], r["prompt_sha256"])]:
            answers[tier] += 1
            if r["valid"] != "1":
                invalid[tier] += 1
                continue
            pred = set(json.loads(r["prediction"] or "[]"))
            for cve in p["expected"]:
                keys = [
                    (tier, "all"),
                    (tier, kind.get(cve, "other")),
                    (tier, "prose" if cve in prose else "structured"),
                ]
                if on_inclusive_bound(p, terms[cve]):
                    keys.append((tier, "inclusive bound, equal"))
                for key in keys:
                    positives[key] += 1
                    missed[key] += cve not in pred
            for cve in set(p["advisories"]) - set(p["expected"]):
                negatives[tier] += 1
                included[tier] += cve in pred
    assert sum(answers.values()) == 192 * sum(len(v) for v in runs.values())
    groups = sorted({g for _, g in positives}, key=lambda g: (g != "all", g))
    out = {
        "executor": EXECUTOR,
        "prompts_per_tier": {
            t: sum(x[0] == t for v in runs.values() for x in v) for t in (1, 2, 3)
        },
        "invalid_share": {t: invalid[t] / answers[t] for t in (1, 2, 3)},
        "missed_share": {
            g: {t: missed[(t, g)] / positives[(t, g)] for t in (1, 2, 3) if positives[(t, g)]}
            for g in groups
        },
        "positives": {g: {t: positives[(t, g)] for t in (1, 2, 3)} for g in groups},
        "wrongly_included_share": {t: included[t] / negatives[t] for t in (1, 2, 3)},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tier_error_profile.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = [
        "| Group | Tier 1 missed | Tier 2 missed | Tier 3 missed | Positives (T1/T2/T3) |",
        "|---|---|---|---|---|",
    ]
    for g in groups:
        m = out["missed_share"][g]
        lines.append(
            f"| {g} | "
            + " | ".join(f"{m.get(t, float('nan')):.2f}" for t in (1, 2, 3))
            + " | "
            + "/".join(str(positives[(t, g)]) for t in (1, 2, 3))
            + " |"
        )
    lines += [
        "",
        "| | Tier 1 | Tier 2 | Tier 3 |",
        "|---|---|---|---|",
        "| wrongly included | "
        + " | ".join(f"{out['wrongly_included_share'][t]:.3f}" for t in (1, 2, 3))
        + " |",
        "| invalid answers | "
        + " | ".join(f"{out['invalid_share'][t]:.3f}" for t in (1, 2, 3))
        + " |",
    ]
    (args.out / "tier_error_profile.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
