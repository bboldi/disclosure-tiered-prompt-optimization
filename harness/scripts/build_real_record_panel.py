#!/usr/bin/env python3
"""Real-Record Panel (round 1 revision): sub-profiles of the early-signals System Profiles,
real CVE records as published, labels from that study's deterministic matcher.

No model is called. The panel is written in the benchmark shape the harness reads
(profiles.jsonl, advisories.jsonl), with the CVE 5 `affected` entries rendered as text in
`presentation.prose_advisories` (the Executor sees them as the advisory's `affected`
field) and the one undecided cell per sub-profile in `presentation.undecided_cve`, never
in `expected`.

The early-signals code is imported, not copied, so the matcher is exactly theirs; run with
that project's interpreter:

    <early-signals repository>/app/.venv/bin/python \
        scripts/build_real_record_panel.py --early-signals <early-signals repository> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sqlite3
import subprocess
import sys
from pathlib import Path

SEED = 20261029
TOTAL, BASE_PER_PROFILE, MAX_PER_PROFILE = 48, 5, 8  # sub-profiles
COMPOSITION = {"affected": 3, "near_miss": 3, "clean": 1, "fictional": 1}  # 8 components
CANDIDATES = 23
POSITIVES = (3, 7)
PARTITION = "real_record"
INGESTION_MAX_ID = 9406102  # raw_item snapshot at build time; later collections are ignored
MAX_CHARS = 3000  # description + rendered affected entries per advisory
MAX_SUBPROFILE_CHARS = 32000  # all advisories of one sub-profile (about 9k tokens)


def combine(values):
    """Cell label over components: applicable if any says so, undecided if none does and one
    cannot be decided, otherwise not applicable."""
    if True in values:
        return True
    return None if None in values else False


def render_affected(doc) -> str:
    """The record's CNA and ADP `affected` entries, all fields with verbatim values, one line
    per entry."""
    lines = []
    containers = doc["containers"]
    for origin, container in [("", containers.get("cna"))] + [
        ("(ADP enrichment) ", a) for a in containers.get("adp") or []
    ]:
        for e in (container or {}).get("affected") or []:
            parts = [f"default status: {e['defaultStatus']}"] if e.get("defaultStatus") else []
            for v in e.get("versions") or []:
                s = f"{v.get('status')}: {v.get('version')}"
                if "lessThan" in v:
                    s += f" up to, not including, {v['lessThan']}"
                if "lessThanOrEqual" in v:
                    s += f" up to and including {v['lessThanOrEqual']}"
                if v.get("versionType"):
                    s += f" ({v['versionType']})"
                if v.get("changes"):
                    s += "; changes: " + ", ".join(
                        f"{c.get('status')} at {c.get('at')}" for c in v["changes"]
                    )
                parts.append(s)
            # every other field as recorded (platforms, modules, packageName, cpes, ...)
            parts += [
                f"{k}: {', '.join(map(str, v)) if isinstance(v, list) else v}"
                for k, v in sorted(e.items())
                if k not in ("vendor", "product", "defaultStatus", "versions")
                and v not in (None, "", [])
            ]
            head = f"{origin}{e.get('vendor') or 'n/a'} / {e.get('product') or 'n/a'}"
            lines.append(head + ": " + ("; ".join(parts) or "no version data"))
    return "\n".join(lines)


def description(doc) -> str:
    texts = (doc["containers"].get("cna") or {}).get("descriptions") or []
    english = [d["value"] for d in texts if str(d.get("lang", "")).lower().startswith("en")]
    return (english or [d["value"] for d in texts] or [""])[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--early-signals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.early_signals.resolve()
    app = root / "app"
    sys.path.insert(0, str(app / "src"))
    from earlysignals.derive import _products
    from earlysignals.profiles import NameIndex
    from earlysignals.versions import affects

    truth = sqlite3.connect(f"file:{app / 'data/truth.sqlite'}?mode=ro", uri=True)
    ingestion = sqlite3.connect(f"file:{app / 'data/ingestion.sqlite'}?mode=ro", uri=True)
    names = NameIndex(truth)

    def related_rows(component, rows):
        """(rows of `rows` about a spelling related to this component's product)."""
        keys = {
            (v.lower() if v else None, p.lower())
            for v, p in names.related(component["vendor"], component["product"])
        }
        return [r for r in rows if ((r[0] or "").lower() or None, (r[1] or "").lower()) in keys]

    decisions: dict[tuple, dict[str, bool | None]] = {}

    def decide(c) -> dict[str, bool | None]:
        """Matcher verdict per CVE for one component, over every related record row."""
        key = (c["vendor"], c["product"], c["version"])
        if key not in decisions:
            verdicts: dict[str, list] = {}
            for v, p in names.related(c["vendor"], c["product"]):
                for cve, status, versions in truth.execute(
                    "SELECT cve_id, default_status, versions_json FROM cve_product"
                    " WHERE lower(vendor) IS ? AND lower(product) = ?",
                    (v.lower() if v else None, p.lower()),
                ):
                    verdicts.setdefault(cve, []).append(
                        affects(json.loads(versions) if versions else None, status, c["version"])
                    )
            decisions[key] = {cve: combine(vs) for cve, vs in verdicts.items()}
        return decisions[key]

    profiles = [
        json.loads(f.read_text()) for f in sorted((app / "profiles").glob("[0-9][0-9]-*.json"))
    ]
    assert len(profiles) == 10
    records: dict[str, tuple | None] = {}

    def record(cve):
        """The stored record version whose affected entries are exactly the label rows."""
        if cve not in records:
            rows = truth.execute(
                "SELECT vendor, product, default_status, versions_json FROM cve_product"
                " WHERE cve_id = ? ORDER BY position",
                (cve,),
            ).fetchall()
            records[cve] = None
            for (content,) in ingestion.execute(
                "SELECT content FROM raw_item WHERE source_id = 'cvelist' AND external_id = ?"
                " AND id <= ? ORDER BY id DESC",
                (cve, INGESTION_MAX_ID),
            ):
                doc = json.loads(content)
                if [tuple(p[1:]) for p in _products(doc)] == [tuple(r) for r in rows]:
                    records[cve] = (doc, content, rows)
                    break
        return records[cve]

    def size(cve):
        doc = record(cve)[0]
        return len(description(doc)) + len(render_affected(doc))

    def fits(cve):
        """Shown in full and short enough for the Executors' context (decided before any run)."""
        r = record(cve)
        return r is not None and len(description(r[0])) + len(render_affected(r[0])) <= MAX_CHARS

    def build(counts):
        used_undecided: set[str] = set()
        subprofiles, made_by_index = [], {}
        for i in order:
            profile, count = profiles[i], counts[i]
            rng = random.Random(f"{SEED}:{profile['id']}")
            by_role = {
                role: [c for c in profile["components"] if c["role"] == role]
                for role in COMPOSITION
            }
            for c in by_role[
                "affected"
            ]:  # the study's ground truth is a subset of the matcher's yes
                assert all(decide(c).get(cve) is True for cve in c["affecting_cves"]), c
            seen: set[frozenset] = set()
            made = 0
            for _ in range(5000):
                if made == count:
                    break
                picked = [
                    c for role, n in COMPOSITION.items() for c in rng.sample(by_role[role], n)
                ]
                ident = frozenset((c["vendor"], c["product"], c["version"]) for c in picked)
                if ident in seen:
                    continue
                cves = set().union(*(decide(c) for c in picked))
                cell = {
                    cve: combine([decide(c)[cve] for c in picked if cve in decide(c)])
                    for cve in cves
                }
                truth_cves = sorted(
                    {cve for c in picked for cve in c["affecting_cves"] if fits(cve)}
                )
                if len(truth_cves) < POSITIVES[0]:
                    continue
                positives = sorted(rng.sample(truth_cves, min(len(truth_cves), POSITIVES[1])))
                undecided = sorted(
                    cve
                    for cve, x in cell.items()
                    if x is None and cve not in used_undecided and fits(cve)
                )
                negatives = sorted(cve for cve, x in cell.items() if x is False and fits(cve))
                need = CANDIDATES - len(positives) - 1
                near = sorted(rng.sample(negatives, min(need, len(negatives))))
                # Decided near-misses are too few in several System Profiles, so the rest
                # are real records about other components of the same System Profile (off-inventory)
                others = [c for c in profile["components"] if c not in picked]
                off_pool = sorted(
                    c for c in set().union(*(decide(o) for o in others)) - cves if fits(c)
                )
                if not undecided or len(off_pool) < need - len(near):
                    continue
                u = rng.choice(undecided)
                off = sorted(rng.sample(off_pool, need - len(near)))
                shown = sum(size(c) for c in positives + near + off + [u])
                if shown > MAX_SUBPROFILE_CHARS:  # must fit the Executors' 16,384-token context
                    continue
                seen.add(ident)
                used_undecided.add(u)
                made += 1
                subprofiles.append(
                    {
                        "id": f"real_{profile['id'][:2]}_{made}",
                        "system_profile": profile["id"],
                        "components": rng.sample(picked, len(picked)),
                        "positives": positives,
                        "near_misses": near,
                        "off_inventory": off,
                        "undecided": u,
                        "order": rng.sample(positives + near + off + [u], CANDIDATES),
                    }
                )
            made_by_index[i] = made
        made = [made_by_index[i] for i in range(len(profiles))]
        return sorted(subprofiles, key=lambda s: s["id"]), made

    # undecided records must be distinct across the panel, so the System Profiles with the
    # fewest eligible undecided records draw first; each profile's capacity under the caps then
    # sets the counts, the rest of the 48 going to profiles with spare capacity
    scarcity = [
        len(
            {
                cve
                for c in p["components"]
                for cve, x in decide(c).items()
                if x is None and fits(cve)
            }
        )
        for p in profiles
    ]
    order = sorted(range(len(profiles)), key=lambda i: (scarcity[i], i))
    _, feasible = build([MAX_PER_PROFILE] * len(profiles))
    counts = [min(f, BASE_PER_PROFILE) for f in feasible]
    while sum(counts) < TOTAL:
        spare = [i for i, f in enumerate(feasible) if counts[i] < f]
        assert spare, "the System Profiles cannot supply the panel"
        counts[min(spare, key=lambda i: counts[i])] += 1
    subprofiles, made = build(counts)
    assert made == counts, (made, counts)

    # self-check: every label reproduces from affects() on the record the Executor is shown
    def verdict(cve, components):
        _, _, rows = records[cve]
        return combine(
            [
                affects(json.loads(r[3]) if r[3] else None, r[2], c["version"])
                for c in components
                for r in related_rows(c, rows)
            ]
            or [False]
        )

    for s in subprofiles:
        assert all(verdict(c, s["components"]) is True for c in s["positives"]), s["id"]
        assert all(verdict(c, s["components"]) is False for c in s["near_misses"]), s["id"]
        assert all(
            not any(related_rows(c, records[cve][2]) for c in s["components"])
            for cve in s["off_inventory"]
        ), s["id"]
        assert verdict(s["undecided"], s["components"]) is None, s["id"]
        assert s["undecided"] not in s["positives"] + s["near_misses"] + s["off_inventory"]
        assert len(set(s["order"])) == CANDIDATES
    assert len({s["undecided"] for s in subprofiles}) == len(subprofiles)

    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    with (out / "advisories.jsonl").open("w") as f:
        for cve in sorted({a for s in subprofiles for a in s["order"]}):
            doc, content, _ = records[cve]
            meta = doc.get("cveMetadata", {})
            row = {
                "description": description(doc),
                "id": cve,
                "last_modified": meta.get("dateUpdated") or "",
                "published": meta.get("datePublished") or "",
                "source_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "terms": [],
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (out / "profiles.jsonl").open("w") as f:
        for s in subprofiles:
            row = {
                "advisories": s["order"],
                "anchor_cve": s["positives"][0],
                "components": [
                    {
                        "part": "a",
                        "product": c["product"],
                        "vendor": c["vendor"],
                        "version": c["version"],
                    }
                    for c in s["components"]
                ],
                "expected": s["positives"],
                "id": s["id"],
                "partition": PARTITION,
                "presentation": {
                    "prose_advisories": {
                        cve: {"text": render_affected(records[cve][0])} for cve in s["order"]
                    },
                    "system_profile": s["system_profile"],
                    "undecided_cve": s["undecided"],
                },
                "stratum": "real_record",
                "text": "System inventory:\n"
                + "\n".join(
                    f"- {c['vendor']} {c['product']} {c['version']}" for c in s["components"]
                ),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (out / "cells.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subprofile_id", "cve_id", "stratum", "kind", "matcher_label"])
        for s in subprofiles:
            kinds = {c: k for k in ("positives", "near_misses", "off_inventory") for c in s[k]}
            for cve in s["order"]:
                if cve == s["undecided"]:
                    w.writerow([s["id"], cve, "hand_labelled", "undecided", ""])
                else:
                    label = int(cve in s["positives"])
                    w.writerow([s["id"], cve, "matcher_decided", kinds[cve], label])
    with (out / "labelling_sheet.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "subprofile_id",
                "cve_id",
                "installed_component",
                "description",
                "affected",
                "label",
                "note",
            ]
        )
        for s in subprofiles:
            cve = s["undecided"]
            undecided_for = [c for c in s["components"] if verdict(cve, [c]) is None]
            w.writerow(
                [
                    s["id"],
                    cve,
                    " | ".join(
                        f"{c['vendor']} {c['product']} {c['version']}" for c in undecided_for
                    ),
                    description(records[cve][0]),
                    render_affected(records[cve][0]),
                    "",
                    "",
                ]
            )

    def git(*a):
        return subprocess.run(
            ["git", "-C", str(root), *a], capture_output=True, text=True
        ).stdout.strip()

    manifest = {
        "adr": "docs/adr/0043-round1-revision-experiments.md (G3.4)",
        "seed": SEED,
        "composition": COMPOSITION,
        "subprofiles_per_system_profile": dict(
            zip([p["id"] for p in profiles], counts, strict=True)
        ),
        "feasible_per_system_profile": dict(
            zip([p["id"] for p in profiles], feasible, strict=True)
        ),
        "eligible_undecided_per_system_profile": dict(
            zip([p["id"] for p in profiles], scarcity, strict=True)
        ),
        "draw_order": [profiles[i]["id"] for i in order],
        "candidates_per_subprofile": CANDIDATES,
        "counts": {
            "subprofiles": len(subprofiles),
            "candidate_cells": sum(len(s["order"]) for s in subprofiles),
            "positives": sum(len(s["positives"]) for s in subprofiles),
            "near_misses": sum(len(s["near_misses"]) for s in subprofiles),
            "off_inventory": sum(len(s["off_inventory"]) for s in subprofiles),
            "near_misses_per_subprofile": sorted(len(s["near_misses"]) for s in subprofiles),
            "undecided": len(subprofiles),
            "distinct_advisories": len({a for s in subprofiles for a in s["order"]}),
            "max_chars_per_advisory": MAX_CHARS,
            "max_chars_per_subprofile": MAX_SUBPROFILE_CHARS,
            "positives_per_subprofile": sorted({len(s["positives"]) for s in subprofiles}),
        },
        "early_signals": {
            "commit": git("rev-parse", "HEAD"),
            "uncommitted_src_changes": git("status", "--porcelain", "app/src"),
            "truth_sqlite_sha256": sha256_file(app / "data/truth.sqlite"),
            "ingestion_raw_item_max_id": INGESTION_MAX_ID,
            "profiles_sha256": {
                f.name: sha256_file(f) for f in sorted((app / "profiles").glob("[0-9][0-9]-*.json"))
            },
            "matcher": "app/src/earlysignals/versions.py affects()",
            "names": "app/src/earlysignals/profiles.py NameIndex.related()",
            "records": "app/data/ingestion.sqlite raw_item (cvelist), the version whose affected entries equal truth.sqlite cve_product",
        },
        "files": {
            f.name: sha256_file(f) for f in sorted(out.iterdir()) if f.name != "manifest.json"
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
