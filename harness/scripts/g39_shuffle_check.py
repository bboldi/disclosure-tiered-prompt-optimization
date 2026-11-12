#!/usr/bin/env python3
"""Shuffled Abstraction manipulation check: how much did the shuffle change the Tier 2 records?

Each stored shuffled request is un-shuffled with its seed (the inverse of
conditions.shuffle_abstractions). The seed derivation is checked on round 0, where the
plain Tier 2 arm received the same unshuffled records. Reported over requests with two or
more records: records whose field tuple changed, and records whose shuffled tuple occurs
nowhere in the original request.

    python scripts/g39_shuffle_check.py --run ../data/followups/shuffled-abstraction \
        --plain ../data/followups/shuffled-abstraction-plain-tier2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from promptbench.live.conditions import shuffle_abstractions  # noqa: E402
from promptbench.storage import digest  # noqa: E402

# key order of relations.error_relation, which sets the order the shuffle consumes the rng
FIELDS = [
    "category",
    "product_installed",
    "version_position",
    "bound_rendering",
    "version_segments",
    "alias_style",
    "advisory_count",
]


def unshuffle(records: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    n = len(records)
    original = [{} for _ in range(n)]
    for k in FIELDS:
        perm = list(range(n))
        rng.shuffle(perm)  # shuffled[i] = original[perm[i]]
        for i, j in enumerate(perm):
            original[j][k] = records[i][k]
    return original


def requests(run: Path) -> dict[str, dict]:
    out = {}
    for path in sorted((run / "disclosures").glob("*.json")):
        payload = json.loads(path.read_text())["payload"]
        out[payload["key"]] = json.loads(payload["request"]["messages"][-1]["content"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--plain", type=Path, required=True)
    args = ap.parse_args()
    plain = requests(args.plain)
    checked = changed = novel = total = count = 0
    for key, content in requests(args.run).items():
        assert content["feedback_version"].endswith("-shuffled"), key
        shuffled = content["feedback"]["abstractions"]
        seed = int(digest(key)[:16], 16)
        original = unshuffle(shuffled, seed)
        assert shuffle_abstractions(original, seed) == shuffled, key
        if "/r0/" in key and key in plain:
            # same batch and naive incumbent in round 0, so the same unshuffled records
            assert original == plain[key]["feedback"]["abstractions"], key
            checked += 1
        if len(shuffled) < 2:
            continue
        tuples = {tuple(r[k] for k in FIELDS) for r in original}
        count += 1
        total += len(shuffled)
        changed += sum(a != b for a, b in zip(shuffled, original, strict=True))
        novel += sum(tuple(r[k] for k in FIELDS) not in tuples for r in shuffled)
    print(
        json.dumps(
            {
                "seed_checked_requests": checked,
                "requests": count,
                "records": total,
                "records_changed": changed,
                "records_novel_tuple": novel,
            }
        )
    )
    assert checked, "no round-0 request to check the seed against"
    return 0


if __name__ == "__main__":
    sys.exit(main())
