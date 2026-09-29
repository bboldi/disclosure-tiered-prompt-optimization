"""Acquire selected official CNA records at one recorded CVE repository commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..live.transport import NoRedirect
from ..runner import utc_now
from ..storage import IntegrityError, Store, digest, read_jsonl


def record_url(commit: str, cve_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(
        r"CVE-20[0-9]{2}-[0-9]{4,}", cve_id
    ):
        raise ValueError("invalid immutable source identifier")
    _, year, number = cve_id.split("-")
    return f"https://raw.githubusercontent.com/CVEProject/cvelistV5/{commit}/cves/{year}/{number[:-3]}xxx/{cve_id}.json"


def acquire(root: Path, candidate_root: Path, *, workers: int = 8) -> dict[str, Any]:
    if not 1 <= workers <= 16:
        raise ValueError("CNA acquisition concurrency must be between one and sixteen")
    store, candidate = Store(root), Store(candidate_root)
    with store.lock():
        commit = store.get("repository-revision.json")["commit"]
        candidate_manifest = candidate.get("manifest.json")
        expected = (
            candidate_manifest.get("advisories_sha256")
            or candidate_manifest["artifact_sha256"]["advisories.jsonl"]
        )
        if (
            hashlib.sha256((candidate.root / "advisories.jsonl").read_bytes()).hexdigest()
            != expected
        ):
            raise IntegrityError("candidate advisories changed")
        identifiers = sorted(row["id"] for row in read_jsonl(candidate.root / "advisories.jsonl"))
        if len(set(identifiers)) != len(identifiers):
            raise IntegrityError("duplicate candidate CVE")
        plan = {
            "commit": commit,
            "candidate_manifest_sha256": digest(candidate.get("manifest.json")),
            "identifiers": identifiers,
            "workers": workers,
            "source_sha256": digest(Path(__file__).read_text()),
        }
        store.put("acquisition-plan.json", plan)
        store.put("acquisition-source.json", {"text": Path(__file__).read_text()})

        def fetch(identifier: str) -> dict[str, Any]:
            url = record_url(commit, identifier)
            receipt = f"records/{identifier}.json"
            if store.exists(receipt):
                record = store.get(receipt)
                if (
                    record["url"] != url
                    or hashlib.sha256(record["body_text"].encode()).hexdigest()
                    != record["body_sha256"]
                ):
                    raise IntegrityError("CNA receipt changed")
                return {"id": identifier, "status": "acquired", "reused": True}
            opener = urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({}))
            for index in range(3):
                name = f"attempts/{identifier}/{index}.json"
                if store.exists(name):
                    continue
                started = time.perf_counter_ns()
                status, raw, error = None, b"", None
                try:
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "promptbench-public-research"}
                    )
                    with opener.open(request, timeout=20) as response:  # nosec B310 -- validated commit/CVE under fixed official origin
                        status = response.status
                        raw = response.read(2_000_001)
                    if len(raw) > 2_000_000:
                        raise ValueError("CNA record exceeds two-megabyte acquisition bound")
                    parsed = json.loads(raw)
                    if (
                        parsed["cveMetadata"]["cveId"] != identifier
                        or parsed["dataType"] != "CVE_RECORD"
                    ):
                        raise ValueError("CNA identity/schema mismatch")
                except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as exc:
                    error = str(exc)
                record = {
                    "url": url,
                    "retrieved_utc": utc_now(),
                    "duration_ns": time.perf_counter_ns() - started,
                    "http_status": status,
                    "body_text": raw.decode("utf-8", errors="replace"),
                    "body_sha256": hashlib.sha256(raw).hexdigest(),
                    "error": error,
                }
                store.put(name, record)
                if error is None:
                    store.put(receipt, record)
                    return {"id": identifier, "status": "acquired", "reused": False}
                if index < 2:
                    time.sleep(2**index)
            return {
                "id": identifier,
                "status": "unavailable_after_bounded_attempts",
                "reused": False,
            }

        outcomes = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, result in enumerate(pool.map(fetch, identifiers)):
                outcomes.append(result)
                if index % 100 == 0 or index + 1 == len(identifiers):
                    print(
                        f"[CNA {100 * (index + 1) / len(identifiers):.1f}%] {index + 1}/{len(identifiers)} records",
                        flush=True,
                    )
        report = {
            "commit": commit,
            "candidate_count": len(identifiers),
            "acquired": sum(r["status"] == "acquired" for r in outcomes),
            "unavailable": [r["id"] for r in outcomes if r["status"] != "acquired"],
            "candidate_manifest_sha256": plan["candidate_manifest_sha256"],
        }
        store.put("acquisition-report.json", report)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    report = acquire(args.run_dir, args.candidate_dir)
    print(json.dumps(report))
    return 0 if not report["unavailable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
