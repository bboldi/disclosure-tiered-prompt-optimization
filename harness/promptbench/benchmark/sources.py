"""Acquire immutable annual NVD feeds and verify their published uncompressed hashes."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from ..live.transport import NoRedirect
from ..runner import utc_now
from ..storage import IntegrityError, Store, digest

FEED = "https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{}"


def parse_meta(raw: bytes) -> dict[str, Any]:
    fields = {}
    for line in raw.decode("utf-8").splitlines():
        if not line:
            continue
        key, separator, value = line.partition(":")
        if not separator or key in fields:
            raise IntegrityError("invalid or duplicate feed metadata field")
        fields[key] = value
    if not {"lastModifiedDate", "size", "gzSize", "sha256"} <= fields.keys():
        raise IntegrityError("missing feed metadata")
    checksum = fields["sha256"].lower()
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise IntegrityError("invalid feed checksum")
    size, compressed = int(fields["size"]), int(fields["gzSize"])
    if not 0 < compressed < 200_000_000 or not 0 < size < 2_000_000_000:
        raise IntegrityError("feed exceeds admitted source size")
    return {
        "uncompressed_bytes": size,
        "compressed_bytes": compressed,
        "uncompressed_sha256": checksum,
        "source_last_modified": fields["lastModifiedDate"],
    }


def verify_feed(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if path.stat().st_size != expected["compressed_bytes"]:
        raise IntegrityError("compressed feed length differs from metadata")
    with path.open("rb") as compressed_stream:
        compressed = hashlib.file_digest(compressed_stream, "sha256").hexdigest()
    total, checksum = 0, hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        while block := stream.read(1_048_576):
            total += len(block)
            if total > expected["uncompressed_bytes"]:
                raise IntegrityError("decompressed feed exceeds metadata size")
            checksum.update(block)
    if (
        total != expected["uncompressed_bytes"]
        or checksum.hexdigest() != expected["uncompressed_sha256"]
    ):
        raise IntegrityError("uncompressed feed size/checksum differs from metadata")
    return {**expected, "compressed_sha256": compressed}


def acquire(root: Path, years: tuple[int, ...] = (2023, 2024, 2025)) -> dict[str, Any]:
    store = Store(root)
    opener = urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({}))
    if (
        not years
        or len(set(years)) != len(years)
        or any(year not in (2023, 2024, 2025) for year in years)
    ):
        raise ValueError("source years must be a unique subset of 2023–2025")
    with store.lock():
        results = {}
        for index, year in enumerate(years):
            folder = f"sources/{year}"
            metadata_name = folder + "/metadata.json"
            receipt_name = folder + "/receipt.json"
            target = store.root / folder / "feed.json.gz"
            if not store.exists(metadata_name):
                started = utc_now()
                with opener.open(FEED.format(year) + ".meta", timeout=20) as response:
                    raw = response.read(16_385)
                    if len(raw) > 16_384:
                        raise IntegrityError("feed metadata exceeds size cap")
                    metadata = {
                        "url": response.url,
                        "http_status": response.status,
                        "headers": dict(response.headers),
                        "raw_text": raw.decode(),
                        "started_utc": started,
                        "acquired_utc": utc_now(),
                        "expected": parse_meta(raw),
                    }
                store.put(metadata_name, metadata)
            expected = store.get(metadata_name)["expected"]
            if target.exists():
                verified = verify_feed(target, expected)
            else:
                if (
                    shutil.disk_usage(store.root).free
                    < 2_147_483_648 + expected["compressed_bytes"]
                ):
                    raise IntegrityError("less than 2 GiB plus feed size remains on disk")
                attempt = folder + "/downloads/" + uuid.uuid4().hex
                store.put(
                    attempt + "/intent.json",
                    {"url": FEED.format(year) + ".json.gz", "started_utc": utc_now()},
                )
                partial = store.root / attempt / "response.json.gz.partial"
                started_ns = time.perf_counter_ns()
                total = 0
                try:
                    with opener.open(FEED.format(year) + ".json.gz", timeout=30) as response:
                        store.put(
                            attempt + "/headers.json",
                            {
                                "http_status": response.status,
                                "headers": dict(response.headers),
                                "url": response.url,
                            },
                        )
                        with partial.open("xb") as stream:
                            while block := response.read(1_048_576):
                                stream.write(block)
                                total += len(block)
                                if total > expected["compressed_bytes"]:
                                    raise IntegrityError("feed response exceeds declared size")
                                if time.perf_counter_ns() - started_ns > 300_000_000_000:
                                    raise TimeoutError("source download exceeded 300 seconds")
                                percent = 100 * total / expected["compressed_bytes"]
                                print(
                                    f"[sources {index + 1}/{len(years)} | {year} {percent:5.1f}%] {total} bytes",
                                    flush=True,
                                )
                            stream.flush()
                            os.fsync(stream.fileno())
                    verified = verify_feed(partial, expected)
                    os.link(partial, target)
                    directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                    partial.unlink()
                    store.put(
                        attempt + "/outcome.json",
                        {
                            "status": "verified",
                            "bytes": total,
                            "finished_utc": utc_now(),
                            "duration_ns": time.perf_counter_ns() - started_ns,
                        },
                    )
                except (OSError, ValueError) as exc:
                    store.put(
                        attempt + "/outcome.json",
                        {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "bytes": total,
                            "partial_retained": partial.exists(),
                            "finished_utc": utc_now(),
                            "duration_ns": time.perf_counter_ns() - started_ns,
                        },
                    )
                    raise
            if not store.exists(receipt_name):
                store.put(
                    receipt_name,
                    {
                        **verified,
                        "year": year,
                        "feed_path": str(target.relative_to(store.root)),
                        "verified_utc": utc_now(),
                        "metadata_sha256": digest(store.get(metadata_name)),
                        "acquisition_source": Path(__file__).read_text(),
                    },
                )
            receipt = store.get(receipt_name)
            if receipt["metadata_sha256"] != digest(store.get(metadata_name)):
                raise IntegrityError("source metadata changed after acquisition")
            if any(receipt[key] != value for key, value in verified.items()):
                raise IntegrityError("source receipt differs from verified feed")
            results[str(year)] = receipt
            print(
                f"[sources {(index + 1) / len(years):.0%}] {year}: verified; existing feeds are reused",
                flush=True,
            )
        return results
