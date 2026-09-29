"""Periodic machine measurements in a checksummed append-only session stream."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import subprocess  # nosec B404 -- fixed local sensor command
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..runner import utc_now
from ..storage import IntegrityError, Store, canonical, digest


def number(value: str) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def sample(root: Path) -> dict[str, Any]:
    frame: dict[str, Any] = {"timestamp_utc": utc_now(), "monotonic_ns": time.monotonic_ns()}
    try:
        cpu = [int(n) for n in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
        memory = {
            line.split(":")[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemTotal:", "MemAvailable:"))
        }
        frame.update(
            cpu_ticks_total=sum(cpu),
            cpu_ticks_idle=cpu[3] + cpu[4],
            load_average=list(os.getloadavg()),
            memory_bytes=memory,
            controller_rss_bytes=int(Path("/proc/self/statm").read_text().split()[1])
            * os.sysconf("SC_PAGE_SIZE"),
            disk_free_bytes=shutil.disk_usage(root).free,
            cpu_error=None,
        )
    except (OSError, ValueError, IndexError) as exc:
        frame.update(
            cpu_ticks_total=None,
            cpu_ticks_idle=None,
            load_average=None,
            memory_bytes=None,
            controller_rss_bytes=None,
            disk_free_bytes=None,
            cpu_error=f"{type(exc).__name__}: {exc}",
        )
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=False)  # nosec B603 B607 -- fixed sensor executable
        frame["gpu_raw"] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        frame["gpus"] = (
            [
                dict(
                    zip(
                        (
                            "uuid",
                            "memory_total_mib",
                            "memory_used_mib",
                            "utilization_percent",
                            "temperature_c",
                            "power_w",
                        ),
                        [row[0].strip()] + [number(v.strip()) for v in row[1:]],
                        strict=True,
                    )
                )
                for row in csv.reader(io.StringIO(result.stdout))
                if len(row) == 6
            ]
            if result.returncode == 0
            else []
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        frame["gpu_raw"] = {"error": f"{type(exc).__name__}: {exc}"}
        frame["gpus"] = []
    return frame


def summarize(path: Path, interval: float) -> dict[str, Any]:
    previous, chain = None, None
    count, truncated = 0, False
    energy: dict[str, float] = {}
    observed_seconds: dict[str, float] = {}
    cpu_busy = []
    first_ns, last_ns = None, None
    peaks: dict[str, float] = {}
    missing_gpu_samples = 0
    missing_fields: dict[str, int] = {}
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                truncated = True
                break
            record = json.loads(line)
            frame = record["payload"]
            if (
                digest(frame) != record["sha256"]
                or frame["sequence"] != count
                or frame["previous_sha256"] != chain
            ):
                raise IntegrityError("telemetry sequence/checksum mismatch")
            chain = record["sha256"]
            first_ns = frame["monotonic_ns"] if first_ns is None else first_ns
            last_ns = frame["monotonic_ns"]
            current_gpus = {g["uuid"]: g for g in frame["gpus"]}
            if not current_gpus:
                missing_gpu_samples += 1
            for identity, gpu in current_gpus.items():
                for field in ("power_w", "memory_used_mib", "utilization_percent"):
                    if gpu.get(field) is None:
                        missing_fields[field] = missing_fields.get(field, 0) + 1
                used = gpu.get("memory_used_mib")
                if used is not None:
                    peaks[identity] = max(peaks.get(identity, 0), used)
            if previous:
                elapsed = (frame["monotonic_ns"] - previous["monotonic_ns"]) / 1e9
                if elapsed <= 0:
                    raise IntegrityError("nonmonotonic telemetry")
                for gpu in previous["gpus"]:
                    other = current_gpus.get(gpu["uuid"])
                    if (
                        other
                        and gpu.get("power_w") is not None
                        and other.get("power_w") is not None
                        and elapsed <= interval * 2.5
                    ):
                        identity = gpu["uuid"]
                        energy[identity] = (
                            energy.get(identity, 0)
                            + (gpu["power_w"] + other["power_w"]) * elapsed / 2
                        )
                        observed_seconds[identity] = observed_seconds.get(identity, 0) + elapsed
                if all(
                    previous.get(k) is not None and frame.get(k) is not None
                    for k in ("cpu_ticks_total", "cpu_ticks_idle")
                ):
                    total = frame["cpu_ticks_total"] - previous["cpu_ticks_total"]
                    idle = frame["cpu_ticks_idle"] - previous["cpu_ticks_idle"]
                    if total > 0 and 0 <= idle <= total:
                        cpu_busy.append(100 * (1 - idle / total))
            previous = frame
            count += 1
    return {
        "samples": count,
        "trailing_partial_line": truncated,
        "last_sha256": chain,
        "observed_span_seconds": (last_ns - first_ns) / 1e9
        if last_ns is not None and first_ns is not None
        else None,
        "gpu_energy_joules_observed": energy,
        "gpu_energy_covered_seconds": observed_seconds,
        "gpu_peak_sampled_memory_mib": peaks,
        "gpu_measurement_limitations": {
            "samples_without_gpu": missing_gpu_samples,
            "missing_field_observations": missing_fields,
            "energy_unavailable": not bool(energy),
            "peak_vram_unavailable": not bool(peaks),
        },
        "system_cpu_busy_percent_mean": sum(cpu_busy) / len(cpu_busy) if cpu_busy else None,
        "method": "Trapezoidal whole-GPU power over adjacent available samples, gaps <= 2.5 intervals. No extrapolation, idle subtraction or exclusive process attribution. Empty sensor mappings mean unavailable, not zero.",
    }


class Telemetry:
    def __init__(
        self,
        store: Store,
        *,
        interval: float = 1.0,
        sampler: Callable[[Path], dict[str, Any]] = sample,
    ):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("telemetry interval must be positive")
        self.store, self.interval, self.sampler = store, interval, sampler
        self.session = uuid.uuid4().hex
        self.directory = "telemetry/" + self.session
        self.path = store.root / self.directory / "samples.jsonl"
        self.stop_event = threading.Event()
        self.failure: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="telemetry", daemon=True)

    def start(self) -> None:
        self.store.put(
            self.directory + "/metadata.json",
            {
                "started_utc": utc_now(),
                "interval_seconds": self.interval,
                "pid": os.getpid(),
                "stream": "samples.jsonl",
            },
        )
        self.thread.start()

    def _run(self) -> None:
        sequence, previous = 0, None
        try:
            with self.path.open("xb") as stream:
                while True:
                    frame = {
                        **self.sampler(self.store.root),
                        "sequence": sequence,
                        "previous_sha256": previous,
                    }
                    previous = digest(frame)
                    stream.write(
                        (canonical({"sha256": previous, "payload": frame}) + "\n").encode()
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                    sequence += 1
                    if self.stop_event.wait(self.interval):
                        break
        except BaseException as exc:
            self.failure = exc

    def ensure_healthy(self) -> None:
        if self.failure:
            raise IntegrityError(f"telemetry could not retain measurements: {self.failure}")

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise IntegrityError("telemetry sampler did not stop within its bound")
        self.ensure_healthy()
        result = summarize(self.path, self.interval)
        self.store.put(self.directory + "/summary.json", result)
        return result
