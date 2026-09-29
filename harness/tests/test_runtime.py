from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from promptbench.live.process import invoke
from promptbench.live.telemetry import Telemetry, summarize
from promptbench.storage import IntegrityError, Store, canonical, digest


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            self.root = Path(retained) / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            temp = tempfile.TemporaryDirectory()
            self.addCleanup(temp.cleanup)
            self.root = Path(temp.name)

    def frame(self, index, power=20, timestamp=None):
        return {
            "monotonic_ns": index * 1_000_000_000 if timestamp is None else timestamp,
            "cpu_ticks_total": index * 100,
            "cpu_ticks_idle": index * 40,
            "gpus": [{"uuid": "fixture", "power_w": power, "memory_used_mib": index * 100}],
        }

    def test_energy_does_not_interpolate_missing_measurements_or_large_gaps(self):
        path = self.root / "samples.jsonl"
        chain = None
        with path.open("wb") as stream:
            for index, (seconds, power) in enumerate(
                [(0, 10), (1, 20), (2, None), (9, 30), (10, 50)]
            ):
                frame = {
                    **self.frame(index, power, seconds * 1_000_000_000),
                    "sequence": index,
                    "previous_sha256": chain,
                }
                chain = digest(frame)
                stream.write((canonical({"payload": frame, "sha256": chain}) + "\n").encode())
            stream.write(b'{"sha256":')
        result = summarize(path, 1)
        self.assertEqual(result["gpu_energy_joules_observed"], {"fixture": 55.0})
        self.assertEqual(result["gpu_energy_covered_seconds"], {"fixture": 2.0})
        self.assertEqual(result["observed_span_seconds"], 10.0)
        self.assertTrue(result["trailing_partial_line"])
        Store(self.root).put("observed-summary.json", result)

    def test_periodic_telemetry_retains_frames_while_main_thread_waits(self):
        ready = threading.Event()
        index = 0

        def sampler(root):
            nonlocal index
            frame = self.frame(index, timestamp=index * 10_000_000)
            index += 1
            if index >= 3:
                ready.set()
            return frame

        monitor = Telemetry(Store(self.root), interval=0.01, sampler=sampler)
        monitor.start()
        self.assertTrue(ready.wait(2))
        result = monitor.stop()
        self.assertGreaterEqual(result["samples"], 3)
        self.assertAlmostEqual(
            result["gpu_energy_joules_observed"]["fixture"],
            20 * result["gpu_energy_covered_seconds"]["fixture"],
        )
        data = monitor.path.read_bytes()
        monitor.path.write_bytes(data.replace(b'"power_w":20', b'"power_w":99', 1))
        with self.assertRaises(IntegrityError):
            summarize(monitor.path, 0.01)

    def test_sensor_storage_failure_is_visible_to_controller(self):
        def failing(root):
            raise OSError("simulated telemetry disk failure")

        monitor = Telemetry(Store(self.root), sampler=failing)
        monitor.start()
        with self.assertRaisesRegex(IntegrityError, "disk failure"):
            monitor.stop()

    def test_missing_gpu_fields_are_limitations_not_zero(self):
        for label, gpus in (("no_gpu", []), ("missing_fields", [{"uuid": "fixture"}])):
            path = self.root / (label + ".jsonl")
            frame = {**self.frame(0), "gpus": gpus, "sequence": 0, "previous_sha256": None}
            path.write_text(canonical({"payload": frame, "sha256": digest(frame)}) + "\n")
            result = summarize(path, 1)
            self.assertEqual(result["gpu_energy_joules_observed"], {})
            self.assertEqual(result["gpu_peak_sampled_memory_mib"], {})
            limits = result["gpu_measurement_limitations"]
            self.assertTrue(limits["energy_unavailable"])
            self.assertTrue(limits["peak_vram_unavailable"])
            self.assertEqual(limits["samples_without_gpu"], 0 if gpus else 1)
            if gpus:
                self.assertEqual(
                    limits["missing_field_observations"],
                    {
                        "power_w": 1,
                        "memory_used_mib": 1,
                        "utilization_percent": 1,
                    },
                )

    def test_worker_deadline_stops_a_stalled_process_and_preserves_logs(self):
        store = Store(self.root)
        store.put("attempt/request.json", {"fixture": True})
        command = [sys.executable, "-u", "-c", "import time; print('waiting'); time.sleep(60)"]
        start = time.perf_counter()
        result = invoke(store, "attempt", self.root / "missing.env", timeout=0.2, command=command)
        self.assertEqual(result["status"], "deadline")
        self.assertLess(time.perf_counter() - start, 3)
        self.assertIn("waiting", (self.root / "attempt/worker.log").read_text())
        self.assertFalse(store.exists("attempt/response.json"))

    def test_owned_worker_dies_when_controller_is_hard_killed(self):
        signal_file = self.root / "child-ready.json"
        child_code = (
            "import json,os,signal,sys; from pathlib import Path; "
            "from promptbench.live.worker import parent_lifetime; "
            "parent_lifetime(int(sys.argv[1])); "
            "Path(sys.argv[2]).write_text(json.dumps({'pid':os.getpid()})); signal.pause()"
        )
        parent_code = (
            "import os,signal,subprocess,sys; "
            "subprocess.Popen([sys.executable,'-c',sys.argv[1],str(os.getpid()),sys.argv[2]]); "
            "signal.pause()"
        )
        with (self.root / "parent.log").open("w") as log:
            parent = subprocess.Popen(
                [sys.executable, "-c", parent_code, child_code, str(signal_file)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 3
                while not signal_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(signal_file.exists())
                child_pid = json.loads(signal_file.read_text())["pid"]
                parent.kill()
                parent.wait(timeout=3)
                status = Path(f"/proc/{child_pid}/stat")
                deadline = time.monotonic() + 3
                while status.exists() and time.monotonic() < deadline:
                    if status.read_text().split()[2] == "Z":
                        break
                    time.sleep(0.01)
                self.assertTrue(not status.exists() or status.read_text().split()[2] == "Z")
                Store(self.root).put(
                    "outcome.json", {"controller_killed": True, "worker_not_running": True}
                )
            finally:
                try:
                    os.killpg(parent.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                parent.wait(timeout=3)
