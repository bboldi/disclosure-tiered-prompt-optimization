from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from promptbench.audit import audit
from promptbench.experiment import Config, Experiment, read_fixture
from promptbench.providers import FakeProvider, Reply
from promptbench.runner import RunPaused
from promptbench.storage import IntegrityError, Store, atomic_write, canonical

CODE = Path(__file__).resolve().parents[1]
FIXTURE = CODE / "fixtures/mini.json"


class Harness(unittest.TestCase):
    def setUp(self) -> None:
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            self.root = Path(retained) / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            self.temp = tempfile.TemporaryDirectory(prefix="promptbench-test-")
            self.addCleanup(self.temp.cleanup)
            self.root = Path(self.temp.name)
        self.fixture = read_fixture(FIXTURE)

    def run_case(self, name: str, config: Config | None = None, **kwargs: Any) -> dict[str, Any]:
        root = self.root / name
        Experiment(root, config or Config(), self.fixture, **kwargs).run()
        return audit(Store(root), export=bool(os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")))

    def projection(self, name: str) -> dict[str, Any]:
        store = Store(self.root / name)
        return {
            "completion": store.get("completion.json"),
            "prompts": [store.get(n) for n in store.names("prompts/*.json")],
            "feedback": [store.get(n) for n in store.names("feedback/*.json")],
            "decisions": [store.get(n) for n in store.names("decisions/*.json")],
            "evaluations": [store.get(n) for n in store.names("evaluations/*.json")],
        }

    def test_end_to_end_and_idempotent_resume(self) -> None:
        first = self.run_case("reference")
        store = Store(self.root / "reference")
        self.assertEqual(first["actual_api_cost_usd"], "0")
        self.assertEqual(first["test_metrics"]["valid_answers"]["micro_f1"], 1.0)
        attempts = store.names("work/*/attempts/*/request.json")
        first_projection = self.projection("reference")
        second = self.run_case("reference")
        self.assertEqual(attempts, store.names("work/*/attempts/*/request.json"))
        self.assertEqual(first_projection, self.projection("reference"))
        self.assertEqual(first["synthetic_known_cost_usd"], second["synthetic_known_cost_usd"])

    def test_hard_kills_at_every_checkpoint(self) -> None:
        reference = self.run_case("reference")
        points = (
            "planned",
            "intent",
            "provider_returned",
            "response",
            "before_commit",
            "committed",
            "iteration",
            "selection",
            "completion",
        )
        for point in points:
            with self.subTest(point=point):
                root = self.root / point
                command = [
                    sys.executable,
                    "-m",
                    "tests.crash_worker",
                    str(root),
                    str(FIXTURE),
                    point,
                ]
                started = time.perf_counter()
                process = subprocess.run(
                    command,
                    cwd=CODE,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                Store(root).put(
                    "diagnostics/crash_process.json",
                    {
                        "command": command,
                        "returncode": process.returncode,
                        "stdout": process.stdout,
                        "stderr": process.stderr,
                        "duration_seconds": time.perf_counter() - started,
                        "injected_boundary": point,
                    },
                )
                self.assertEqual(process.returncode, -signal.SIGKILL, process.stderr)
                resumed = self.run_case(point)
                self.assertEqual(self.projection("reference"), self.projection(point))
                if point in ("intent", "provider_returned"):
                    self.assertEqual(resumed["unknown_attempts"], 1)
                    self.assertEqual(
                        resumed["provider_attempts"], reference["provider_attempts"] + 1
                    )
                else:
                    self.assertEqual(resumed["provider_attempts"], reference["provider_attempts"])

    def test_cli_checkpoint_resume_and_audit(self) -> None:
        root = self.root / "cli"
        commands = [
            ["run", "--run-dir", str(root), "--fixture", str(FIXTURE), "--stop-after-commits", "5"],
            ["run", "--run-dir", str(root)],
            ["audit", "--run-dir", str(root)],
        ]
        for index, (arguments, expected_exit) in enumerate(zip(commands, [130, 0, 0], strict=True)):
            command = [sys.executable, "-m", "promptbench", *arguments]
            started = time.perf_counter()
            process = subprocess.run(
                command, cwd=CODE, capture_output=True, text=True, timeout=60, check=False
            )
            Store(root).put(
                f"diagnostics/cli-{index}.json",
                {
                    "command": command,
                    "returncode": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                    "duration_seconds": time.perf_counter() - started,
                },
            )
            self.assertEqual(process.returncode, expected_exit, process.stderr)
        self.assertEqual(audit(Store(root), export=False)["actual_api_cost_usd"], "0")

    def test_partial_candidate_does_not_skip_rest_of_iteration(self) -> None:
        def interrupt(point: str, key: str) -> None:
            if point == "committed" and key.startswith("evaluate/iteration-1/i1-c0/"):
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_case("partial", hook=interrupt)
        self.run_case("partial")
        store = Store(self.root / "partial")
        self.assertTrue(store.exists("prompts/i1-c1.json"))
        self.assertEqual(len(store.get("decisions/iteration-001.json")["contenders"]), 3)

    def test_raw_response_recovered_without_second_provider_call(self) -> None:
        def interrupt(point: str, key: str) -> None:
            if point == "response":
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_case("response", hook=interrupt)
        store = Store(self.root / "response")
        pending = store.names("work/*/spec.json")[0]
        saved_request = store.get(pending)["request"]

        class GuardedProvider(FakeProvider):
            def call(self, request: dict[str, Any], retry: int, fault: str | None) -> Reply:
                if request == saved_request:
                    raise AssertionError("persisted response was unnecessarily regenerated")
                return super().call(request, retry, fault)

        guarded = Experiment(store.root, Config(), self.fixture, provider=GuardedProvider())
        with store.lock():
            result = guarded.runner.execute(store.get(pending)["key"], saved_request)
        self.assertEqual(result["status"], "valid")
        # Later iterations may intentionally evaluate an identical request as a NEW replicate.
        self.run_case("response")

    def test_exhausted_executor_transport_is_explicit_and_auditable(self) -> None:
        config = Config(
            faults={
                "evaluate/baseline/baseline/opt_in_range": "timeout_always",
                "evaluate/baseline/baseline/opt_patched": "reasoning_only",
            }
        )
        summary = self.run_case("executor-failures", config)
        self.assertEqual(summary["failures"]["transport_error"], 3)
        store = Store(self.root / "executor-failures")
        rows = [
            store.get(n)["score"]
            for n in store.names("evaluations/*.json")
            if store.get(n)["stage"] == "baseline"
        ]
        by_id = {r["profile_id"]: r for r in rows}
        self.assertEqual(by_id["opt_in_range"]["status"], "transport_failure")
        self.assertEqual(by_id["opt_patched"]["status"], "invalid_output")
        self.assertIsNone(by_id["opt_in_range"]["prediction"])

    def test_duplicate_candidates_do_not_create_extra_evaluations(self) -> None:
        class DuplicateProvider(FakeProvider):
            def call(self, request: dict[str, Any], retry: int, fault: str | None) -> Reply:
                if request["role"] == "optimizer":
                    request = {**request, "candidate_slot": 0}
                return super().call(request, retry, fault)

        self.run_case("duplicates", provider=DuplicateProvider())
        store = Store(self.root / "duplicates")
        self.assertEqual(store.get("prompts/i1-c1.json")["duplicate_of"], "i1-c0")
        self.assertEqual(len(store.get("decisions/iteration-001.json")["contenders"]), 2)

    def test_retries_preserve_raw_failures_and_usage(self) -> None:
        config = Config(faults={"propose/1/0": "malformed_once", "propose/1/1": "timeout_once"})
        summary = self.run_case("faults", config)
        self.assertEqual(summary["failures"]["transport_error"], 1)
        self.assertEqual(summary["failures"]["retryable"], 2)
        self.assertGreater(summary["provider_attempts"], summary["committed_work"])
        store = Store(self.root / "faults")
        raw = [store.get(n) for n in store.names("work/*/attempts/*/response.json")]
        self.assertTrue(any(r.get("reply", {}).get("text") == '{"incomplete":' for r in raw))

    def test_protocol_exhaustion_does_not_fabricate_completion(self) -> None:
        config = Config(faults={"propose/1/0": "malformed_always"})
        with self.assertRaisesRegex(RunPaused, "retries exhausted"):
            self.run_case("exhaust", config)
        self.assertFalse(Store(self.root / "exhaust").exists("completion.json"))
        with self.assertRaises(RunPaused):
            self.run_case("exhaust", config)

    def test_permanent_error_does_not_retry_storm(self) -> None:
        config = Config(faults={"propose/1/0": "auth"})
        with self.assertRaisesRegex(RunPaused, "authentication"):
            self.run_case("auth", config)
        store = Store(self.root / "auth")
        attempts = store.names("work/*/attempts/*/request.json")
        with self.assertRaises(RunPaused):
            self.run_case("auth", config)
        self.assertEqual(attempts, store.names("work/*/attempts/*/request.json"))

    def test_absolute_attempt_cap(self) -> None:
        with self.assertRaisesRegex(RunPaused, "budget"):
            self.run_case("budget", Config(max_attempts=2))
        store = Store(self.root / "budget")
        self.assertEqual(len(store.names("work/*/attempts/*/request.json")), 2)
        self.assertFalse(store.exists("completion.json"))

    def test_write_failure_before_commit_preserves_saved_response(self) -> None:
        def fail(point: str, key: str) -> None:
            if point == "before_commit":
                raise OSError(28, "simulated disk full")

        with self.assertRaises(OSError):
            self.run_case("disk-full", hook=fail)
        store = Store(self.root / "disk-full")
        response_name = store.names("work/*/attempts/*/response.json")[0]
        original = (store.root / response_name).read_bytes()
        self.run_case("disk-full")
        self.assertEqual(original, (store.root / response_name).read_bytes())

    def test_missing_completion_is_rebuilt_without_new_calls(self) -> None:
        self.run_case("summary")
        store = Store(self.root / "summary")
        before = store.names("work/*/attempts/*/request.json")
        (store.root / "completion.json").unlink()
        self.run_case("summary")
        self.assertEqual(before, store.names("work/*/attempts/*/request.json"))

    def test_configuration_and_fixture_drift_rejected(self) -> None:
        self.run_case("drift")
        with self.assertRaises(IntegrityError):
            self.run_case("drift", Config(seed=12))
        fixture = copy.deepcopy(self.fixture)
        fixture["provenance"] += " changed"
        with self.assertRaises(IntegrityError):
            Experiment(self.root / "drift", Config(), fixture).run()

    def test_corruption_and_missing_evidence_fail_audit(self) -> None:
        self.run_case("tamper")
        store = Store(self.root / "tamper")
        path = store.root / store.names("work/*/attempts/*/response.json")[0]
        original = path.read_bytes()
        path.write_text('{"payload":')
        with self.assertRaises(IntegrityError):
            audit(store, export=False)
        path.write_bytes(original)
        path.unlink()
        with self.assertRaises(IntegrityError):
            audit(store, export=False)

    def test_lock_prevents_concurrent_controller(self) -> None:
        store = Store(self.root / "locked")
        with store.lock(), self.assertRaisesRegex(IntegrityError, "another controller"):
            with Store(store.root).lock():
                self.fail("second controller acquired lock")

    def test_atomic_immutable_write_conflict(self) -> None:
        path = self.root / "value"
        atomic_write(path, b"original")
        atomic_write(path, b"original")
        with self.assertRaises(IntegrityError):
            atomic_write(path, b"replacement")
        self.assertEqual(path.read_bytes(), b"original")

    def test_complete_optimizer_requests_have_no_profile_canaries_at_tiers_1_2(self) -> None:
        for tier in (1, 2):
            name = f"tier-{tier}"
            self.run_case(name, Config(tier=tier))
            store = Store(self.root / name)
            for item in store.names("work/*/spec.json"):
                spec = store.get(item)
                if spec["request"]["role"] == "optimizer":
                    content = canonical(spec["request"])
                    for forbidden in ("CANARY_PRIVATE", "CVE-2099", "Acme", "Delta", "Golf"):
                        self.assertNotIn(forbidden, content)

    def test_test_inputs_cannot_change_selected_prompt(self) -> None:
        self.run_case("original")
        changed = copy.deepcopy(self.fixture)
        for row in changed["profiles"]:
            if row["partition"] == "test":
                row["components"][0]["version"] = "100.0"
                row["text"] = "An independently changed held-out profile."
                row["expected"] = []
        Experiment(self.root / "changed-test", Config(), changed).run()
        self.assertEqual(
            Store(self.root / "original").get("decisions/selection.json"),
            Store(self.root / "changed-test").get("decisions/selection.json"),
        )

    def test_export_is_reconstructible_and_explicitly_synthetic(self) -> None:
        self.run_case("export")
        store = Store(self.root / "export")
        summary = audit(store)
        self.assertEqual(summary, store.get("exports/summary.json"))
        lines = (store.root / "exports/attempts.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), summary["provider_attempts"])
        self.assertTrue(all("request" in json.loads(line) for line in lines))
        self.assertIn("does not measure LLM quality", (store.root / "CONCLUSIONS.md").read_text())
