from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from promptbench.live.execution import Execution, ledger
from promptbench.runner import RunPaused
from promptbench.storage import IntegrityError, Store, digest

LOCAL = {
    "provider": "ollama",
    "url": "http://127.0.0.1:11434/api/chat",
    "expected_models": ["fixture"],
    "expected_provider": "local-ollama",
}


def response(text='{"applicable_cves":[]}'):
    return {
        "body_text": json.dumps(
            {
                "model": "fixture",
                "done": True,
                "done_reason": "stop",
                "message": {"content": text},
                "eval_count": 10,
            }
        ),
        "http_status": 200,
        "error": None,
        "headers": {},
        "duration_ns": 1000000,
    }


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            self.root = Path(retained) / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            temp = tempfile.TemporaryDirectory()
            self.addCleanup(temp.cleanup)
            self.root = Path(temp.name)
        self.store = Store(self.root / "campaign")
        self.calls = 0

    def worker(self, store, attempt, env_file, **kwargs):
        self.calls += 1
        store.put(attempt + "/response.json", response())
        return {"status": "response"}

    def engine(self, **kwargs):
        return Execution(
            self.store,
            self.root / "unused.env",
            max_seconds=600,
            max_cost_usd="8",
            max_attempts=20,
            study_root=self.root,
            worker=self.worker,
            **kwargs,
        )

    def test_accepted_tariff_change_reuses_committed_and_rebinds_uncommitted_requests(self):
        # The tariff tolerance is provider-agnostic; the local fixture provider keeps the
        # test free of hosted response validation and key lookups.
        hosted = {
            **LOCAL,
            "reservation_usd": "0.10",
            "endpoint": {"tag": "x/fp8", "pricing": {"prompt": "1", "completion": "2"}},
        }
        body = {"model": "m", "provider": {"max_price": {"prompt": "1", "completion": "2"}}}
        fake_key = lambda: {"data": {"usage": 0.0}}  # noqa: E731
        engine = self.engine()
        engine.key_reader = fake_key
        first = engine.call("P0/one", hosted, body, role="executor")
        repriced_cond = {
            **hosted,
            "reservation_usd": "0.12",
            "endpoint": {"tag": "x/fp8", "pricing": {"prompt": "1.1", "completion": "2.2"}},
        }
        repriced_body = {
            "model": "m",
            "provider": {"max_price": {"prompt": "1.1", "completion": "2.2"}},
        }
        # committed at the old tariff: the recorded result is returned, nothing is re-sent
        resumed = self.engine()
        resumed.key_reader = fake_key
        again = resumed.call("P0/one", repriced_cond, repriced_body, role="executor")
        self.assertEqual(again["attempt"], first["attempt"])
        self.assertEqual(self.calls, 1)
        # never committed: a spec exists without a result; the key rebinds to the new price
        spec_dir = self.store.names("work/*/spec.json")
        self.assertEqual(len(spec_dir), 1)
        pending = {
            "key": "P0/two",
            "condition": hosted,
            "body": body,
            "role": "executor",
            "allowed_ids": [],
        }
        self.store.put("work/" + digest(pending) + "/spec.json", pending)
        self.store.put(
            "logical-keys/" + digest("P0/two") + ".json", {"spec_sha256": digest(pending)}
        )
        rebinding = self.engine()
        rebinding.key_reader = fake_key
        rebinding.call("P0/two", repriced_cond, repriced_body, role="executor")
        pointer = self.store.get("logical-keys/" + digest("P0/two") + ".json")
        self.assertEqual(pointer["superseded_spec_sha256"], digest(pending))
        self.assertEqual(self.calls, 2)
        # any other difference is still refused
        with self.assertRaises(IntegrityError):
            self.engine().call(
                "P0/one", repriced_cond, {**repriced_body, "model": "other"}, role="executor"
            )

    def test_durable_response_reused_after_interrupt_before_commit(self):
        def stop(boundary, key):
            if boundary == "before_commit":
                raise KeyboardInterrupt

        engine = self.engine(hook=stop)
        with self.assertRaises(KeyboardInterrupt):
            engine.call("P0/one", LOCAL, {}, role="executor")
        engine.checkpoint()
        resumed = self.engine()
        result = resumed.call("P0/one", LOCAL, {}, role="executor")
        self.assertEqual(result["value"], [])
        resumed.call("P0/one", LOCAL, {}, role="executor")
        self.assertEqual(self.calls, 1)

    def test_lost_attempt_retained_and_time_budget_not_reset(self):
        def stop(boundary, key):
            if boundary == "intent":
                raise KeyboardInterrupt

        engine = self.engine(hook=stop)
        with self.assertRaises(KeyboardInterrupt):
            engine.call("P0/one", LOCAL, {}, role="executor")
        resumed = self.engine()
        self.assertLess(resumed.remaining_seconds(), 480)
        resumed.call("P0/one", LOCAL, {}, role="executor")
        self.assertEqual(len(self.store.names("work/*/attempts/*/request.json")), 2)
        self.assertEqual(len(self.store.names("work/*/attempts/*/unknown.json")), 1)
        self.assertEqual(self.calls, 1)

    def test_malformed_executor_commits_failure_without_semantic_retry(self):
        engine = self.engine()

        def worker(store, attempt, env_file, **kwargs):
            self.calls += 1
            store.put(attempt + "/response.json", response("broken json"))

        engine.worker = worker
        result = engine.call("P0/one", LOCAL, {}, role="executor")
        self.assertEqual(result["status"], "invalid_output")
        self.assertIsNone(result["value"])
        self.assertEqual(self.calls, 1)
        with self.assertRaises(IntegrityError):
            engine.call("P0/one", LOCAL, {"changed": True}, role="executor")

    def test_transient_then_valid_preserves_both_responses(self):
        engine = self.engine()

        def worker(store, attempt, env_file, **kwargs):
            self.calls += 1
            raw = response()
            if self.calls == 1:
                raw.update(http_status=429, body_text='{"error":"rate limit"}')
            store.put(attempt + "/response.json", raw)

        engine.worker = worker
        with patch("promptbench.live.execution.time.sleep"):
            result = engine.call("P0/one", LOCAL, {}, role="executor")
        self.assertEqual(result["status"], "valid")
        self.assertEqual(len(self.store.names("work/*/attempts/*/response.json")), 2)

    def test_permanent_error_and_returned_model_drift_pause(self):
        engine = self.engine()

        def worker(store, attempt, env_file, **kwargs):
            raw = response()
            raw["http_status"] = 401
            store.put(attempt + "/response.json", raw)

        engine.worker = worker
        with self.assertRaises(RunPaused):
            engine.call("P0/auth", LOCAL, {}, role="executor")
        engine.worker = self.worker
        with self.assertRaises(IntegrityError):
            engine.call(
                "P0/drift", {**LOCAL, "expected_models": ["different"]}, {}, role="executor"
            )

    def test_study_ledger_deduplicates_continuations_excludes_offline_fixtures(self):
        request = {"provider": "openrouter", "reservation_usd": "1", "id": "old"}
        for name in ["original", "continuation", "validation/raw_test_runs/fixture"]:
            Store(self.root / name).put("work/one/attempts/0/request.json", request)
        Store(self.root / "validation/raw_test_runs/fixture").put(
            "work/two/attempts/0/request.json", {**request, "id": "fake", "reservation_usd": "1000"}
        )
        self.assertEqual(ledger(self.root, study=True)["reserved_unknown_usd"], "1")
        Store(self.root / "continuation").put(
            "work/one/attempts/0/response.json", {"body_text": '{"usage":{"cost":0.01}}'}
        )
        account = ledger(self.root, study=True)
        self.assertEqual(account["reported_cost_usd"], "0.01")
        self.assertEqual(account["reserved_unknown_usd"], "0")
        self.assertEqual(account["hosted_attempts"], 1)

    def test_unknown_reservation_prevents_dispatch_over_cap(self):
        engine = self.engine()
        engine.max_cost = __import__("decimal").Decimal("0.50")
        hosted = {**LOCAL, "provider": "openrouter", "reservation_usd": "1"}
        with self.assertRaises(RunPaused):
            engine.call("P0/cloud", hosted, {}, role="optimizer")
        self.assertEqual(self.calls, 0)
        self.assertFalse(self.store.names("work/*/attempts/*/request.json"))

    def test_auth_failure_can_retry_on_operator_resume_without_erasing_failed_attempt(self):
        engine = self.engine()

        def worker(store, attempt, env_file, **kwargs):
            raw = response()
            raw["http_status"] = 401
            store.put(attempt + "/response.json", raw)

        engine.worker = worker
        with self.assertRaises(RunPaused):
            engine.call("P0/auth-retry", LOCAL, {}, role="executor")
        engine.checkpoint()
        result = self.engine().call("P0/auth-retry", LOCAL, {}, role="executor")
        self.assertEqual(result["status"], "valid")
        self.assertEqual(len(self.store.names("work/*/attempts/*/request.json")), 2)

    def test_context_rejection_is_terminal_and_does_not_block_next_item(self):
        engine = self.engine()

        def worker(store, attempt, env_file, **kwargs):
            self.calls += 1
            raw = response()
            raw.update(http_status=400, body_text='{"error":"context limit exceeded"}')
            store.put(attempt + "/response.json", raw)

        engine.worker = worker
        result = engine.call("P0/oversized", LOCAL, {}, role="executor")
        self.assertEqual(result["status"], "request_rejected")
        self.assertIsNone(result["value"])
        self.assertEqual(self.calls, 1)
        engine.worker = self.worker
        self.assertEqual(engine.call("P0/next", LOCAL, {}, role="executor")["status"], "valid")
