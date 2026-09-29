from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

from promptbench.experiment import read_fixture
from promptbench.live.__main__ import main
from promptbench.live.adapters import normalize, optimizer_body, reservation
from promptbench.live.preflight import local_identity
from promptbench.live.progress import Progress
from promptbench.live.smoke import Smoke
from promptbench.live.transport import (
    OLLAMA,
    OPENROUTER,
    NoRedirect,
    Transport,
    read_key,
    response_json,
)
from promptbench.runner import RunPaused
from promptbench.storage import IntegrityError, Store


def record(value: dict) -> dict:
    return {"http_status": 200, "error": None, "body_text": json.dumps(value), "duration_ns": 1234}


def registry() -> dict:
    endpoint = {
        "tag": "streamlake/fp8",
        "provider_name": "StreamLake",
        "context_length": 1_024_000,
        "pricing": {"prompt": "0.000000738282", "completion": "0.000001476564"},
    }
    return {
        "local_models": [
            {
                "requested_tag": "granite4.2:30b",
                "ollama_manifest_digest": "fixture",
                "capabilities": ["completion", "thinking"],
            }
        ],
        "hosted_models": [
            {
                "requested_model": "deepseek/deepseek-v4-pro",
                "catalog": {"canonical_slug": "deepseek/deepseek-v4-pro-20260423"},
                "endpoints": {"endpoints": [endpoint]},
            }
        ],
    }


class FixtureTransport(Transport):
    def __init__(self):
        self.calls = []
        self.cloud_calls = 0

    def request(self, url, body=None, **kwargs):
        if body is None:
            return record({"data": {"usage": self.cloud_calls * 0.002}})
        self.calls.append((url, body))
        if url.startswith(OLLAMA):
            return record(
                {
                    "model": body["model"],
                    "done": True,
                    "done_reason": "stop",
                    "message": {"content": '{"applicable_cves":[]}'},
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                }
            )
        self.cloud_calls += 1
        if "CANARY_" in json.dumps(body):
            raise AssertionError("fixture canary in hosted request")
        return record(
            {
                "model": body["model"],
                "provider": "StreamLake",
                "id": "generation-fixture",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"prompt":"Check applicability and return applicable_cves JSON."}'
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "completion_tokens_details": {"reasoning_tokens": 5},
                    "cost": 0.002,
                },
            }
        )


class FixtureSmoke(Smoke):
    def verify_identities(self):
        self.start_key_usage = Decimal(0)
        self.store.put(
            f"metadata/{self.session}/key-before.json", {"response": record({"data": {"usage": 0}})}
        )


class LiveContracts(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            self.root = Path(retained) / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            temp = tempfile.TemporaryDirectory()
            self.addCleanup(temp.cleanup)
            self.root = Path(temp.name)
        self.fixture = read_fixture(Path(__file__).resolve().parents[1] / "fixtures/mini.json")

    def test_live_round_resume_reuses_all_committed_requests(self):
        transport = FixtureTransport()
        with patch("promptbench.live.smoke.gpu_snapshot", return_value={"fixture": True}):
            with self.assertRaises(KeyboardInterrupt):
                FixtureSmoke(self.root, registry(), self.fixture, transport, stop_after=2).run()
            self.assertEqual(transport.cloud_calls, 0)
            result = FixtureSmoke(self.root, registry(), self.fixture, transport).run()
            self.assertEqual(len(result["required_work"]), 7)
            self.assertEqual(len(transport.calls), 7)
            self.assertEqual(transport.cloud_calls, 1)
            again = FixtureSmoke(self.root, registry(), self.fixture, transport).run()
            self.assertEqual(result, again)
            self.assertEqual(len(transport.calls), 7)

    def test_changed_live_manifest_cannot_resume(self):
        transport = FixtureTransport()
        with patch("promptbench.live.smoke.gpu_snapshot", return_value={}):
            FixtureSmoke(self.root, registry(), self.fixture, transport).run()
        changed = registry()
        changed["local_models"][0]["ollama_manifest_digest"] = "changed"
        with self.assertRaises(IntegrityError):
            FixtureSmoke(self.root, changed, self.fixture, transport).run()

    def test_cli_thinking_condition_survives_resume_without_repeating_option(self):
        registry_dir = self.root / "registry"
        Store(registry_dir).put("model_registry.json", registry())
        run_dir = self.root / "run"
        fixture = Path(__file__).resolve().parents[1] / "fixtures/mini.json"
        args = [
            "promptbench.live",
            "smoke",
            "--run-dir",
            str(run_dir),
            "--registry-dir",
            str(registry_dir),
            "--fixture",
            str(fixture),
        ]
        transport = FixtureTransport()
        with (
            patch("promptbench.live.__main__.Smoke", FixtureSmoke),
            patch("promptbench.live.__main__.Transport", return_value=transport),
            patch("promptbench.live.__main__.read_key", return_value="fixture"),
            patch("promptbench.live.smoke.gpu_snapshot", return_value={}),
            patch("sys.stdout", new_callable=io.StringIO),
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            with patch("sys.argv", args + ["--local-think", "off", "--stop-after-commits", "2"]):
                self.assertEqual(main(), 130)
            with patch("sys.argv", args):
                self.assertEqual(main(), 0)
            self.assertEqual(len(transport.calls), 7)
            self.assertFalse(Store(run_dir).get("manifest.json")["local_think"])
            for url, body in transport.calls:
                if url.startswith(OLLAMA):
                    self.assertIs(body["think"], False)
                    self.assertEqual(body["options"]["num_predict"], 512)
                else:
                    self.assertNotIn("think", body)
            with patch("sys.argv", args + ["--local-think", "on"]):
                self.assertEqual(main(), 2)
            self.assertEqual(len(transport.calls), 7)

    def test_budget_continuation_cannot_change_thinking_condition(self):
        parent = self.root / "parent"
        with patch("promptbench.live.smoke.gpu_snapshot", return_value={}):
            with self.assertRaises(KeyboardInterrupt):
                FixtureSmoke(
                    parent,
                    registry(),
                    self.fixture,
                    FixtureTransport(),
                    stop_after=1,
                    local_think=False,
                ).run()
        changed = FixtureSmoke(
            self.root / "changed",
            registry(),
            self.fixture,
            FixtureTransport(),
            max_cost_usd="3",
            local_think=True,
        )
        with self.assertRaisesRegex(IntegrityError, "scientific inputs: local_think"):
            changed.prepare_continuation(parent, "budget cannot change reasoning")
        self.assertEqual(changed.store.names("work/*/result.json"), [])

    def test_explicit_thinking_requires_advertised_support(self):
        unsupported = registry()
        unsupported["local_models"][0]["capabilities"] = ["completion"]
        for option in (False, True):
            with self.assertRaisesRegex(ValueError, "advertise thinking"):
                FixtureSmoke(
                    self.root, unsupported, self.fixture, FixtureTransport(), local_think=option
                )

    def test_length_limited_thinking_is_retained_but_never_scored_as_an_answer(self):
        response = record(
            {
                "model": "granite4.2:30b",
                "done": True,
                "done_reason": "length",
                "eval_count": 512,
                "message": {"content": "", "thinking": '{"applicable_cves":[]}'},
            }
        )
        transport = FixtureTransport()
        transport.request = Mock(return_value=response)
        smoke = FixtureSmoke(self.root, registry(), self.fixture, transport)
        smoke.start_key_usage = Decimal(0)
        with patch("promptbench.live.smoke.gpu_snapshot", return_value={}):
            result = smoke.call("baseline/limited", "ollama", {"model": "granite4.2:30b"})
        self.assertEqual(result["status"], "invalid_output")
        self.assertIsNone(result["value"])
        normalized = smoke.store.get(result["attempt"] + "/normalized.json")
        self.assertEqual(normalized["reasoning"], '{"applicable_cves":[]}')
        self.assertEqual(normalized["output_tokens"], 512)

    def test_unknown_cloud_attempt_keeps_its_cost_reservation(self):
        smoke = FixtureSmoke(self.root, registry(), self.fixture, FixtureTransport())
        smoke.start_key_usage = Decimal(0)
        smoke.store.put(
            "work/unknown/attempts/0000/request.json",
            {"provider": "openrouter", "reservation_usd": "0.84"},
        )
        body = optimizer_body("deepseek/deepseek-v4-pro", "streamlake/fp8", {})
        with self.assertRaisesRegex(RunPaused, "USD 1"):
            smoke.call("another", "openrouter", body)
        self.assertEqual(smoke.accounting()["unresolved_reserved_usd"], "0.84")
        self.assertEqual(smoke.transport.calls, [])

    def test_budget_continuation_reuses_local_results_and_preserves_unknown_charge(self):
        parent = self.root / "parent"
        child = self.root / "continued"
        first_transport = FixtureTransport()
        original_request = first_transport.request

        def interrupt_cloud(url, body=None, **kwargs):
            if body is not None and url.startswith(OPENROUTER):
                raise KeyboardInterrupt("simulated lost hosted response")
            return original_request(url, body, **kwargs)

        first_transport.request = interrupt_cloud
        with patch("promptbench.live.smoke.gpu_snapshot", return_value={}):
            with self.assertRaises(KeyboardInterrupt):
                FixtureSmoke(parent, registry(), self.fixture, first_transport).run()
            parent_store = Store(parent)
            original_results = {
                n: (parent / n).read_bytes() for n in parent_store.names("work/*/result.json")
            }
            continuation = FixtureSmoke(
                child, registry(), self.fixture, FixtureTransport(), max_cost_usd="3"
            )
            prepared = continuation.prepare_continuation(
                parent, "Researcher authorized a larger smoke allocation"
            )
            self.assertEqual(prepared["accounting"]["unresolved_reserved_usd"], "0.8323072")
            self.assertEqual(prepared["accounting"]["attempts"], 4)
            transport = FixtureTransport()
            resumed = FixtureSmoke(child, registry(), self.fixture, transport, max_cost_usd="3")
            completed = resumed.run()
            self.assertEqual(len(completed["required_work"]), 7)
            self.assertEqual(len(transport.calls), 4)
            self.assertEqual(resumed.accounting()["attempts"], 8)
            self.assertEqual(resumed.accounting()["unresolved_reserved_usd"], "0.8323072")
            self.assertEqual(parent_store.get("manifest.json")["max_cost_usd"], "1")
            for name, original in original_results.items():
                self.assertEqual((parent / name).read_bytes(), original)
                self.assertEqual((child / name).read_bytes(), original)

    def test_budget_continuation_rejects_changed_scientific_inputs(self):
        parent = self.root / "parent"
        with patch("promptbench.live.smoke.gpu_snapshot", return_value={}):
            with self.assertRaises(KeyboardInterrupt):
                FixtureSmoke(
                    parent, registry(), self.fixture, FixtureTransport(), stop_after=1
                ).run()
        modified = registry()
        modified["local_models"][0]["ollama_manifest_digest"] = "different"
        child = FixtureSmoke(
            self.root / "changed", modified, self.fixture, FixtureTransport(), max_cost_usd="3"
        )
        with self.assertRaisesRegex(IntegrityError, "scientific inputs"):
            child.prepare_continuation(parent, "budget only")
        self.assertEqual(child.store.names("work/*/result.json"), [])

    def test_budget_continuation_rejects_changed_adapter_source(self):
        parent = self.root / "parent"
        with patch("promptbench.live.smoke.gpu_snapshot", return_value={}):
            with self.assertRaises(KeyboardInterrupt):
                FixtureSmoke(
                    parent, registry(), self.fixture, FixtureTransport(), stop_after=1
                ).run()
        from promptbench.live.preflight import sources

        modified = {**sources(), "live/adapters.py": "different response contract"}
        child = FixtureSmoke(
            self.root / "changed", registry(), self.fixture, FixtureTransport(), max_cost_usd="3"
        )
        with patch("promptbench.live.smoke.sources", return_value=modified):
            with self.assertRaisesRegex(IntegrityError, "adapters"):
                child.prepare_continuation(parent, "budget only")

    def test_smoke_cap_cannot_silently_consume_beyond_its_reserve(self):
        for cap in ("0", "-1", "11", "NaN", "Infinity"):
            with self.assertRaises(ValueError):
                FixtureSmoke(
                    self.root, registry(), self.fixture, FixtureTransport(), max_cost_usd=cap
                )

    def test_paid_bound_uses_full_context_and_enforced_price_caps(self):
        body = optimizer_body("deepseek/deepseek-v4-pro", "streamlake/fp8", {})
        endpoint = registry()["hosted_models"][0]["endpoints"]["endpoints"][0]
        self.assertEqual(reservation(body, endpoint), Decimal("0.8323072"))
        self.assertFalse(body["provider"]["allow_fallbacks"])
        self.assertEqual(body["provider"]["only"], ["streamlake/fp8"])
        endpoint["pricing"]["prompt"] = "0.000002"
        with self.assertRaises(ValueError):
            reservation(body, endpoint)

    def test_real_usage_missing_cost_and_reasoning_overlap(self):
        raw = {
            "model": "m",
            "choices": [
                {"finish_reason": "length", "message": {"content": "", "reasoning": "retained"}}
            ],
            "usage": {
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        }
        value = normalize(record(raw), "openrouter")
        self.assertIsNone(value["reported_api_cost_usd"])
        self.assertEqual(value["output_tokens"], 20)
        self.assertEqual(value["reasoning_tokens"], 5)
        self.assertEqual(value["finish_reason"], "length")

    def test_returned_provider_mismatch_stops_before_committing(self):
        transport = FixtureTransport()
        normal = transport.request

        def changed(*args, **kwargs):
            response = normal(*args, **kwargs)
            data = json.loads(response["body_text"])
            data["provider"] = "Wrong Provider"
            return record(data)

        transport.request = changed
        smoke = FixtureSmoke(self.root, registry(), self.fixture, transport)
        smoke.start_key_usage = Decimal(0)
        with patch("promptbench.live.smoke.gpu_snapshot", return_value={}):
            with self.assertRaises(IntegrityError):
                smoke.call(
                    "optimizer/1",
                    "openrouter",
                    optimizer_body("deepseek/deepseek-v4-pro", "streamlake/fp8", {}),
                )
        self.assertEqual(Store(self.root).names("work/*/result.json"), [])
        self.assertEqual(len(Store(self.root).names("work/*/attempts/*/response.json")), 1)

    def test_key_parser_does_not_execute_or_accept_duplicate_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "credential-fixture"
            for content in [
                "OPENROUTER_API_KEY=fixture\n",
                "export OPENROUTER_API_KEY='fixture' # note\n",
            ]:
                path.write_text(content)
                self.assertEqual(read_key(path), "fixture")
            for content in [
                "OPENROUTER_API_KEY=$(echo oops)",
                "OPENROUTER_API_KEY=\n",
                "OPENROUTER_API_KEY=one\nOPENROUTER_API_KEY=two",
                "OPENROUTER_API_KEY='unfinished",
            ]:
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_key(path)

    def test_origin_and_credential_scope_are_enforced_before_network(self) -> None:
        transport = Transport("fixture-secret")
        transport._opener = Mock()
        for url, authenticated in [
            ("https://evil.invalid/api", True),
            (OLLAMA + "/api/tags", True),
            ("https://openrouter.ai.evil.invalid/api", True),
            ("https://someone@openrouter.ai/api", True),
        ]:
            with self.assertRaises(ValueError):
                transport.request(url, authenticated=authenticated)
        transport._opener.open.assert_not_called()
        self.assertIsNone(
            NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.invalid")
        )

    def test_raw_response_retention_redacts_an_echoed_credential(self) -> None:
        transport = Transport("fixture-secret")
        reply = io.BytesIO(b'{"echo":"fixture-secret"}')
        reply.status = 200
        reply.headers = {"Content-Type": "application/json", "Set-Cookie": "private-cookie"}
        transport._opener = Mock()
        transport._opener.open.return_value = reply
        record = transport.request(OPENROUTER + "/api/v1/key", authenticated=True)
        self.assertNotIn("fixture-secret", json.dumps(record))
        self.assertNotIn("private-cookie", json.dumps(record))
        self.assertTrue(record["credential_redaction_applied"])
        self.assertEqual(response_json(record)["echo"], "[REDACTED_CREDENTIAL]")

    def test_progress_counts_commits_not_attempts_and_survives_restart(self):
        store = Store(self.root)
        sizes = {"baseline": 3, "optimizer": 1, "proposed": 3, "report": 1, "preflight": 7}
        store.put("work/a/result.json", {"key": "baseline/a"})
        store.put("work/b/attempts/0000/request.json", {"fixture": "unknown attempt"})
        store.put("work/b/attempts/0001/request.json", {"fixture": "retry"})
        stream = io.StringIO()
        first = Progress(store, sizes, stream=stream)
        first.emit("baseline", "b", "waiting")
        resumed = Progress(store, sizes, stream=stream)
        resumed.emit("baseline", "b", "resuming")
        records = [store.get(n) for n in store.names("progress/*/*.json")]
        self.assertEqual([r["total_percent"] for r in records], [12.5, 12.5])
        self.assertTrue(all(r["task_done"] == 1 and r["task_steps"] == 3 for r in records))
        self.assertIn("TOTAL  12.5% 1/8", stream.getvalue())

    def test_progress_heartbeat_runs_while_request_is_blocked(self):
        ready = threading.Event()

        class SignalStream(io.StringIO):
            lines = 0

            def write(self, value):
                if value.startswith("[TOTAL"):
                    self.lines += 1
                    if self.lines >= 2:
                        ready.set()
                return super().write(value)

        stream = SignalStream()
        progress = Progress(
            Store(self.root), {"optimizer": 1, "report": 1}, stream=stream, interval=0.01
        )
        with progress.waiting("optimizer", "generation"):
            self.assertTrue(ready.wait(2), "no heartbeat arrived during the blocked request")
        rows = [progress.store.get(n) for n in progress.store.names("progress/*/*.json")]
        self.assertTrue(any(r["request_elapsed_seconds"] > 0 for r in rows))
        self.assertTrue(all(r["total_percent"] == 0 for r in rows))

    def test_remote_alias_cannot_be_a_local_executor(self) -> None:
        with self.assertRaises(ValueError):
            local_identity({"remote_host": "https://ollama.com"}, {}, "fixture-version")

    def test_local_identity_changes_with_template_parameters_and_digest(self) -> None:
        tag = {"name": "fixture:1", "digest": "one", "size": 123}
        details = {"template": "first", "parameters": "temperature 0"}
        original = local_identity(tag, details, "1")
        self.assertNotEqual(original, local_identity({**tag, "digest": "two"}, details, "1"))
        self.assertNotEqual(
            original["template_sha256"],
            local_identity(tag, {**details, "template": "second"}, "1")["template_sha256"],
        )
        self.assertNotEqual(
            original["parameters_sha256"],
            local_identity(tag, {**details, "parameters": "temperature 1"}, "1")[
                "parameters_sha256"
            ],
        )
