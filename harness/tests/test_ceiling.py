from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from promptbench.live.campaign import Campaign
from promptbench.live.ceiling import ceiling_request, execute
from promptbench.live.conditions import NAIVE, hosted_conditions
from promptbench.live.execution import Execution
from promptbench.storage import Store, digest
from tests.test_study import conditions, fixture_profiles


class CeilingTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            self.root = Path(retained) / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            temp = tempfile.TemporaryDirectory()
            self.addCleanup(temp.cleanup)
            self.root = Path(temp.name)
        self.profiles, self.advisory = fixture_profiles()
        self.calls = 0

    def hosted(self, supports_temperature: bool):
        cond = conditions()["hosted"][0]
        cond["endpoint"]["supported_parameters"] = ["seed"] + (
            ["temperature"] if supports_temperature else []
        )
        return cond

    def test_request_sets_temperature_only_where_supported(self):
        profile = self.profiles[0]
        admitted, body = ceiling_request(self.hosted(True), NAIVE, profile)
        self.assertEqual(body["temperature"], 0)
        self.assertTrue(admitted["temperature_supported"])
        self.assertEqual(body["response_format"]["json_schema"]["name"], "applicable_cves")
        self.assertEqual(body["messages"][0]["content"], NAIVE)
        admitted, body = ceiling_request(self.hosted(False), NAIVE, profile)
        self.assertNotIn("temperature", body)
        self.assertFalse(admitted["temperature_supported"])
        self.assertNotIn("seed", body)  # ceiling row sends no sampling seed

    def test_execute_evaluates_every_panel_per_model_and_resumes_without_inference(self):
        store = Store(self.root / "ceiling")
        store.put("plan.json", {"kind": "hosted_ceiling"})
        store.put("inputs/profiles.json", [p.record() for p in self.profiles])
        store.put("inputs/advisories.json", {self.advisory.id: self.advisory.record()})
        cond = conditions()
        store.put("inputs/conditions.json", cond)
        models = [h["id"] for h in cond["hosted"]]
        panels = {
            panel: [p.id for p in self.profiles if p.partition == panel]
            for panel in ("test", "temporal", "product_heldout")
        }
        store.put(
            "manifest.json",
            {
                "models": models,
                "conditions": {m: next(h for h in cond["hosted"] if h["id"] == m) for m in models},
                "panels": panels,
                "initial_step_estimate": {"ceiling": 18, "report": 1},
            },
        )

        def worker(s, attempt, env_file, **kwargs):
            self.calls += 1
            request = s.get(attempt + "/request.json")
            text = json.loads(request["body"]["messages"][1]["content"])["system_profile"]
            answer = [self.advisory.id] if text == "version 1" else []
            body = {
                "model": request["body"]["model"],
                "provider": "fixture-provider",
                "id": f"gen-{self.calls}",
                "choices": [
                    {
                        "message": {"content": json.dumps({"applicable_cves": answer})},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"cost": 0.01},
            }
            s.put(
                attempt + "/response.json",
                {
                    "http_status": 200,
                    "error": None,
                    "headers": {},
                    "body_text": json.dumps(body),
                    "duration_ns": 1_000_000,
                },
            )

        def run_once():
            engine = Execution(
                store,
                self.root / "no-key",
                max_seconds=3600,
                max_cost_usd="8",
                max_attempts=1000,
                study_root=self.root,
                worker=worker,
            )
            with patch.object(Campaign, "check"):
                campaign_key = {"data": {"usage": 0.01}}
                result = None
                original = Campaign.__init__

                def init(self_, execution):
                    original(self_, execution)
                    execution.key_reader = lambda: campaign_key

                with patch.object(Campaign, "__init__", init):
                    result = execute(engine)
            return result

        result = run_once()
        self.assertEqual(self.calls, 18)  # 2 models × (3 + 3 + 3) profiles
        for model in models:
            for panel in panels:
                summary = result["panels"][model][panel]
                self.assertEqual(summary["metrics"]["failure_aware_lower_bound"]["micro_f1"], 1.0)
                self.assertEqual(summary["metrics"]["coverage"], 1.0)
        self.assertEqual(len(store.names("generation-pending/*.json")), 18)
        self.assertEqual(len(store.names("panels/*.json")), 6)
        again = run_once()
        self.assertEqual(self.calls, 18)
        self.assertEqual(again["panels"], result["panels"])
        for name in store.names("evaluations/*.json"):
            self.assertEqual(store.get(name)["prompt_sha256"], digest(NAIVE))

    def test_local_condition_uses_executor_request_at_campaign_cap(self):
        store = Store(self.root / "local-ceiling")
        store.put("plan.json", {"kind": "hosted_ceiling"})
        store.put("inputs/profiles.json", [p.record() for p in self.profiles])
        store.put("inputs/advisories.json", {self.advisory.id: self.advisory.record()})
        cond = conditions()
        store.put("inputs/conditions.json", cond)
        local = {**cond["local"][0], "num_predict": 2048, "timeout_seconds": 300}
        panels = {"test": [p.id for p in self.profiles if p.partition == "test"]}
        store.put(
            "manifest.json",
            {
                "models": [local["id"]],
                "conditions": {local["id"]: local},
                "panels": panels,
                "initial_step_estimate": {"ceiling": 3, "report": 1},
            },
        )
        seen = []

        def worker(s, attempt, env_file, **kwargs):
            request = s.get(attempt + "/request.json")
            seen.append(request["body"])
            s.put(
                attempt + "/response.json",
                {
                    "http_status": 200,
                    "error": None,
                    "headers": {},
                    "body_text": json.dumps(
                        {
                            "model": request["body"]["model"],
                            "done": True,
                            "done_reason": "stop",
                            "message": {"content": json.dumps({"applicable_cves": []})},
                            "eval_count": 5,
                        }
                    ),
                    "duration_ns": 1_000_000,
                },
            )

        engine = Execution(
            store,
            self.root / "no-key",
            max_seconds=3600,
            max_cost_usd="1",
            max_attempts=100,
            study_root=self.root,
            worker=worker,
        )
        original = Campaign.__init__

        def init(self_, execution):
            original(self_, execution)
            execution.key_reader = lambda: {"data": {"usage": 0.0}}

        with patch.object(Campaign, "check"), patch.object(Campaign, "__init__", init):
            result = execute(engine)
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[0]["options"]["num_predict"], 2048)
        self.assertEqual(seen[0]["options"]["temperature"], 0)
        self.assertEqual(seen[0]["messages"][0]["content"], NAIVE)
        self.assertEqual(len(store.names("generation-pending/*.json")), 0)
        self.assertIn("test", result["panels"][local["id"]])


class EndpointOverrideTests(unittest.TestCase):
    def registry(self):
        endpoint = lambda tag, params: {  # noqa: E731
            "tag": tag,
            "provider_name": tag.split("/")[0],
            "context_length": 1000,
            "quantization": "fp8",
            "pricing": {"prompt": ".000001", "completion": ".000002"},
            "supported_parameters": params,
        }
        full = ["response_format", "structured_outputs", "reasoning", "temperature"]
        return {
            "hosted_models": [
                {
                    "requested_model": "z-ai/glm-5.3",
                    "catalog": {"canonical_slug": "z-ai/glm-5.3-dated"},
                    "endpoints": {
                        "endpoints": [
                            endpoint("reka/fp8", full),
                            endpoint("akashml/fp8", full),
                            endpoint("novita/fp8", ["response_format", "reasoning"]),
                        ]
                    },
                }
            ]
        }

    def test_override_selects_alternative_endpoint_and_rejects_incapable_ones(self):
        default = hosted_conditions(self.registry())[0]
        self.assertEqual(default["endpoint"]["tag"], "reka/fp8")
        alt = hosted_conditions(self.registry(), {"z-ai/glm-5.3": "akashml/fp8"})[0]
        self.assertEqual(alt["endpoint"]["tag"], "akashml/fp8")
        self.assertEqual(alt["expected_provider"], "akashml")
        with self.assertRaises(ValueError):
            hosted_conditions(self.registry(), {"z-ai/glm-5.3": "novita/fp8"})
        with self.assertRaises(ValueError):
            hosted_conditions(self.registry(), {"z-ai/glm-5.3": "nonexistent"})


if __name__ == "__main__":
    unittest.main()
