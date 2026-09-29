from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from promptbench.benchmark.model import BenchmarkProfile
from promptbench.benchmark.oracle import Criterion, Installed, PublicAdvisory
from promptbench.live.campaign import Campaign
from promptbench.live.conditions import NAIVE
from promptbench.live.execution import Execution
from promptbench.live.pilot import Pilot, run
from promptbench.live.reconstruct import reconstruct
from promptbench.live.scheduling import allocate, main_forecast, pareto_and_promote
from promptbench.storage import IntegrityError, Store


class PilotTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            self.root = Path(retained) / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            temp = tempfile.TemporaryDirectory()
            self.addCleanup(temp.cleanup)
            self.root = Path(temp.name)

    def test_allocation_preserves_seeds_and_censors_impossible_schedule(self):
        fast = allocate(14000, [2, 3], 20, available_usd=7, mean_hosted_cost=0.01)
        self.assertEqual(fast["status"], "admitted")
        self.assertEqual(fast["selected"]["seeds"], [11, 29])
        slow = allocate(14000, [80, 80], 90, available_usd=7, mean_hosted_cost=0.01)
        self.assertIsNone(slow["selected"])
        self.assertTrue(all(c["seeds"] == [11, 29] for c in slow["alternatives"]))
        self.assertFalse(
            main_forecast([120, 120], 120, 0.1, 40)["alternatives"][-1]["fits_72h_and_funds"]
        )
        Store(self.root).put("forecasts.json", {"fast": fast, "slow": slow})

    def test_completion_cannot_precede_failed_telemetry_closeout(self):
        root = self.root / "campaign"
        plan = {
            "source_quality": {"status": "admitted_after_source_review"},
            "max_seconds": 21600,
            "max_cost_usd": "8",
            "max_attempts": 10000,
            "initial_step_estimate": {"report": 1},
        }
        with (
            patch("promptbench.live.pilot.verify_runtime", return_value=plan),
            patch("promptbench.live.pilot.Pilot") as controller,
            patch("promptbench.live.pilot.Telemetry") as monitor,
        ):
            controller.return_value.execute.return_value = {"status": "fixture_complete"}
            monitor.return_value.stop.side_effect = IntegrityError(
                "fixture disk write failed at closeout"
            )
            with self.assertRaisesRegex(IntegrityError, "closeout"):
                run(root, self.root / "unused.env")
        store = Store(root)
        self.assertFalse(store.exists("reports/complete.json"))
        self.assertEqual(len(store.names("pauses/*.json")), 1)

    def test_promotion_requires_reliability_and_distinct_families(self):
        def row(name, family, f1, coverage, latency):
            return {
                "condition_id": name,
                "family": family,
                "metrics": {
                    "coverage": coverage,
                    "failure_aware_lower_bound": {"micro_f1": f1, "recall": f1},
                },
                "latency_p90_seconds": latency,
            }

        result = pareto_and_promote(
            [
                row("a-off", "a", 0.9, 1, 2),
                row("a-on", "a", 0.99, 1, 3),
                row("b", "b", 0.8, 1, 2),
                row("c", "c", 1, 0.5, 1),
            ]
        )
        self.assertEqual(result["selected"], ["a-on", "b"])
        self.assertNotIn("c", result["ranked"])

    def test_fake_pilot_pairs_shortlists_and_resume_adds_no_inference(self):
        store = Store(self.root / "campaign")
        store.put("plan.json", {"max_seconds": 21600})
        store.put("inputs/prompts.json", {"naive": NAIVE, "historical_expert": "fixture expert"})
        profiles = []
        term = Criterion("vendor", "product", "1", None, None, False, False, "fixture", None)
        advisory = PublicAdvisory(
            "CVE-2099-1001", "fictional", "2024-01-01", "2024-01-01", (term,), "fixture"
        )
        for partition, size in [("pilot_development", 72), ("pilot_validation", 48)]:
            for index in range(size):
                positive = index % 2 == 1
                profiles.append(
                    BenchmarkProfile(
                        f"{partition}_{index}",
                        partition,
                        "version " + ("1" if positive else "2"),
                        (Installed("vendor", "product", "1" if positive else "2"),),
                        (advisory,),
                        (advisory.id,) if positive else (),
                        ("easy", "medium", "hard")[(index // 2) % 3],
                        advisory.id,
                    )
                )
        store.put("inputs/profiles.json", [p.record() for p in profiles])
        store.put("inputs/advisories.json", {advisory.id: advisory.record()})
        local = [
            {
                "id": f"family{i}/off",
                "family": f"family{i}",
                "provider": "ollama",
                "url": "http://127.0.0.1:11434/api/chat",
                "model": f"family{i}",
                "expected_models": [f"family{i}"],
                "expected_provider": "local-ollama",
                "think": False,
                "num_ctx": 16384,
                "num_predict": 512,
            }
            for i in range(2)
        ]
        hosted = [
            {
                "id": model,
                "model": model,
                "canonical_model": model + "-dated",
                "provider": "openrouter",
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "expected_models": [model, model + "-dated"],
                "expected_provider": "fixture-provider",
                "endpoint": {
                    "tag": "fixture",
                    "context_length": 1000,
                    "pricing": {"prompt": ".000001", "completion": ".000002"},
                },
            }
            for model in ("deepseek/deepseek-v4-pro", "z-ai/glm-5.3")
        ]
        store.put("inputs/conditions.json", {"local": local, "hosted": hosted})
        calls = 0

        def worker(s, attempt, env_file, **kwargs):
            nonlocal calls
            calls += 1
            request = s.get(attempt + "/request.json")
            if request["provider"] == "ollama":
                input_data = json.loads(request["body"]["messages"][1]["content"])
                answer = [advisory.id] if input_data["system_profile"] == "version 1" else []
                body = {
                    "model": request["body"]["model"],
                    "done": True,
                    "done_reason": "stop",
                    "message": {"content": json.dumps({"applicable_cves": answer})},
                }
            else:
                body = {
                    "model": request["body"]["model"],
                    "provider": "fixture-provider",
                    "id": f"gen-fixture-{calls}",
                    "choices": [
                        {
                            "message": {"content": '{"prompt":"fixture candidate"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"cost": 0.001},
                }
            s.put(
                attempt + "/response.json",
                {
                    "http_status": 200,
                    "error": None,
                    "headers": {},
                    "body_text": json.dumps(body),
                    "duration_ns": 1000000,
                },
            )

        engine = Execution(
            store,
            self.root / "no-key",
            max_seconds=21600,
            max_cost_usd="8",
            max_attempts=2000,
            study_root=self.root,
            worker=worker,
        )
        pilot = Pilot(engine)
        engine.key_reader = lambda: {"data": {"usage": 0.01}}
        design = {
            "status": "admitted",
            "selected": {
                "iterations": 1,
                "batch_size": 6,
                "validation_size": 6,
                "seeds": [11, 29],
                "candidate_slots": 2,
                "conservative_seconds": 600,
            },
        }
        with (
            patch.object(Campaign, "check"),
            patch.object(Campaign, "reconcile_generation"),
            patch("promptbench.live.pilot.allocate", return_value=design),
        ):
            result = pilot.execute()
            before = calls
            repeated = pilot.execute()
        self.assertEqual(calls, before)
        self.assertEqual(result, repeated)
        self.assertEqual(len(result["cells"]), 4)
        self.assertEqual(len(store.names("trajectories/*.json")), 10)
        self.assertEqual(len(result["fresh_confirmations"]), 2)
        for name in store.names("trajectories/*.json"):
            trajectory = store.get(name)
            self.assertEqual(trajectory["design"]["seeds"], [trajectory["seed"]])
            if trajectory["fresh_confirmation"]:
                self.assertEqual(trajectory["design"]["seeds"], [47])
            self.assertEqual(len(trajectory["shortlist"]), 2)
            self.assertEqual(trajectory["selected"]["index"], 0)
            self.assertEqual(trajectory["trace"][1]["slot"], -1)
        store.put(
            "test-outcome.json", {"fake_calls": calls, "resume_new_inference": 0, "result": result}
        )
        audited = reconstruct(store.root, self.root / "reconstruction")
        self.assertEqual(audited["physical_attempts"], calls)
        self.assertEqual(audited["unknown_attempts"], 0)
        name = store.names("evaluations/*.json")[0]
        altered = store.get(name)
        altered["row"]["tp"] += 1
        store.put(name, altered, immutable=False)
        with self.assertRaisesRegex(IntegrityError, "stored score"):
            reconstruct(store.root, self.root / "corruption-audit")
