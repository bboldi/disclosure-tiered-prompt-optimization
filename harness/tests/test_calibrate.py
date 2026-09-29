from __future__ import annotations

import copy
import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from promptbench.live.calibrate import GATE, LOCAL_IDS, calibration_gate, reference_request
from promptbench.live.campaign import prepare
from promptbench.live.taskbudget import INITIAL_GPU_SECONDS, SCOPE, ScopedExecution, scope_usage
from promptbench.runner import RunPaused
from promptbench.storage import Store
from tests import test_conditions as fixtures


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get("PROMPTBENCH_TEST_ARTIFACT_ROOT")
        if retained:
            self.root = Path(retained) / self._testMethodName
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            self.root = Path(tmp.name)

    def summaries(self):
        local = {}
        for i, identifier in enumerate(LOCAL_IDS):
            row = {
                "rows": [{}] * 48,
                "metrics": {"failure_aware_lower_bound": {"micro_f1": 0.5}, "coverage": 1},
                "condition_id": identifier,
                "family": identifier.split(":")[0],
                "latency_p90_seconds": i + 1,
                "nonzero_batch_fraction": 1,
            }
            local[identifier] = {"naive": row, "historical_expert": copy.deepcopy(row)}
        hosted = {"rows": [{}] * 48, "metrics": {"failure_aware_lower_bound": {"micro_f1": 0.90}}}
        return local, hosted

    def test_calibration_preparation_loads_only_pilot_development(self):
        benchmark = self.root / "benchmark"
        Store(benchmark).put("manifest.json", {"source_quality": {"status": "fixture"}})
        with (
            patch("promptbench.live.campaign.load", return_value=[]) as loader,
            patch("promptbench.live.campaign.sources", return_value={}),
            patch(
                "promptbench.live.campaign.save_registry",
                return_value={"local_models": [], "hosted_models": []},
            ),
        ):
            prepare(
                self.root / "prepared",
                benchmark,
                self.root / "registry",
                {"kind": "calibration", "input_partitions": ["pilot_development"]},
            )
        loader.assert_called_once_with(benchmark, {"pilot_development"})

    def test_frozen_runtime_is_self_contained_and_importable(self):
        """`resume.py` imports the package from `runtime/`; the config must travel with it."""
        import subprocess
        import sys

        from promptbench.live.preflight import sources

        benchmark = self.root / "benchmark"
        Store(benchmark).put("manifest.json", {"source_quality": {"status": "fixture"}})
        with (
            patch("promptbench.live.campaign.load", return_value=[]),
            patch(
                "promptbench.live.campaign.save_registry",
                return_value={"local_models": [], "hosted_models": []},
            ),
        ):
            prepare(self.root / "frozen", benchmark, self.root / "registry", {"kind": "x"})
        runtime = self.root / "frozen" / "runtime"
        self.assertTrue((runtime / "models.toml").exists())
        self.assertEqual(
            set(sources()),
            {str(q.relative_to(runtime / "promptbench")) for q in runtime.rglob("*.py")},
        )
        probe = subprocess.run(
            [sys.executable, "-c", "import promptbench.live.study as s; print(s.__file__)"],
            cwd=runtime,
            capture_output=True,
            text=True,
            env={k: v for k, v in os.environ.items() if k != "PROMPTBENCH_CONFIG"},
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        self.assertTrue(probe.stdout.strip().startswith(str(runtime)))

    def test_gate_requires_complete_screen_two_families_and_hosted_decidability(self):
        local, hosted = self.summaries()
        passed = calibration_gate(local, hosted)
        self.assertTrue(passed["passed"])
        self.assertEqual(len({x.split(":")[0] for x in passed["promoted"]}), 2)
        hosted["metrics"]["failure_aware_lower_bound"]["micro_f1"] = 0.899
        self.assertFalse(calibration_gate(local, hosted)["passed"])
        local, hosted = self.summaries()
        local.pop(LOCAL_IDS[-1])
        self.assertFalse(calibration_gate(local, hosted)["passed"])
        local, hosted = self.summaries()
        for identifier, prompts in local.items():
            if not identifier.startswith("qwen"):
                prompts["naive"]["metrics"]["failure_aware_lower_bound"]["micro_f1"] = 0.71
        self.assertFalse(calibration_gate(local, hosted)["passed"])
        Store(self.root).put("gate-evidence.json", {"criteria": GATE, "passing_fixture": passed})

    def test_gate_boundaries_do_not_relax_for_coverage_or_zero_feedback(self):
        for f1, coverage, nonzero, eligible in [
            (0.3, 0.95, 0.8, True),
            (0.7, 1, 1, True),
            (0.2999, 1, 1, False),
            (0.7001, 1, 1, False),
            (0.5, 0.949, 1, False),
            (0.5, 1, 0.799, False),
        ]:
            local, hosted = self.summaries()
            cell = local[LOCAL_IDS[0]]["naive"]
            cell["metrics"]["failure_aware_lower_bound"]["micro_f1"] = f1
            cell["metrics"]["coverage"] = coverage
            cell["nonzero_batch_fraction"] = nonzero
            self.assertEqual(
                LOCAL_IDS[0] in calibration_gate(local, hosted)["eligible_conditions"], eligible
            )
        local, hosted = self.summaries()
        local[LOCAL_IDS[0]]["historical_expert"]["metrics"]["coverage"] = 0.9
        self.assertNotIn(LOCAL_IDS[0], calibration_gate(local, hosted)["eligible_conditions"])

    def test_hosted_executor_uses_matching_text_without_inventory_or_id_scaffolds(self):
        condition = {
            "id": "deepseek",
            "model": "deepseek",
            "endpoint": {
                "tag": "streamlake/fp8",
                "context_length": 32768,
                "pricing": {"prompt": ".000001", "completion": ".000002"},
            },
        }
        profile = fixtures.ConditionTests().profile()
        admitted, body = reference_request(condition, "naive", profile)
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(body["temperature"], 0)
        self.assertNotIn("components", body["messages"][1]["content"])
        self.assertNotIn(
            "enum",
            body["response_format"]["json_schema"]["schema"]["properties"]["applicable_cves"][
                "items"
            ],
        )
        self.assertNotIn("secret", str(body))
        self.assertFalse(body["provider"]["allow_fallbacks"])
        self.assertTrue(body["provider"]["require_parameters"])
        self.assertGreater(Decimal(admitted["reservation_usd"]), 0)
        Store(self.root).put("request.json", body)

    def test_scope_caps_count_other_runs_unknown_requests_and_initial_telemetry(self):
        older = Store(self.root / "calibration-a")
        older.put("plan.json", {"budget_scope": SCOPE, "kind": "calibration"})
        older.put("sessions/a/000001.json", {"elapsed_seconds": 50, "pending_timeout_seconds": 20})
        older.put("resource-adjustments/cleanup.json", {"charged_seconds": 15})
        older.put(
            "work/old/attempts/0000/request.json",
            {"provider": "openrouter", "reservation_usd": "1.5"},
        )
        usage = scope_usage(self.root)
        self.assertEqual(usage["charged_gpu_seconds"], INITIAL_GPU_SECONDS + 85)
        self.assertEqual(usage["reserved_unknown_usd"], "1.5")
        store = Store(self.root / "calibration-b")
        store.put(
            "plan.json",
            {"budget_scope": SCOPE, "kind": "calibration", "scope_gpu_ceiling_seconds": 14400},
        )
        engine = ScopedExecution(
            store,
            self.root / "unused",
            max_seconds=5400,
            max_cost_usd="2",
            max_attempts=10,
            study_root=self.root,
        )
        engine.key_reader = lambda: {"data": {"usage": ".25388512"}}
        with self.assertRaisesRegex(RunPaused, "combined calibration"):
            engine.admission("openrouter", Decimal(".6"))
        # The dedicated-key total includes the original pilot, not only this scope's delta.
        engine.key_reader = lambda: {"data": {"usage": "19.99"}}
        with patch(
            "promptbench.live.taskbudget.scope_usage",
            return_value={**usage, "reserved_unknown_usd": "0"},
        ):
            with self.assertRaisesRegex(RunPaused, "USD 20 scope cap"):
                engine.admission("openrouter", Decimal(".02"))
        with patch(
            "promptbench.live.taskbudget.scope_usage",
            return_value={
                "charged_gpu_seconds": 14399,
                "reported_cost_usd": "0",
                "reserved_unknown_usd": "0",
            },
        ):
            self.assertLessEqual(engine.remaining_seconds(), 1)
            with self.assertRaises(RunPaused):
                engine.admission("ollama", Decimal(0))
