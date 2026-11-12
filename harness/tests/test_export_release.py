import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import json  # noqa: E402
import tempfile  # noqa: E402

from export_release import split_key, update, work_summary  # noqa: E402


class SplitKeyTest(unittest.TestCase):
    def test_every_key_shape_keeps_the_executor_mode(self):
        cases = {
            "main/qwen3.8:27b/off/T1/R1/r6/incumbent/optimization_0026": (
                "main",
                "qwen3.8:27b/off",
                "1",
                "1",
                "r6/incumbent",
                "",
            ),
            "gepa/qwen3.8:27b/off/T2/R2/e00316/optimization_0107": (
                "gepa",
                "qwen3.8:27b/off",
                "2",
                "2",
                "e00316",
                "",
            ),
            "reference/qwen3.8:27b/off/T2/R3/r1/candidate0/optimization_0100": (
                "reference",
                "qwen3.8:27b/off",
                "2",
                "3",
                "r1/candidate0",
                "",
            ),
            "sealed/granite4.2:30b/off/temporal/cb10/temporal_0049": (
                "sealed",
                "granite4.2:30b/off",
                "",
                "",
                "",
                "temporal",
            ),
            "selection/granite4.2:30b/off/3267/validation_0089": (
                "selection",
                "granite4.2:30b/off",
                "",
                "",
                "",
                "validation",
            ),
            "ceiling/z-ai/glm-5.3/test/test_0042": ("ceiling", "z-ai/glm-5.3", "", "", "", "test"),
            "ceiling/rule-baseline/temporal/temporal_0061": (
                "ceiling",
                "rule-baseline",
                "",
                "",
                "",
                "temporal",
            ),
            "calibration/gemma4:31b/on/historical_expert/pilot_development_0002": (
                "calibration",
                "gemma4:31b/on",
                "",
                "",
                "historical_expert",
                "",
            ),
            "scaffolding/qwen3.8:27b/off/neutral/test_0031": (
                "scaffolding",
                "qwen3.8:27b/off",
                "",
                "",
                "neutral",
                "",
            ),
            "real_record/granite4.2:30b/off/output_scaffold/db11/real_04_1": (
                "real_record",
                "granite4.2:30b/off/output_scaffold",
                "",
                "",
                "",
                "real_record",
            ),
        }
        for key, expected in cases.items():
            with self.subTest(key=key):
                self.assertEqual(split_key(key), expected)


def put(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"payload": payload, "sha256": "x"}))


class UpdateTest(unittest.TestCase):
    def test_update_adds_a_followup_with_its_work_summary_and_rebuilds_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, out = Path(tmp) / "run-20261028T000000Z", Path(tmp) / "data"
            key = "ceiling/qwen3.8:27b/off/test/test_0001"
            put(run / "work/s1/result.json", {"key": key, "attempt": "work/s1/attempts/0001"})
            put(run / "work/s1/attempts/0000/worker-outcome.json", {"duration_ns": 2_000_000_000})
            put(run / "work/s1/attempts/0001/worker-outcome.json", {"duration_ns": 1_000_000_000})
            put(
                run / "work/s1/attempts/0001/normalized.json",
                {"input_tokens": 900, "output_tokens": 40, "raw_usage": {"load_duration": 5e8}},
            )
            row = {"profile_id": "test_0001", "status": "valid", "valid": True}
            row |= {"tp": 1, "fp": 0, "fn": 0, "prediction": ["CVE-1"], "expected": ["CVE-1"]}
            put(run / "evaluations/e1.json", {"key": key, "row": row})
            out.mkdir()
            (out / "INDEX.json").write_text(json.dumps({"sections": {"benchmark": {}}}))
            update(out, json.loads((out / "INDEX.json").read_text()), [f"verify={run}"])
            summary = json.loads((out / "followups/verify/work-summary.jsonl").read_text())
            self.assertEqual(summary["key"], key)
            self.assertEqual((summary["attempts"], summary["worker_seconds"]), (2, 3.0))
            self.assertEqual((summary["model_load_seconds"], summary["input_tokens"]), (0.5, 900))
            index = json.loads((out / "INDEX.json").read_text())
            self.assertIn("followups/verify", index["sections"])
            self.assertEqual(index["exports"]["evaluations.csv"], 1)
            with self.assertRaises(SystemExit):  # a section is never overwritten
                update(out, index, [f"verify={run}"])
            self.assertEqual(work_summary(Path(tmp) / "none", out), 0)


if __name__ == "__main__":
    unittest.main()
