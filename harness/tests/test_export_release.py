import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_release import split_key  # noqa: E402


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
        }
        for key, expected in cases.items():
            with self.subTest(key=key):
                self.assertEqual(split_key(key), expected)


if __name__ == "__main__":
    unittest.main()
