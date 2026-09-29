from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from promptbench.live.calibration_recovery import adopt_local_calibration
from promptbench.storage import IntegrityError, Store, atomic_write


class CalibrationRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.parent, self.child = Store(root / "parent"), Store(root / "child")
        manifest = dict.fromkeys(
            (
                "profile_ids",
                "profile_sha256",
                "local",
                "hosted",
                "gate",
                "local_calls",
                "hosted_calls",
                "total_calls",
            ),
            "unchanged",
        )
        for store in (self.parent, self.child):
            store.put("manifest.json", manifest)
            store.put("inputs/prompts.json", {"naive": "fixed"})
            store.put(
                "inputs/source.json",
                {
                    "live/calibrate.py": "def execute(x):\n    return x\n",
                    "benchmark/oracle.py": "immutable evaluator",
                },
            )
        self.parent.put("operator-exit.json", {"exit_code": -2})
        self.parent.put("sessions/old/000001.json", {"elapsed_seconds": 50})
        self.parent.put("work/local/attempts/0000/request.json", {"provider": "ollama"})
        self.parent.put("work/local/result.json", {"valid": True})
        atomic_write(self.parent.root / "work/local/attempts/0000/stderr.log", b"raw evidence")

    def test_adoption_preserves_parent_and_raw_evidence_without_copying_clocks(self):
        before = {str(p): p.read_bytes() for p in self.parent.root.rglob("*") if p.is_file()}
        result = adopt_local_calibration(self.parent.root, self.child.root)
        self.assertEqual(result["adopted_commits"], 1)
        self.assertFalse((self.child.root / "sessions").exists())
        self.assertEqual(self.child.get("work/local/result.json"), {"valid": True})
        self.assertEqual(
            (self.child.root / "work/local/attempts/0000/stderr.log").read_bytes(),
            b"raw evidence",
        )
        self.assertEqual(
            before, {str(p): p.read_bytes() for p in self.parent.root.rglob("*") if p.is_file()}
        )

    def test_rejects_scientific_input_and_runtime_changes_before_copying(self):
        for name, value in (
            ("inputs/prompts.json", {"naive": "different"}),
            (
                "inputs/source.json",
                {
                    "live/calibrate.py": "def execute(x):\n    return 0\n",
                    "benchmark/oracle.py": "immutable evaluator",
                },
            ),
        ):
            original = self.child.get(name)
            self.child.put(name, value, immutable=False)
            with self.assertRaises(IntegrityError):
                adopt_local_calibration(self.parent.root, self.child.root)
            self.assertFalse((self.child.root / "work").exists())
            self.child.put(name, original, immutable=False)

    def test_rejects_hosted_attempts_and_completed_runs(self):
        self.parent.put("work/hosted/attempts/0000/request.json", {"provider": "openrouter"})
        with self.assertRaisesRegex(IntegrityError, "local-only"):
            adopt_local_calibration(self.parent.root, self.child.root)
        self.parent.put("reports/complete.json", {"passed": False})
        with self.assertRaisesRegex(IntegrityError, "stopped incomplete"):
            adopt_local_calibration(self.parent.root, self.child.root)
