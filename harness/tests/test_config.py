from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from promptbench import config


class ConfigTests(unittest.TestCase):
    def test_default_file_loads_and_matches_exported_constants(self):
        data = config.load()
        self.assertEqual(tuple(data["local"]["tags"]), config.LOCAL_MODELS)
        self.assertEqual(dict(data["hosted"]["endpoints"]), config.PREFERRED_ENDPOINTS)
        self.assertEqual(set(config.PREFERRED_ENDPOINTS), set(config.HOSTED_MODELS))
        self.assertTrue(config.OLLAMA.startswith("http"))
        self.assertIn(config.CALIBRATION_REFERENCE, config.HOSTED_MODELS)
        self.assertTrue(set(config.ABLATION_EXECUTORS) <= set(config.CALIBRATION_CONDITIONS))

    def test_missing_section_or_unpinned_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "models.toml"
            path.write_text('[hosts]\nollama="x"\nopenrouter="y"\n')
            with self.assertRaises(ValueError):
                config.load(path)
            path.write_text(
                '[hosts]\nollama="x"\nopenrouter="y"\n[local]\ntags=["a"]\n'
                '[hosted]\nmodels=["m1","m2"]\n[hosted.endpoints]\n"m1"="e1"\n'
                '[calibration]\nconditions=["a/off"]\nreference="m1"\n[ablation]\nexecutors=["a/off"]\n'
            )
            with self.assertRaises(ValueError):
                config.load(path)


if __name__ == "__main__":
    unittest.main()
