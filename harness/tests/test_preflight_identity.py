from __future__ import annotations

import unittest

from promptbench.live.preflight import local_identity

TAG = {"name": "fixture:1", "digest": "d", "size": 1, "modified_at": "t"}
MODELFILE = "FROM blob\nTEMPLATE x\nPARAMETER top_k 20\nPARAMETER temperature 1\nLICENSE y"


class IdentityTests(unittest.TestCase):
    """`ollama show` parameter order must not change a model's identity."""

    def show(self, modelfile: str, parameters: str) -> dict:
        return {"modelfile": modelfile, "parameters": parameters, "template": "x"}

    def test_parameter_order_does_not_change_identity(self):
        a = local_identity(TAG, self.show(MODELFILE, "top_k 20\ntemperature 1"), "0.32.13")
        swapped = MODELFILE.replace(
            "PARAMETER top_k 20\nPARAMETER temperature 1",
            "PARAMETER temperature 1\nPARAMETER top_k 20",
        )
        b = local_identity(TAG, self.show(swapped, "temperature 1\ntop_k 20"), "0.32.13")
        self.assertEqual(a, b)

    def test_parameter_value_and_template_still_change_identity(self):
        a = local_identity(TAG, self.show(MODELFILE, "top_k 20"), "0.32.13")
        value = local_identity(
            TAG, self.show(MODELFILE.replace("top_k 20", "top_k 40"), "top_k 40"), "0.32.13"
        )
        template = local_identity(
            TAG, self.show(MODELFILE.replace("TEMPLATE x", "TEMPLATE z"), "top_k 20"), "0.32.13"
        )
        self.assertNotEqual(a, value)
        self.assertNotEqual(a["metadata_sha256"], template["metadata_sha256"])


if __name__ == "__main__":
    unittest.main()
