"""Published CMA input-shape and vocabulary compatibility checks."""

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from isaac_r2r.cma_inputs import audit_cma_split, prepare_rgbd, validate_instruction_tokens


class CmaInputsTest(unittest.TestCase):
    def test_prepares_published_rgb_and_normalized_depth_shapes(self):
        output = io.BytesIO()
        Image.new("RGB", (256, 256), (12, 34, 56)).save(output, format="JPEG")
        metric_depth = np.full((256, 256), 5.0, dtype="<f4")
        metric_depth[0, 0] = np.inf
        metric_depth[0, 1] = np.nan
        rgb, depth = prepare_rgbd(output.getvalue(), metric_depth.tobytes(), 256, 256)
        self.assertEqual((rgb.shape, rgb.dtype), ((224, 224, 3), np.dtype("uint8")))
        self.assertEqual((depth.shape, depth.dtype), ((256, 256, 1), np.dtype("float32")))
        self.assertEqual(float(depth[1, 1, 0]), 0.5)
        self.assertEqual(float(depth[0, 0, 0]), 1.0)
        self.assertEqual(float(depth[0, 1, 0]), 0.0)

    def test_minimal_split_token_ids_must_not_enter_cma_embedding(self):
        validate_instruction_tokens([2, 10, 2503, 0])
        with self.assertRaisesRegex(ValueError, "preprocessed"):
            validate_instruction_tokens([2, 2707])
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_instruction_tokens([0, 0])

    def test_cma_split_requires_exact_episode_alignment(self):
        episode = {
            "episode_id": "7",
            "scene_id": "mp3d/abc/abc.glb",
            "instruction": {"instruction_text": "Go ahead.", "instruction_tokens": [2, 3]},
            "start_position": [0.0, 0.0, 0.0],
            "goals": [{"position": [4.0, 0.0, 0.0]}],
            "info": {"geodesic_distance": 4.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.json"
            policy = root / "policy.json"
            reference.write_text(json.dumps({"episodes": [episode]}))
            policy.write_text(json.dumps({"episodes": [episode]}))
            audit = audit_cma_split(reference, policy)
            self.assertEqual(audit["maximum_token_id"], 3)
            self.assertEqual(audit["split_sha256"], audit["policy_split_sha256"])
            changed = {
                **episode,
                "instruction": {**episode["instruction"], "instruction_tokens": [2707]},
            }
            policy.write_text(json.dumps({"episodes": [changed]}))
            with self.assertRaisesRegex(ValueError, "preprocessed"):
                audit_cma_split(reference, policy)
            changed["instruction"]["instruction_tokens"] = [2, 3]
            changed["start_position"] = [1.0, 0.0, 0.0]
            policy.write_text(json.dumps({"episodes": [changed]}))
            with self.assertRaisesRegex(ValueError, "differs"):
                audit_cma_split(reference, policy)


if __name__ == "__main__":
    unittest.main()
