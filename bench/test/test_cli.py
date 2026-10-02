"""Check the common CLI without importing the Isaac runtime."""

import argparse
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from bench.cli import build_parser
from isaac_r2r.cli import register


class CliTest(unittest.TestCase):
    def test_r2r_preflight_uses_common_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.json.gz"
            with gzip.open(split, "wt", encoding="utf-8") as stream:
                json.dump(
                    {
                        "episodes": [
                            {
                                "episode_id": "1",
                                "scene_id": "mp3d/abc/abc.glb",
                                "instruction": {"instruction_text": "Go forward."},
                                "start_position": [0, 0, 0],
                                "start_rotation": [0, 0, 0, 1],
                                "goals": [{"position": [4, 0, 0]}],
                                "info": {"geodesic_distance": 4},
                            }
                        ]
                    },
                    stream,
                )
            usd = root / "abc.usd"
            usd.write_text("#usda 1.0\n", encoding="utf-8")
            scenes = root / "scenes.json"
            scenes.write_text(
                json.dumps(
                    {
                        "scenes": {
                            "abc": {
                                "usd": str(usd.resolve()),
                                "isaac_from_habitat": [
                                    [1, 0, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1],
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = build_parser([register]).parse_args(
                [
                    "r2r",
                    "--split",
                    str(split),
                    "--scenes",
                    str(scenes),
                    "--episode",
                    "1",
                    "--check",
                ]
            )
            self.assertIsInstance(args, argparse.Namespace)
            result = args.handler(args)
            self.assertEqual(result["status"], "assets_validated")
            self.assertEqual(result["scene_id"], "abc")


if __name__ == "__main__":
    unittest.main()
