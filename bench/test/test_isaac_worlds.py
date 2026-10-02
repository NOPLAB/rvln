"""Validate that the owned Gazebo pilot geometry survives conversion parsing."""

import unittest
from pathlib import Path

from isaac_rvln.worlds import parse_boxes


ROOT = Path(__file__).resolve().parents[2]


class WorldTest(unittest.TestCase):
    def test_all_owned_worlds_have_matching_visual_and_collision_boxes(self):
        expected = {"corridor": 3, "junction": 3, "weave": 3}
        for scene, count in expected.items():
            with self.subTest(scene=scene):
                boxes = parse_boxes(ROOT / "bench" / "worlds" / f"{scene}.world")
                self.assertEqual(len(boxes), count)
                self.assertTrue(all(len(item["pose"]) == 6 for item in boxes))


if __name__ == "__main__":
    unittest.main()
