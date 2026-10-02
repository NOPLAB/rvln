"""Focused tests for the portable R2R-CE episode contract."""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from isaac_r2r.protocol import load_episodes, score
from isaac_r2r.inventory import inventory
from isaac_r2r.scans import validate_scan
from isaac_r2r.summary import summarize


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.episode = {
            "episode_id": "7",
            "scene_id": "mp3d/abc/abc.glb",
            "instruction": {"instruction_text": "Go to the door."},
            "start_position": [0.0, 0.0, 0.0],
            "goals": [{"position": [4.0, 0.0, 0.0], "radius": 3.0}],
            "info": {"geodesic_distance": 4.0},
        }

    def test_gzip_split_and_stop_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val_unseen.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump({"episodes": [self.episode]}, stream)
            episode = load_episodes(path)[0]
        path = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        self.assertEqual(score(episode, path, True)["spl"], 1.0)
        self.assertFalse(score(episode, path, False)["success"])

    def test_success_requires_strictly_less_than_three_metres(self):
        boundary = score(self.episode, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], True)
        near = score(self.episode, [[0.0, 0.0, 0.0], [1.01, 0.0, 0.0]], True)
        self.assertEqual(boundary["navigation_error_m"], 3.0)
        self.assertEqual(boundary["navigation_error_kind"], "euclidean")
        self.assertFalse(boundary["success"])
        self.assertEqual(boundary["spl"], 0.0)
        self.assertTrue(near["success"])

    def test_duplicate_episode_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val.json"
            path.write_text(json.dumps({"episodes": [self.episode, self.episode]}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_episodes(path)

    def test_inventory_counts_real_split_scenes_and_missing_scans(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "val_unseen.json.gz"
            other = {**self.episode, "episode_id": "8", "scene_id": "mp3d/def/def.glb"}
            with gzip.open(split, "wt", encoding="utf-8") as stream:
                json.dump(
                    {"episodes": [self.episode, other, {**self.episode, "episode_id": "9"}]},
                    stream,
                )
            scan = root / "scans/abc/abc.glb"
            scan.parent.mkdir(parents=True)
            scan.write_bytes(b"glb")
            result = inventory(split, root / "scans")
        self.assertEqual(result["episodes"], 3)
        self.assertEqual(result["scan_count"], 1)
        self.assertEqual(
            [(row["scene_id"], row["episodes"], row["scan_exists"]) for row in result["scenes"]],
            [("abc", 2, True), ("def", 1, False)],
        )

    def test_scan_preflight_rejects_corrupt_glb(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scan.glb"
            output = Path(directory) / "scan.usd"
            source.write_bytes(b"glTF" + (2).to_bytes(4, "little") + (13).to_bytes(4, "little"))
            with self.assertRaisesRegex(ValueError, "invalid GLB"):
                validate_scan(source, output)
            source.write_bytes(b"glTF" + (2).to_bytes(4, "little") + (12).to_bytes(4, "little"))
            self.assertEqual(validate_scan(source, output), (source.resolve(), output.resolve()))

    def test_aggregate_rejects_split_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, split in enumerate(("a", "b")):
                path = Path(directory) / f"{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "episode_id": str(index),
                            "metric_namespace": "isaac_r2r_transfer",
                            "status": "completed",
                            "split_sha256": split,
                            "scene_id": "abc",
                            "success": False,
                            "spl": 0.0,
                            "navigation_error_m": 5.0,
                            "blocked_steps": 0,
                        }
                    )
                )
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "mixed R2R-CE split"):
                summarize(paths)

    def test_aggregate_rejects_policy_identity_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, policy_id in enumerate(("model-a", "model-b")):
                path = Path(directory) / f"{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "episode_id": str(index),
                            "metric_namespace": "isaac_r2r_transfer",
                            "status": "completed",
                            "split_sha256": "one-split",
                            "scene_id": "abc",
                            "policy_id": policy_id,
                            "success": False,
                            "spl": 0.0,
                            "navigation_error_m": 5.0,
                            "blocked_steps": 0,
                        }
                    )
                )
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "mixed policy model identities"):
                summarize(paths)

    def test_aggregate_rejects_incomplete_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.json"
            path.write_text(json.dumps({"episode_id": "7", "status": "started"}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                summarize([path])

    def test_aggregate_requires_exact_split_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.json"
            split.write_text(
                json.dumps({"episodes": [self.episode, {**self.episode, "episode_id": "8"}]}),
                encoding="utf-8",
            )
            path = root / "7.json"
            path.write_text(
                json.dumps(
                    {
                        "episode_id": "7",
                        "metric_namespace": "isaac_r2r_transfer",
                        "status": "completed",
                        "split_sha256": "wrong",
                        "scene_id": "abc",
                        "success": True,
                        "spl": 1.0,
                        "navigation_error_m": 0.0,
                        "blocked_steps": 0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exact split episodes"):
                summarize([path], split)


if __name__ == "__main__":
    unittest.main()
