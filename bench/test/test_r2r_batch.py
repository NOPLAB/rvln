"""Batch recovery checks that do not require Isaac Sim or licensed scenes."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench.artifacts import sha256
from isaac_r2r.batch import run_batch


class BatchRecoveryTest(unittest.TestCase):
    def test_native_exit_after_started_trace_retries_and_keeps_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "episodes": [
                            {
                                "episode_id": "one",
                                "scene_id": "mp3d/room/room.glb",
                                "instruction": {
                                    "instruction_text": "Go ahead.",
                                    "instruction_tokens": [2, 3],
                                },
                                "start_position": [0.0, 0.0, 0.0],
                                "start_rotation": [
                                    0.0,
                                    -0.7071067811865475,
                                    0.0,
                                    0.7071067811865476,
                                ],
                                "goals": [{"position": [1.0, 0.0, 0.0]}],
                                "info": {"geodesic_distance": 1.0},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            usd = root / "room.usd"
            usd.write_bytes(b"USD fixture")
            scenes = root / "scenes.json"
            scenes.write_text(
                json.dumps(
                    {
                        "scenes": {
                            "room": {
                                "usd": str(usd),
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
            calls = []

            def child(command, **kwargs):
                calls.append(command)
                path = Path(command[command.index("--out") + 1])
                row = {
                    "episode_id": "one",
                    "scene_id": "room",
                    "split_sha256": sha256(split),
                    "usd_sha256": sha256(usd),
                    "policy_split_sha256": sha256(split),
                    "policy_id": "fixture-policy",
                    "metric_namespace": "isaac_r2r_transfer",
                    "status": "started" if len(calls) == 1 else "completed",
                    "success": True,
                    "spl": 1.0,
                    "navigation_error_m": 0.0,
                    "blocked_steps": 0,
                }
                path.write_text(json.dumps(row), encoding="utf-8")
                return subprocess.CompletedProcess(command, 3221227274 if len(calls) == 1 else 0)

            output = root / "batch"
            with patch("isaac_r2r.batch.subprocess.run", side_effect=child):
                result = run_batch(
                    split,
                    scenes,
                    "http://127.0.0.1:8765",
                    "fixture-policy",
                    output,
                    12,
                    False,
                    False,
                    60,
                    policy_split=split,
                )
            self.assertEqual(len(calls), 2)
            self.assertEqual(
                json.loads(calls[0][calls[0].index("--policy-tokens-json") + 1]), [2, 3]
            )
            self.assertEqual(result["episodes"], 1)
            self.assertEqual(result["success_rate"], 1.0)
            self.assertTrue((output / "episodes/00000.log").is_file())
            self.assertTrue((output / "episodes/00000.attempt-1.log").is_file())
            first_log = output / "episodes/00000.log"
            first_log.write_text("native startup crash", encoding="utf-8")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["startup_retry_counts"], {"one": 1})
            trace = output / "episodes/00000.json"
            stale = json.loads(trace.read_text())
            stale["usd_sha256"] = "stale scene"
            trace.write_text(json.dumps(stale))
            with patch("isaac_r2r.batch.subprocess.run", side_effect=child):
                run_batch(
                    split,
                    scenes,
                    "http://127.0.0.1:8765",
                    "fixture-policy",
                    output,
                    12,
                    False,
                    True,
                    60,
                    policy_split=split,
                )
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads(trace.read_text())["usd_sha256"], sha256(usd))
            self.assertEqual(first_log.read_text(encoding="utf-8"), "native startup crash")
            self.assertTrue((output / "episodes/00000.attempt-2.log").is_file())


if __name__ == "__main__":
    unittest.main()
