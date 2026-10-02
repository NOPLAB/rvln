"""R2R evaluation lifecycle tests with real scene validation and HTTP encoding."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from isaac_r2r import evaluation
from isaac_r2r.cli import register
from bench.cli import build_parser
from bench.episode import Observation


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.split, self.scenes, self.out = (
            root / name for name in ("split.json", "scenes.json", "result.json")
        )
        self.episode = {
            "episode_id": "one",
            "scene_id": "mp3d/room/room.glb",
            "instruction": {"instruction_text": "Go forward."},
            "start_position": [0, 0, 0],
            "start_rotation": [0, 0, 0, 1],
            "goals": [{"position": [0.25, 0, 0]}],
            "info": {"geodesic_distance": 0.25},
        }
        self.split.write_text(json.dumps({"episodes": [self.episode]}))
        usd = root / "room.usd"
        usd.write_bytes(b"fixture")
        self.scenes.write_text(
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
            )
        )
        self.args = build_parser([register]).parse_args(
            [
                "r2r",
                "--split",
                str(self.split),
                "--scenes",
                str(self.scenes),
                "--episode",
                "one",
                "--out",
                str(self.out),
                "--policy-id",
                "fixture-policy",
            ]
        )

    def test_check_never_imports_renderer(self):
        self.args.check = True
        with patch.dict(sys.modules, {"isaac_r2r.run": None}):
            row = self.args.handler(self.args)
        self.assertEqual(row["status"], "assets_validated")
        self.assertFalse(self.out.exists())

    def test_run_and_error_reports_close_simulator(self):
        actions = []
        requests = []
        simulator = NS(
            blocked_steps=0,
            reset=lambda request: actions.append(("reset", request.episode_id)) or [0, 0, 0],
            observe=lambda: Observation(b"jpeg"),
            apply=lambda action: actions.append(action) or [0.25, 0, 0],
            close=lambda: actions.append("close"),
        )

        def initialize(scene, headless):
            self.assertEqual(json.loads(self.out.read_text())["status"], "started")
            return simulator

        def response(request, timeout):
            payload = json.loads(request.data)
            requests.append((request.full_url, payload))
            row = {"episode_id": "one", "policy_id": "fixture-policy"}
            if request.full_url.endswith("/act"):
                row.update(
                    frame_id=payload["frame_id"],
                    action="forward" if payload["frame_id"] == 0 else "stop",
                )
            return io.BytesIO(json.dumps(row).encode())

        with (
            patch("isaac_r2r.run.IsaacR2RSimulator", side_effect=initialize),
            patch("urllib.request.urlopen", side_effect=response),
        ):
            row = self.args.handler(self.args)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["actions"], ["forward", "stop"])
        self.assertEqual(actions, [("reset", "one"), "forward", "close"])
        self.assertEqual(requests[0][1], {"episode_id": "one"})
        self.assertEqual(requests[1][1]["jpeg_base64"], "anBlZw==")
        self.assertEqual(json.loads(self.out.read_text()), row)
        actions.clear()
        with (
            patch("isaac_r2r.run.IsaacR2RSimulator", side_effect=initialize),
            patch("urllib.request.urlopen", side_effect=OSError("offline")),
        ):
            with self.assertRaises(OSError):
                evaluation.run(self.args)
        error = json.loads(self.out.read_text())
        self.assertEqual((error["status"], error["error"]), ("error", "OSError: offline"))
        self.assertEqual(actions, ["close"])
