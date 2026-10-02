"""Verify common orchestration without importing either simulator."""

import base64
import struct
import unittest
from unittest.mock import patch

from bench.episode import (
    EpisodeRequest,
    EpisodeSimulator,
    HttpPolicy,
    Observation,
    Policy,
    run_episode,
)


class FakeSimulator(EpisodeSimulator):
    def __init__(self):
        self.x = 0.0

    def reset(self, episode):
        self.x = 0.0
        return [0.0, 0.0, 0.0]

    def observe(self):
        return Observation(b"jpeg")

    def apply(self, action):
        self.x += 1.0
        return [self.x, 0.0, 0.0]

    def close(self):
        pass


class FakePolicy(Policy):
    def __init__(self):
        self.reset_id = None

    def reset(self, episode_id):
        self.reset_id = episode_id

    def act(self, episode_id, frame_id, instruction, observation):
        assert self.reset_id == episode_id
        assert observation.jpeg == b"jpeg"
        return "stop" if frame_id == 1 else "forward"


class EpisodeTest(unittest.TestCase):
    def test_reset_then_forward_then_model_stop(self):
        result = run_episode(
            FakeSimulator(),
            FakePolicy(),
            EpisodeRequest("a", "Go forward.", None, frozenset({"forward", "stop"}), "stop"),
            5,
        )
        self.assertEqual(result["actions"], ["forward", "stop"])
        self.assertEqual(result["positions"][-1], [1.0, 0.0, 0.0])
        self.assertTrue(result["stopped"])

    def test_http_policy_transports_metric_depth_without_goal(self):
        depth = struct.pack("<2f", 1.25, 2.5)
        observation = Observation(b"jpeg", depth, 2, 1)
        policy = HttpPolicy("http://127.0.0.1:8765")
        with patch.object(
            policy,
            "_request",
            return_value={
                "episode_id": "a",
                "frame_id": 3,
                "action": "left",
            },
        ) as request:
            self.assertEqual(policy.act("a", 3, "Turn left.", observation), "left")
        route, payload = request.call_args.args
        self.assertEqual(route, "/act")
        self.assertEqual(payload["depth_width"], 2)
        self.assertEqual(payload["depth_height"], 1)
        self.assertEqual(base64.b64decode(payload["depth_f32_base64"]), depth)
        self.assertNotIn("goal", payload)

    def test_depth_dimensions_must_match_bytes(self):
        with self.assertRaisesRegex(ValueError, "one float32"):
            Observation(b"jpeg", b"\x00" * 4, 2, 1)


if __name__ == "__main__":
    unittest.main()
