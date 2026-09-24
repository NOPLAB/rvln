"""Tests for AsyncVLAEdgeAdapter -- delta accumulation + waypoint units, no weights.

The model itself is stubbed (``__new__`` bypass, same pattern as
test_omnivla_edge_local_adapter); torch must still be importable because
``predict_path`` builds its input tensors with it (the test image ships CPU
torch).
"""
import math

import numpy as np
import pytest

import rvln_edge.adapters.asyncvla as mod
from rvln_edge.adapters.asyncvla import _delta_to_pose_np


def test_delta_to_pose_accumulates_straight_line():
    # 4 steps of (dx=1, dy=0, dtheta=0) -> x = 1, 2, 3, 4 (spacing units).
    delta = np.zeros((1, 4, 4), dtype=np.float32)
    delta[..., 0] = 1.0
    delta[..., 2] = 1.0  # cos(0)
    poses = _delta_to_pose_np(delta)
    assert poses.shape == (1, 4, 4)
    np.testing.assert_allclose(poses[0, :, 0], [1.0, 2.0, 3.0, 4.0], atol=1e-6)
    np.testing.assert_allclose(poses[0, :, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(poses[0, :, 2], 1.0, atol=1e-6)  # cos
    np.testing.assert_allclose(poses[0, :, 3], 0.0, atol=1e-6)  # sin


def test_delta_to_pose_rotates_body_frame_deltas_into_world():
    # Step 1 turns 90deg left in place; step 2 moves (dx=1) in the *new* body
    # frame, which is world +y.
    delta = np.zeros((1, 2, 4), dtype=np.float32)
    delta[0, 0] = [0.0, 0.0, math.cos(math.pi / 2), math.sin(math.pi / 2)]
    delta[0, 1] = [1.0, 0.0, 1.0, 0.0]
    poses = _delta_to_pose_np(delta)
    np.testing.assert_allclose(poses[0, 1, 0], 0.0, atol=1e-6)  # x
    np.testing.assert_allclose(poses[0, 1, 1], 1.0, atol=1e-6)  # y


class _StubEdgeAdapterModel:
    """Stands in for prismatic's Edge_adapter: fixed (1, 8, 4) delta output."""

    def __init__(self, delta: np.ndarray):
        self._delta = delta

    def __call__(self, cur, past, feat):
        import torch

        assert cur.shape == (1, 3, 96, 96)
        assert past.shape == (1, 3, 96, 96)
        assert feat.shape == (1, 8, 1024)
        return torch.from_numpy(self._delta)


def _make_adapter(delta: np.ndarray) -> mod.AsyncVLAEdgeAdapter:
    torch = pytest.importorskip('torch')
    adapter = mod.AsyncVLAEdgeAdapter.__new__(mod.AsyncVLAEdgeAdapter)
    adapter._device = torch.device('cpu')
    adapter._dtype = torch.float32
    adapter._model = _StubEdgeAdapterModel(delta)
    return adapter


def test_predict_path_scales_waypoints_to_metres():
    """Regression: Edge_adapter deltas are in waypoint-spacing (0.1 m) units.

    Serializing them unscaled made the Path 10x too large, so the follower's
    lookahead point sat far ahead and cmd_vel theta barely moved (same unit
    bug as OmniVLA-original, f2a5834).
    """
    delta = np.zeros((1, 8, 4), dtype=np.float32)
    delta[..., 0] = 1.0  # 1 spacing-unit forward per step
    delta[..., 2] = 1.0  # cos(0)
    adapter = _make_adapter(delta)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    path = adapter.predict_path(
        embedding=np.zeros(8 * 1024, dtype=np.float32),
        embedding_shape=(1, 8, 1024),
        cur_image_rgb=img,
    )
    assert len(path.poses) == 8
    xs = [ps.pose.position.x for ps in path.poses]
    expected = [(i + 1) * mod._METRIC_WAYPOINT_SPACING for i in range(8)]
    np.testing.assert_allclose(xs, expected, atol=1e-6)
    # Orientation (cos, sin) passes through unscaled.
    assert path.poses[0].pose.orientation.w == pytest.approx(1.0)
    assert path.poses[0].pose.orientation.z == pytest.approx(0.0)


def test_predict_path_falls_back_to_cur_image_when_no_past():
    delta = np.zeros((1, 8, 4), dtype=np.float32)
    delta[..., 2] = 1.0
    adapter = _make_adapter(delta)
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    path = adapter.predict_path(
        embedding=np.zeros(8 * 1024, dtype=np.float32),
        embedding_shape=(1, 8, 1024),
        cur_image_rgb=img,
        past_image_rgb=None,
    )
    assert len(path.poses) == 8
