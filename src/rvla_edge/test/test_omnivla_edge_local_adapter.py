"""Tests for OmniVLAEdgeLocalAdapter (Plan 2B Path 2).

The CPU-only tests exercise the pure helpers (modality mapping, pose-goal
vector, ring-buffer stacking, trajectory->Path) that do not need torch / clip /
weights. The full model forward pass is a slow, GPU + weights smoke test gated
behind OMNIVLA_EDGE_E2E=1.
"""
import math
import os

import numpy as np
import pytest

from rvla_edge.adapters.base import EdgeGoal
from rvla_edge.adapters.omnivla_edge_local import (
    _METRIC_WAYPOINT_SPACING,
    _modality_id_for,
    _pose_goal_vector,
    _stack_frames,
    _trajectory_to_path,
)


# --------------------------------------------------------------- modality map

def test_modality_id_for_known_modes():
    assert _modality_id_for('text') == 7
    assert _modality_id_for('pose') == 4
    assert _modality_id_for('image') == 6


def test_modality_id_for_rejects_unknown():
    with pytest.raises(ValueError, match='unknown goal mode'):
        _modality_id_for('satellite')


# --------------------------------------------------------------- pose vector

def test_pose_goal_vector_packs_fwd_left_cos_sin():
    """The model's goal_pose shares the waypoint frame: (x_fwd, y_left)/spacing."""
    vec = _pose_goal_vector((2.0, 1.0, 0.0))  # 2 m forward, 1 m left, yaw 0
    assert vec[0] == pytest.approx(2.0 / _METRIC_WAYPOINT_SPACING)   # forward/spacing
    assert vec[1] == pytest.approx(1.0 / _METRIC_WAYPOINT_SPACING)   # left/spacing
    assert vec[2] == pytest.approx(1.0)   # cos(0)
    assert vec[3] == pytest.approx(0.0)   # sin(0)


def test_pose_goal_vector_clamps_to_threshold():
    vec = _pose_goal_vector((100.0, 0.0, 0.0))  # 100 m forward -> clamp to 30 m
    assert vec[0] == pytest.approx(30.0 / _METRIC_WAYPOINT_SPACING)
    assert vec[1] == pytest.approx(0.0)


# --------------------------------------------------------------- ring buffer

def test_stack_frames_front_pads_when_short():
    f0 = np.full((3, 96, 96), 0.0, dtype=np.float32)
    f1 = np.full((3, 96, 96), 1.0, dtype=np.float32)
    stacked = _stack_frames([f0, f1], 6)
    assert stacked.shape == (1, 18, 96, 96)
    # Current (newest) frame is last 3 channels.
    assert np.all(stacked[0, -3:] == 1.0)
    # Front-padded with the oldest frame f0.
    assert np.all(stacked[0, :3] == 0.0)


def test_stack_frames_keeps_last_n_when_overfull():
    frames = [np.full((3, 4, 4), float(i), dtype=np.float32) for i in range(8)]
    stacked = _stack_frames(frames, 6)
    assert stacked.shape == (1, 18, 4, 4)
    # Oldest two (0, 1) dropped; window is frames 2..7, newest last.
    assert np.all(stacked[0, :3] == 2.0)
    assert np.all(stacked[0, -3:] == 7.0)


def test_stack_frames_empty_raises():
    with pytest.raises(RuntimeError, match='no observation frames'):
        _stack_frames([], 6)


# --------------------------------------------------------------- trajectory

def test_trajectory_to_path_scales_xy_by_spacing():
    wp = np.zeros((8, 4), dtype=np.float32)
    wp[:, 0] = [1, 2, 3, 4, 5, 6, 7, 8]   # x in spacing units
    wp[:, 2] = 1.0                         # cos(0)
    path = _trajectory_to_path(wp, frame_id='base_link')
    assert path.header.frame_id == 'base_link'
    assert len(path.poses) == 8
    xs = [ps.pose.position.x for ps in path.poses]
    assert xs == [pytest.approx(i * _METRIC_WAYPOINT_SPACING) for i in range(1, 9)]


def test_trajectory_to_path_maps_cos_sin_to_quat():
    wp = np.zeros((1, 4), dtype=np.float32)
    wp[0, :] = [0.0, 0.0, math.cos(math.pi / 4), math.sin(math.pi / 4)]
    path = _trajectory_to_path(wp)
    ps = path.poses[0]
    assert ps.pose.orientation.w == pytest.approx(math.cos(math.pi / 4))
    assert ps.pose.orientation.z == pytest.approx(math.sin(math.pi / 4))


def test_trajectory_to_path_rejects_too_few_dims():
    with pytest.raises(ValueError, match='ACTION_DIM>=4'):
        _trajectory_to_path(np.zeros((8, 3), dtype=np.float32))


# --------------------------------------------------------------- EdgeGoal

def test_edge_goal_defaults():
    g = EdgeGoal(mode='text', text='go to the door')
    assert g.mode == 'text'
    assert g.text == 'go to the door'
    assert g.pose_xy_theta is None
    assert g.image_rgb is None


# --------------------------------------------------------------- gated E2E

@pytest.mark.skipif(
    os.environ.get('OMNIVLA_EDGE_E2E') != '1',
    reason='set OMNIVLA_EDGE_E2E=1 (needs CUDA + omnivla-edge.pth + CLIP)',
)
def test_omnivla_edge_local_full_forward():
    from rvla_edge.adapters.omnivla_edge_local import OmniVLAEdgeLocalAdapter

    adapter = OmniVLAEdgeLocalAdapter(
        weights_path=os.environ.get(
            'OMNIVLA_EDGE_WEIGHTS', '/workspace/models/omnivla-edge/omnivla-edge.pth'),
        device='cuda:0',
    )
    adapter.set_goal(EdgeGoal(mode='text', text='blue trash bin'))
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    path = adapter.predict_path(cur_image_rgb=img, frame_id='base_link')
    assert path.header.frame_id == 'base_link'
    assert len(path.poses) == 8  # len_traj_pred


def test_local_adapter_is_local_true():
    """The edge node uses is_local to bypass the cloud/cache.

    Checked without loading the model (via __new__).
    """
    from rvla_edge.adapters import omnivla_edge_local as mod
    adapter = mod.OmniVLAEdgeLocalAdapter.__new__(mod.OmniVLAEdgeLocalAdapter)
    assert adapter.is_local is True


def test_cloud_adapters_are_not_local():
    """Stub / Path-1 OmniVLA adapters consume the cloud embedding (is_local False)."""
    from rvla_edge.adapters.stub import StubAdapter
    from rvla_edge.adapters.omnivla import OmniVLAEdgeAdapter
    assert StubAdapter().is_local is False
    assert OmniVLAEdgeAdapter().is_local is False


def test_predict_path_without_goal_returns_empty(monkeypatch):
    """Before a goal arrives, predict_path must yield an empty (safe-stop) Path.

    Must not touch torch/clip; we stub __init__ to avoid loading the model.
    """
    from rvla_edge.adapters import omnivla_edge_local as mod

    adapter = mod.OmniVLAEdgeLocalAdapter.__new__(mod.OmniVLAEdgeLocalAdapter)
    import threading
    adapter._lock = threading.Lock()
    adapter._goal = None
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    path = adapter.predict_path(cur_image_rgb=img, frame_id='odom')
    assert path.header.frame_id == 'odom'
    assert len(path.poses) == 0
