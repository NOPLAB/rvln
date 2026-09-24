"""Unit tests for path_follower_node's command decision.

A fresh inference result (any new Path) is authoritative and honored verbatim
— including a fresh stop — so new inferences are never masked by the hold
latch. The hold only bridges a single-tick gap when the *same* path is
re-evaluated at 20 Hz with no new path in between.

Drives ``_decide_cmd`` directly with synthetic timestamps so the logic is
exercised deterministically, without a running executor or the 20 Hz
wall-clock timer.
"""
import pytest
import rclpy
from rclpy.time import Time
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from rvla_edge.path_follower_node import PathFollowerNode


@pytest.fixture(scope='module')
def ros_runtime():
    rclpy.init()
    yield
    rclpy.shutdown()


def _forward_path(n: int = 10, step: float = 0.1, frame: str = 'base_link') -> Path:
    path = Path()
    path.header.frame_id = frame
    for i in range(1, n + 1):
        ps = PoseStamped()
        ps.pose.position.x = i * step
        path.poses.append(ps)
    return path


def _empty_path(frame: str = 'base_link') -> Path:
    path = Path()
    path.header.frame_id = frame
    return path


def _t(sec: float) -> Time:
    return Time(nanoseconds=int(sec * 1e9))


def _make_node(hold_timeout_sec: float = 1.0) -> PathFollowerNode:
    node = PathFollowerNode()
    node._hold_timeout_ns = int(hold_timeout_sec * 1e9)
    return node


def test_new_empty_path_stops_immediately(ros_runtime):
    node = _make_node(hold_timeout_sec=1.0)
    try:
        # Moving forward.
        node._on_path(_forward_path())
        cmd = node._decide_cmd(_t(0.0))
        assert cmd.linear > 0.0

        # A NEW empty path is a fresh inference result: honor it now and stop,
        # well within the hold window. The old moving command must not persist.
        node._on_path(_empty_path())
        stopped = node._decide_cmd(_t(0.1))
        assert stopped.linear == 0.0 and stopped.angular == 0.0
    finally:
        node.destroy_node()


def test_new_inference_is_not_masked_by_hold(ros_runtime):
    node = _make_node(hold_timeout_sec=1.0)
    try:
        # Moving forward, latched.
        node._on_path(_forward_path())
        node._decide_cmd(_t(0.0))

        # A fresh stop, then a fresh forward again: each new path takes effect
        # immediately -- the hold never republishes the stale command.
        node._on_path(_empty_path())
        assert node._decide_cmd(_t(0.2)).linear == 0.0

        node._on_path(_forward_path())
        assert node._decide_cmd(_t(0.4)).linear > 0.0

        node._on_path(_empty_path())
        assert node._decide_cmd(_t(0.6)).linear == 0.0
    finally:
        node.destroy_node()


def test_hold_bridges_retick_gap_without_new_path(ros_runtime):
    node = _make_node(hold_timeout_sec=1.0)
    try:
        # Moving forward, latched.
        node._on_path(_forward_path())
        cmd = node._decide_cmd(_t(0.0))
        assert cmd.linear > 0.0

        # Same path re-evaluated at the 20 Hz follower rate with NO new path in
        # between (``_path_new`` stays False). Simulate a momentary zero for the
        # path we're already following: within the window -> hold last command.
        node._latest = []
        held = node._decide_cmd(_t(0.5))
        assert held.linear == pytest.approx(cmd.linear)

        # Still no new path at 1.5 s: past the 1.0 s hold window -> safe-stop.
        stopped = node._decide_cmd(_t(1.5))
        assert stopped.linear == 0.0 and stopped.angular == 0.0
    finally:
        node.destroy_node()


def test_frame_mismatch_stops_immediately(ros_runtime):
    node = _make_node(hold_timeout_sec=5.0)
    try:
        node._on_path(_forward_path())
        moving = node._decide_cmd(_t(0.0))
        assert moving.linear > 0.0

        # A wrong-frame path is a correctness fault: stop now, don't coast even
        # though we're well within the hold window.
        node._on_path(_forward_path(frame='map'))
        stopped = node._decide_cmd(_t(0.1))
        assert stopped.linear == 0.0 and stopped.angular == 0.0
    finally:
        node.destroy_node()


def test_hold_disabled_emits_zero_immediately(ros_runtime):
    node = _make_node(hold_timeout_sec=0.0)
    try:
        node._on_path(_forward_path())
        node._decide_cmd(_t(0.0))

        node._on_path(_empty_path())
        stopped = node._decide_cmd(_t(0.01))
        assert stopped.linear == 0.0 and stopped.angular == 0.0
    finally:
        node.destroy_node()
