"""Unit tests for WaypointPD, the OmniVLA-edge single-waypoint PD controller.

Anchored on the real failure that motivated it: on OmniVLA-edge paths (waypoints
several metres ahead) Pure Pursuit produced a near-zero angular velocity and the
robot only crawled straight. WaypointPD must instead command a real turn on the
same path.
"""
import math

import pytest

from rvln_edge.waypoint_pd import PathPoint, WaypointPD


# The path actually observed from OmniVLA on /rvln/predicted_path
# (metres, robot frame). Index 4 is the target waypoint the reference law picks.
_OMNIVLA_PATH = [
    PathPoint(1.52, 0.056),
    PathPoint(2.20, 0.141),
    PathPoint(2.94, 0.291),
    PathPoint(3.63, 0.414),
    PathPoint(4.25, 0.551),
    PathPoint(4.75, 0.566),
    PathPoint(5.56, 0.066),
]


def _pp(**kw) -> WaypointPD:
    kw.setdefault('max_v', 0.4)
    kw.setdefault('max_w', 1.0)
    return WaypointPD(**kw)


def test_turns_meaningfully_on_omnivla_path():
    # Pure Pursuit gave ~0.02-0.13 rad/s here; the PD law must turn far harder.
    cmd = _pp().compute(path=_OMNIVLA_PATH)
    assert cmd.linear == pytest.approx(0.4, abs=1e-6)
    assert cmd.angular > 0.25          # left turn toward +y, actually reflected
    assert cmd.angular == pytest.approx(0.309, abs=0.02)


def test_right_turn_is_negative():
    path = [PathPoint(p.x, -p.y) for p in _OMNIVLA_PATH]
    cmd = _pp().compute(path=path)
    assert cmd.angular < -0.25


def test_straight_path_has_zero_angular():
    path = [PathPoint(0.1 * i, 0.0) for i in range(1, 11)]
    cmd = _pp().compute(path=path)
    assert cmd.angular == pytest.approx(0.0, abs=1e-9)
    assert cmd.linear == pytest.approx(0.4, abs=1e-6)


def test_empty_path_stops():
    cmd = _pp().compute(path=[])
    assert cmd.linear == 0.0 and cmd.angular == 0.0


def test_waypoint_select_clamps_to_path_length():
    # Fewer poses than waypoint_select (4): fall back to the last waypoint
    # instead of indexing out of range.
    path = [PathPoint(0.3, 0.0), PathPoint(0.6, 0.2), PathPoint(0.9, 0.5)]
    cmd = _pp(waypoint_select=4).compute(path=path)
    # Uses index 2 = (0.9, 0.5): a real forward+turn command, not a crash.
    assert cmd.linear > 0.0 and cmd.angular > 0.0


def test_limiter_preserves_turn_ratio():
    # linear over max_v, so both components shrink along the same arc: the ratio
    # linear/angular (turn radius) is preserved.
    dt = 1.0 / 3.0
    pd = WaypointPD(max_v=0.3, max_w=0.3, waypoint_select=0, dt=dt)
    cmd = pd.compute(path=[PathPoint(4.25, 0.551)])
    # The limiter preserves the ratio of its *input* (post pre-clip): linear is
    # pre-clipped to 0.5, angular stays atan(dy/dx)/dt (< angular_clip).
    lin_in = min(0.5, 4.25 / dt)
    ang_in = math.atan(0.551 / 4.25) / dt
    pre_ratio = lin_in / ang_in
    assert abs(cmd.linear) <= 0.3 + 1e-9
    assert abs(cmd.angular) <= 0.3 + 1e-9
    assert (cmd.linear / cmd.angular) == pytest.approx(pre_ratio, rel=1e-6)


def test_target_on_robot_rotates_in_place():
    # Degenerate target at the origin: rotate toward its heading, don't advance.
    # Heading +pi/2 (facing +y) -> positive angular, zero linear.
    cmd = _pp(waypoint_select=0).compute(
        path=[PathPoint(0.0, 0.0, cos=math.cos(math.pi / 2), sin=math.sin(math.pi / 2))]
    )
    assert cmd.linear == 0.0
    assert cmd.angular > 0.0
