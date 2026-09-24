"""WaypointPD: OmniVLA-edge's single-waypoint PD path controller.

Pure Pursuit (``pure_pursuit.py``) under-steers badly on OmniVLA-edge paths:
those paths are long-horizon plans whose waypoints reach several metres ahead,
and Pure Pursuit's geometric curvature ``2y/L**2`` collapses as ``1/L**2`` for a
far lookahead target, so the commanded angular velocity is near zero and the
robot only crawls straight.

This controller instead mirrors the reference edge policy
(``external/OmniVLA/inference/run_omnivla_edge.py``): pick a single waypoint a
fixed number of steps ahead and drive a proportional law

    linear  = dx / dt
    angular = atan2-style(dy / dx) / dt        (dt = control horizon, e.g. 1/3 s)

so the steering gain is set by the *heading* to that waypoint, not by a
geometric curvature that vanishes with distance. Velocities are then clamped to
``max_v`` / ``max_w`` while preserving the linear/angular ratio (turn radius),
again matching the reference limiter.

Waypoints are expected in the robot frame (x forward, y left), already scaled to
metres (the adapter applies ``metric_waypoint_spacing``). Pure numpy/math — no
ROS, no torch — so it is unit-testable without a running node.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

# Reuse the command type so the follower node and tests stay uniform.
from .pure_pursuit import TwistCmd


@dataclass
class PathPoint:
    """A waypoint in the robot frame (metres) with optional heading.

    ``cos``/``sin`` carry the waypoint's heading (from the Path pose orientation,
    w=cos, z=sin). They are only consulted in the degenerate case where the
    chosen waypoint sits on the robot (dx==dy==0) and we must rotate in place.
    """

    x: float
    y: float
    cos: float = 1.0
    sin: float = 0.0


def _wrap_angle(theta: float) -> float:
    """Wrap to [-pi, pi] (the reference's clip_angle)."""
    return math.atan2(math.sin(theta), math.cos(theta))


class WaypointPD:
    """Single-waypoint proportional controller (OmniVLA-edge reference law)."""

    def __init__(
        self,
        *,
        max_v: float,
        max_w: float,
        waypoint_select: int = 4,
        dt: float = 1.0 / 3.0,
        linear_clip: float = 0.5,
        angular_clip: float = 1.0,
    ) -> None:
        if dt <= 0.0:
            raise ValueError('dt must be > 0')
        self.max_v = max_v
        self.max_w = max_w
        self.waypoint_select = waypoint_select
        self.dt = dt
        # Pre-clips applied before the ratio-preserving limiter, matching the
        # reference (linear -> [0, 0.5], angular -> [-1, 1]).
        self.linear_clip = linear_clip
        self.angular_clip = angular_clip

    def compute(self, *, path: Sequence[PathPoint]) -> TwistCmd:
        if not path:
            return TwistCmd(0.0, 0.0)

        # Pick the target waypoint a fixed number of steps ahead, clamped to the
        # path length so shorter paths (e.g. dummy/stub) still resolve a target.
        idx = self.waypoint_select
        if idx >= len(path):
            idx = len(path) - 1
        wp = path[idx]
        dx, dy = float(wp.x), float(wp.y)

        eps = 1e-8
        if abs(dx) < eps and abs(dy) < eps:
            # Target is on the robot: rotate toward its heading, don't advance.
            linear = 0.0
            angular = _wrap_angle(math.atan2(wp.sin, wp.cos)) / self.dt
        elif abs(dx) < eps:
            # Target directly beside the robot: spin toward it.
            linear = 0.0
            angular = math.copysign(math.pi / (2.0 * self.dt), dy)
        else:
            linear = dx / self.dt
            angular = math.atan(dy / dx) / self.dt

        linear = max(0.0, min(self.linear_clip, linear))
        angular = max(-self.angular_clip, min(self.angular_clip, angular))
        return self._limit(linear, angular)

    def _limit(self, linear: float, angular: float) -> TwistCmd:
        """Clamp to (max_v, max_w) preserving the linear/angular ratio.

        Faithful port of run_omnivla_edge.py's velocity limitation: shrinking one
        component pulls the other along the same turn radius so the arc the robot
        drives is preserved rather than distorted.
        """
        maxv, maxw = self.max_v, self.max_w
        if abs(linear) <= maxv:
            if abs(angular) <= maxw:
                return TwistCmd(linear, angular)
            rd = linear / angular
            return TwistCmd(maxw * math.copysign(1.0, linear) * abs(rd),
                            maxw * math.copysign(1.0, angular))
        if abs(angular) <= 0.001:
            return TwistCmd(maxv * math.copysign(1.0, linear), 0.0)
        rd = linear / angular
        if abs(rd) >= maxv / maxw:
            return TwistCmd(maxv * math.copysign(1.0, linear),
                            maxv * math.copysign(1.0, angular) / abs(rd))
        return TwistCmd(maxw * math.copysign(1.0, linear) * abs(rd),
                        maxw * math.copysign(1.0, angular))
