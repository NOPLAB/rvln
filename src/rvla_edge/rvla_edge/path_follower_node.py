"""ROS2 wrapper around WaypointPD: subscribe to Path, publish Twist."""
from __future__ import annotations

from typing import List, Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.node import Node

from .pure_pursuit import TwistCmd
from .waypoint_pd import PathPoint, WaypointPD


class PathFollowerNode(Node):

    def __init__(self) -> None:
        super().__init__('path_follower_node')
        # Steering law: OmniVLA-edge's single-waypoint PD (waypoint_pd.py), used
        # for every backend. Pure Pursuit under-steers on the long-horizon paths
        # these policies emit (waypoints reach several metres ahead), so we pick
        # a waypoint a fixed number of steps ahead and drive a proportional law
        # whose gain comes from the heading to it, not a vanishing curvature.
        self.declare_parameter('waypoint_select', 4)
        self.declare_parameter('control_dt', 1.0 / 3.0)
        self.declare_parameter('max_v', 0.4)
        self.declare_parameter('max_w', 1.0)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('path_topic', '/rvla/predicted_path')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        # Plan 1: paths are always treated as being expressed in the robot
        # frame (typically base_link). If a path arrives with a different
        # frame_id we warn and zero the command, since blindly following
        # would steer toward the wrong pose.
        self.declare_parameter('expected_frame', 'base_link')
        # Hold-last-command (safety net only): the follower re-evaluates the
        # *same* path at rate_hz (20 Hz) while new paths arrive only at the
        # inference rate, so between path updates we may recompute a momentary
        # zero for a path we're already following. To avoid a single-tick
        # dropout we *latch* the last moving command and keep republishing it
        # for that gap, up to hold_timeout_sec, then safe-stop.
        #
        # IMPORTANT: this hold NEVER masks a fresh inference. When a new Path
        # arrives (see _on_path -> _path_new), its computed command is honored
        # verbatim — including a fresh *stop* — so new inference results always
        # take effect immediately instead of coasting on the old command. The
        # hold applies only when no new path has arrived since the last tick.
        # Set to 0 to disable (zero command is emitted immediately).
        self.declare_parameter('hold_timeout_sec', 1.0)
        # A command is "moving" (worth latching) if |linear| or |angular|
        # exceeds this. Below it we treat the command as a stop.
        self.declare_parameter('cmd_epsilon', 1e-3)

        self._pp = WaypointPD(
            max_v=float(self.get_parameter('max_v').value),
            max_w=float(self.get_parameter('max_w').value),
            waypoint_select=int(self.get_parameter('waypoint_select').value),
            dt=float(self.get_parameter('control_dt').value),
        )
        self._expected_frame: str = str(self.get_parameter('expected_frame').value)
        self._hold_timeout_ns: int = int(
            float(self.get_parameter('hold_timeout_sec').value) * 1e9)
        self._cmd_eps: float = float(self.get_parameter('cmd_epsilon').value)
        self._latest: List[PathPoint] = []
        self._frame_mismatch: bool = False
        # Set by _on_path whenever a new Path (a fresh inference result) lands;
        # consumed by _decide_cmd. A fresh result is authoritative and bypasses
        # the hold latch so new inferences are never masked by a stale command.
        self._path_new: bool = False
        # Last "moving" command and the time it was computed, for hold-last.
        self._held_cmd: Optional[TwistCmd] = None
        self._held_at = None  # rclpy Time or None
        # Motion state of the last published command, for transition logging.
        self._was_moving = False
        self._sub = self.create_subscription(
            Path,
            self.get_parameter('path_topic').value,
            self._on_path, 10,
        )
        self._pub = self.create_publisher(
            Twist, self.get_parameter('cmd_vel_topic').value, 10,
        )
        rate = float(self.get_parameter('rate_hz').value)
        self._timer = self.create_timer(1.0 / rate, self._tick)

    def _on_path(self, msg: Path) -> None:
        # Any received path is a fresh inference result; mark it so _decide_cmd
        # honors it verbatim rather than coasting on the held command.
        self._path_new = True
        if msg.header.frame_id and msg.header.frame_id != self._expected_frame:
            if not self._frame_mismatch:
                self.get_logger().warn(
                    f'path frame_id={msg.header.frame_id!r} != '
                    f'expected {self._expected_frame!r}; zeroing cmd_vel'
                )
            self._frame_mismatch = True
            self._latest = []
            return
        self._frame_mismatch = False
        wps: List[PathPoint] = []
        for ps in msg.poses:
            # Heading is a yaw-only quaternion (w=cos, z=sin); carry it so the
            # controller can rotate in place when the target sits on the robot.
            wps.append(PathPoint(
                x=ps.pose.position.x,
                y=ps.pose.position.y,
                cos=ps.pose.orientation.w,
                sin=ps.pose.orientation.z,
            ))
        self._latest = wps

    def _tick(self) -> None:
        cmd = self._decide_cmd(self.get_clock().now())
        self._log_motion_transition(cmd)
        twist = Twist()
        twist.linear.x = float(cmd.linear)
        twist.angular.z = float(cmd.angular)
        self._pub.publish(twist)

    def _log_motion_transition(self, cmd: TwistCmd) -> None:
        """One INFO line per moving<->stopped transition, with the stop reason.

        Surge-and-stop driving shows up as pairs of these lines; the reason
        (empty path vs model's own near-zero waypoint vs frame mismatch)
        tells which upstream stage commanded the stop without needing a
        debugger on the robot.
        """
        moving = abs(cmd.linear) > self._cmd_eps or abs(cmd.angular) > self._cmd_eps
        if moving == self._was_moving:
            return
        self._was_moving = moving
        if moving:
            self.get_logger().info(
                f'resume: cmd=({cmd.linear:+.3f},{cmd.angular:+.3f}) '
                f'path_len={len(self._latest)}'
            )
            return
        if self._frame_mismatch:
            reason = 'frame mismatch'
        elif not self._latest:
            reason = 'empty path (edge safe-stop)'
        else:
            reason = 'model output near-zero waypoint'
        self.get_logger().info(f'stop: {reason} (path_len={len(self._latest)})')

    def _decide_cmd(self, now) -> TwistCmd:
        """Pick the command to publish, applying hold-last-command.

        Pure w.r.t. ROS I/O (takes ``now``, mutates only the latch state) so the
        stop-go smoothing is unit-testable without a running executor.
        """
        cmd = self._pp.compute(path=self._latest)

        # Did a new path (a fresh inference result) arrive since the last tick?
        # A fresh result is authoritative and is honored verbatim below, so new
        # inferences are never masked by the hold latch. Consume the flag
        # unconditionally, whichever branch we take.
        fresh = self._path_new
        self._path_new = False

        # A frame mismatch is a correctness fault, not a transient gap: stop
        # immediately and drop any held command (don't coast on a bad target).
        if self._frame_mismatch:
            self._held_cmd = None
            self._held_at = None
            return cmd

        if abs(cmd.linear) > self._cmd_eps or abs(cmd.angular) > self._cmd_eps:
            # Moving command: use it and latch it for the hold window.
            self._held_cmd = cmd
            self._held_at = now
            return cmd

        # Freshly computed command is zero (empty path / pure-pursuit stop).
        if fresh:
            # A NEW inference explicitly yields a stop. Honor it now and drop
            # the latch so the fresh result — not the stale moving command —
            # controls the robot. This is what keeps new inferences from being
            # overridden by an old cmd_vel.
            self._held_cmd = None
            self._held_at = None
            return cmd

        # No new path since the last tick: we're only re-evaluating the same
        # path at the 20 Hz follower rate. Bridge that single-tick gap by
        # holding the last moving command up to hold_timeout_sec, else stop.
        if (
            self._hold_timeout_ns > 0
            and self._held_cmd is not None
            and self._held_at is not None
            and (now - self._held_at).nanoseconds <= self._hold_timeout_ns
        ):
            return self._held_cmd

        self._held_cmd = None
        self._held_at = None
        return cmd


def main() -> None:
    rclpy.init()
    node = PathFollowerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
