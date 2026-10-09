"""Odometry-consumed NaVIDA actions and post-stop image barriers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, final


TURN_DRIVE_SPEED: Final = 0.35


@dataclass(frozen=True, slots=True)
class Pose:
    x: float
    y: float
    yaw: float
    yaw_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class Action:
    distance: float
    angle: float


@dataclass(frozen=True, slots=True)
class Request:
    frame_id: int
    generation: int
    capture_ns: int


@dataclass(frozen=True, slots=True)
class StepDataError(ValueError):
    reason: str

    def __post_init__(self) -> None:
        ValueError.__init__(self, self.reason)


def decode_action(values: list[float]) -> Action:
    """Parse the backend's single metric token, not a ROS quaternion."""
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise StepDataError("invalid action token")
    x, y, cosine, sine = values
    angle = math.atan2(sine, cosine)
    if (
        y != 0
        or not 0 <= x <= 0.75
        or abs(cosine * cosine + sine * sine - 1) > 0.001
        or (0 < x < 0.01 - 1e-6)
        or (0 < abs(angle) < math.radians(1) - 1e-6)
        or abs(angle) > math.radians(45) + 1e-6
        or (x > 0 and abs(angle) > 1e-6)
    ):
        raise StepDataError("action is not a NaVIDA primitive")
    return Action(x, angle)


def angle_delta(current: float, previous: float) -> float:
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


@final
class StepCycle:
    """Mutable accumulator for one request, motion and confirmed stop."""

    def __init__(self, max_v: float = 0.1, max_w: float = 0.35) -> None:
        if max_v < 0.015 or max_w < TURN_DRIVE_SPEED:
            raise ValueError("speed limits are below the robot execution minimum")
        self.max_v, self.max_w = max_v, max_w
        self.pose: Pose | None = None
        self.odom_ns = 0
        self.odom_at = -math.inf
        self.generation = 0
        self.floor_ns = 0
        self.pending: Request | None = None
        self.action: Action | None = None
        self.origin: Pose | None = None
        self.turn_progress = 0.0
        self.yaw_rate = 0.0
        self.motion_at = 0.0
        self.settling = True
        self.stopped = False
        self.fault: str | None = None
        self._stop_ns = 0
        self._settle_at = 0.0
        self._settle_pose: Pose | None = None
        self._settle_stamp_ns = 0
        self._stable_since_ns = 0
        self._turn_target: float | None = None
        self._turn_integral = 0.0
        self._turn_control_at = 0.0
        self._turn_emitted = False
        self._settle_not_before_ns = 0

    def cancel(self, now_ns: int, now: float, *, stopped: bool = False) -> None:
        """Invalidate late replies and require another physically stopped barrier."""
        self.generation += 1
        self.pending = None
        self.action = None
        self.origin = None
        self.turn_progress = 0.0
        self.stopped = stopped
        self.fault = None
        self._turn_target = None
        self._begin_settling(now_ns, now)

    def _begin_settling(self, now_ns: int, now: float, *, retain_action: bool = False) -> None:
        if not retain_action:
            self.action = None
        if self._turn_emitted:
            self._settle_not_before_ns = max(self._settle_not_before_ns, now_ns + 555_000_000)
        self._turn_emitted = False
        self.settling = True
        self._stop_ns = now_ns
        self._settle_at = now
        self._settle_pose = None
        self._settle_stamp_ns = 0
        self._stable_since_ns = 0

    def confirm_motor_disabled(self, now_ns: int, now: float) -> None:
        """An acknowledged physical inhibit removes queued-command uncertainty."""
        self._begin_settling(now_ns, now, retain_action=True)
        self._settle_not_before_ns = now_ns

    def observe(self, pose: Pose, stamp_ns: int, now: float, now_ns: int) -> None:
        """Only advancing odometry stamps refresh the watchdog or prove cessation."""
        if stamp_ns <= self.odom_ns:
            return
        self.yaw_rate = pose.yaw_rate
        if self.pose is None and self.settling:
            self._settle_at = now
        if (
            (self.action is not None or self.settling)
            and self.origin is not None
            and self.pose is not None
        ):
            self.turn_progress += angle_delta(pose.yaw, self.pose.yaw)
        self.pose, self.odom_ns, self.odom_at = pose, stamp_ns, now
        if not self.settling or stamp_ns <= max(self._stop_ns, self._settle_not_before_ns):
            return
        previous = self._settle_pose
        previous_stamp = self._settle_stamp_ns
        self._settle_pose = pose
        self._settle_stamp_ns = stamp_ns
        if previous is None:
            self._stable_since_ns = stamp_ns
            return
        dt = (stamp_ns - previous_stamp) / 1e9
        stable = (
            math.hypot(pose.x - previous.x, pose.y - previous.y) / dt <= 0.005
            and abs(angle_delta(pose.yaw, previous.yaw)) / dt <= 0.01
        )
        if not stable:
            self._stable_since_ns = stamp_ns
        if stable and stamp_ns - self._stable_since_ns >= 150_000_000:
            if self._turn_target is not None:
                sign = math.copysign(1, self._turn_target)
                target, progress = abs(self._turn_target), sign * self.turn_progress
                tolerance = min(math.radians(5), target / 3)
                if abs(target - progress) > tolerance:
                    self.action = Action(0, self._turn_target)
                    self.settling = False
                    return
                self._turn_target = None
            self.action = None
            self.settling = False
            self.floor_ns = now_ns

    def can_request(self, capture_ns: int, now: float, now_ns: int) -> bool:
        return (
            not self.stopped
            and not self.settling
            and self.action is None
            and self.pending is None
            and self.pose is not None
            and now - self.odom_at <= 0.3
            and self.floor_ns < capture_ns <= now_ns
            and now_ns - capture_ns <= 750_000_000
        )

    def request(self, frame_id: int, capture_ns: int, now: float, now_ns: int) -> Request:
        if not self.can_request(capture_ns, now, now_ns):
            raise StepDataError("observation is not eligible")
        self.pending = Request(frame_id, self.generation, capture_ns)
        return self.pending

    def accept(self, request: Request, action: Action, now: float, now_ns: int) -> bool:
        if request != self.pending or request.generation != self.generation:
            return False
        self.pending = None
        if now_ns - request.capture_ns > 1_500_000_000:
            self.fail("reply expired", now, now_ns)
            return False
        return self.start(action, now, now_ns)

    def start(self, action: Action, now: float, now_ns: int) -> bool:
        """Begin one primitive from the current measured pose."""
        if self.stopped or self.pose is None or now - self.odom_at > 0.3:
            return False
        self.origin = self.pose
        self.action = action
        self.turn_progress = 0.0
        self.motion_at = now
        self._turn_target = action.angle if action.angle else None
        self._turn_integral = math.copysign(TURN_DRIVE_SPEED, action.angle)
        self._turn_control_at = now
        if action.distance == 0 and action.angle == 0:
            self._begin_settling(now_ns, now)
        return True

    def fail(self, reason: str, now: float, now_ns: int) -> None:
        progress = self.turn_progress
        self.cancel(now_ns, now, stopped=True)
        self.turn_progress = progress
        self.fault = reason

    def command(
        self, now: float, now_ns: int, *, motion_enabled: bool = True
    ) -> tuple[float, float]:
        """Stop on target crossing; never wind a relative target around again."""
        if self.stopped or self.pose is None:
            return 0.0, 0.0
        if now - self.odom_at > 0.3:
            self.fail("odometry stale", now, now_ns)
            return 0.0, 0.0
        action, origin = self.action, self.origin
        if action is not None and origin is not None:
            duration = max(action.distance / self.max_v, abs(action.angle) / self.max_w)
            budget = 3 * duration + 8
            if now - self.motion_at > budget:
                self.fail("motion timeout", now, now_ns)
                return 0.0, 0.0
        if self.settling:
            if now - self._settle_at > 2:
                self.fail("stop not confirmed", now, now_ns)
            return 0.0, 0.0
        if action is None or origin is None:
            return 0.0, 0.0
        if not motion_enabled:
            self._turn_control_at = now
            return 0.0, 0.0
        if action.distance > 0:
            progress = (self.pose.x - origin.x) * math.cos(origin.yaw) + (
                self.pose.y - origin.y
            ) * math.sin(origin.yaw)
            remaining = action.distance - progress
            tolerance = min(0.05, action.distance / 4)
            if remaining > tolerance:
                return min(self.max_v, max(0.015, 0.5 * remaining)), 0.0
        else:
            remaining = action.angle - self.turn_progress
            tolerance = min(math.radians(5), abs(action.angle) / 3)
            if abs(remaining) > tolerance or abs(self.yaw_rate) > 0.08:
                angular = 0.0
                if abs(remaining) > tolerance:
                    if remaining * action.angle < 0 and self._turn_integral * action.angle > 0:
                        self._turn_integral = 0.0
                    desired_rate = max(
                        -self.max_w, min(self.max_w, 2 * (remaining - 0.65 * self.yaw_rate))
                    )
                    error = desired_rate - self.yaw_rate
                    raw = self._turn_integral + 0.5 * error
                    if abs(raw) < self.max_w or raw * error < 0:
                        self._turn_integral = max(
                            -self.max_w,
                            min(
                                self.max_w,
                                self._turn_integral + 2 * error * (now - self._turn_control_at),
                            ),
                        )
                    angular = max(-self.max_w, min(self.max_w, self._turn_integral + 0.5 * error))
                self._turn_control_at = now
                self._turn_emitted = True
                return 0.0, angular
        self._begin_settling(now_ns, now)
        return 0.0, 0.0
