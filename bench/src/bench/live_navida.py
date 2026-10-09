"""Execute NaVIDA primitives, stop, then infer from a post-stop exposure."""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import signal
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import FrameType

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rvln_msgs.msg import GoalSpec, Observation
from std_srvs.srv import SetBool

from .navida_step import (
    Action,
    Pose,
    StepCycle,
    StepDataError,
    angle_delta,
    decode_action,
)


@dataclass(frozen=True, slots=True)
class Reply:
    frame_id: int
    action: Action
    raw_response: str


def infer(url: str, frame_id: int, text: str, jpeg: bytes) -> Reply:
    payload = {
        "frame_id": frame_id,
        "text": text,
        "jpeg_base64": base64.b64encode(jpeg).decode("ascii"),
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/infer",
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=4) as response:
        result = json.load(response)
    embedding = result["embedding"]
    if len(embedding) != 1:
        raise StepDataError("expected exactly one NaVIDA primitive")
    action = decode_action([float(v) for v in embedding[0]])
    return Reply(int(result["frame_id"]), action, str(result["diagnostics"]["raw_response"]))


def chunk_actions(raw: str) -> list[Action]:
    """Primitives after the first in NaVIDA's text chunk, in execution order."""
    actions = []
    for verb, value in re.findall(r"(forward|turn left|turn right)\s+(\d+)", raw)[1:]:
        if verb == "forward":
            actions.append(Action(min(0.75, int(value) / 100), 0.0))
        else:
            sign = 1 if verb == "turn left" else -1
            actions.append(Action(0.0, sign * math.radians(min(45, int(value)))))
    return actions


class NavidaStepBridge(Node):
    """The sole navigation command publisher; HTTP never blocks odometry callbacks."""

    def __init__(
        self, url: str, odom_topic: str = "/odom", max_v: float = 0.1, max_w: float = 0.35
    ) -> None:
        super().__init__("rvln_navida_step")
        self.url = url
        self.cycle = StepCycle(max_v, max_w)
        self.goal, self.latest = None, None
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.future, self.request_token = None, None
        self.queue: list[Action] = []
        self._last_fault: str | None = None
        self.power = self.create_client(SetBool, "/motor_power")
        self.power_future, self.power_state, self.power_at = None, "stop", self._now()
        self.power_pose, self.power_stamp, self.power_stable = None, 0, 0
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        goal_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Observation, "/rvln/observation", self._observation, sensor_qos)
        self.create_subscription(Odometry, odom_topic, self._odom, sensor_qos)
        self.create_subscription(GoalSpec, "/rvln/goal", self._goal, goal_qos)
        self.create_service(SetBool, "/rvln/follower_stop", self._forced_stop)
        self.create_timer(0.05, self._tick)

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _now(self) -> float:
        return time.monotonic()

    def _event(self, event: str, **fields: str | int | float | bool) -> None:
        print(json.dumps({"event": event, **fields}), flush=True)

    def _publish(self, linear: float = 0, angular: float = 0) -> None:
        msg = Twist()
        msg.linear.x, msg.angular.z = float(linear), float(angular)
        self.publisher.publish(msg)

    def _set_goal(self, text: str | None, *, restart: bool = False) -> None:
        if restart or text != self.goal:
            self.goal, self.latest, self.queue = text, None, []
            self.cycle.cancel(
                self._now_ns(), self._now(), stopped=self.cycle.stopped or text is None
            )
            self._stop_power()
            self._event("goal_changed", generation=self.cycle.generation)

    def _goal(self, msg: GoalSpec) -> None:
        self._set_goal(
            msg.text if msg.mode == GoalSpec.MODE_TEXT and msg.text else None, restart=True
        )

    def _observation(self, msg: Observation) -> None:
        self._set_goal(
            msg.goal.text if msg.goal.mode == GoalSpec.MODE_TEXT and msg.goal.text else None
        )
        self.latest = msg

    def _odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        yaw_rate = msg.twist.twist.angular.z
        values = (p.x, p.y, q.x, q.y, q.z, q.w, yaw_rate)
        now_ns = self._now_ns()
        if (
            stamp <= 0
            or now_ns - stamp > 300_000_000
            or stamp > now_ns
            or not all(math.isfinite(v) for v in values)
            or abs(sum(v * v for v in values[2:6]) - 1) > 0.01
        ):
            self.cycle.fail("invalid odometry", self._now(), now_ns)
            self._stop_power()
            return
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.cycle.observe(Pose(p.x, p.y, yaw, yaw_rate=yaw_rate), stamp, self._now(), now_ns)
        if self.power_state == "verify" and stamp > self.power_stamp:
            previous, previous_stamp = self.power_pose, self.power_stamp
            self.power_pose, self.power_stamp = self.cycle.pose, stamp
            dt = (stamp - previous_stamp) / 1e9
            if (
                previous is None
                or math.hypot(p.x - previous.x, p.y - previous.y) / dt > 0.005
                or abs(angle_delta(yaw, previous.yaw)) / dt > 0.01
            ):
                self.power_stable = stamp
            elif stamp - self.power_stable >= 200_000_000:
                self.power_state = "ready"
                if self.cycle.action is None:
                    self.cycle.floor_ns = now_ns
                    self._event(
                        "stopped",
                        barrier_ns=now_ns,
                        turn_deg=math.degrees(self.cycle.turn_progress),
                    )

    def _forced_stop(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        self.cycle.cancel(self._now_ns(), self._now(), stopped=bool(request.data))
        self.queue = []
        self._stop_power()
        response.success = True
        response.message = "follower stopped" if request.data else "follower resumed"
        return response

    def _stop_power(self) -> None:
        self._publish()
        if self.power_state not in ("stop", "disabling"):
            self.power_state = "stop"
        self._power_tick(self._now(), self._now_ns())

    def _power_tick(self, now: float, now_ns: int) -> None:
        if self.power_future is not None:
            if not self.power_future.done() and now - self.power_at <= 2:
                return
            future, self.power_future = self.power_future, None
            try:
                if not future.done():
                    self.power.remove_pending_request(future)
                    raise TimeoutError("motor power timeout")
                if not future.result().success:
                    raise ValueError("motor power rejected")
            except (OSError, ValueError, RuntimeError) as exc:
                self.cycle.fail(str(exc), now, now_ns)
                self._publish()
                self.power_state = "stop"
            else:
                if self.power_state == "disabling":
                    self.power_state = "off"
                    self.cycle.confirm_motor_disabled(now_ns, now)
                elif self.power_state == "enabling":
                    self.power_state, self.power_at = "verify", now
                    self.power_pose, self.power_stamp, self.power_stable = None, now_ns, 0
        if self.power_state == "verify" and now - self.power_at > 2:
            self.cycle.fail("post-enable stop not confirmed", now, now_ns)
            self.power_state = "stop"
        enable = self.power_state == "off" and not self.cycle.settling
        if self.power_state == "stop" or enable:
            self._publish()
            self.power_at = now
            self.power_state = "enabling" if enable else "disabling"
            self.power_future = self.power.call_async(SetBool.Request(data=enable))

    def _finish_request(self, now: float, now_ns: int) -> None:
        future, token = self.future, self.request_token
        if future is None or not future.done() or token is None:
            return
        self.future, self.request_token = None, None
        try:
            reply = future.result()
            if reply.frame_id != token.frame_id:
                raise StepDataError("reply frame ID mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if token.generation == self.cycle.generation:
                self.cycle.fail(str(exc), now, now_ns)
            self._event("request_failed", reason=str(exc))
            return
        if self.cycle.accept(token, reply.action, now, now_ns):
            self.queue = chunk_actions(reply.raw_response)
            self._event(
                "action_started",
                frame_id=token.frame_id,
                distance_m=reply.action.distance,
                angle_deg=math.degrees(reply.action.angle),
                raw=reply.raw_response,
            )
        else:
            self._event("reply_discarded", frame_id=token.frame_id)

    def _tick(self) -> None:
        now, now_ns = self._now(), self._now_ns()
        self._power_tick(now, now_ns)
        if self.power_state == "ready":
            self._finish_request(now, now_ns)
        action = self.cycle.action
        linear, angular = self.cycle.command(
            now, now_ns, motion_enabled=self.power_state == "ready"
        )
        self._publish(linear, angular)
        if self.cycle.settling and self.power_state in ("ready", "enabling", "verify"):
            self._stop_power()
        if action is not None and self.cycle.action is None:
            self._event("action_zero", turn_deg=math.degrees(self.cycle.turn_progress))
        if self.cycle.fault != self._last_fault:
            self._last_fault = self.cycle.fault
            if self.cycle.fault is not None:
                self._event("fault", reason=self.cycle.fault)
        if self.cycle.stopped:
            self.queue = []
        if (
            self.queue
            and self.power_state == "ready"
            and self.future is None
            and self.cycle.action is None
            and not self.cycle.settling
            and self.cycle.start(self.queue[0], now, now_ns)
        ):
            action = self.queue.pop(0)
            self._event(
                "chunk_action_started",
                distance_m=action.distance,
                angle_deg=math.degrees(action.angle),
                remaining=len(self.queue),
            )
            return
        obs = self.latest
        if (
            self.power_state != "ready"
            or obs is None
            or self.goal is None
            or self.future is not None
        ):
            return
        capture_ns = obs.image.header.stamp.sec * 1_000_000_000 + obs.image.header.stamp.nanosec
        if not self.cycle.can_request(capture_ns, now, now_ns):
            return
        token = self.cycle.request(int(obs.frame_id), capture_ns, now, now_ns)
        self.request_token = token
        self._event(
            "request",
            frame_id=token.frame_id,
            capture_ns=capture_ns,
            barrier_ns=self.cycle.floor_ns,
        )
        self.future = self.worker.submit(
            infer, self.url, token.frame_id, self.goal, bytes(obs.image.data)
        )

    def destroy_node(self) -> None:
        self.cycle.cancel(self._now_ns(), self._now(), stopped=True)
        if rclpy.ok():
            self._stop_power()
            deadline = time.monotonic() + 5
            while self.power_state != "ready" and rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                self._power_tick(self._now(), self._now_ns())
            if self.power_state != "ready":
                self._event("shutdown_stop_unconfirmed", power_state=self.power_state)
        self.worker.shutdown(wait=True, cancel_futures=True)
        super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--max-v", type=float, default=0.1)
    parser.add_argument("--max-w", type=float, default=0.35)
    parser.add_argument("--duration", type=float, default=7200)
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.max_v, args.max_w)):
        parser.error("speed limits must be finite and positive")
    with urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=4) as response:
        if json.load(response)["backend"] != "navida":
            parser.error("this executor requires the NaVIDA backend")
    stopping = threading.Event()

    def request_shutdown(signum: int, frame: FrameType | None) -> None:
        stopping.set()

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = NavidaStepBridge(args.url, args.odom_topic, args.max_v, args.max_w)
    previous = {s: signal.signal(s, request_shutdown) for s in (signal.SIGINT, signal.SIGTERM)}
    deadline = time.monotonic() + args.duration
    node._event("executor_ready", odom_topic=args.odom_topic)
    try:
        while rclpy.ok() and not stopping.is_set() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
