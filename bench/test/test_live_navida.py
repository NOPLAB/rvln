"""Real ROS objects and HTTP events exercise cancellation while inference is busy."""

import importlib
import json
import math
import signal
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from bench.navida_step import Action

rclpy = pytest.importorskip("rclpy")
messages = importlib.import_module("rvln_msgs.msg")
GoalSpec, Observation = messages.GoalSpec, messages.Observation
SetBool = importlib.import_module("std_srvs.srv").SetBool
Odometry = importlib.import_module("nav_msgs.msg").Odometry
live_navida = importlib.import_module("bench.live_navida")
NavidaStepBridge, main = live_navida.NavidaStepBridge, live_navida.main


class MotorPower:
    """Requests do not change hardware until the service executes and acknowledges."""

    def __init__(self):
        self.requests = []
        self.enabled = True
        self.removed = []

    def call_async(self, request):
        future = Future()
        self.requests.append((request.data, future))
        return future

    def acknowledge(self, success=True):
        enable, future = self.requests[-1]
        if success:
            self.enabled = enable
        future.set_result(SetBool.Response(success=success))

    def remove_pending_request(self, future):
        self.removed.append(future)


def odom(node, t, x=0, yaw=0, *, yaw_rate=0):
    node._now = lambda: t
    node._now_ns = lambda: round(t * 1e9)
    msg = Odometry()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(round(t * 1e9), 10**9)
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.orientation.z = math.sin(yaw / 2)
    msg.pose.pose.orientation.w = math.cos(yaw / 2)
    msg.twist.twist.angular.z = float(yaw_rate)
    node._odom(msg)


def finish_stop(node, start):
    node.power.acknowledge()
    node._power_tick(start, round(start * 1e9))
    for t in (start + 0.01, start + 0.09, start + 0.17):
        odom(node, t)
    node._power_tick(node._now(), node._now_ns())
    assert node.power.requests[-1][0] is True
    node.power.acknowledge()
    node._power_tick(node._now(), node._now_ns())
    for t in (start + 0.18, start + 0.28, start + 0.38):
        odom(node, t)
    assert node.power_state == "ready"


def shutdown_spin(node, **kwargs):
    if node.power_future is not None and not node.power_future.done():
        node.power.acknowledge()
    odom(node, node._now() + 0.1)


@pytest.fixture()
def bridge(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    frame_ids = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            frame_ids.append(request["frame_id"])
            entered.set()
            assert release.wait(5), "test did not release HTTP response"
            payload = json.dumps(
                {
                    "frame_id": request["frame_id"],
                    "embedding": [[0, 0, math.cos(0.2), math.sin(0.2)]],
                    "diagnostics": {"raw_response": ""},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    rclpy.init()
    node = NavidaStepBridge(f"http://127.0.0.1:{server.server_port}")
    node.power = MotorPower()
    monkeypatch.setattr(rclpy, "spin_once", shutdown_spin)
    now = [1.0]
    monkeypatch.setattr(node, "_now", lambda: now[0])
    monkeypatch.setattr(node, "_now_ns", lambda: round(now[0] * 1e9))
    commands = []
    monkeypatch.setattr(node.publisher, "publish", commands.append)
    node._set_goal("target")
    finish_stop(node, 1.0)
    odom(node, 1.4)
    obs = Observation()
    obs.frame_id = 10
    obs.goal.mode = GoalSpec.MODE_TEXT
    obs.goal.text = "target"
    obs.image.header.stamp.sec = 1
    obs.image.header.stamp.nanosec = 400000000
    obs.image.data = b"jpeg"
    node._observation(obs)
    destroyed = [False]

    def destroy():
        if not destroyed[0]:
            node.destroy_node()
            destroyed[0] = True

    try:
        yield node, entered, release, commands, frame_ids, destroy
    finally:
        release.set()
        destroy()
        rclpy.shutdown()
        server.shutdown()
        server_thread.join(2)
        server.server_close()
        assert not server_thread.is_alive()


def test_control_callback_returns_and_stop_wins_while_http_is_blocked(bridge):
    node, entered, release, commands, frame_ids, destroy = bridge
    returned = threading.Event()

    def tick():
        node._tick()
        returned.set()

    with ThreadPoolExecutor(max_workers=1) as runner:
        runner.submit(tick)
        assert entered.wait(2)
        assert returned.wait(2), "HTTP blocked the command/watchdog callback"
        response = node._forced_stop(SetBool.Request(data=True), SetBool.Response())
        done = threading.Event()
        node.future.add_done_callback(lambda future: done.set())
        release.set()
        assert done.wait(2)
        node._tick()

    assert response.success
    assert node.cycle.stopped
    assert node.cycle.action is None
    assert commands[-1].linear.x == commands[-1].angular.z == 0
    assert frame_ids == [10]


def test_goal_change_discards_reply_without_queuing_replacement(bridge):
    node, entered, release, commands, frame_ids, destroy = bridge
    node._tick()
    assert entered.wait(2)
    node._set_goal("other")
    done = threading.Event()
    node.future.add_done_callback(lambda future: done.set())
    release.set()
    assert done.wait(2)

    node._tick()

    assert node.cycle.action is None
    assert commands[-1].linear.x == commands[-1].angular.z == 0
    assert frame_ids == [10]


def test_republished_goal_cannot_clear_operator_stop(bridge):
    node, entered, release, commands, frame_ids, destroy = bridge
    node._forced_stop(SetBool.Request(data=True), SetBool.Response())
    goal = GoalSpec()
    goal.mode = GoalSpec.MODE_TEXT
    goal.text = "target"

    node._goal(goal)
    node._tick()

    assert node.cycle.stopped
    assert frame_ids == []
    assert commands[-1].linear.x == commands[-1].angular.z == 0


def test_shutdown_publishes_zero_before_destroying_publisher(bridge):
    node, entered, release, commands, frame_ids, destroy = bridge
    node.cycle.stopped = False

    destroy()

    assert node.power_state == "ready"
    assert node.power.enabled
    assert commands[-1].linear.x == commands[-1].angular.z == 0


def test_eligible_new_goal_image_waits_for_old_worker_to_finish(bridge):
    node, entered, release, commands, frame_ids, destroy = bridge
    node._tick()
    assert entered.wait(2)
    pending = node.future
    node._set_goal("other")
    finish_stop(node, 1.4)
    odom(node, 1.85)
    obs = Observation()
    obs.frame_id = 11
    obs.goal.mode = GoalSpec.MODE_TEXT
    obs.goal.text = "other"
    obs.image.header.stamp.sec = 1
    obs.image.header.stamp.nanosec = 850000000
    obs.image.data = b"jpeg"
    node._observation(obs)
    assert node.cycle.can_request(1850000000, 1.85, 1850000000)

    node._tick()

    assert node.future is pending
    assert frame_ids == [10]


def start_turn(node):
    token = node.cycle.request(20, 1400000000, 1.4, 1400000000)
    assert node.cycle.accept(token, Action(0, 0.4), 1.4, 1400000000)
    node.latest = None
    node._tick()
    return node.cycle.generation, node.cycle.motion_at, node.cycle.origin


def internal_stop(node):
    identity = start_turn(node)
    odom(node, 2.16, yaw=0.4)
    node._tick()
    assert node.cycle.settling
    assert node.power.requests[-1][0] is False
    return identity


@pytest.mark.parametrize("t,yaw", [(2.16, 0.4), (1.5, 0.4)])
def test_settling_publishes_zero_then_requests_motor_off_in_same_tick(bridge, monkeypatch, t, yaw):
    node, entered, release, commands, frame_ids, destroy = bridge
    start_turn(node)
    events = []
    call_async = node.power.call_async

    def publish(msg):
        events.append(("command", msg.linear.x, msg.angular.z, node._now()))
        commands.append(msg)

    def request_power(request):
        assert node.cycle.settling
        assert events[-1] == ("command", 0, 0, t)
        events.append(("power", request.data, node._now()))
        return call_async(request)

    monkeypatch.setattr(node.publisher, "publish", publish)
    monkeypatch.setattr(node.power, "call_async", request_power)
    odom(node, t, yaw=yaw)
    requests_before = len(node.power.requests)
    node._tick()
    assert node.cycle.settling
    assert events[-1] == ("power", False, t)
    assert len(node.power.requests) == requests_before + 1
    assert node.power_state == "disabling"
    assert not node.power_future.done()
    monkeypatch.setattr(node.power, "call_async", call_async)


def test_heading_correction_waits_for_both_acks_and_stable_post_enable_pose(bridge):
    node, entered, release, commands, frame_ids, destroy = bridge
    identity = internal_stop(node)
    stopped_at = len(commands)
    for t in (2.2, 2.3, 2.4):
        odom(node, t, yaw=0.1)
        node._tick()
    assert node.cycle.action is None
    assert node.cycle.settling
    assert not node.power_future.done()
    assert not node.cycle._turn_emitted
    assert all(command.linear.x == command.angular.z == 0 for command in commands[stopped_at:])

    node.power.acknowledge()
    node._tick()
    for t in (2.41, 2.51, 2.61):
        odom(node, t, yaw=0.1)
        node._tick()
    assert node.power_state == "enabling"
    assert not node.cycle._turn_emitted
    odom(node, 2.71, yaw=0.1)
    node._tick()
    assert commands[-1].angular.z == 0
    node.power.acknowledge()
    node._tick()
    for t, yaw in ((2.72, 0.1), (2.82, 0.11), (2.92, 0.11), (3.02, 0.11)):
        odom(node, t, yaw=yaw)
        if t < 3.02:
            node._tick()
            assert commands[-1].angular.z == 0
    assert node.power_state == "ready"
    assert (node.cycle.generation, node.cycle.motion_at, node.cycle.origin) == identity
    assert node.cycle.turn_progress == pytest.approx(0.11)
    node._tick()
    assert commands[-1].angular.z == 0.35
    assert frame_ids == []


def test_final_stop_reenable_updates_capture_barrier_before_inference(bridge):
    node, entered, release, commands, frame_ids, destroy = bridge
    start_turn(node)
    odom(node, 1.5, yaw=0.4)
    node._tick()
    assert node.power.requests[-1][0] is False
    node.power.acknowledge()
    node._tick()
    for t in (1.51, 1.61, 1.71):
        odom(node, t, yaw=0.4)
        node._tick()
    assert node.power_state == "enabling"
    obs = Observation()
    obs.frame_id = 21
    obs.goal.mode, obs.goal.text = GoalSpec.MODE_TEXT, "target"
    obs.image.header.stamp.sec, obs.image.header.stamp.nanosec = 1, 710000000
    obs.image.data = b"jpeg"
    node._observation(obs)
    node._tick()
    assert node.future is None
    node.power.acknowledge()
    node._tick()
    for t in (1.72, 1.82):
        odom(node, t, yaw=0.4)
        node._tick()
        assert node.future is None
    odom(node, 1.82, yaw=0.4)
    assert node.power_state == "verify"
    odom(node, 1.92, yaw=0.4)
    node._tick()
    assert node.power_state == "ready" and node.power.enabled
    assert node.cycle.floor_ns == 1920000000
    assert node.future is None
    odom(node, 1.93, yaw=0.4)
    obs.image.header.stamp.nanosec = 930000000
    node._observation(obs)
    node._tick()
    assert entered.wait(2)
    assert frame_ids == [21]
    assert commands[-1].linear.x == commands[-1].angular.z == 0


@pytest.mark.parametrize("failure", ["reject", "exception", "timeout"])
@pytest.mark.parametrize("enable", [False, True])
def test_power_failures_disable_and_never_emit_motion(bridge, failure, enable):
    node, entered, release, commands, frame_ids, destroy = bridge
    internal_stop(node)
    if enable:
        node.power.acknowledge()
        node._tick()
        for t in (2.17, 2.27, 2.37):
            odom(node, t, yaw=0.1)
            node._tick()
        assert node.power_state == "enabling"
    if failure == "reject":
        node.power.acknowledge(False)
    elif failure == "exception":
        node.power_future.set_exception(OSError("service disconnected"))
    else:
        odom(node, node._now() + 2.01, yaw=0.1)
    failed = node.power_future
    node._tick()
    assert node.cycle.stopped
    assert node.cycle.fault
    assert node.power_state == "disabling"
    assert node.power.requests[-1][0] is False
    assert commands[-1].linear.x == commands[-1].angular.z == 0
    assert not frame_ids
    if failure == "timeout":
        assert failed in node.power.removed


def test_gate_keeps_stale_odometry_watchdog_active(bridge):
    node, entered, release, commands, frame_ids, destroy = bridge
    internal_stop(node)
    node.power.acknowledge()
    node._tick()
    for t in (2.17, 2.27, 2.37):
        odom(node, t, yaw=0.1)
        node._tick()
    assert node.power_state == "enabling"
    node._now = lambda: 2.68
    node._now_ns = lambda: 2680000000
    node._tick()
    assert node.cycle.stopped
    assert node.cycle.fault == "odometry stale"
    assert commands[-1].angular.z == 0


def test_post_enable_confirmation_is_bounded_and_disables_on_failure(bridge):
    node, entered, release, commands, frame_ids, destroy = bridge
    internal_stop(node)
    node.power.acknowledge()
    node._tick()
    for t in (2.17, 2.27, 2.37):
        odom(node, t, yaw=0.1)
        node._tick()
    node.power.acknowledge()
    node._tick()
    assert node.power_state == "verify"
    odom(node, 4.38, yaw=0.2)
    node._tick()
    assert node.cycle.stopped
    assert node.cycle.fault == "post-enable stop not confirmed"
    assert node.power.requests[-1][0] is False
    assert commands[-1].angular.z == 0


def test_gate_keeps_absolute_motion_deadline_active(bridge):
    node, entered, release, commands, frame_ids, destroy = bridge
    internal_stop(node)
    node.power.acknowledge()
    node._tick()
    for t in (2.17, 2.27, 2.37):
        odom(node, t, yaw=0.1)
        node._tick()
    assert node.power_state == "enabling"
    node.cycle.motion_at = -100
    node._tick()
    assert node.cycle.stopped
    assert node.cycle.fault == "motion timeout"
    assert commands[-1].angular.z == 0


def test_shutdown_without_stop_evidence_never_restores_power(bridge, monkeypatch):
    node, entered, release, commands, frame_ids, destroy = bridge
    clock = [1.4]
    node._now = lambda: clock[0]
    node._now_ns = lambda: round(clock[0] * 1e9)
    monkeypatch.setattr(live_navida.time, "monotonic", lambda: clock[0])

    def service_unavailable(node, **kwargs):
        clock[0] += 1

    monkeypatch.setattr(rclpy, "spin_once", service_unavailable)
    first = len(node.power.requests)
    destroy()
    assert all(not enable for enable, future in node.power.requests[first:])
    assert node.power_state != "ready"
    assert commands[-1].linear.x == commands[-1].angular.z == 0


def test_forced_stop_during_enable_ack_disables_again(bridge):
    node, entered, release, commands, frame_ids, destroy = bridge
    internal_stop(node)
    node.power.acknowledge()
    node._tick()
    for t in (2.17, 2.27, 2.37):
        odom(node, t, yaw=0.1)
        node._tick()
    assert node.power_state == "enabling"
    node._forced_stop(SetBool.Request(data=True), SetBool.Response())
    node.power.acknowledge()
    node._tick()
    assert node.power.requests[-1][0] is False
    assert node.cycle.stopped
    assert commands[-1].angular.z == 0


@pytest.mark.parametrize("terminate", [False, True])
def test_cli_entrypoint_publishes_zero_before_shutdown(monkeypatch, terminate):
    requests = []
    published = []
    original_publish = NavidaStepBridge._publish

    def record(node, linear=0, angular=0):
        published.append((linear, angular, rclpy.ok()))
        original_publish(node, linear, angular)

    monkeypatch.setattr(NavidaStepBridge, "_publish", record)
    monkeypatch.setattr(NavidaStepBridge, "create_client", lambda *args: MotorPower())

    def spin(node, **kwargs):
        if terminate:
            signal.raise_signal(signal.SIGTERM)
        shutdown_spin(node)

    monkeypatch.setattr(rclpy, "spin_once", spin)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            payload = b'{"backend":"navida"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_navida",
            "--url",
            f"http://127.0.0.1:{server.server_port}",
            "--duration",
            "7200" if terminate else "0",
        ],
    )
    try:
        main()
    finally:
        server.shutdown()
        thread.join(2)
        server.server_close()

    assert requests == ["/health"]
    assert published[-1] == (0, 0, True)
    assert not thread.is_alive()


def test_chunk_actions_keep_remaining_primitives_in_order():
    actions = live_navida.chunk_actions("turn right 15 degree, forward 25 cm, turn left 30 degree")

    assert actions == [Action(0.25, 0.0), Action(0.0, pytest.approx(math.radians(30)))]
