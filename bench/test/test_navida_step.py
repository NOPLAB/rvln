"""Synthetic clocks prove finite motion and acquisition-time inference gating."""

import math
from collections import deque

import pytest

from bench.navida_step import Action, Pose, StepCycle, StepDataError, decode_action


def ns(seconds):
    return round(seconds * 1e9)


def ready(pose=Pose(0, 0, 0)):
    cycle = StepCycle()
    cycle.cancel(ns(1), 1)
    for t in (1.1, 1.2, 1.3):
        cycle.observe(pose, ns(t), t, ns(t))
    return cycle


def moving(action, pose=Pose(0, 0, 0)):
    cycle = ready(pose)
    request = cycle.request(10, ns(1.4), 1.4, ns(1.4))
    assert cycle.accept(request, action, 1.5, ns(1.5))
    return cycle


def test_initial_odometry_discovery_does_not_expire_stop_confirmation():
    cycle = StepCycle()
    cycle.cancel(ns(1), 1)
    cycle.observe(Pose(0, 0, 0), ns(10), 10, ns(10))

    assert cycle.command(10, ns(10)) == (0, 0)
    assert cycle.fault is None
    for t in (10.1, 10.2):
        cycle.observe(Pose(0, 0, 0), ns(t), t, ns(t))
    assert not cycle.settling


@pytest.mark.parametrize("direction", [-1, 1])
def test_turn_stops_after_requested_angle_even_when_same_action_remains(direction):
    cycle = moving(Action(0, direction * math.radians(15)))
    cycle.observe(Pose(0, 0, direction * math.radians(15)), ns(1.6), 1.6, ns(1.6))

    assert cycle.command(1.6, ns(1.6)) == (0, 0)
    assert cycle.action is None
    assert not cycle.can_request(ns(1.65), 1.65, ns(1.65))


@pytest.mark.parametrize("direction", [-1, 1])
def test_turn_counts_progress_across_yaw_wrap(direction):
    start = direction * math.radians(179)
    cycle = moving(Action(0, direction * math.radians(15)), Pose(0, 0, start))
    end = math.atan2(
        math.sin(start + direction * math.radians(15)),
        math.cos(start + direction * math.radians(15)),
    )
    cycle.observe(Pose(0, 0, end), ns(1.6), 1.6, ns(1.6))

    assert cycle.command(1.6, ns(1.6)) == (0, 0)


def test_turn_overshoot_corrects_heading_before_next_request():
    cycle = moving(Action(0, math.radians(15)))
    cycle.observe(Pose(0, 0, math.radians(25)), ns(1.6), 1.6, ns(1.6))

    linear, angular = cycle.command(1.6, ns(1.6))
    assert linear == 0
    assert -0.35 <= angular < 0
    assert not cycle.can_request(ns(1.65), 1.65, ns(1.65))


def test_yaw_feedback_reduces_command_as_angular_velocity_rises():
    cycle = moving(Action(0, math.radians(15)))
    cycle.observe(Pose(0, 0, 0.1, yaw_rate=0.2), ns(1.6), 1.6, ns(1.6))
    _, slow = cycle.command(1.6, ns(1.6))
    cycle.observe(Pose(0, 0, 0.1, yaw_rate=0.3), ns(1.65), 1.65, ns(1.65))
    _, fast = cycle.command(1.65, ns(1.65))

    assert fast < slow < 0.35
    assert slow > 0
    assert abs(fast) <= 0.35


def test_target_angle_does_not_finish_while_imu_still_reports_rotation():
    target = math.radians(15)
    cycle = moving(Action(0, target))
    cycle.observe(Pose(0, 0, target, yaw_rate=0.1), ns(1.6), 1.6, ns(1.6))

    assert cycle.command(1.6, ns(1.6)) == (0, 0)
    assert not cycle.settling
    assert not cycle.can_request(ns(1.65), 1.65, ns(1.65))
    cycle.observe(Pose(0, 0, target), ns(1.7), 1.7, ns(1.7))
    assert cycle.command(1.7, ns(1.7)) == (0, 0)
    assert cycle.settling


def test_forward_consumes_distance_along_original_heading():
    cycle = moving(Action(0.25, 0), Pose(1, 2, math.pi / 2))
    cycle.observe(Pose(1, 2.25, math.pi / 2), ns(1.6), 1.6, ns(1.6))

    assert cycle.command(1.6, ns(1.6)) == (0, 0)


def test_sideways_displacement_does_not_consume_forward_distance():
    cycle = moving(Action(0.25, 0))
    cycle.observe(Pose(0, 0.3, 0), ns(1.6), 1.6, ns(1.6))

    assert cycle.command(1.6, ns(1.6)) == (0.1, 0)


@pytest.mark.parametrize(
    "action,expected",
    [
        (Action(0.01, 0), (0.015, 0)),
        (Action(0, math.radians(1)), (0, 0.35)),
    ],
)
def test_smallest_supported_action_is_not_consumed_at_origin(action, expected):
    cycle = moving(action)

    assert cycle.command(1.5, ns(1.5)) == expected


def test_precompletion_image_delivered_later_is_rejected():
    cycle = moving(Action(0, math.radians(15)))
    pose = Pose(0, 0, math.radians(15))
    cycle.observe(pose, ns(1.6), 1.6, ns(1.6))
    cycle.command(1.6, ns(1.6))
    for t in (1.7, 1.8, 1.9):
        cycle.observe(pose, ns(t), t, ns(t))

    assert not cycle.can_request(ns(1.85), 2, ns(2))
    assert cycle.can_request(ns(1.95), 2, ns(2))


def test_publication_of_zero_is_not_physical_completion():
    cycle = moving(Action(0, math.radians(15)))
    cycle.observe(Pose(0, 0, math.radians(15)), ns(1.6), 1.6, ns(1.6))
    cycle.command(1.6, ns(1.6))
    cycle.observe(Pose(0, 0, math.radians(18)), ns(1.7), 1.7, ns(1.7))
    cycle.observe(Pose(0, 0, math.radians(21)), ns(1.8), 1.8, ns(1.8))

    assert not cycle.can_request(ns(1.85), 1.85, ns(1.85))


def test_small_per_sample_motion_is_not_mistaken_for_stopped():
    cycle = moving(Action(0, math.radians(15)))
    cycle.observe(Pose(0, 0, math.radians(15)), ns(1.6), 1.6, ns(1.6))
    cycle.command(1.6, ns(1.6))
    for i in range(1, 21):
        t = 1.6 + i * 0.01
        cycle.observe(Pose(0, 0, math.radians(15) + i * 0.001), ns(t), t, ns(t))

    assert cycle.settling
    assert not cycle.can_request(ns(1.85), 1.85, ns(1.85))


def test_three_fast_stationary_samples_do_not_prove_cessation():
    cycle = moving(Action(0, math.radians(15)))
    pose = Pose(0, 0, math.radians(15))
    cycle.observe(pose, ns(1.6), 1.6, ns(1.6))
    cycle.command(1.6, ns(1.6))
    for t in (1.61, 1.62, 1.63):
        cycle.observe(pose, ns(t), t, ns(t))

    assert cycle.settling


def test_stopped_progress_includes_coasting_after_zero_command():
    cycle = moving(Action(0, math.radians(15)))
    cycle.observe(Pose(0, 0, math.radians(14)), ns(1.6), 1.6, ns(1.6))
    cycle.command(1.6, ns(1.6))
    pose = Pose(0, 0, math.radians(16))
    for t in (1.7, 1.8, 1.9):
        cycle.observe(pose, ns(t), t, ns(t))

    assert cycle.turn_progress == pytest.approx(math.radians(16))
    assert not cycle.settling


@pytest.mark.parametrize(
    "action",
    [
        Action(0, math.radians(15)),
        Action(0, -math.radians(15)),
        Action(0.25, 0),
    ],
)
def test_delayed_actuator_stops_within_requested_action_tolerance(action):
    cycle = moving(action)
    commands = deque([(0, 0)] * 10)
    x, yaw = 0.0, 0.0
    for i in range(1, 501):
        t = 1.5 + i * 0.05
        v, w = commands.popleft()
        x += v * 0.05
        yaw += w * 0.05
        cycle.observe(Pose(x, 0, yaw, yaw_rate=w), ns(t), t, ns(t))
        commands.append(cycle.command(t, ns(t)))
        if cycle.action is None and not cycle.settling:
            break

    assert not cycle.fault
    assert not cycle.settling
    assert x == pytest.approx(action.distance, abs=0.05)
    assert yaw == pytest.approx(action.angle, abs=math.radians(5))


@pytest.mark.parametrize("direction", [-1, 1])
def test_turn_completes_when_low_speed_commands_cannot_start_wheels(direction):
    target = direction * math.radians(15)
    cycle = moving(Action(0, target))
    commands = deque([0.0] * 10)
    yaw = 0.0
    for i in range(1, 501):
        t = 1.5 + i * 0.05
        angular = commands.popleft()
        rate = angular if abs(angular) >= 0.2 else 0.0
        yaw += rate * 0.05
        cycle.observe(Pose(0, 0, yaw, yaw_rate=rate), ns(t), t, ns(t))
        _, angular = cycle.command(t, ns(t))
        commands.append(angular)
        if cycle.action is None and not cycle.settling:
            break

    assert cycle.fault is None
    assert not cycle.settling
    assert yaw == pytest.approx(target, abs=math.radians(5))


@pytest.mark.parametrize("max_v,max_w", [(0.01, 0.35), (0.1, 0.04), (0.1, 0.3)])
def test_driver_deadband_speed_limits_are_rejected(max_v, max_w):
    with pytest.raises(ValueError, match="below the robot execution minimum"):
        StepCycle(max_v, max_w)


def test_turn_brakes_and_reassesses_without_completing_or_requesting():
    cycle = moving(Action(0, math.radians(15)))
    assert cycle.command(1.5, ns(1.5)) == (0, 0.35)
    cycle.observe(Pose(0, 0, math.radians(8), yaw_rate=0.4), ns(2.25), 2.25, ns(2.25))
    barrier = cycle.floor_ns

    assert cycle.command(2.25, ns(2.25))[1] < 0
    assert cycle.action == Action(0, math.radians(15))
    assert not cycle.settling
    for t in (2.3, 2.4, 2.5):
        cycle.observe(Pose(0, 0, math.radians(8)), ns(t), t, ns(t))

    assert cycle.action is not None
    assert not cycle.can_request(ns(2.55), 2.55, ns(2.55))
    assert cycle.floor_ns == barrier
    assert cycle.motion_at == 1.5


def test_forced_stop_discards_inflight_reply():
    cycle = ready()
    request = cycle.request(10, ns(1.4), 1.4, ns(1.4))
    cycle.cancel(ns(1.5), 1.5, stopped=True)

    assert not cycle.accept(request, Action(0, 0.2), 1.5, ns(1.5))
    assert cycle.command(1.5, ns(1.5)) == (0, 0)


def test_goal_change_discards_old_reply():
    cycle = ready()
    request = cycle.request(10, ns(1.4), 1.4, ns(1.4))
    cycle.cancel(ns(1.5), 1.5)

    assert not cycle.accept(request, Action(0.25, 0), 1.5, ns(1.5))


def test_expired_reply_cannot_start_motion():
    cycle = ready()
    request = cycle.request(10, ns(1.4), 1.4, ns(1.4))
    cycle.observe(Pose(0, 0, 0), ns(3), 3, ns(3))

    assert not cycle.accept(request, Action(0.25, 0), 3, ns(3))
    assert cycle.fault == "reply expired"


def test_stale_odometry_latches_stop():
    cycle = moving(Action(0, 0.2))

    assert cycle.command(1.7, ns(1.7)) == (0, 0)
    assert cycle.fault == "odometry stale"
    assert cycle.stopped


def test_frozen_odometry_stamp_does_not_refresh_watchdog():
    cycle = moving(Action(0, 0.2))
    cycle.observe(Pose(0, 0, 0), ns(1.3), 1.7, ns(1.7))

    assert cycle.command(1.7, ns(1.7)) == (0, 0)
    assert cycle.stopped


def test_motion_without_progress_times_out():
    cycle = moving(Action(0, 0.2))
    cycle.observe(Pose(0, 0, 0), ns(16), 16, ns(16))

    assert cycle.command(16, ns(16)) == (0, 0)
    assert cycle.fault == "motion timeout"


def test_motor_gate_does_not_start_a_pulse_before_enable_confirmation():
    cycle = moving(Action(0, 0.2))
    assert cycle.command(1.5, ns(1.5), motion_enabled=False) == (0, 0)
    cycle.observe(Pose(0, 0, 0), ns(1.6), 1.6, ns(1.6))

    assert cycle.command(1.6, ns(1.6), motion_enabled=True) == (0, 0.35)
    assert cycle.motion_at == 1.5


def test_motor_gate_does_not_suspend_odometry_watchdog():
    cycle = moving(Action(0, 0.2))
    assert cycle.command(2, ns(2), motion_enabled=False) == (0, 0)
    assert cycle.fault == "odometry stale"


def test_second_request_is_not_queued():
    cycle = ready()
    cycle.request(10, ns(1.4), 1.4, ns(1.4))

    with pytest.raises(StepDataError):
        cycle.request(11, ns(1.45), 1.45, ns(1.45))


def test_decoder_does_not_double_packed_turn_angle():
    action = decode_action([0, 0, math.cos(0.2), math.sin(0.2)])

    assert action.angle == pytest.approx(0.2)


@pytest.mark.parametrize("token", [[0, 0, 0, 0], [0.25, 0.1, 1, 0], [float("nan"), 0, 1, 0]])
def test_decoder_rejects_invalid_primitives(token):
    with pytest.raises(StepDataError):
        decode_action(token)
