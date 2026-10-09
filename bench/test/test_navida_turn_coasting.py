"""Turn completion under encoder latching, delayed output and motor deadband."""

import math
from collections import deque

import pytest

from bench.navida_step import Action, Pose, StepCycle


def ns(seconds):
    return round(seconds * 1e9)


def turn(direction=1):
    cycle = StepCycle()
    cycle.cancel(ns(1), 1)
    for t in (1.1, 1.2, 1.3):
        cycle.observe(Pose(0, 0, 0), ns(t), t, ns(t))
    request = cycle.request(1, ns(1.4), 1.4, ns(1.4))
    assert cycle.accept(request, Action(0, direction * math.radians(15)), 1.5, ns(1.5))
    return cycle


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("gain", [0.125, 0.75, 1.0])
@pytest.mark.parametrize("latch_phase", [0, 5, 10])
def test_turn_completes_with_latched_encoders_and_start_deadband(direction, gain, latch_phase):
    cycle = turn(direction)
    commands = deque([0.0] * 100)
    yaw = reported_yaw = command = 0.0
    for i in range(1, 5001):
        t = 1.5 + i * 0.005
        angular = commands.popleft()
        rate = gain * angular if abs(angular) >= 0.35 else 0.0
        yaw += rate * 0.005
        if i % 11 == latch_phase:
            reported_yaw = yaw
        cycle.observe(Pose(0, 0, reported_yaw, yaw_rate=rate), ns(t), t, ns(t))
        if i % 10 == 0:
            _, command = cycle.command(t, ns(t))
        commands.append(command)
        if cycle.action is None and not cycle.settling:
            break
        assert not cycle.can_request(ns(t), t, ns(t))

    assert cycle.fault is None
    assert not cycle.settling
    assert yaw == pytest.approx(direction * math.radians(15), abs=math.radians(5))


def test_turn_cancel_waits_for_delayed_commands_before_confirming_stop():
    cycle = turn()
    assert cycle.command(1.5, ns(1.5)) == (0, 0.35)
    cycle.cancel(ns(1.6), 1.6)
    for t in (1.7, 1.8, 1.9):
        cycle.observe(Pose(0, 0, 0), ns(t), t, ns(t))

    assert cycle.settling
    assert not cycle.can_request(ns(1.95), 1.95, ns(1.95))
    for t in (2.2, 2.3, 2.4):
        cycle.observe(Pose(0, 0, 0), ns(t), t, ns(t))
    assert cycle.can_request(ns(2.45), 2.45, ns(2.45))


def test_excessive_turn_coast_requires_correction_before_declaring_completion():
    cycle = turn()
    cycle.command(1.5, ns(1.5))
    pose = Pose(0, 0, math.radians(25))
    cycle.observe(pose, ns(1.6), 1.6, ns(1.6))
    assert cycle.command(1.6, ns(1.6))[1] < 0
    for t in (2.2, 2.3, 2.4):
        cycle.observe(pose, ns(t), t, ns(t))

    assert not cycle.stopped
    assert cycle.fault is None
    assert cycle.action is not None
    assert not cycle.can_request(ns(2.45), 2.45, ns(2.45))


def test_immobile_turn_still_times_out_without_coast_credit():
    cycle = turn()
    budget = 3 * math.radians(15) / 0.35 + 8
    for i in range(1, 501):
        t = 1.5 + i * 0.05
        cycle.observe(Pose(0, 0, 0), ns(t), t, ns(t))
        cycle.command(t, ns(t))
        if cycle.fault:
            break

    assert cycle.fault == "motion timeout"
    assert t - cycle.motion_at <= budget + 0.05


def test_tiny_progress_cannot_extend_turn_beyond_absolute_deadline():
    cycle = turn()
    budget = 3 * math.radians(15) / 0.35 + 8
    for i in range(1, 501):
        t = 1.5 + i * 0.05
        cycle.observe(Pose(0, 0, i * 0.00001), ns(t), t, ns(t))
        cycle.command(t, ns(t))
        if cycle.fault:
            break

    assert cycle.fault == "motion timeout"
    assert t - cycle.motion_at <= 2 * budget + 0.05


def test_fault_keeps_last_measured_turn_progress_for_diagnostics():
    cycle = turn()
    cycle.observe(Pose(0, 0, math.radians(3)), ns(16), 16, ns(16))
    assert cycle.command(16, ns(16)) == (0, 0)

    assert cycle.fault == "motion timeout"
    assert cycle.turn_progress == pytest.approx(math.radians(3))
