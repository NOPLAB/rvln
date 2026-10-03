"""RVLN owns configuration and resources, not generic engine implementations."""

from unittest.mock import patch

import pytest

from bench.cli import build_parser
from isaac_rvln.cli import register
from usim.simulation import SimulationConfig


def test_rvln_dispatches_typed_configuration_to_generic_port(tmp_path):
    world, robot = tmp_path / "room.usd", tmp_path / "robot.urdf"
    world.touch()
    robot.touch()
    args = build_parser([register]).parse_args(
        [
            "rvln",
            "--world",
            str(world),
            "--robot-urdf",
            str(robot),
            "--cmd-vel-topic",
            "/custom/drive",
            "--robot-prim-path",
            "/World/CustomRobot",
            "--environment-prim-path",
            "/World/CustomRoom",
            "--camera-prim-path",
            "/World/CustomCamera",
            "--ground-name",
            "Floor",
            "--contact-out",
            str(tmp_path / "contacts.json"),
        ]
    )

    with patch("usim.ports.isaac.runner.run") as execute:
        args.handler(args)
    execute.assert_called_once()
    configuration, port = execute.call_args.args
    assert isinstance(configuration, SimulationConfig)
    assert configuration.world == world
    assert configuration.robot_urdf == robot
    assert configuration.wheel_radius == 0.0762
    assert configuration.wheel_separation == 0.27918
    assert configuration.camera_offset == (0.1, 0.0, 0.1433)
    assert configuration.robot_name == "raspicat"
    assert configuration.ros.node_name == "isaac_rvln_bridge"
    assert configuration.ros.cmd_vel_topic == "/custom/drive"
    assert port.robot_prim_path == "/World/CustomRobot"
    assert port.environment_prim_path == "/World/CustomRoom"
    assert port.camera_prim_path == "/World/CustomCamera"
    assert port.ground_name == "Floor"
    assert port.contact_out == tmp_path / "contacts.json"
    assert execute.call_args.kwargs["stop"] is None


def test_rvln_passive_physics_dispatches_without_ros_or_camera(tmp_path):
    world, robot = tmp_path / "room.usd", tmp_path / "robot.urdf"
    world.touch()
    robot.touch()
    args = build_parser([register]).parse_args(
        [
            "rvln",
            "--world",
            str(world),
            "--robot-urdf",
            str(robot),
            "--no-ros",
            "--physics-only",
            "--headless",
            "--max-seconds",
            "1.5",
        ]
    )

    with patch("usim.ports.isaac.runner.run") as execute:
        args.handler(args)

    configuration, port = execute.call_args.args
    assert configuration.ros is None
    assert not configuration.camera_enabled
    assert configuration.headless
    assert configuration.max_seconds == 1.5
    assert port.robot_prim_path == "/World/Raspicat"
    assert port.camera_prim_path == "/World/RVLN_Camera"


@pytest.mark.parametrize(
    "option,value",
    [("--wheel-radius", "0"), ("--max-seconds", "nan"), ("--camera-hz", "0")],
)
def test_rvln_rejects_invalid_configuration_before_engine_execution(option, value):
    args = build_parser([register]).parse_args(
        ["rvln", "--world", "room.usd", "--robot-urdf", "robot.urdf", option, value]
    )

    with patch("usim.ports.isaac.runner.run") as execute:
        with pytest.raises(ValueError):
            args.handler(args)

    execute.assert_not_called()


def test_plugin_discovery_uses_benchmark_group_without_usim_metadata():
    class Entry:
        name = "fixture"

        def load(self):
            return register

    with patch("bench.cli.entry_points", return_value=[Entry()]) as discovery:
        parser = build_parser()
    discovery.assert_called_once_with(group="rvln_bench.plugins")
    args = parser.parse_args(["rvln", "--world", "room.usd", "--robot-urdf", "robot.urdf"])
    assert args.robot_name == "raspicat"
