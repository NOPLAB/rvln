"""RVLN owns configuration and resources, not generic engine implementations."""

from unittest.mock import patch

from bench.cli import build_parser
from isaac_rvln.cli import register
from isaac_rvln.sim import IsaacRVLNSimulator
from usim.simulation import SimulationConfig


def test_rvln_dispatches_typed_configuration_to_generic_port(tmp_path):
    world, robot = tmp_path / "room.usd", tmp_path / "robot.urdf"
    args = build_parser([register]).parse_args(
        [
            "rvln",
            "--world",
            str(world),
            "--robot-urdf",
            str(robot),
            "--cmd-vel-topic",
            "/custom/drive",
        ]
    )

    def stop():
        return True

    with patch("isaac_rvln.sim.IsaacSimulator") as port:
        IsaacRVLNSimulator().run(args, stop=stop)
    port.assert_called_once_with(
        robot_prim_path="/World/Raspicat",
        environment_prim_path="/World/Environment",
        camera_prim_path="/World/RVLN_Camera",
        ground_name="Ground",
        contact_out=None,
    )
    configuration = port.return_value.run.call_args.args[0]
    assert isinstance(configuration, SimulationConfig)
    assert configuration.world == world
    assert configuration.robot_urdf == robot
    assert configuration.wheel_radius == 0.0762
    assert configuration.ros.cmd_vel_topic == "/custom/drive"
    assert port.return_value.run.call_args.kwargs["stop"] is stop


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
