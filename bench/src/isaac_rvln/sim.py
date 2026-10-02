"""Compatibility runtime entry points for the RVLN Raspicat bridge."""

from bench.episode import ContinuousSimulator
from usim.ports.isaac import IsaacSimulator
from usim.simulation import wheel_velocities as wheel_velocities
from usim.bridges.ros_runtime import load_ros_python as _load_ros_python


__all__ = ["IsaacRVLNSimulator", "wheel_velocities", "_load_ros_python"]


def _run(args, *, stop=None) -> None:
    from isaac_rvln.cli import configuration

    IsaacSimulator(
        robot_prim_path=getattr(args, "robot_prim_path", "/World/Raspicat"),
        environment_prim_path=getattr(args, "environment_prim_path", "/World/Environment"),
        camera_prim_path=getattr(args, "camera_prim_path", "/World/RVLN_Camera"),
        ground_name=getattr(args, "ground_name", "Ground"),
        contact_out=getattr(args, "contact_out", None),
    ).run(configuration(args), stop=stop)


class IsaacRVLNSimulator(ContinuousSimulator):
    def run(self, configuration, *, stop=None) -> None:
        _run(configuration, stop=stop)
