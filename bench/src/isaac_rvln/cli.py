"""Isaac implementation of the RVLN robot simulation commands."""

from __future__ import annotations

import argparse
from pathlib import Path


RVLN_DEFAULTS = {
    "wheel_radius": 0.0762,
    "wheel_separation": 0.27918,
    "camera_offset": (0.1, 0.0, 0.1433),
    "robot_name": "raspicat",
    "robot_prim_path": "/World/Raspicat",
    "camera_prim_path": "/World/RVLN_Camera",
    "node_name": "isaac_rvln_bridge",
}


def configuration(args):
    from usim.ports.isaac.cli import configured

    from usim.simulation import SimulationConfig

    if isinstance(args, SimulationConfig):
        return args
    return configured(argparse.Namespace(**{**RVLN_DEFAULTS, **vars(args)}))


def run(args) -> None:
    from isaac_rvln.sim import IsaacRVLNSimulator, wheel_velocities

    for path in (args.world, args.robot_urdf):
        if not path.is_file():
            raise ValueError(f"missing asset: {path}")
    wheel_velocities(0, 0, args.wheel_radius, args.wheel_separation)
    if args.max_seconds < 0:
        raise ValueError("max-seconds must be nonnegative")
    IsaacRVLNSimulator().run(args)


def convert(args) -> dict:
    from isaac_rvln.worlds import convert_world

    return convert_world(args.world, args.out)


def prepare(args) -> dict:
    from isaac_rvln.robot import prepare_robot

    return prepare_robot(args.description_root, args.out)


def register(commands: argparse._SubParsersAction) -> None:
    rvln = commands.add_parser("rvln", help="run the Isaac ROS 2 robot bridge")
    from usim.ports.isaac.cli import add_arguments, register_world

    add_arguments(rvln, defaults=RVLN_DEFAULTS)
    rvln.set_defaults(handler=run)
    register_world(commands)

    robot = commands.add_parser("prepare-robot", help="prepare the pinned Raspicat URDF")
    robot.add_argument("--description-root", type=Path, required=True)
    robot.add_argument("--out", type=Path, required=True)
    robot.set_defaults(handler=prepare)
