"""Isaac implementation of the RVLN robot simulation commands."""
from __future__ import annotations

import argparse
from pathlib import Path


def run(args) -> None:
    from isaac_rvln.sim import IsaacRVLNSimulator, wheel_velocities

    for path in (args.world, args.robot_urdf):
        if not path.is_file():
            raise ValueError(f'missing asset: {path}')
    wheel_velocities(0, 0, args.wheel_radius, args.wheel_separation)
    if args.max_seconds < 0:
        raise ValueError('max-seconds must be nonnegative')
    IsaacRVLNSimulator().run(args)


def convert(args) -> dict:
    from isaac_rvln.worlds import convert_world

    return convert_world(args.world, args.out)


def prepare(args) -> dict:
    from isaac_rvln.robot import prepare_robot

    return prepare_robot(args.description_root, args.out)


def register(commands: argparse._SubParsersAction) -> None:
    rvln = commands.add_parser('rvln', help='run the Isaac ROS 2 robot bridge')
    rvln.add_argument('--world', type=Path, required=True)
    rvln.add_argument('--robot-urdf', type=Path, required=True)
    rvln.add_argument('--wheel-radius', type=float, default=0.0762)
    rvln.add_argument('--wheel-separation', type=float, default=0.27918)
    rvln.add_argument('--left-joint', default='left_wheel_joint')
    rvln.add_argument('--right-joint', default='right_wheel_joint')
    rvln.add_argument('--headless', action='store_true')
    rvln.add_argument('--physics-only', action='store_true',
                      help='run wheel and ROS checks without camera frames')
    rvln.add_argument('--max-seconds', type=float, default=0)
    rvln.add_argument('--contact-out', type=Path,
                      help='write obstacle-contact onsets from Isaac PhysX')
    rvln.set_defaults(handler=run)

    world = commands.add_parser('convert-world', help='convert a pilot SDF to USD')
    world.add_argument('--world', type=Path, required=True)
    world.add_argument('--out', type=Path, required=True)
    world.set_defaults(handler=convert)

    robot = commands.add_parser('prepare-robot', help='prepare the pinned Raspicat URDF')
    robot.add_argument('--description-root', type=Path, required=True)
    robot.add_argument('--out', type=Path, required=True)
    robot.set_defaults(handler=prepare)
