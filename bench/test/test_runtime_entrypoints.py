"""Portable CLI regressions for the mobile and evaluation boundaries."""

import json
from dataclasses import replace
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from usim.ports.isaac.cli import register as mobile_register
from isaac_rvln.cli import register as rvln_register
from unittest.mock import patch

from bench.cli import build_parser


class EntryPointTest(unittest.TestCase):
    def test_mobile_and_rvln_defaults_and_single_converter(self):
        for registrations in ([mobile_register, rvln_register], [rvln_register, mobile_register]):
            parser = build_parser(registrations)
            mobile = parser.parse_args(
                ["simulate", "--world", "room.usd", "--robot-urdf", "robot.urdf"]
            )
            legacy = parser.parse_args(
                ["rvln", "--world", "room.usd", "--robot-urdf", "robot.urdf"]
            )
            self.assertEqual(
                (mobile.robot_name, mobile.robot_prim_path), ("mobile_robot", "/World/Robot")
            )
            self.assertEqual(
                (legacy.robot_name, legacy.robot_prim_path), ("raspicat", "/World/Raspicat")
            )
            self.assertEqual(legacy.camera_offset, (0.1, 0.0, 0.1433))
            self.assertEqual(legacy.wheel_radius, 0.0762)
            conversion = parser.parse_args(
                ["convert-world", "--world", "room.world", "--out", "room.usd"]
            )
            self.assertTrue(callable(conversion.handler))
            custom = parser.parse_args(
                [
                    "simulate",
                    "--world",
                    "room.usd",
                    "--robot-urdf",
                    "robot.urdf",
                    "--left-joint",
                    "drive_l",
                    "--right-joint",
                    "drive_r",
                    "--camera-offset",
                    "0.3",
                    "0.2",
                    "0.5",
                    "--prim-path",
                    "/World/Custom",
                    "--robot-name",
                    "custom",
                    "--cmd-vel-topic",
                    "/drive",
                    "--odom-topic",
                    "/state",
                    "--rgb-topic",
                    "/rgb",
                    "--depth-topic",
                    "/depth",
                    "--motor-service",
                    "/enable",
                ]
            )
            self.assertEqual(custom.camera_offset, [0.3, 0.2, 0.5])
            self.assertEqual(
                (
                    custom.cmd_vel_topic,
                    custom.odom_topic,
                    custom.rgb_topic,
                    custom.depth_topic,
                    custom.motor_service,
                ),
                ("/drive", "/state", "/rgb", "/depth", "/enable"),
            )

    def _isolated(self, code, *arguments):
        # Prohibit heavy imports even if optional packages happen to be installed.
        guard = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
                'numpy', 'PIL', 'pxr', 'rclpy', 'cv2', 'rvln_msgs', 'isaacsim', 'omni'}:
            raise AssertionError('unexpected execution import: ' + fullname)
sys.meta_path.insert(0, Guard())
"""
        child = subprocess.run(
            [sys.executable, "-c", guard + code, *arguments],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
        return child

    def test_help_and_batch_import_without_execution_dependencies(self):
        self._isolated("import usim.bridges.ros2; import usim.ports.isaac")
        self._isolated("import isaac_r2r.batch; assert 'isaac_r2r.run' not in sys.modules")
        self._isolated(
            "import runpy; sys.argv=['live_episode', '--help']; "
            "runpy.run_module('isaac_rvln.live_episode', run_name='__main__')"
        )
        self._isolated(
            "from bench.cli import build_parser; "
            "from usim.ports.isaac.cli import register; "
            "build_parser([register]).parse_args(['simulate', '--help'])"
        )

    def test_all_registrars_and_offline_commands_without_optional_packages(self):
        setup = (
            "from bench.cli import build_parser; "
            "from usim.ports.isaac.cli import register as mobile; "
            "from isaac_rvln.cli import register as rvln; "
            "from isaac_r2r.cli import register as r2r; "
            "parser=build_parser([mobile, rvln, r2r]); "
        )
        self._isolated(setup + "parser.parse_args(['--help'])")
        self._isolated(setup + "parser.parse_args(['simulate', '--help'])")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, split = root / "robot.urdf", root / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "episodes": [
                            {
                                "episode_id": "one",
                                "scene_id": "mp3d/room/room.glb",
                                "instruction": {"instruction_text": "Go forward."},
                                "start_position": [0, 0, 0],
                                "goals": [{"position": [1, 0, 0]}],
                                "info": {"geodesic_distance": 1},
                            }
                        ]
                    }
                )
            )
            description = root / "description"
            (description / "urdf").mkdir(parents=True)
            links = (
                "base_link",
                "left_wheel_link",
                "right_wheel_link",
                "caster_link",
                "caster_wheel_link",
                "camera_link",
            )
            source = '<robot name="raspicat">' + "".join(
                f'<link name="{name}"/>' for name in links
            )
            source += (
                '<joint name="camera_joint" type="fixed">'
                '<parent link="base_link"/><child link="camera_link"/>'
                '<origin xyz="0 0 0"/></joint></robot>'
            )
            (description / "urdf/raspicat.urdf").write_text(source)
            self._isolated(
                setup + "args=parser.parse_args(['prepare-robot', '--out', sys.argv[1], "
                "'--description-root', sys.argv[2]]); "
                "row=args.handler(args); assert row['wheel_radius_m']==0.0762",
                str(output),
                str(description),
            )
            self.assertTrue(output.is_file())
            self._isolated(
                setup + "args=parser.parse_args(['inventory-r2r', '--split', sys.argv[1], "
                "'--scans-root', sys.argv[2]]); "
                "row=args.handler(args); assert row['episodes']==1 and row['scan_count']==0",
                str(split),
                str(root / "scans"),
            )

    def test_finalize_contacts_without_execution_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, contacts = root / "result.json", root / "contacts.json"
            result.write_text(
                json.dumps(
                    {
                        "model": "fixture",
                        "id": "one",
                        "pose_source": "isaac_ground_truth",
                        "wall_started_at": 10,
                        "wall_ended_at": 20,
                        "contact_log": None,
                    }
                )
            )
            contacts.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "simulator": "isaac_sim",
                        "receive_errors": 0,
                        "samples": {"physics_steps": 1},
                        "events": [],
                        "collisions": 0,
                    }
                )
            )
            self._isolated(
                "import runpy; sys.argv=['live_episode', '--out', sys.argv[1], "
                "'--finalize-contacts', sys.argv[2]]; "
                "runpy.run_module('isaac_rvln.live_episode', run_name='__main__')",
                str(result),
                str(contacts),
            )
            self.assertEqual(json.loads(result.read_text())["collisions"], 0)


class LibraryPortTest(unittest.TestCase):
    def test_typed_configuration_and_standalone_dispatch(self):
        from usim.ports.isaac import IsaacSimulator, SimulationConfig, simulate
        from usim.ports.isaac.contacts import ObstacleContactLog

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world, robot = root / "room.usd", root / "robot.urdf"
            world.write_bytes(b"fixture")
            robot.write_bytes(b"fixture")
            configuration = SimulationConfig(
                world=world, robot_urdf=robot, ros=None, camera_enabled=False
            )
            with patch("usim.ports.isaac.runner.run") as execute:
                simulate(configuration)
                self.assertIs(execute.call_args.args[0], configuration)
                self.assertIsNone(execute.call_args.kwargs["stop"])
            with self.assertRaises(ValueError):
                simulate(replace(configuration, wheel_radius=0))
            with self.assertRaises(ValueError):
                IsaacSimulator(robot_prim_path="/World/Environment/Robot")
            contacts = ObstacleContactLog(
                {"Wall"},
                clock=lambda: 12,
                robot_prim_path="/World/Custom",
                scene_prim_path="/World/Room",
            )
            contacts.record(
                "/World/Custom/Base",
                "/World/Room/Wall/Body",
                "/World/Custom/Shape",
                "/World/Room/Wall/Shape",
                1,
                2,
            )
            contacts.record(
                "/World/Raspicat/Base",
                "/World/Room/Wall/Body",
                "/World/Raspicat/Shape",
                "/World/Room/Wall/Shape",
                1,
                2,
            )
            self.assertEqual(contacts.report()["collisions"], 1)
