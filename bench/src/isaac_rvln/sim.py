"""Isaac Sim 6.1 ROS 2 bridge for the Raspicat VLA edge/follower contract.

Requires an expanded Raspicat URDF and a collidable, Z-up USD world. This
module is kept separate from ROS launch because Isaac uses Python 3.12.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

from bench.episode import ContinuousSimulator


_ROS_DLL_DIRECTORY = None


def _load_ros_python() -> None:
    """Use a sourced ROS environment or Isaac's bundled ROS 2 Python modules."""
    try:
        import rclpy  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    distro = os.environ.setdefault('ROS_DISTRO', 'humble')
    os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp')
    isaac_path = Path(os.environ['ISAAC_PATH'])
    bundle = isaac_path / 'exts' / 'isaacsim.ros2.core' / distro
    if not bundle.is_dir():
        bundle = isaac_path / 'exts' / 'isaacsim.ros2.bridge' / distro
    python = bundle / 'rclpy'
    libraries = bundle / 'lib'
    if not python.is_dir() or not libraries.is_dir():
        raise RuntimeError(f'Isaac bundled ROS {distro} is unavailable')
    sys.path.insert(0, str(python))
    if sys.platform == 'win32':
        os.environ['PATH'] += os.pathsep + str(libraries)
        global _ROS_DLL_DIRECTORY
        _ROS_DLL_DIRECTORY = os.add_dll_directory(str(libraries))
    elif str(libraries) not in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
        raise RuntimeError(f'set LD_LIBRARY_PATH={libraries}:$LD_LIBRARY_PATH '
                           'before starting Isaac to load bundled ROS libraries')
    import rclpy  # noqa: F401


def wheel_velocities(linear: float, angular: float, radius: float,
                     separation: float) -> tuple[float, float]:
    """Map body Twist to left/right wheel radians per second."""
    if radius <= 0 or separation <= 0:
        raise ValueError('wheel geometry must be positive')
    return ((linear - angular * separation / 2) / radius,
            (linear + angular * separation / 2) / radius)


class IsaacRVLNSimulator(ContinuousSimulator):
    """Concrete Isaac Sim ROS bridge behind the common simulator contract."""

    def run(self, configuration) -> None:
        _run(configuration)


def _run(args) -> None:
    from isaacsim import SimulationApp
    robot_assets = None
    app = SimulationApp({'headless': args.headless, 'multi_gpu': False,
                         'create_new_stage': False, 'enable_crashreporter': False,
                         'width': 320, 'height': 240,
                         'samples_per_pixel_per_frame': 1})
    try:
        import numpy as np
        import isaacsim.core.experimental.utils.app as app_utils
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension('isaacsim.asset.importer.urdf')
        _load_ros_python()

        import rclpy
        from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.experimental.rtx import CameraSensor
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rclpy.time import Time
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Image
        from std_srvs.srv import SetBool
        from tf2_msgs.msg import TFMessage
        from pxr import UsdGeom, UsdPhysics

        robot_assets = tempfile.TemporaryDirectory(prefix='rvln-isaac-urdf-')
        import_config = URDFImporterConfig(
            urdf_path=str(args.robot_urdf.resolve()), usd_path=robot_assets.name,
            merge_fixed_joints=True, fix_base=False, collision_from_visuals=False)
        robot_usd = URDFImporter(import_config).import_urdf()
        if not Path(robot_usd).is_file():
            raise RuntimeError('Raspicat URDF import failed')

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60,
                      rendering_dt=1 / 30)
        add_reference_to_stage(str(args.world.resolve()), '/World/Environment')
        add_reference_to_stage(str(robot_usd), '/World/Raspicat')
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath('/World/Raspicat')
        robot_prim.GetVariantSet('Physics').SetVariantSelection('physx')
        stage.Load()
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            raise RuntimeError('Isaac world must be Z-up')
        if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
            raise RuntimeError('Isaac world must use metres')
        if not any(prim.HasAPI(UsdPhysics.CollisionAPI)
                   for prim in stage.Traverse() if prim.GetPath().pathString.startswith(
                       '/World/Environment')):
            raise RuntimeError('world USD has no collision geometry')

        articulation_roots = [prim for prim in stage.Traverse()
                              if prim.GetPath().pathString.startswith('/World/Raspicat/')
                              and prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(articulation_roots) != 1:
            raise RuntimeError(f'expected one Raspicat articulation root, got '
                               f'{len(articulation_roots)}')
        for name in (args.left_joint, args.right_joint):
            joints = [prim for prim in stage.Traverse()
                      if prim.GetPath().pathString.startswith('/World/Raspicat/')
                      and prim.GetName() == name]
            if len(joints) != 1 or not joints[0].HasAPI(UsdPhysics.DriveAPI, 'angular'):
                raise RuntimeError(f'wheel drive missing from imported URDF: {name}')
            drive = UsdPhysics.DriveAPI.Get(joints[0], 'angular')
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(10000.0)
            drive.GetMaxForceAttr().Set(1000.0)
        robot = world.scene.add(SingleArticulation(
            prim_path=articulation_roots[0].GetPath().pathString, name='raspicat'))
        world.reset()
        camera = None
        camera_prim = None
        if not args.physics_only:
            UsdGeom.Camera.Define(stage, '/World/RVLN_Camera')
            camera = CameraSensor('/World/RVLN_Camera', resolution=(480, 640),
                                  annotators=['rgb', 'distance_to_image_plane'])
            camera_prim = camera.authoring_object
            app_utils.play(commit=True)
            rep.orchestrator.step(rt_subframes=2, pause_timeline=False)
        left = robot.get_dof_index(args.left_joint)
        right = robot.get_dof_index(args.right_joint)
        if left == right or min(left, right) < 0:
            raise RuntimeError('wheel joints missing from imported URDF')

        rclpy.init()

        class Bridge(Node):
            def __init__(self):
                super().__init__('isaac_rvln_bridge')
                self.command = (0.0, 0.0)
                self.last_command = 0.0
                self.motor_on = False
                self.create_subscription(Twist, '/cmd_vel', self.on_command, 10)
                self.create_service(SetBool, '/motor_power', self.on_motor)
                qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
                self.image_pub = None
                self.depth_pub = None
                if camera is not None:
                    self.image_pub = self.create_publisher(
                        Image, '/camera/color/image_raw', qos)
                    self.depth_pub = self.create_publisher(
                        Image, '/camera/depth/image_raw', qos)
                self.clock_pub = self.create_publisher(Clock, '/clock', 10)
                self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
                self.pose_pub = self.create_publisher(
                    PoseStamped, '/sim/ground_truth_pose', 10)
                self.tf_pub = self.create_publisher(TFMessage, '/tf', 10)

            def on_command(self, message):
                self.command = (max(-0.4, min(0.4, float(message.linear.x))),
                                max(-1.0, min(1.0, float(message.angular.z))))
                self.last_command = time.monotonic()

            def on_motor(self, request, response):
                self.motor_on = bool(request.data)
                response.success = True
                response.message = 'motor enabled' if self.motor_on else 'motor disabled'
                return response

        bridge = Bridge()
        begin = time.monotonic()
        last_camera = 0.0
        steps = 0
        camera_frames = 0
        empty_camera_frames = 0
        while app.is_running() and (args.max_seconds == 0 or
                                    time.monotonic() - begin < args.max_seconds):
            rclpy.spin_once(bridge, timeout_sec=0)
            if bridge.motor_on and time.monotonic() - bridge.last_command <= 0.5:
                command = bridge.command
            else:
                command = (0.0, 0.0)
            velocity = wheel_velocities(*command, args.wheel_radius,
                                        args.wheel_separation)
            robot.apply_action(ArticulationAction(
                joint_velocities=np.asarray(velocity),
                joint_indices=np.asarray([left, right])))
            previous_position, previous_orientation = robot.get_world_pose()
            previous_yaw = math.atan2(
                2 * (previous_orientation[0] * previous_orientation[3] +
                     previous_orientation[1] * previous_orientation[2]),
                1 - 2 * (previous_orientation[2] ** 2 + previous_orientation[3] ** 2))
            if camera is not None:
                w, x, y, z = previous_orientation
                camera_orientation = np.array([w - x + y + z, w + x - y + z,
                                               -w + x + y + z, -w - x - y + z]) * 0.5
                camera_prim.set_world_poses(
                    positions=np.asarray([previous_position + np.array([
                        0.1 * math.cos(previous_yaw),
                        0.1 * math.sin(previous_yaw), 0.1433])]),
                    orientations=np.asarray([camera_orientation]))
            world.step(render=camera is not None)
            steps += 1
            position, orientation = robot.get_world_pose()
            yaw = math.atan2(2 * (orientation[0] * orientation[3] +
                                  orientation[1] * orientation[2]),
                             1 - 2 * (orientation[2] ** 2 + orientation[3] ** 2))
            linear_world = robot.get_linear_velocity()
            angular_world = robot.get_angular_velocity()
            stamp = Time(seconds=world.current_time).to_msg()
            clock = Clock()
            clock.clock = stamp
            bridge.clock_pub.publish(clock)
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = 'odom'
            odom.child_frame_id = 'base_link'
            odom.pose.pose.position.x, odom.pose.pose.position.y, \
                odom.pose.pose.position.z = (float(v) for v in position)
            odom.pose.pose.orientation.w, odom.pose.pose.orientation.x, \
                odom.pose.pose.orientation.y, odom.pose.pose.orientation.z = (
                    float(v) for v in orientation)
            odom.twist.twist.linear.x = float(
                linear_world[0] * math.cos(yaw) + linear_world[1] * math.sin(yaw))
            odom.twist.twist.angular.z = float(angular_world[2])
            bridge.odom_pub.publish(odom)
            pose = PoseStamped()
            pose.header = odom.header
            pose.pose = odom.pose.pose
            bridge.pose_pub.publish(pose)
            transform = TransformStamped()
            transform.header = odom.header
            transform.child_frame_id = 'base_link'
            transform.transform.translation.x = odom.pose.pose.position.x
            transform.transform.translation.y = odom.pose.pose.position.y
            transform.transform.translation.z = odom.pose.pose.position.z
            transform.transform.rotation = odom.pose.pose.orientation
            bridge.tf_pub.publish(TFMessage(transforms=[transform]))
            now = time.monotonic()
            if camera is not None and now - last_camera >= 0.1:
                color, _ = camera.get_data('rgb')
                rgba = (color.numpy() if color is not None and hasattr(color, 'numpy')
                        else np.asarray(color))
                if rgba.size == 0:
                    empty_camera_frames += 1
                    if empty_camera_frames >= 30:
                        raise RuntimeError('Isaac camera produced no RGB frame '
                                           'after 30 capture attempts')
                    continue
                if rgba.shape not in ((480, 640, 3), (480, 640, 4)):
                    raise RuntimeError(f'unexpected camera frame: {rgba.shape}')
                image = Image()
                image.header.stamp = stamp
                image.header.frame_id = 'camera_link'
                image.height, image.width = 480, 640
                image.encoding = 'rgb8'
                image.step = 640 * 3
                image.data = rgba[:, :, :3].astype('uint8').tobytes()
                bridge.image_pub.publish(image)
                depth, _ = camera.get_data('distance_to_image_plane')
                depth = (depth.numpy() if depth is not None and hasattr(depth, 'numpy')
                         else np.asarray(depth))
                if depth.shape == (480, 640, 1):
                    depth = depth[:, :, 0]
                if depth.shape != (480, 640):
                    raise RuntimeError('depth camera frame unavailable')
                depth_image = Image()
                depth_image.header = image.header
                depth_image.height, depth_image.width = 480, 640
                depth_image.encoding = '32FC1'
                depth_image.step = 640 * 4
                depth_image.data = np.asarray(depth, dtype='<f4').tobytes()
                bridge.depth_pub.publish(depth_image)
                last_camera = now
                camera_frames += 1
        bridge.destroy_node()
        rclpy.shutdown()
        status = 'finished' if steps and (args.physics_only or camera_frames) else 'incomplete'
        print(json.dumps({'status': status, 'physics_steps': steps,
                          'camera_frames': camera_frames,
                          'physics_only': args.physics_only}), flush=True)
        if status != 'finished':
            raise RuntimeError('Isaac exited before producing camera frames')
    except Exception as error:
        print(json.dumps({'status': 'error', 'error':
                          f'{type(error).__name__}: {error}'}), flush=True)
        raise
    finally:
        app.close()
        if robot_assets is not None:
            robot_assets.cleanup()
