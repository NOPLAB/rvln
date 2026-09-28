"""Isaac Sim 5.0 ROS 2 bridge for the Raspicat VLA edge/follower contract.

Requires an expanded Raspicat URDF and a collidable, Z-up USD world. This
module is kept separate from ROS launch because Isaac uses Python 3.11.
"""
from __future__ import annotations

import math
import time

from bench.episode import ContinuousSimulator


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
    app = SimulationApp({'headless': args.headless})
    try:
        import numpy as np
        import omni.kit.commands
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension('isaacsim.ros2.bridge')
        enable_extension('isaacsim.asset.importer.urdf')

        import rclpy
        from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rclpy.time import Time
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Image
        from std_srvs.srv import SetBool
        from tf2_msgs.msg import TFMessage
        from pxr import UsdGeom, UsdPhysics

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60,
                      rendering_dt=1 / 30)
        add_reference_to_stage(str(args.world.resolve()), '/World/Environment')
        stage = omni.usd.get_context().get_stage()
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            raise RuntimeError('Isaac world must be Z-up')
        if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
            raise RuntimeError('Isaac world must use metres')
        if not any(prim.HasAPI(UsdPhysics.CollisionAPI)
                   for prim in stage.Traverse() if prim.GetPath().pathString.startswith(
                       '/World/Environment')):
            raise RuntimeError('world USD has no collision geometry')

        ok, config = omni.kit.commands.execute('URDFCreateImportConfig')
        if not ok:
            raise RuntimeError('URDFCreateImportConfig failed')
        config.merge_fixed_joints = False
        config.fix_base = False
        config.import_inertia_tensor = True
        config.collision_from_visuals = False
        ok, prim_path = omni.kit.commands.execute(
            'URDFParseAndImportFile', urdf_path=str(args.robot_urdf.resolve()),
            import_config=config)
        if not ok or not prim_path:
            raise RuntimeError('Raspicat URDF import failed')
        robot = world.scene.add(SingleArticulation(prim_path=prim_path, name='raspicat'))
        camera = world.scene.add(Camera(prim_path='/World/RVLN_Camera',
                                       resolution=(640, 480), frequency=10))
        world.reset()
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()
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
            camera.set_world_pose(
                position=previous_position + np.array([
                    0.1 * math.cos(previous_yaw), 0.1 * math.sin(previous_yaw), 0.1433]),
                orientation=previous_orientation, camera_axes='world')
            world.step(render=True)
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
            if now - last_camera >= 0.1:
                rgba = np.asarray(camera.get_rgba())
                if rgba.shape != (480, 640, 4):
                    raise RuntimeError(f'unexpected camera frame: {rgba.shape}')
                image = Image()
                image.header.stamp = stamp
                image.header.frame_id = 'camera_link'
                image.height, image.width = 480, 640
                image.encoding = 'rgb8'
                image.step = 640 * 3
                image.data = rgba[:, :, :3].astype('uint8').tobytes()
                bridge.image_pub.publish(image)
                depth = camera.get_depth()
                if depth is None or np.asarray(depth).shape != (480, 640):
                    raise RuntimeError('depth camera frame unavailable')
                depth_image = Image()
                depth_image.header = image.header
                depth_image.height, depth_image.width = 480, 640
                depth_image.encoding = '32FC1'
                depth_image.step = 640 * 4
                depth_image.data = np.asarray(depth, dtype='<f4').tobytes()
                bridge.depth_pub.publish(depth_image)
                last_camera = now
        bridge.destroy_node()
        rclpy.shutdown()
    finally:
        app.close()
