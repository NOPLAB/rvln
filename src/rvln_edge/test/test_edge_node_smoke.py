"""Smoke test: edge and remote ROS nodes publish a Path."""
import threading
import time

import pytest
import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image
import numpy as np

from rvln_remote.backends.dummy import DummyBackend
from rvln_remote.inference_node import VLAInferenceNode
from rvln_edge.edge_node import VLAEdgeNode
from rvln_msgs.msg import GoalSpec as GoalSpecMsg
from geometry_msgs.msg import PoseStamped


@pytest.fixture(scope='module')
def ros_runtime():
    rclpy.init()
    yield
    rclpy.shutdown()


def _make_dummy_image_msg() -> Image:
    msg = Image()
    msg.height = 240
    msg.width = 320
    msg.encoding = 'rgb8'
    msg.is_bigendian = 0
    msg.step = 320 * 3
    msg.data = (np.zeros((240, 320, 3), dtype=np.uint8)).tobytes()
    return msg


def test_edge_node_publishes_path(ros_runtime):
    server = VLAInferenceNode(backend=DummyBackend(
        num_tokens=4, embed_dim=8, inference_ms=1.0, model_version='test'))
    try:
        node = VLAEdgeNode()
        node.set_parameters([
            rclpy.parameter.Parameter('obs_publish_rate_hz', value=10.0),
            rclpy.parameter.Parameter('action_rate_hz', value=20.0),
            rclpy.parameter.Parameter('embedding_max_age_sec', value=6.0),
            rclpy.parameter.Parameter('embedding_hard_timeout_sec', value=15.0),
        ])
        # configure -> activate
        node.trigger_configure()
        node.trigger_activate()

        # External publisher to push goal + image
        pub_node = rclpy.create_node('test_pub')
        # Edge subscribes to goals TRANSIENT_LOCAL; match it (a VOLATILE writer
        # would not connect to a TRANSIENT_LOCAL reader).
        goal_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        goal_pub = pub_node.create_publisher(GoalSpecMsg, '/rvln/goal', goal_qos)
        img_pub = pub_node.create_publisher(Image, '/camera/image_raw', 1)
        odom_pub = pub_node.create_publisher(Odometry, '/odom', 10)

        received_paths = []
        path_node = rclpy.create_node('test_sub')
        path_node.create_subscription(Path, '/rvln/predicted_path',
                                      lambda m: received_paths.append(m), 10)

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.add_node(server)
        executor.add_node(pub_node)
        executor.add_node(path_node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        # Send goal once and image periodically
        goal = GoalSpecMsg()
        goal.mode = GoalSpecMsg.MODE_POSE
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'odom'
        goal.pose.pose.position.x = 1.0
        goal.pose.pose.orientation.w = 1.0
        goal_pub.publish(goal)

        odom = Odometry()
        odom.header.frame_id = 'odom'
        odom.pose.pose.orientation.w = 1.0

        deadline = time.time() + 5.0
        while time.time() < deadline and not any(p.poses for p in received_paths):
            odom_pub.publish(odom)
            img_pub.publish(_make_dummy_image_msg())
            time.sleep(0.05)

        executor.shutdown(timeout_sec=1.0)
        node.trigger_deactivate()
        node.trigger_cleanup()
        node.destroy_node()
        pub_node.destroy_node()
        path_node.destroy_node()

        assert any(p.poses for p in received_paths), 'no inferred Path within 5s'
    finally:
        server.destroy_node()
