"""Publish a constant black image at 5 Hz on /camera/image_raw, plus a fixed goal."""
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image

from rvln_msgs.msg import GoalSpec as GoalSpecMsg


class FakePub(Node):
    def __init__(self) -> None:
        super().__init__('fake_pub')
        self._img_pub = self.create_publisher(Image, '/camera/image_raw', 1)
        # Latched goal: the edge subscribes TRANSIENT_LOCAL, so publish with the
        # same durability (a VOLATILE writer would not match its reader).
        goal_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._goal_pub = self.create_publisher(GoalSpecMsg, '/rvln/goal', goal_qos)
        self._timer = self.create_timer(0.2, self._tick)
        self._goal_sent = False

    def _tick(self) -> None:
        msg = Image()
        msg.height = 240
        msg.width = 320
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = 320 * 3
        msg.data = np.zeros((240, 320, 3), dtype=np.uint8).tobytes()
        self._img_pub.publish(msg)
        if not self._goal_sent:
            g = GoalSpecMsg()
            g.mode = GoalSpecMsg.MODE_POSE
            g.pose = PoseStamped()
            g.pose.header.frame_id = 'odom'
            g.pose.pose.position.x = 1.0
            g.pose.pose.orientation.w = 1.0
            self._goal_pub.publish(g)
            self._goal_sent = True


def main() -> None:
    rclpy.init()
    node = FakePub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
