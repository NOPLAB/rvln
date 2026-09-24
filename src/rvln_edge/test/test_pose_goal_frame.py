"""Pose goals in odom must be refreshed in the robot frame for inference."""
import math

import pytest
from nav_msgs.msg import Odometry

from rvln_edge.edge_node import _pose_goal_in_base_link
from rvln_msgs.msg import GoalSpec


def test_odom_goal_is_rotated_and_translated_into_base_link():
    goal = GoalSpec()
    goal.mode = GoalSpec.MODE_POSE
    goal.pose.header.frame_id = 'odom'
    goal.pose.pose.position.x = 2.0
    goal.pose.pose.position.y = 3.0
    goal.pose.pose.orientation.w = 1.0

    odom = Odometry()
    odom.pose.pose.position.x = 1.0
    odom.pose.pose.position.y = 1.0
    odom.pose.pose.orientation.z = math.sin(math.pi / 4.0)
    odom.pose.pose.orientation.w = math.cos(math.pi / 4.0)

    relative = _pose_goal_in_base_link(goal, odom)

    assert relative.pose.header.frame_id == 'base_link'
    assert relative.pose.pose.position.x == pytest.approx(2.0)
    assert relative.pose.pose.position.y == pytest.approx(-1.0)
    assert relative.pose.pose.orientation.z == pytest.approx(-math.sin(math.pi / 4.0))
    assert relative.pose.pose.orientation.w == pytest.approx(math.cos(math.pi / 4.0))
    assert goal.pose.header.frame_id == 'odom'
    assert goal.pose.pose.position.x == 2.0
