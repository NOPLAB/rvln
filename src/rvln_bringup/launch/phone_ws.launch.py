"""Launch the Pi WebSocket endpoint for Web/mobile inference.

The server converts incoming action chunks to nav_msgs/Path for the existing
pure-pursuit follower. The sender owns observation and inference, so this
launch file does not start the VLA edge camera or gRPC node.

For safety, the follower defaults to the non-motor /cmd_vel_vla topic.
Set cmd_vel_topic:=/cmd_vel explicitly to drive hardware.

Launch args:
  port              - WebSocket port (default: 8765)
  cmd_vel_topic     - follower Twist output (default: /cmd_vel_vla)
  chunk_max_age_sec - safe-stop after this gap without a chunk (default: 1.0)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from rvln_edge.launch_util import follower_node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='8765'),
        # Default to a non-motor topic so the real robot is never driven.
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel_vla'),
        DeclareLaunchArgument('chunk_max_age_sec', default_value='1.0'),
        Node(
            package='rvln_edge',
            executable='edge_action_ws_server',
            name='edge_action_ws',
            output='screen',
            parameters=[{
                'port': ParameterValue(LaunchConfiguration('port'), value_type=int),
                'chunk_max_age_sec': ParameterValue(
                    LaunchConfiguration('chunk_max_age_sec'), value_type=float),
            }],
        ),
        follower_node(cmd_vel_topic=LaunchConfiguration('cmd_vel_topic')),
    ])
