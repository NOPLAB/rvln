from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rvln_edge',
            executable='path_follower_node',
            name='path_follower_node',
            output='screen',
            parameters=[{
                'max_v': 0.4,
                'max_w': 1.0,
                'rate_hz': 20.0,
                'path_topic': '/rvln/predicted_path',
                'cmd_vel_topic': '/cmd_vel',
            }],
        ),
    ])
