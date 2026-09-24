"""Pi side of the mobile port, cmd_vel preview mode — no VLA model on this host.

This is ``vla.sh run omnivla_edge_mobile --mode cmd_vel``: the smartphone app
does camera capture + OmniVLA-edge inference and streams action chunks to the
``EdgeActionService`` gRPC server (edge_action_grpc_node) started here; the
chunks become a Path and the path_follower_node publishes Twist to a
**non-motor topic** (default ``/cmd_vel_vla``) so the whole
phone -> gRPC -> Path -> cmd_vel pipeline is observable
(``ros2 topic echo /cmd_vel_vla``) without driving the robot's motors.

There is no VLA edge lifecycle node and no cloud server in this topology —
the phone replaces both (docs/design/mobile_port_spec.md). Safety lives in the
gRPC node's chunk watchdog (empty Path on staleness) + the follower.

Launch args:
  listen_host       - EdgeActionService bind address (default: 0.0.0.0)
  listen_port       - EdgeActionService port (default: 50061)
  cmd_vel_topic     - follower output topic (default: /cmd_vel_vla, non-motor)
  chunk_max_age_sec - watchdog: no chunk for this long => empty Path (safe-stop)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from rvla_edge.launch_util import follower_node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('listen_host', default_value='0.0.0.0'),
        DeclareLaunchArgument('listen_port', default_value='50061'),
        # Default to a non-motor topic so the real robot is never driven.
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel_vla'),
        DeclareLaunchArgument('chunk_max_age_sec', default_value='1.0'),
        Node(
            package='rvla_edge',
            executable='edge_action_grpc_server',
            name='edge_action_grpc',
            output='screen',
            parameters=[{
                'host': LaunchConfiguration('listen_host'),
                'port': ParameterValue(
                    LaunchConfiguration('listen_port'), value_type=int),
                'chunk_max_age_sec': ParameterValue(
                    LaunchConfiguration('chunk_max_age_sec'), value_type=float),
            }],
        ),
        follower_node(cmd_vel_topic=LaunchConfiguration('cmd_vel_topic')),
    ])
