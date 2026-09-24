"""Web/スマホ推論の Pi 側受け口 (Phase 4 の WebSocket 版) を起動する。

web/ (rvln-web) や app/ が推論した action chunk を WebSocket で受け、
``edge_action_ws_server`` が nav_msgs/Path 化 -> 既存 path_follower_node が
pure-pursuit で追従する。VLA edge ノード (カメラ・gRPC) は起動しない —
観測とモデルはすべて送信側にある。

安全のため follower の出力は既定で非モーター topic ``/cmd_vel_vla``。
実機で走らせるときは ``cmd_vel_topic:=/cmd_vel`` を明示する。

Launch args:
  port              - WS ポート (default: 8765)
  cmd_vel_topic     - follower の Twist 出力 (default: /cmd_vel_vla)
  chunk_max_age_sec - この間隔で chunk が来なければ safe-stop (default: 1.0)
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
