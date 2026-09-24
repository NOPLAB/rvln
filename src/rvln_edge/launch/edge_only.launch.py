r"""Launch the VLA edge lifecycle node (auto-transitions to active).

Optional launch args (override edge_params.yaml):
  observation_topic / remote_embedding_topic are set in edge_params.yaml.
  adapter_kind    - stub|asyncvla|omnivla
  image_topic     - camera image topic (default: /camera/image_raw;
                    raspicat_sim uses /camera/color/image_raw)
  camera_kind     - ''|v4l2|realsense. Empty (default) = the edge subscribes to
                    image_topic (frames come from elsewhere). v4l2 = the edge
                    node grabs camera_device IN-PROCESS (no driver node, no DDS
                    image hop). realsense launches a realsense2_camera node
                    remapped to publish on image_topic.
  camera_device   - v4l2 device path (e.g. /dev/video0); only used when
                    camera_kind=v4l2.
  with_follower   - true|false (also bring up path_follower_node)
  cmd_vel_topic   - follower's Twist output topic (default: /cmd_vel; set to a
                    non-motor topic like /cmd_vel_vla to run without driving the robot)
  asyncvla_weights_path / asyncvla_resume_step / asyncvla_device

Use cases:
  ros2 launch rvln_bringup edge_only.launch.py            # yaml defaults
  ros2 launch rvln_bringup edge_only.launch.py \\
      adapter_kind:=asyncvla with_follower:=true
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from rvln_edge.launch_util import (
    camera_nodes, edge_camera_overrides, edge_lifecycle_actions,
    edge_params_path, follower_node,
)


def generate_launch_description():
    adapter_kind = LaunchConfiguration('adapter_kind')
    image_topic = LaunchConfiguration('image_topic')
    with_follower = LaunchConfiguration('with_follower')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    asyncvla_weights_path = LaunchConfiguration('asyncvla_weights_path')
    asyncvla_resume_step = LaunchConfiguration('asyncvla_resume_step')
    asyncvla_device = LaunchConfiguration('asyncvla_device')

    # Per-launch parameter overrides; only emit the ones that were set explicitly
    # (default '' means "leave the YAML value alone").
    overrides = {
        'adapter_kind': adapter_kind,
        'image_topic': image_topic,
        'asyncvla_weights_path': asyncvla_weights_path,
        'asyncvla_resume_step': asyncvla_resume_step,
        'asyncvla_device': asyncvla_device,
        # camera_kind=v4l2 -> the edge grabs camera_device in-process.
        **edge_camera_overrides(),
    }

    edge_actions = edge_lifecycle_actions(parameters=[edge_params_path(), overrides])

    follower = follower_node(
        cmd_vel_topic=cmd_vel_topic,
        condition=IfCondition(with_follower),
    )

    return LaunchDescription([
        DeclareLaunchArgument('adapter_kind', default_value='stub'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('camera_kind', default_value=''),
        DeclareLaunchArgument('camera_device', default_value=''),
        DeclareLaunchArgument('with_follower', default_value='false'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel'),
        DeclareLaunchArgument('asyncvla_weights_path',
                              default_value='/workspace/models/AsyncVLA_release'),
        DeclareLaunchArgument('asyncvla_resume_step', default_value='750000'),
        DeclareLaunchArgument('asyncvla_device', default_value='cpu'),
        *edge_actions,
        follower,
        *camera_nodes(image_topic=image_topic),
    ])
