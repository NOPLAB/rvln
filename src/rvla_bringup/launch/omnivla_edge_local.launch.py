"""Launch OmniVLA-edge standalone (Plan 2B Path 2 — policy runs ON the edge).

 - vla_edge_node       (lifecycle; adapter_kind=omnivla_edge_local, standalone)
 - path_follower_node  (Path -> /cmd_vel)

The full OmniVLA-edge policy + CLIP run locally in the edge node, which operates
in *standalone* mode: no cloud, no gRPC client, no embedding cache. The action
loop drives the local policy directly from the camera frame + goal. (Contrast
with local_stack.launch.py backend:=omnivla — Path 1, where a server process
runs OmniVLA-original — and backend:=omnivla_edge — Path 3, the same policy as
a server.)

Requirements: a CUDA-capable edge (the vendored OmniVLA_edge forward pass is
GPU-only) and the omnivla-edge weights at ``omnivla_edge_weights_path``
(``scripts/download_omnivla_checkpoints.sh edge``).

Inputs: publish RGB frames on the edge node's ``image_topic`` and a GoalSpec on
``goal_topic`` (text goals are the cleanest — see the adapter docstring).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from rvla_edge.launch_util import (
    camera_nodes, edge_camera_overrides, edge_lifecycle_actions,
    edge_params_path, follower_node,
)


def generate_launch_description():
    weights_path = LaunchConfiguration('weights_path')
    device = LaunchConfiguration('device')
    image_topic = LaunchConfiguration('image_topic')

    edge_actions = edge_lifecycle_actions(parameters=[edge_params_path(), {
        'adapter_kind': 'omnivla_edge_local',
        'omnivla_edge_weights_path': weights_path,
        'omnivla_edge_device': device,
        'image_topic': image_topic,
        # camera_kind=v4l2 -> the edge grabs camera_device in-process.
        **edge_camera_overrides(),
    }])

    return LaunchDescription([
        DeclareLaunchArgument(
            'weights_path',
            default_value='/workspace/models/omnivla-edge/omnivla-edge.pth'),
        DeclareLaunchArgument('device', default_value='cuda:0'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('camera_kind', default_value=''),
        DeclareLaunchArgument('camera_device', default_value=''),
        *edge_actions,
        follower_node(),
        *camera_nodes(image_topic=image_topic),
    ])
