"""Single-host all-in-one VLA stack: server + edge node + follower, ONE launch.

This is the no-Docker development bring-up — every piece runs as a local
process on this host. Pick the server with ``backend:=``:

  backend:=dummy          Plan 1 MVP: deterministic embeddings, no weights
                          (vla_inference_node; edge adapter stays the yaml default).
  backend:=asyncvla       Plan 2A: AsyncVLA cloud backbone (GPU) + the small
                          Edge_adapter on the edge (adapter_kind=asyncvla).
  backend:=omnivla        Plan 2B Path 1: OmniVLA-original cloud (GPU); the edge
                          runs the path-only adapter (adapter_kind=omnivla).
  backend:=omnivla_edge   Plan 2B Path 3: the OmniVLA-edge policy as the server
                          ("Jetson infers, Pi controls" — here both on
                          localhost); edge is path-only. --vla-path is the .pth
                          weights file, resume_step is unused.

For a real split-host deployment run the inference node on the GPU box
(``vla.sh run MODEL --mode remote``) and ``edge_only.launch.py`` on the robot.
Use the same ROS domain on both hosts. For Plan 2B Path 2 — the policy ON the robot, no
server at all — see ``omnivla_edge_local.launch.py``. The containerised modes
(and their topology) live in docker/compose.yaml.

Launch args:
  backend      - dummy|asyncvla|omnivla|omnivla_edge (default: dummy)
  vla_path     - checkpoint dir / weights file ('' = the backend's default)
  resume_step  - checkpoint step ('' = the backend's default; unused for
                 dummy/omnivla_edge)
  device       - server-side torch device (default: cuda:0; ignored by dummy)
  edge_device  - asyncvla only: Edge_adapter device (default: cpu)
  inference_ms - dummy only: simulated inference latency (default: 50.0)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from raspicat_vla_edge.launch_util import (
    edge_lifecycle_actions, edge_params_path, follower_node, vla_server_process,
)


# Per-backend checkpoint defaults, applied when vla_path/resume_step are ''.
_BACKEND_DEFAULTS = {
    'asyncvla': ('/workspace/models/AsyncVLA_release', '750000'),
    'omnivla': ('/workspace/models/omnivla-original', '120000'),
    'omnivla_edge': ('/workspace/models/omnivla-edge/omnivla-edge.pth', '0'),
}


def _setup(context):
    backend = LaunchConfiguration('backend').perform(context)
    device = LaunchConfiguration('device').perform(context)
    default_path, default_step = _BACKEND_DEFAULTS.get(backend, ('', ''))
    vla_path = LaunchConfiguration('vla_path').perform(context) or default_path
    resume_step = LaunchConfiguration('resume_step').perform(context) or default_step

    edge_overrides = {}

    if backend == 'dummy':
        server = ExecuteProcess(
            cmd=[
                'ros2', 'run', 'raspicat_vla_remote', 'vla_inference_node',
                '--inference-ms', LaunchConfiguration('inference_ms').perform(context),
                '--num-tokens', '8',
                '--embed-dim', '1024',
            ],
            output='screen',
        )
    elif backend == 'asyncvla':
        server = vla_server_process(
            backend='asyncvla',
            extra_args=['--vla-path', vla_path,
                        '--resume-step', resume_step,
                        '--device', device],
        )
        edge_overrides.update({
            'adapter_kind': 'asyncvla',
            'asyncvla_weights_path': vla_path,
            'asyncvla_resume_step': int(resume_step),
            'asyncvla_device': LaunchConfiguration('edge_device').perform(context),
            # Edge_adapter CPU inference takes O(100 ms) on the robot; the default
            # 10 Hz action tick therefore runs back-to-back, monopolising CPU that
            # the camera driver needs (frames then stall and the freshness guard
            # safe-stops). 3 Hz still refreshes the path ~4x per cloud embedding.
            'action_rate_hz': 3.0,
        })
    elif backend == 'omnivla':
        server = vla_server_process(
            backend='omnivla',
            extra_args=['--vla-path', vla_path,
                        '--resume-step', resume_step,
                        '--device', device],
        )
        edge_overrides['adapter_kind'] = 'omnivla'
    elif backend == 'omnivla_edge':
        # --vla-path is the .pth weights file; --resume-step is unused. The edge
        # runs the same path-only adapter as omnivla (waypoints arrive computed).
        server = vla_server_process(
            backend='omnivla_edge',
            extra_args=['--vla-path', vla_path,
                        '--device', device],
        )
        edge_overrides['adapter_kind'] = 'omnivla'
    else:
        raise RuntimeError(
            f'unknown backend {backend!r} (want dummy|asyncvla|omnivla|omnivla_edge)')

    return [
        server,
        *edge_lifecycle_actions(parameters=[edge_params_path(), edge_overrides]),
        follower_node(),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='dummy'),
        DeclareLaunchArgument('vla_path', default_value=''),
        DeclareLaunchArgument('resume_step', default_value=''),
        DeclareLaunchArgument('device', default_value='cuda:0'),
        DeclareLaunchArgument('edge_device', default_value='cpu'),
        DeclareLaunchArgument('inference_ms', default_value='50.0'),
        OpaqueFunction(function=_setup),
    ])
