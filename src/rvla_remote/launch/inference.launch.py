"""Launch the remote VLA inference node with a selected model backend."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_RESUME_STEP_DEFAULTS = {'asyncvla': '750000', 'omnivla': '120000'}


def _start_inference(context):
    backend = LaunchConfiguration('backend').perform(context)
    vla_path = LaunchConfiguration('vla_path').perform(context)
    resume_step = LaunchConfiguration('resume_step').perform(context)
    device = LaunchConfiguration('device').perform(context)

    arguments = ['--backend', backend, '--device', device]
    if vla_path:
        arguments.extend(['--vla-path', vla_path])
    if resume_step or backend in _RESUME_STEP_DEFAULTS:
        arguments.extend([
            '--resume-step', resume_step or _RESUME_STEP_DEFAULTS[backend],
        ])

    return [Node(
        package='rvla_remote',
        executable='vla_inference_node',
        name='vla_inference_node',
        output='screen',
        arguments=arguments,
        parameters=[{
            'observation_topic': LaunchConfiguration('observation_topic'),
            'embedding_topic': LaunchConfiguration('embedding_topic'),
        }],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('backend', default_value='dummy'),
        DeclareLaunchArgument('vla_path', default_value=''),
        DeclareLaunchArgument('resume_step', default_value=''),
        DeclareLaunchArgument('device', default_value='cuda:0'),
        DeclareLaunchArgument(
            'observation_topic', default_value='/rvla/observation'),
        DeclareLaunchArgument(
            'embedding_topic', default_value='/rvla/remote_embedding'),
        OpaqueFunction(function=_start_inference),
    ])
