"""Launch raspicat_sim (Gazebo) + the VLA edge stack.

Used by ``scripts/vla.sh run {asyncvla,omnivla} --mode sim`` to
bring up Gazebo with raspicat in an empty world and our edge node + path
follower pointed at a remote cloud server.

Launch args:
  observation and embedding topics use the edge node's configured defaults.
  adapter_kind    stub | asyncvla | omnivla       (default omnivla)
  world           gazebo .world path              (raspicat_gazebo/empty.world default)
  rviz            true|false                       (default false; sim is mostly headless)
  gui             true|false                       (default false)
  asyncvla_weights_path / asyncvla_resume_step / asyncvla_device

The handrail-mounted D435 publishes color at ``/camera/color/image_raw`` and
depth at ``/camera/depth/image_raw``. The edge node subscribes to color.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    adapter_kind = LaunchConfiguration('adapter_kind')
    world = LaunchConfiguration('world')
    rviz = LaunchConfiguration('rviz')
    gui = LaunchConfiguration('gui')
    asyncvla_weights_path = LaunchConfiguration('asyncvla_weights_path')
    asyncvla_resume_step = LaunchConfiguration('asyncvla_resume_step')
    asyncvla_device = LaunchConfiguration('asyncvla_device')

    raspicat_gazebo_share = get_package_share_directory('raspicat_gazebo')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world}.items(),
    )
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, 'launch', 'gzclient.launch.py')),
        condition=IfCondition(gui),
    )
    description_script = os.path.join(
        get_package_share_directory('rvln_bringup'), 'urdf', 'raspicat_d435.py',
    )
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['python3 ', description_script])}],
        output='screen',
    )
    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            raspicat_gazebo_share, 'launch', 'spawn_raspicat.launch.py')),
    )
    sim_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            raspicat_gazebo_share, 'launch', 'raspicat_simulation.launch.py')),
        launch_arguments={'rviz': rviz}.items(),
    )

    edge_launch_path = os.path.join(
        get_package_share_directory('rvln_edge'),
        'launch', 'edge_only.launch.py',
    )
    edge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(edge_launch_path),
        launch_arguments={
            'adapter_kind': adapter_kind,
            'image_topic': '/camera/color/image_raw',   # raspicat_sim RealSense topic
            'with_follower': 'true',
            'asyncvla_weights_path': asyncvla_weights_path,
            'asyncvla_resume_step': asyncvla_resume_step,
            'asyncvla_device': asyncvla_device,
        }.items(),
    )

    # rt-net's spawn_raspicat.launch.py calls spawn_entity.py with its
    # built-in 30s service-wait timeout. Under CPU contention (gzserver +
    # gzclient GL + the edge node importing torch, all starting at once) the
    # gazebo_ros_factory's /spawn_entity service can take well over two minutes
    # to become discoverable, so the original spawn dies and the world stays
    # empty. Schedule a fallback respawn shortly after the first attempt gives
    # up (~35s) and let it wait a long time (-timeout 600) for the service. If
    # the first attempt already succeeded, the /model_states guard short-
    # circuits and this returns harmlessly. The guard call itself is wrapped in
    # `timeout` so a not-yet-ready topic can't hang the check indefinitely.
    respawn_fallback = TimerAction(
        period=35.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'bash', '-lc',
                    'timeout 10 ros2 topic echo /model_states --once 2>/dev/null '
                    '| grep -q raspicat || '
                    'ros2 run gazebo_ros spawn_entity.py '
                    '-entity raspicat -topic /robot_description '
                    '-x 0.0 -y 0.0 -z 0.0 -timeout 600',
                ],
                output='screen',
            ),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('adapter_kind', default_value='omnivla'),
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(raspicat_gazebo_share, 'worlds', 'empty.world'),
        ),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('asyncvla_weights_path',
                              default_value='/workspace/models/AsyncVLA_release'),
        DeclareLaunchArgument('asyncvla_resume_step', default_value='750000'),
        DeclareLaunchArgument('asyncvla_device', default_value='cpu'),
        gzserver,
        gzclient,
        robot_state_publisher,
        spawn,
        sim_node,
        edge,
        respawn_fallback,
    ])
