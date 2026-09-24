import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _include(package, launch_file, launch_arguments):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)
        ),
        launch_arguments=launch_arguments.items(),
    )


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    gui = LaunchConfiguration('gui')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')

    gazebo = _include('raspicat_gazebo', 'raspicat_with_iscas_museum.launch.py', {
        'use_sim_time': use_sim_time,
        'gui': gui,
        'x_pose': x_pose,
        'y_pose': y_pose,
    })
    slam = _include('raspicat_slam', 'raspicat_slam_toolbox.launch.py', {
        'use_sim_time': use_sim_time,
        'use_rviz': use_rviz,
    })

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='-2.0'),
        gazebo,
        slam,
    ])
