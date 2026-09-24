import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _include(package, launch_file, launch_arguments=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)
        ),
        launch_arguments=(launch_arguments or {}).items(),
    )


def generate_launch_description():
    use_urg = LaunchConfiguration('use_urg')
    urg = LaunchConfiguration('urg')
    use_rviz = LaunchConfiguration('use_rviz')

    robot = _include('raspicat', 'raspicat.launch.py', {
        'use_urg': use_urg,
        'urg': urg,
    })
    slam = _include('raspicat_slam', 'raspicat_slam_toolbox.launch.py', {
        'use_sim_time': 'false',
        'use_rviz': use_rviz,
    })

    return LaunchDescription([
        DeclareLaunchArgument('use_urg', default_value='true'),
        DeclareLaunchArgument('urg', default_value='serial'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        robot,
        slam,
    ])
