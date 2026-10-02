"""Compatibility entry point for mobile ROS runtime loading."""

from usim.bridges.ros_runtime import load_ros_python as load_ros_python


def __getattr__(name: str):
    from usim.bridges import ros_runtime

    return getattr(ros_runtime, name)
