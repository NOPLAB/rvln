"""Compatibility factory for the legacy RVLN ROS bridge."""

from usim.bridges.ros2 import Bridge as create_bridge, Ros2Config


def Bridge(*, camera_enabled: bool, configuration=None):
    if configuration is None:
        configuration = Ros2Config(node_name="isaac_rvln_bridge")
    return create_bridge(camera_enabled=camera_enabled, configuration=configuration)
