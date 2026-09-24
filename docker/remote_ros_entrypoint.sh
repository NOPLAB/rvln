#!/usr/bin/env bash
set -euo pipefail

source "/opt/ros/${ROS_DISTRO}/setup.bash"
cd /workspace
colcon --log-base /tmp/remote_ros_log build --packages-select raspicat_vla_msgs \
    --build-base /tmp/remote_ros_build \
    --install-base /tmp/remote_ros_install
source /tmp/remote_ros_install/setup.bash
exec "$@"
