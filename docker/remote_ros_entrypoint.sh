#!/usr/bin/env bash
set -euo pipefail

source "/opt/ros/${ROS_DISTRO}/setup.bash"
cd /workspace
colcon --log-base /tmp/remote_ros_log build \
    --packages-select rvln_msgs rvln_core rvln_remote \
    --build-base /tmp/remote_ros_build \
    --install-base /tmp/remote_ros_install
source /tmp/remote_ros_install/setup.bash
# The movla image keeps model dependencies in /opt/venv. ROS console scripts
# are generated with the system Python, so expose that venv's packages to them.
if [[ -d /opt/venv/lib/python3.12/site-packages ]]; then
    export PYTHONPATH="/opt/venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
fi
exec "$@"
