#!/usr/bin/env bash
set -euo pipefail

# For CUDA/L4T images that cannot inherit a ROS base image.
ros_distro=$1
apt-get update
apt-get install -y --no-install-recommends \
    software-properties-common curl ca-certificates python3
add-apt-repository -y universe
version=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')
. /etc/os-release
curl -fL -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${version}/ros2-apt-source_${version}.${UBUNTU_CODENAME:-${VERSION_CODENAME}}_all.deb"
dpkg -i /tmp/ros2-apt-source.deb
apt-get update
apt-get install -y --no-install-recommends \
    "ros-${ros_distro}-ros-base" "ros-${ros_distro}-rosidl-default-generators" \
    python3-colcon-common-extensions
rm -rf /var/lib/apt/lists/* /tmp/ros2-apt-source.deb
