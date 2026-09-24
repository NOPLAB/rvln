#!/usr/bin/env bash
# bash.sh — drop into an interactive bash inside a lightweight ROS2 Humble image.
#
# Unlike scripts/vla.sh (which builds and runs the heavy real/sim/remote images),
# this spins up the stock `ros:humble-ros-base` image — no Gazebo, no desktop, no
# CUDA, no colcon build — just a shell with the core ROS2 CLI (`ros2 topic`,
# `ros2 node`, rclpy, …) already on PATH. Handy for poking at a running stack,
# echoing topics, or trying `ros2` commands without waiting on a workspace build.
#
# The repo is bind-mounted at /workspace (the working directory), so files you
# touch are the real ones on the host. If a colcon overlay has been built
# (install/setup.bash present) it is sourced on top of the base ROS2 env; if not,
# only /opt/ros/humble is available — that's expected for a bare shell.
#
# Usage:
#   scripts/bash.sh                     # interactive bash
#   scripts/bash.sh ros2 topic list     # run one command, then exit
#
# Seeing a running vla.sh stack's topics from here needs two things to line up:
#
#   1. DDS domain. ROS_DOMAIN_ID is forwarded from the host when set — so it must
#      be set (and exported) in the shell you launch this from, matching the value
#      the vla.sh container got. If you started vla.sh with `sudo ROS_DOMAIN_ID=N
#      ./scripts/vla.sh …`, that N only reached the container; export the SAME N
#      here (fish: `set -x ROS_DOMAIN_ID N`) or `ros2 topic list` comes back empty.
#
#   2. Transport. FastDDS talks to same-host peers over shared memory by default,
#      but each container has its own /dev/shm, so discovery succeeds (topics show
#      up) yet `ros2 topic echo` stays silent. We sidestep that by pinning THIS
#      container to a UDP-only FastDDS profile — works over --network host without
#      sharing /dev/shm, and needs no change to the vla.sh container. Opt out with
#      RVLN_UDP_ONLY=0 (e.g. if you later run vla.sh with --ipc host).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${RVLN_BASE_IMAGE:-ros:humble-ros-base}"

# Map the host user so files created under the bind-mounted /workspace are owned
# by you, not root. The mapped uid has no /etc/passwd entry, hence HOME=/tmp.
docker_args=(
    --rm -i
    --user "$(id -u):$(id -g)" -e HOME=/tmp
    --network host
    -v "$REPO_ROOT:/workspace" -w /workspace
)
# Allocate a TTY only when we actually have one, so piped/CI use (e.g.
# `scripts/bash.sh ros2 topic list | grep foo`) doesn't trip docker's
# "cannot attach stdin to a TTY-enabled container".
[[ -t 0 && -t 1 ]] && docker_args+=(-t)
[[ -n ${ROS_DOMAIN_ID:-} ]] && docker_args+=(-e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")

# Build a safely-quoted command for the inner login shell: default to bash if no
# args were given, otherwise exec the passed argv verbatim.
if (($# == 0)); then
    inner="exec bash"
else
    quoted=""
    for a in "$@"; do
        quoted+=" $(printf '%q' "$a")"
    done
    inner="exec${quoted}"
fi

# UDP-only FastDDS profile (see header). Written into the container at /tmp (the
# mapped user's HOME) and selected via FASTRTPS_DEFAULT_PROFILES_FILE. Attributes
# use single quotes so the XML carries no `"` that would close the outer bash -lc
# string. Disable with RVLN_UDP_ONLY=0.
udp_only_setup=":"
if [[ ${RVLN_UDP_ONLY:-1} != 0 ]]; then
    udp_only_setup="
    cat > /tmp/fastdds_udp_only.xml <<'XML'
<?xml version='1.0' encoding='UTF-8'?>
<dds xmlns='http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles'>
  <profiles>
    <transport_descriptors>
      <transport_descriptor>
        <transport_id>udp_only</transport_id>
        <type>UDPv4</type>
      </transport_descriptor>
    </transport_descriptors>
    <participant profile_name='udp_only' is_default_profile='true'>
      <rtps>
        <userTransports><transport_id>udp_only</transport_id></userTransports>
        <useBuiltinTransports>false</useBuiltinTransports>
      </rtps>
    </participant>
  </profiles>
</dds>
XML
    export FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/fastdds_udp_only.xml"
fi

exec docker run "${docker_args[@]}" "$IMAGE" bash -lc "
    ${udp_only_setup}
    source /opt/ros/humble/setup.bash
    [ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash
    ${inner}
"
