#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
workspace=${RVLN_BENCH_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
runtime=/runtime
mkdir -p "$runtime"
colcon --log-base "$runtime/log" build \
  --base-paths "$workspace/src" \
  --packages-select rvln_msgs rvln_edge \
  --build-base "$runtime/build" \
  --install-base "$runtime/install" \
  --merge-install \
  --event-handlers console_direct+

source "$runtime/install/setup.bash"
python3 -c 'from rvln_msgs.msg import Observation, ActionEmbedding; from rvln_edge.edge_node import VLAEdgeNode; from rvln_edge.path_follower_node import PathFollowerNode; print("rvln ROS edge imports passed")'
