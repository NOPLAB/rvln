#!/usr/bin/env bash
# CPU-only regression checks for the real NaVIDA executor; no robot is started.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
set +u
source /opt/ros/humble/setup.bash
set -u
colcon build --merge-install --symlink-install \
  --packages-select rvln_msgs rvln_core rvln_proto rvln_edge \
  --cmake-args -DBUILD_TESTING=OFF
set +u
source install/setup.bash
set -u
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
export PYTHONPATH="$PWD/bench/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -m compileall -q bench/src/bench/live_navida.py bench/src/bench/navida_step.py
python3 -m bench.live_navida --help
python3 -m pytest -q \
  bench/test/test_navida_step.py bench/test/test_navida_turn_coasting.py \
  bench/test/test_live_navida.py \
  src/rvln_edge/test/test_observation_state.py \
  src/rvln_edge/test/test_edge_node_frame_correlation.py \
  src/rvln_edge/test/test_path_follower_hold.py \
  src/rvln_edge/test/test_waypoint_pd.py \
  src/rvln_edge/test/test_omnivla_edge_adapter.py
