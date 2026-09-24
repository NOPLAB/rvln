#!/usr/bin/env bash
# docker/ros_entrypoint.sh — in-container bootstrap shared by every ROS service
# in docker/compose.yaml (edge / sim / edge-local / test).
#
# Sources ROS + the prebuilt rt-net overlay (VLA_WS_OVERLAY, skipped when the
# image doesn't have one, e.g. the test-image fallback), then (re)builds the
# bind-mounted rvln_* packages into /workspace/install and execs the
# service command. The build is idempotent: it only runs when a package is
# missing from install/ (a newly added package with a stale overlay counts) or
# when RVLN_REBUILD is set.
set -e

source /opt/ros/humble/setup.bash
if [[ -n ${VLA_WS_OVERLAY:-} && -f ${VLA_WS_OVERLAY} ]]; then
    source "${VLA_WS_OVERLAY}"
fi

cd /workspace
if [[ ! -f src/rvln_proto/rvln_proto/edge_action_pb2.py ]]; then
    bash scripts/gen_proto.sh
fi
_vla_pkgs=(rvln_msgs rvln_proto rvln_core
           rvln_remote rvln_edge rvln_bringup)
_need_build=${RVLN_REBUILD:-}
_msg_fingerprint=$(sha256sum src/rvln_msgs/CMakeLists.txt src/rvln_msgs/msg/*.msg)
if [[ ! -f install/.rvln_msgs_fingerprint ]] ||
   [[ $(cat install/.rvln_msgs_fingerprint) != "$_msg_fingerprint" ]]; then
    _need_build=1
fi
for _p in "${_vla_pkgs[@]}"; do
    [[ -d "install/${_p}" ]] || _need_build=1
done
if [[ -n ${_need_build} ]]; then
    echo "==> colcon build rvln_*" >&2
    colcon build --symlink-install --packages-select "${_vla_pkgs[@]}"
    printf '%s' "$_msg_fingerprint" > install/.rvln_msgs_fingerprint
fi
source install/setup.bash

exec "$@"
