#!/usr/bin/env bash
set -euo pipefail

shared=/mnt/workspace/nop/rvln
runtime=/var/tmp/rvln-ros-edge
out=${1:?output directory required}
backend_url=${RVLN_BACKEND_URL:-http://127.0.0.1:8766}
model=${RVLN_BENCH_MODEL:-stub}
deployment=${RVLN_BENCH_DEPLOYMENT:-isaac_gpu_edge_local_stub}
adapter=${RVLN_EDGE_ADAPTER:-stub}
episode_file=${model}-c01.json
benchmark_kind=${RVLN_BENCH_KIND:-synthetic_stub_integration}
mkdir -p "$out"
mkdir -p "$out/ros_home"
export ROS_DOMAIN_ID=57
edge_name=rvln-live-edge-${SLURM_JOB_ID:-$$}
isaac_name=rvln-live-isaac-${SLURM_JOB_ID:-$$}
backend_pid=
isaac_pid=
cleanup() {
  docker rm -f "$edge_name" >/dev/null 2>&1 || true
  docker rm -f "$isaac_name" >/dev/null 2>&1 || true
  if [[ -n $backend_pid ]]; then kill "$backend_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT

if [[ $model == stub ]]; then
  python3 "$shared/bench/scripts/smoke_live_stub_server.py" --port 8766 \
    >"$out/backend.log" 2>&1 &
  backend_pid=$!
fi
docker run --rm --name "$edge_name" --network=host \
  --mount "type=bind,source=$runtime,target=/runtime,readonly" \
  --mount "type=bind,source=$shared,target=$shared" \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
  -e FASTRTPS_DEFAULT_PROFILES_FILE="$shared/bench/scripts/fastdds_udp.xml" \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e RVLN_EDGE_ADAPTER="$adapter" \
  --entrypoint /bin/bash rvln-ros-edge:humble -lc \
  'source /opt/ros/humble/setup.bash; source /runtime/install/setup.bash; exec ros2 launch rvln_edge edge_only.launch.py adapter_kind:=$RVLN_EDGE_ADAPTER image_topic:=/camera/color/image_raw with_follower:=true' \
  >"$out/edge.log" 2>&1 &

RVLN_ISAAC_ROS_HOST=1 RVLN_ISAAC_CONTAINER_NAME="$isaac_name" \
  bash "$shared/bench/scripts/isaac6_container.sh" \
  /var/tmp/rvln-isaac6-container/venv/bin/python -m bench.cli rvln \
  --world "$shared/bench/assets/corridor.usd" \
  --robot-urdf "$shared/bench/assets/raspicat_isaac_generated.urdf" \
  --headless --max-seconds 60 --contact-out "$out/contacts.json" \
  >"$out/isaac.log" 2>&1 &
isaac_pid=$!

docker run --rm --network=host \
  --user 2001:2001 \
  --mount "type=bind,source=$runtime,target=/runtime,readonly" \
  --mount "type=bind,source=$shared,target=$shared" \
  -e HOME="$out/ros_home" \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
  -e FASTRTPS_DEFAULT_PROFILES_FILE="$shared/bench/scripts/fastdds_udp.xml" \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e RVLN_LIVE_OUT="$out" \
  -e RVLN_BACKEND_URL="$backend_url" \
  -e RVLN_BENCH_MODEL="$model" \
  -e RVLN_BENCH_DEPLOYMENT="$deployment" \
  --entrypoint /bin/bash rvln-ros-edge:humble -lc \
  'source /opt/ros/humble/setup.bash; source /runtime/install/setup.bash; export PYTHONPATH=/mnt/workspace/nop/rvln/bench/src:${PYTHONPATH:-}; exec python3 -m isaac_rvln.live_episode --url "$RVLN_BACKEND_URL" --model "$RVLN_BENCH_MODEL" --deployment "$RVLN_BENCH_DEPLOYMENT" --episode c01 --manifest /mnt/workspace/nop/rvln/bench/episodes/pilot.json --world-source /mnt/workspace/nop/rvln/bench/worlds/corridor.world --usd /mnt/workspace/nop/rvln/bench/assets/corridor.usd --out "$RVLN_LIVE_OUT/$RVLN_BENCH_MODEL-c01.json" --video "$RVLN_LIVE_OUT/$RVLN_BENCH_MODEL-c01.mp4" --duration 25 --startup-timeout 45' \
  >"$out/episode.log" 2>&1

wait "$isaac_pid"
docker run --rm --network=host \
  --user 2001:2001 \
  --mount "type=bind,source=$shared,target=$shared" \
  -e PYTHONPATH="$shared/bench/src" \
  --entrypoint /usr/bin/python3 rvln-ros-edge:humble \
  "$shared/bench/scripts/finalize_live_isaac.py" --out-dir "$out" \
  --episode-file "$episode_file" --expect-model "$model" \
  --expect-deployment "$deployment" --benchmark-kind "$benchmark_kind"
