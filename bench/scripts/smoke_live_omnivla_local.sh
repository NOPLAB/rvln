#!/usr/bin/env bash
set -euo pipefail

shared=${RVLN_BENCH_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
out=${1:?output directory required}
mkdir -p "$out"
server_pid=
cleanup() {
  if [[ -n $server_pid ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

RVLN_INFER_BIND=127.0.0.1 RVLN_INFER_ALLOWED_CLIENT=127.0.0.1 \
  bash "$shared/bench/scripts/run_omnivla_edge_server.sh" \
  >"$out/model_server.log" 2>&1 &
server_pid=$!
ready=false
for attempt in $(seq 1 45); do
  if curl -fsS --max-time 2 http://127.0.0.1:8765/health > /dev/null; then
    ready=true
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    cat "$out/model_server.log" >&2
    exit 1
  fi
  sleep 2
done
if [[ $ready != true ]]; then
  cat "$out/model_server.log" >&2
  exit 1
fi

export RVLN_BACKEND_URL=http://127.0.0.1:8765
export RVLN_BENCH_MODEL=omnivla_edge
export RVLN_BENCH_DEPLOYMENT=isaac_gpu_edge_local_uv_omnivla
export RVLN_EDGE_ADAPTER=omnivla
export RVLN_BENCH_KIND=synthetic_scene_real_model_integration
bash "$shared/bench/scripts/smoke_live_isaac.sh" "$out"
