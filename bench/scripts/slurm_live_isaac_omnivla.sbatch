#!/usr/bin/env bash
#SBATCH --job-name=rvln-live-omnivla
#SBATCH --partition=main
#SBATCH --nodelist=pve2ubuntu
#SBATCH --gres=gpu:1
#SBATCH --time=00:04:00
#SBATCH --output=/mnt/workspace/nop/rvln/bench/runs/validation/live_omnivla_slurm-%j.log

set -euo pipefail
shared=/mnt/workspace/nop/rvln
for attempt in $(seq 1 30); do
  if curl -fsS --max-time 2 http://100.76.158.87:8765/health > /dev/null; then
    break
  fi
  sleep 2
done
curl -fsS --max-time 2 http://100.76.158.87:8765/health
export RVLN_BACKEND_URL=http://100.76.158.87:8765
export RVLN_BENCH_MODEL=omnivla_edge
export RVLN_BENCH_DEPLOYMENT=isaac_gpu_edge_remote_omnivla
export RVLN_EDGE_ADAPTER=omnivla
export RVLN_BENCH_KIND=synthetic_scene_real_model_integration
out="$shared/bench/runs/validation/live_isaac_omnivla_${SLURM_JOB_ID}"
bash "$shared/bench/scripts/smoke_live_isaac.sh" "$out"
