#!/usr/bin/env bash
#SBATCH --job-name=rvln-live-isaac
#SBATCH --partition=main
#SBATCH --nodelist=pve2ubuntu
#SBATCH --gres=gpu:1
#SBATCH --time=00:04:00
#SBATCH --output=/mnt/workspace/nop/rvln/bench/runs/validation/live_isaac_slurm-%j.log

set -euo pipefail
shared=/mnt/workspace/nop/rvln
out="$shared/bench/runs/validation/live_isaac_stub_${SLURM_JOB_ID}"
bash "$shared/bench/scripts/smoke_live_isaac.sh" "$out"
