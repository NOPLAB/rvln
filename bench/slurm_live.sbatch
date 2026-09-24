#!/usr/bin/env bash
#SBATCH --job-name=rvln-vln-live
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/workspace/nop/raspicat_vla/bench/runs/slurm-%j.out
#SBATCH --error=/mnt/workspace/nop/raspicat_vla/bench/runs/slurm-%j.err
set -euo pipefail
workspace=${RVLN_BENCH_WORKSPACE:-/mnt/workspace/nop/raspicat_vla}
vendor=${RVLN_BENCH_VENDOR:-/home/nop/.cache/rvln_bench_vendor}
export PYTHONPATH="$vendor:$workspace/src/rvln_remote:$workspace/src/rvln_core:$workspace/external/OmniVLA:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/mnt/workspace/nop/hf_cache}
exec /mnt/workspace/nop/vla/.venv/bin/python "$workspace/bench/live_server.py" "$@"
