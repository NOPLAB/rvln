#!/usr/bin/env bash
#SBATCH --job-name=rvln-omnivla-edge
#SBATCH --partition=main
#SBATCH --nodelist=pve1ubuntu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/workspace/nop/rvln/bench/runs/validation/omnivla_edge_smoke-%j.log

set -euo pipefail
shared=/mnt/workspace/nop/rvln
echo "Slurm GPU allocation: ${CUDA_VISIBLE_DEVICES:-unset}"
export PYTHONPATH="$shared/src/rvln_remote:$shared/src/rvln_core:${PYTHONPATH:-}"
export HOME="$shared/bench/runs/datasets/omnivla-edge"
export HF_HOME=/mnt/workspace/nop/hf_cache
export TOKENIZERS_PARALLELISM=false
"/var/tmp/rvln-bench-infer-venv/bin/python" \
  "$shared/bench/scripts/smoke_omnivla_edge_checkpoint.py" \
  --checkpoint "$shared/bench/runs/datasets/omnivla-edge/omnivla-edge.pth" \
  --device cuda:0
