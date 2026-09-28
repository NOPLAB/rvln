#!/usr/bin/env bash
# Fetch only the four missing integrated public checkpoints.
set -euo pipefail

workspace=${RVLN_BENCH_WORKSPACE:-/mnt/workspace/nop/raspicat_vla}
cd "$workspace"
export PYTHON_BIN=${PYTHON_BIN:-/mnt/workspace/nop/vla/.venv/bin/python}
export HF_HOME=${HF_HOME:-/mnt/workspace/nop/hf_cache}

for model in asyncvla omnivla_edge navila navida; do
    echo "==> $(date -Is) $model"
    if bash scripts/download_checkpoints.sh "$model"; then
        echo "download_status=$model:ok"
    else
        echo "download_status=$model:failed" >&2
    fi
done
