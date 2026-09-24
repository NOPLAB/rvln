#!/usr/bin/env bash
#SBATCH --job-name=rvln-vln-replay
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/workspace/nop/raspicat_vla/bench/runs/slurm-%j.out
#SBATCH --error=/mnt/workspace/nop/raspicat_vla/bench/runs/slurm-%j.err
set -euo pipefail

workspace=${RVLN_BENCH_WORKSPACE:-/mnt/workspace/nop/raspicat_vla}
python_bin=${RVLN_BENCH_PYTHON:-/mnt/workspace/nop/vla/.venv/bin/python}
vendor=${RVLN_BENCH_VENDOR:-/home/nop/.cache/rvln_bench_vendor}
backend=''
arguments=("$@")
for ((i=0; i<${#arguments[@]}-1; i++)); do
    if [[ ${arguments[i]} == --backend ]]; then
        backend=${arguments[i+1]}
        break
    fi
done
case $backend in
    asyncvla) model_source="$workspace/external/AsyncVLA" ;;
    omnivla|omnivla_edge) model_source="$workspace/external/OmniVLA" ;;
    navila) model_source="$workspace/bench/upstream/NaVILA:$workspace/bench/upstream/scaling_on_scales" ;;
    *) model_source='' ;;
esac
case $backend in
    movla) vendor='' ;;
    navida) vendor=${RVLN_BENCH_NAVIDA_VENDOR:-/home/nop/.cache/rvln_bench_navida} ;;
esac
export PYTHONPATH="${vendor:+$vendor:}$workspace/src/rvln_remote:$workspace/src/rvln_core:${model_source:+$model_source:}$workspace/external/MBRA/train:$workspace/external/movla/src:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/mnt/workspace/nop/hf_cache}
export TOKENIZERS_PARALLELISM=false

echo "job_id=${SLURM_JOB_ID:-none} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader
"$python_bin" "$workspace/bench/replay.py" "$@"
