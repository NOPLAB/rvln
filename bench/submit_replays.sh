#!/usr/bin/env bash
# Submit one fixed-camera replay per available VLN backend to Slurm.
set -euo pipefail

workspace=${RVLN_BENCH_WORKSPACE:-/mnt/workspace/nop/raspicat_vla}
scene=${1:?usage: submit_replays.sh SCENE 'instruction'}
instruction=${2:?usage: submit_replays.sh SCENE 'instruction'}
frames="$workspace/bench/runs/${scene}_capture"
test -f "$frames/capture.json"
mkdir -p "$workspace/bench/runs/$scene"
cd "$workspace"

for backend in asyncvla omnivla omnivla_edge movla navila navida; do
    case $backend in
        asyncvla) checkpoint="$workspace/models/AsyncVLA_release" ;;
        omnivla) checkpoint="$workspace/models/omnivla-original" ;;
        omnivla_edge) checkpoint="$workspace/models/omnivla-edge/omnivla-edge.pth" ;;
        movla) checkpoint=/mnt/workspace/nop/vla/runs/stage_a_v9b ;;
        navila) checkpoint="$workspace/models/navila-llama3-8b-8f" ;;
        navida) checkpoint="$workspace/models/NaVIDA" ;;
    esac
    args=(--backend "$backend" --checkpoint "$checkpoint" --frames "$frames"
          --out "$workspace/bench/runs/$scene/$backend.json" --max-frames 20
          --text "$instruction")
    if [[ $backend == asyncvla ]]; then
        args+=(--resume-step 750000)
    fi
    sbatch "$workspace/bench/slurm_replay.sbatch" "${args[@]}"
done
