#!/usr/bin/env bash
# Fetch model checkpoints into models/. Run without arguments for available models.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage: scripts/download_checkpoints.sh MODEL [RUN]

Models:
  asyncvla            NHirose/AsyncVLA_release
  omnivla             NHirose/omnivla-original
  omnivla_edge        NHirose/omnivla-edge
  movla [RUN]         Private training host; default RUN is stage_a_v2
  navila              a8cheng/navila-llama3-8b-8f (ROS backend)
  navida              waynechu/NaVIDA (ROS backend)
  streamvln_real      StreamVLN real-world checkpoint (research checkpoint)
  internvla_dualvln   InternVLA-N1 DualVLN (research checkpoint)

Set PYTHON_BIN for a Python with huggingface_hub installed. Public models are
large; each command downloads only the selected checkpoint. The research
checkpoints need their upstream inference code before they can run in ROS.
EOF
}

download_hf() {
    local repo_id="$1" out_dir="$2"
    mkdir -p "$out_dir"
    "$PYTHON_BIN" - "$repo_id" "$out_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, out_dir = sys.argv[1:]
path = snapshot_download(repo_id=repo_id, local_dir=out_dir)
print(f'== {repo_id} -> {path}')
PY
}

download_movla() {
    local run="${1:-stage_a_v2}"
    if [[ ! "$run" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || "$run" == *..* ]]; then
        echo "Invalid movla run: $run" >&2
        exit 2
    fi
    local host="${MOVLA_TRAIN_HOST:-nop@pve1ubuntu}"
    local source_dir="${MOVLA_TRAIN_REPO:-/home/nop/dev/mywork/movla}/runs/${run}"
    local out_dir="${REPO_ROOT}/models/movla/${run}"
    mkdir -p "$out_dir"
    rsync -av --progress \
        "${host}:${source_dir}/checkpoint.pt" \
        "${host}:${source_dir}/normalizer.json" \
        "${out_dir}/"
    echo "== movla ${run} -> ${out_dir}"
}

model="${1:-}"
if [[ -z "$model" || "$model" == --help || "$model" == -h ]]; then
    usage
    exit 0
fi

if [[ "$model" != movla && "$#" -ne 1 ]] || [[ "$model" == movla && "$#" -gt 2 ]]; then
    usage >&2
    exit 2
fi

case "$model" in
    asyncvla) download_hf NHirose/AsyncVLA_release "${REPO_ROOT}/models/AsyncVLA_release" ;;
    omnivla) download_hf NHirose/omnivla-original "${REPO_ROOT}/models/omnivla-original" ;;
    omnivla_edge) download_hf NHirose/omnivla-edge "${REPO_ROOT}/models/omnivla-edge" ;;
    movla) download_movla "${2:-stage_a_v2}" ;;
    navila) download_hf a8cheng/navila-llama3-8b-8f \
        "${REPO_ROOT}/models/navila-llama3-8b-8f" ;;
    navida) download_hf waynechu/NaVIDA "${REPO_ROOT}/models/NaVIDA" ;;
    streamvln_real) download_hf \
        mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world \
        "${REPO_ROOT}/models/StreamVLN_real_world" ;;
    internvla_dualvln) download_hf InternRobotics/InternVLA-N1-DualVLN \
        "${REPO_ROOT}/models/InternVLA-N1-DualVLN" ;;
    *) echo "Unknown model: $model" >&2; usage >&2; exit 2 ;;
esac
