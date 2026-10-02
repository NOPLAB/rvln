#!/usr/bin/env bash
set -euo pipefail

shared=${RVLN_BENCH_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
assets="$shared/bench/runs/datasets/omnivla-edge"
export PYTHONPATH="$shared/bench/src:$shared/src/rvln_remote:$shared/src/rvln_core:${PYTHONPATH:-}"
export HOME="$assets"
export HF_HOME=/mnt/workspace/nop/hf_cache
export TOKENIZERS_PARALLELISM=false
test -s "$assets/omnivla-edge.pth"
test -s "$assets/.cache/clip/ViT-B-32.pt"
bind=${RVLN_INFER_BIND:-100.76.158.87}
allowed_client=${RVLN_INFER_ALLOWED_CLIENT:-100.120.242.44}
exec /var/tmp/rvln-bench-infer-venv/bin/python -m bench.live_server \
  --backend omnivla_edge \
  --checkpoint "$assets/omnivla-edge.pth" \
  --device cuda:0 --bind "$bind" --port 8765 \
  --allowed-client "$allowed_client"
