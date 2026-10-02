#!/usr/bin/env python3
"""ONNX をモバイル用に量子化してサイズ/メモリを削減する (spec §2 推論精度)。

fp16 (onnxconverter-common) は onnx 1.22 で Range/Cast まわりが壊れるため、より堅牢な
**ONNX Runtime の int8 動的量子化 (quantize_dynamic)** を使う。MatMul/Gemm 系の重み
(このモデルの大半 = transformer/Linear) を int8 にする。activation は実行時に量子化。

手順: onnxsim で定数畳み込み → quantize_dynamic(QInt8) → fp32 と onnxruntime で比較。
入出力は float32 のままなのでアプリは無改変。

使い方:
    ~/omnivla-export-venv/bin/python app/scripts/quantize_onnx.py \
        app/assets/models/omnivla_edge.onnx app/assets/models/clip_text.onnx
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def rand_for(inp):
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    if "int64" in inp.type:
        return np.zeros(shape, dtype=np.int64) + 5
    return np.random.randn(*shape).astype(np.float32)


def convert(path: str):
    print(f"\n=== {path} ===")
    fp32_mb = os.path.getsize(path) / (1024 * 1024)

    # 1. 定数畳み込み (FiLM の arange->Range など)。
    from onnxsim import simplify

    model, ok = simplify(onnx.load(path))
    sim_path = path + ".sim.tmp"
    onnx.save(model, sim_path)
    print(f"  onnxsim simplified (check={ok})")

    # 2. int8 動的量子化。
    q_path = path + ".q.tmp"
    quantize_dynamic(
        sim_path,
        q_path,
        weight_type=QuantType.QInt8,
    )

    # 3. 検証: fp32(元) vs int8。
    s32 = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    sq = ort.InferenceSession(q_path, providers=["CPUExecutionProvider"])
    feed = {i.name: rand_for(i) for i in s32.get_inputs()}
    o32 = s32.run(None, feed)
    oq = sq.run(None, feed)
    max_diff = max(float(np.abs(a - b).max()) for a, b in zip(o32, oq))
    rel = max(float(np.abs(a - b).max() / (np.abs(a).max() + 1e-9)) for a, b in zip(o32, oq))

    os.replace(q_path, path)
    os.remove(sim_path)
    q_mb = os.path.getsize(path) / (1024 * 1024)
    print(
        f"  size {fp32_mb:.1f}MB -> {q_mb:.1f}MB   max|int8-fp32|={max_diff:.3e} (rel {rel:.2%})"
    )
    if rel > 0.15:
        print("  WARNING: int8 相対誤差が大きい。品質を実機で要確認")


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: quantize_onnx.py <onnx> [<onnx> ...]")
        sys.exit(1)
    np.random.seed(0)
    for p in paths:
        convert(p)


if __name__ == "__main__":
    main()
