#!/usr/bin/env python3
"""OmniVLA-edge 本体モデルを ONNX 化する (Phase 1 / docs/design/mobile_port_spec.md §3.1)。

- `omnivla_edge_model.py` を **無改変のまま** 読み込み、CPU export のために
  `tensor.get_device()` を `tensor.device` へ置換したコピーを exec する
  (パラメータ名は変わらないので strict=True ロードはそのまま通る)。
- EfficientNet の MemoryEfficientSwish は ONNX 非対応なので
  `set_swish(memory_efficient=False)` に切り替える。
- 7 入力を固定 batch=1 でトレースし、出力 action_pred (1,8,4) を書き出す。
- 最後に onnxruntime と PyTorch の出力を突き合わせる (ゴールデン)。

入力名は app 側 `lib/src/inference/ort_runner.dart` の _modelInputNames と一致させる。

使い方:
    ~/omnivla-export-venv/bin/python app/scripts/export_omnivla_edge_onnx.py \
        --weights models/omnivla-edge/omnivla-edge.pth \
        --out app/assets/models/omnivla_edge.onnx
"""
from __future__ import annotations

import argparse
import os
import types

import numpy as np
import torch
import torch.nn as nn

# nn.TransformerEncoderLayer の融合カーネル (_transformer_encoder_layer_fwd) は
# ONNX 未対応。高速パスを無効化し通常実装でトレースさせる。
torch.backends.mha.set_fastpath_enabled(False)

# omnivla_edge_engine.py の _MODEL_PARAMS と一致必須 (checkpoint 契約)。
MODEL_PARAMS = dict(
    context_size=5,
    len_traj_pred=8,
    learn_angle=True,
    obs_encoder="efficientnet-b0",
    obs_encoding_size=1024,
    late_fusion=False,
    mha_num_attention_heads=4,
    mha_num_attention_layers=4,
    mha_ff_dim_factor=4,
)

OBS = 96
LARGE = 224
CTX = MODEL_PARAMS["context_size"]
HIST = CTX + 1  # 6
LEN = MODEL_PARAMS["len_traj_pred"]  # 8
ADIM = 4

INPUT_NAMES = [
    "obs_images",
    "goal_pose",
    "map_images",
    "goal_image",
    "modality_id",
    "feat_text",
    "cur_large",
]
OUTPUT_NAMES = ["action_pred"]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SRC = os.path.join(
    REPO, "src/rvln_core/rvln_core/models/omnivla_edge_model.py"
)


def load_patched_model_class():
    """get_device()->device に置換したコピーを exec し OmniVLA_edge を返す。"""
    with open(MODEL_SRC, "r") as f:
        src = f.read()
    n = src.count(".get_device()")
    src = src.replace(".get_device()", ".device")
    print(f"[patch] .get_device() -> .device : {n} 箇所")

    # map_encoding の余分な unsqueeze(1) は 5D を作り、ONNX の GlobalAveragePool が
    # 全空間次元を潰して壊れる (eager では後段 flatten(start_dim=1) が吸収するので無害)。
    # export に限り除去 — 数値結果は同一。
    pat = "self.goal_encoder.extract_features(map_images).unsqueeze(1)"
    rep = "self.goal_encoder.extract_features(map_images)"
    assert src.count(pat) == 1, f"expected 1 map_encoding unsqueeze, got {src.count(pat)}"
    src = src.replace(pat, rep)
    print("[patch] map_encoding unsqueeze(1) を除去 (ONNX GlobalAveragePool 対策)")
    mod = types.ModuleType("omnivla_edge_model_patched")
    exec(compile(src, MODEL_SRC, "exec"), mod.__dict__)
    return mod.OmniVLA_edge


class ExportWrapper(nn.Module):
    """forward の 3 出力から action_pred のみ返す (app が使うのは軌道だけ)。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, obs_images, goal_pose, map_images, goal_image,
                modality_id, feat_text, cur_large):
        action_pred, _dist, _mask = self.model(
            obs_images, goal_pose, map_images, goal_image,
            modality_id, feat_text, cur_large,
        )
        return action_pred


def dummy_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(1, 3 * HIST, OBS, OBS, generator=g),   # obs_images
        torch.randn(1, ADIM, generator=g),                 # goal_pose
        torch.randn(1, 9, OBS, OBS, generator=g),          # map_images
        torch.randn(1, 3, OBS, OBS, generator=g),          # goal_image
        torch.tensor([7], dtype=torch.int64),              # modality_id (text)
        torch.randn(1, 512, generator=g),                  # feat_text
        torch.randn(1, 3, LARGE, LARGE, generator=g),      # cur_large
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(REPO, "models/omnivla-edge/omnivla-edge.pth"))
    ap.add_argument("--out", default=os.path.join(REPO, "app/assets/models/omnivla_edge.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")

    OmniVLA_edge = load_patched_model_class()
    model = OmniVLA_edge(**MODEL_PARAMS)

    state = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict) and "model" in state and not any(
        k.startswith(("obs_encoder", "decoder")) for k in state
    ):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    print("[load] state_dict strict=True OK")

    # ONNX 非対応の MemoryEfficientSwish を無効化。
    for enc in (model.obs_encoder, model.goal_encoder, model.goal_encoder_img):
        enc.set_swish(memory_efficient=False)

    model.eval()
    wrapper = ExportWrapper(model).eval()

    inputs = dummy_inputs()
    with torch.no_grad():
        ref = wrapper(*inputs)
    print(f"[torch] action_pred shape={tuple(ref.shape)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            args.out,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,  # batch=1 固定
            dynamo=False,  # 旧型の動的制御フロー(index_select/in-place slice)には
                           # レガシー TorchScript エクスポータが安定
        )
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[onnx] wrote {args.out} ({size_mb:.1f} MB)")

    # --- ゴールデン検証 ---
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    feed = {name: t.numpy() for name, t in zip(INPUT_NAMES, inputs)}
    out = sess.run(OUTPUT_NAMES, feed)[0]
    diff = np.abs(out - ref.numpy())
    print(f"[verify] max|diff|={diff.max():.3e}  mean|diff|={diff.mean():.3e}")
    if diff.max() < 1e-3:
        print("[verify] OK (< 1e-3)")
    else:
        print("[verify] WARNING: 差が大きい。opset/演算子を要確認")


if __name__ == "__main__":
    main()
