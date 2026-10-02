#!/usr/bin/env python3
"""CLIP ViT-B/32 の text encoder を ONNX 化する (Phase 1 / spec §3.4)。

engine (`omnivla_edge_engine.py`) は OpenAI CLIP の `encode_text` を **正規化なし**
で使う。open_clip の `ViT-B-32-quickgelu` + `pretrained='openai'` は同一重み・同一演算
なので出力は一致する (normalize=False)。

**argmax-free**: モバイル ORT プラグインは `ArgMax` カーネルを持たないため、EOT 位置を
`eot_index` 入力として外から渡し、`index_select`(=Gather) で取り出す。app 側は EOT
トークン (49407) の位置を計算して渡す。

入力 tokens (1,77) int64, eot_index (1,) int64 -> 出力 feat (1,512) float32。

使い方:
    ~/omnivla-export-venv/bin/python app/scripts/export_clip_text_onnx.py \
        --out app/assets/models/clip_text.onnx
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

# MHA 高速パスは attn_mask 形状検証や融合カーネルで ONNX/直呼びと衝突するため無効化。
torch.backends.mha.set_fastpath_enabled(False)

CTX_LEN = 77
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TextEncoderWrapper(nn.Module):
    """open_clip CLIP.encode_text を argmax なしで再実装 (EOT 位置は入力)。"""

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.m = clip_model

    def forward(self, tokens, eot_index):  # tokens (1,77) int64, eot_index (1,) int64
        m = self.m
        cast_dtype = m.transformer.get_cast_dtype()
        x = m.token_embedding(tokens).to(cast_dtype)
        x = x + m.positional_embedding.to(cast_dtype)
        # open_clip 3.x の transformer は batch_first (permute 不要)。
        x = m.transformer(x, attn_mask=m.attn_mask)
        x = m.ln_final(x)  # (1, 77, 512)
        # argmax の代わりに eot_index で Gather。
        x = torch.index_select(x, 1, eot_index.view(-1))  # (1, 1, 512)
        x = x[:, 0, :]  # (1, 512)
        # text_projection は Parameter (行列) or Linear。
        if isinstance(m.text_projection, nn.Linear):
            x = m.text_projection(x)
        else:
            x = x @ m.text_projection  # 非正規化
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "app/assets/models/clip_text.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import open_clip

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    tokens = tokenizer(["go to the blue trash bin"])  # (1,77) int64
    assert tokens.shape == (1, CTX_LEN), tokens.shape
    eot_index = tokens.argmax(dim=-1).to(torch.int64)  # 参照: 本物の EOT 位置

    wrapper = TextEncoderWrapper(model).eval()
    with torch.no_grad():
        ref = wrapper(tokens, eot_index)
        # 純正 encode_text (argmax 内蔵) と一致するか確認。
        ref_builtin = model.encode_text(tokens)
    consistency = (ref - ref_builtin).abs().max().item()
    print(f"[torch] feat shape={tuple(ref.shape)}  vs encode_text max|diff|={consistency:.3e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (tokens, eot_index),
            args.out,
            input_names=["tokens", "eot_index"],
            output_names=["feat"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[onnx] wrote {args.out} ({size_mb:.1f} MB)")

    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    out = sess.run(["feat"], {"tokens": tokens.numpy(), "eot_index": eot_index.numpy()})[0]
    diff = np.abs(out - ref.numpy())
    print(f"[verify] max|diff|={diff.max():.3e}  mean|diff|={diff.mean():.3e}")
    print("[verify] OK" if diff.max() < 1e-3 else "[verify] WARNING: 差が大きい")


if __name__ == "__main__":
    main()
