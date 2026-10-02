"""Prepare Isaac RGB-D observations for the published VLN-CE CMA inputs.

This module does not load the pretrained policy or its licensed assets.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from bench.artifacts import sha256
from isaac_r2r.protocol import load_episodes, scene_name


CMA_VOCAB_SIZE = 2504
CMA_DEPTH_MAX_M = 10.0


def validate_instruction_tokens(tokens: list[int], vocab_size: int = CMA_VOCAB_SIZE) -> None:
    """Reject token IDs that cannot index the published CMA embedding table."""
    if not tokens or all(token == 0 for token in tokens):
        raise ValueError("CMA instruction tokens are empty")
    if any(type(token) is not int or token < 0 or token >= vocab_size for token in tokens):
        raise ValueError(
            "instruction tokens exceed CMA vocabulary; use the R2R_VLNCE_v1-3_preprocessed split"
        )


def load_cma_tokens(reference: Path, policy_split: Path) -> tuple[dict, dict[str, list[int]]]:
    """Match preprocessed tokens to every benchmark episode and instruction."""
    expected = {str(item["episode_id"]): item for item in load_episodes(reference)}
    actual = {str(item["episode_id"]): item for item in load_episodes(policy_split)}
    if expected.keys() != actual.keys():
        raise ValueError("CMA policy split does not cover the exact benchmark episodes")
    token_count = 0
    max_token = 0
    tokens_by_id = {}
    for episode_id, episode in expected.items():
        policy = actual[episode_id]
        if (
            scene_name(episode) != scene_name(policy)
            or episode["instruction"]["instruction_text"]
            != policy["instruction"]["instruction_text"]
            or episode["start_position"] != policy["start_position"]
            or episode["goals"][0]["position"] != policy["goals"][0]["position"]
        ):
            raise ValueError(f"CMA policy split episode differs: {episode_id}")
        tokens = policy["instruction"].get("instruction_tokens", [])
        validate_instruction_tokens(tokens)
        tokens_by_id[episode_id] = tokens
        token_count += len(tokens)
        max_token = max(max_token, *tokens)
    report = {
        "episodes": len(expected),
        "tokens": token_count,
        "maximum_token_id": max_token,
        "vocab_size": CMA_VOCAB_SIZE,
    }
    return report, tokens_by_id


def audit_cma_split(reference: Path, policy_split: Path) -> dict:
    """Check preprocessed token compatibility without exposing dataset contents."""
    report, _ = load_cma_tokens(reference, policy_split)
    return {
        **report,
        "split_sha256": sha256(reference),
        "policy_split_sha256": sha256(policy_split),
    }


def prepare_rgbd(
    jpeg: bytes, depth_f32: bytes, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return uint8 RGB 224x224 and normalized float32 depth 256x256.

    The Isaac observation carries metric image-plane depth. The published
    Habitat configuration normalizes its 0-10 m depth sensor to [0, 1].
    """
    import numpy as np
    from PIL import Image

    if (width, height) != (256, 256) or len(depth_f32) != width * height * 4:
        raise ValueError("CMA requires a 256x256 float32 depth observation")
    with Image.open(io.BytesIO(jpeg)) as image:
        if image.size != (256, 256):
            raise ValueError("CMA requires a 256x256 Isaac RGB observation")
        rgb = np.asarray(
            image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8
        )
    depth = np.frombuffer(depth_f32, dtype="<f4").reshape((height, width))
    depth = np.nan_to_num(depth, nan=0.0, posinf=CMA_DEPTH_MAX_M, neginf=0.0)
    depth = np.clip(depth, 0.0, CMA_DEPTH_MAX_M) / CMA_DEPTH_MAX_M
    return rgb, depth.astype(np.float32, copy=False)[:, :, None]
