"""Tests for OmniVLAEdgeBackend (Plan 2B Path 3 — remote OmniVLA-edge).

The CPU test exercises the pure goal-mode selection. The full forward pass is a
slow GPU + weights + CLIP smoke test gated behind OMNIVLA_EDGE_E2E=1 (run inside
Dockerfile.omnivla with --gpus all and models/omnivla-edge/ mounted).
"""
import os

import pytest


# --------------------------------------------------------------- goal-mode map

def test_goal_mode_prefers_text_then_image_then_pose():
    from rvln_remote.backends.omnivla_edge import _goal_mode_for

    assert _goal_mode_for(has_lang=True, has_image_goal=True, has_pose=True) == 'text'
    assert _goal_mode_for(has_lang=False, has_image_goal=True, has_pose=True) == 'image'
    assert _goal_mode_for(has_lang=False, has_image_goal=False, has_pose=True) == 'pose'
    # Nothing set -> default to text (engine substitutes an empty prompt).
    assert _goal_mode_for(has_lang=False, has_image_goal=False, has_pose=False) == 'text'


# --------------------------------------------------------------- gated E2E

@pytest.mark.skipif(
    os.environ.get('OMNIVLA_EDGE_E2E') != '1',
    reason='set OMNIVLA_EDGE_E2E=1 (needs CUDA + omnivla-edge.pth + CLIP + rvln_core)',
)
def test_omnivla_edge_backend_returns_scaled_action_chunk():
    import numpy as np
    import PIL.Image

    from rvln_remote.backends.omnivla_edge import OmniVLAEdgeBackend

    backend = OmniVLAEdgeBackend(
        weights_path=os.environ.get(
            'OMNIVLA_EDGE_WEIGHTS', '/workspace/models/omnivla-edge/omnivla-edge.pth'),
        device='cuda:0',
    )
    backend.warmup(num_iters=1)

    img = PIL.Image.new('RGB', (224, 224), (128, 128, 128))
    arr, metrics = backend.infer(
        current_image=img,
        past_image=img,
        lang_instruction='blue trash bin',
        goal_image=None,
        goal_pose_xy_theta=None,
    )
    info = backend.model_info()
    assert arr.ndim == 2
    assert arr.shape == (info.num_tokens, info.embed_dim)  # (len_traj_pred, 4)
    assert info.embed_dim == 4
    assert arr.dtype.name == 'float32'
    assert metrics['inference_ms'] > 0
    assert metrics['modality_id'] == 7  # text/language nav
    # cos/sin columns stay unit-ish (untouched by the metric scaling).
    assert np.all(np.abs(arr[:, 2:4]) <= 1.5)
    print(f'chunk shape={arr.shape} inf_ms={metrics["inference_ms"]:.1f}')
