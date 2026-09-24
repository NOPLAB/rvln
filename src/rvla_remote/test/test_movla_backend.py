"""Tests for MovlaBackend (external/movla Stage A policy).

The CPU tests exercise the pure prompt/waypoint helpers (no torch, no movla
source needed — the backend module keeps its heavy imports inside __init__).
The full forward pass is a slow smoke test gated behind MOVLA_E2E=1 (run inside
Dockerfile.movla with models/movla/<run>/ present; CPU works, CUDA faster).
"""
import math
import os

import numpy as np
import pytest


# ------------------------------------------------------------- status line

def test_status_line_matches_training_format():
    from rvla_remote.backends.movla import _status_line

    # 学習時の GnmDatasetBase.__getitem__ の書式と一字一句同じであること。
    assert _status_line() == \
        'Status: going straight (recent cumulative +0deg), v=0.00m/s'
    assert _status_line(35.0, 0.42) == \
        'Status: turning left (recent cumulative +35deg), v=0.42m/s'
    assert _status_line(-90.0, 1.0) == \
        'Status: turning right (recent cumulative -90deg), v=1.00m/s'


# --------------------------------------------------------- chunk -> embedding

def test_chunk_to_embedding_packs_x_y_cos_sin():
    from rvla_remote.backends.movla import _chunk_to_embedding

    wp = np.array([
        [0.3, 0.0, 0.0],
        [0.6, 0.1, math.pi / 2],
        [0.9, 0.3, -math.pi],
    ], dtype=np.float32)
    out = _chunk_to_embedding(wp)
    assert out.shape == (3, 4)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, :2], wp[:, :2])
    np.testing.assert_allclose(out[:, 2], np.cos(wp[:, 2]), atol=1e-6)
    np.testing.assert_allclose(out[:, 3], np.sin(wp[:, 2]), atol=1e-6)
    # (cos, sin) は単位ベクトル — path-only アダプタの四元数がそのまま正規化済みになる。
    np.testing.assert_allclose(out[:, 2] ** 2 + out[:, 3] ** 2, 1.0, atol=1e-6)


def test_chunk_to_embedding_rejects_wrong_shape():
    from rvla_remote.backends.movla import _chunk_to_embedding

    with pytest.raises(ValueError):
        _chunk_to_embedding(np.zeros((8, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        _chunk_to_embedding(np.zeros(3, dtype=np.float32))


# --------------------------------------------------------------- gated E2E

@pytest.mark.skipif(
    os.environ.get('MOVLA_E2E') != '1',
    reason='set MOVLA_E2E=1 (needs torch + transformers>=5.12 + movla source + '
           'models/movla checkpoint; run inside the movla image)',
)
def test_movla_backend_returns_metric_waypoint_chunk():
    import PIL.Image

    from rvla_remote.backends.movla import MovlaBackend

    backend = MovlaBackend(
        checkpoint_dir=os.environ.get(
            'MOVLA_CHECKPOINT_DIR', '/workspace/models/movla/stage_a_v2'),
        device=os.environ.get('MOVLA_DEVICE', 'cpu'),
    )

    img = PIL.Image.new('RGB', (640, 480), (128, 128, 128))
    arr, metrics = backend.infer(
        current_image=img,
        past_image=None,
        lang_instruction='go straight ahead',
        goal_image=None,
        goal_pose_xy_theta=None,
    )
    info = backend.model_info()
    assert arr.shape == (info.num_tokens, info.embed_dim)  # (horizon, 4)
    assert info.embed_dim == 4
    assert arr.dtype.name == 'float32'
    assert metrics['inference_ms'] > 0
    # (cos, sin) 列は単位円上。
    np.testing.assert_allclose(arr[:, 2] ** 2 + arr[:, 3] ** 2, 1.0, atol=1e-4)
    # x/y はメートル: turtlebot2 の正規化統計 (~1m/step) から桁外れでないこと。
    assert np.all(np.abs(arr[:, :2]) < 20.0)
    print(f'chunk shape={arr.shape} inf_ms={metrics["inference_ms"]:.1f}')
