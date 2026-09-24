"""movla remote backend — LFM2.5-VL ベースの自律移動 VLA (external/movla).

movla (https://github.com/NOPLAB/movla) の Stage A ポリシー
(凍結 LFM2.5-VL-1.6B バックボーン + flow matching action expert) をこの GPU/CPU
ボックスで走らせ、予測ウェイポイントチャンクを gRPC で edge に流す。checkpoint は
``{checkpoint_dir}/checkpoint.pt`` (expert + state_encoder + expert_cfg) と
``normalizer.json`` (scripts/download_movla_checkpoint.sh で取得)。

ワイヤ契約は OmniVLA-edge Path 3 と同じ「ウェイポイント直送」: モデル出力
``(horizon, 3)`` の (x, y, yaw) [m, rad] を ``(horizon, 4)`` の
(x, y, cos(yaw), sin(yaw)) float32 に詰め替えて ActionEmbedding として返すので、
edge 側は path-only の ``adapter_kind=omnivla`` がそのまま使える (x/y はメートル)。

学習時条件との対応 (movla の GnmDatasetBase.__getitem__ が定義):

- 画像: 過去サムネイル ``context_frames`` 枚 + 現在フレーム。ここでは受信
  フレームのリングバッファで近似する (学習時は ~0.3m 間隔のストライド)。
- history / velocity: リモートにはオドメトリが来ないため停止状態パディング
  (history は [0,0,1,0] 行、v=0) に固定する。学習データの走行開始点と同分布。
- prev_tail: フレーム間の座標変換が取れないため常に無効 (学習時も確率
  prev_drop_p で落としており in-distribution)。
- instruction: goal の言語指示をそのまま使う。Stage A はテンプレート指示
  ("go straight ahead" / "turn left ahead" / "turn right ahead") のみで
  学習されている点に注意。goal 画像・pose 目標は未対応 (無視して警告)。
- embodiment: raspicat は学習データに無いので、normalizer 統計を持つ機体
  (既定 turtlebot2 — 小型差動二輪で最も近い) の spec と統計を使う。

movla のソース (movla_v1 / movla_libs) は Dockerfile.movla が
/opt/movla/src に vendor し、compose.yaml が PYTHONPATH に載せる。
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np
import PIL.Image

from .base import ModelInfoDict, VLABackend


_LOG = logging.getLogger(__name__)

_DEFAULT_INSTRUCTION = 'go straight ahead'


def _status_line(cum_yaw_deg: float = 0.0, v_mps: float = 0.0) -> str:
    """学習時の status 行 (GnmDatasetBase.__getitem__) と同一書式を再現する。"""
    if cum_yaw_deg > 20.0:
        turning = 'turning left'
    elif cum_yaw_deg < -20.0:
        turning = 'turning right'
    else:
        turning = 'going straight'
    return f'Status: {turning} (recent cumulative {cum_yaw_deg:+.0f}deg), v={v_mps:.2f}m/s'


def _chunk_to_embedding(waypoints_xyyaw: np.ndarray) -> np.ndarray:
    """(H, 3) の (x, y, yaw) → (H, 4) の (x, y, cos, sin) float32。

    x/y はモデル出力のままメートル。edge の path-only アダプタ
    (`trajectory_to_path`, spacing=1) がそのまま Path に描ける形。
    """
    wp = np.asarray(waypoints_xyyaw, dtype=np.float32)
    if wp.ndim != 2 or wp.shape[-1] != 3:
        raise ValueError(f'expected (H, 3) waypoints (x, y, yaw); got shape={wp.shape}')
    out = np.empty((wp.shape[0], 4), dtype=np.float32)
    out[:, 0] = wp[:, 0]
    out[:, 1] = wp[:, 1]
    out[:, 2] = np.cos(wp[:, 2])
    out[:, 3] = np.sin(wp[:, 2])
    return out


class MovlaBackend(VLABackend):
    """movla Stage A ポリシーをリモートで走らせてウェイポイントチャンクを返す。"""

    def __init__(
        self,
        *,
        checkpoint_dir: str = '/workspace/models/movla/stage_a_v2',
        device: str = 'cuda:0',
        embodiment: str = 'turtlebot2',
        backbone_layer_index: int = 8,
        context_frames: int = 3,
        context_size: int = 192,
    ) -> None:
        # torch / transformers / movla はここで初めて import する
        # (推論ノードの --help やユニットテストを重い依存なしで通すため)。
        import collections
        from pathlib import Path

        import torch

        from movla_libs.data.gnm import _SPECS
        from movla_v1.data.normalize import ActionNormalizer
        from movla_v1.model.action_expert import ActionExpertConfig
        from movla_v1.model.backbone import LFMBackbone
        from movla_v1.model.policy import MovlaPolicy

        ckpt_dir = Path(checkpoint_dir)
        ckpt = torch.load(
            ckpt_dir / 'checkpoint.pt', map_location=device, weights_only=True)
        normalizer = ActionNormalizer.load(ckpt_dir / 'normalizer.json')
        if embodiment not in normalizer.stats:
            raise ValueError(
                f'embodiment {embodiment!r} not in normalizer stats '
                f'{sorted(normalizer.stats)} ({ckpt_dir / "normalizer.json"})')
        if embodiment not in _SPECS:
            raise ValueError(f'embodiment {embodiment!r} not in movla _SPECS')

        expert_cfg = ActionExpertConfig(**ckpt['expert_cfg'])
        # bf16 は CUDA 前提の既定。CPU では fp32 の方が速く数値も安全。
        dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
        if device.startswith('cuda'):
            torch.set_float32_matmul_precision('high')
        backbone = LFMBackbone(
            layer_index=backbone_layer_index, dtype=dtype, device=device)
        policy = MovlaPolicy(backbone, expert_cfg, normalizer).to(device)
        policy.expert.load_state_dict(ckpt['expert'])
        policy.state_encoder.load_state_dict(ckpt['state_encoder'])
        policy.eval()

        self._torch = torch
        self._policy = policy
        self._expert_cfg = expert_cfg
        self._spec = _SPECS[embodiment]
        self._embodiment = embodiment
        self._device = str(device)
        self._checkpoint_dir = str(ckpt_dir)
        self._context_size = int(context_size)
        # 過去フレームのサムネイル (古い順)。学習時の context_frames 枚に対応。
        self._past = collections.deque(maxlen=int(context_frames))
        self._warned_goal = False

    # ------------------------------------------------------------- VLABackend

    def warmup(self, num_iters: int = 1) -> None:
        gray = PIL.Image.new('RGB', (640, 480), (128, 128, 128))
        for _ in range(max(1, num_iters)):
            self.infer(
                current_image=gray, past_image=None,
                lang_instruction=_DEFAULT_INSTRUCTION,
                goal_image=None, goal_pose_xy_theta=None)
        self._past.clear()

    def infer(
        self,
        *,
        current_image: PIL.Image.Image,
        past_image: Optional[PIL.Image.Image] = None,  # noqa: ARG002 (履歴は内部リングバッファ)
        lang_instruction: str,
        goal_image: Optional[PIL.Image.Image],
        goal_pose_xy_theta: Optional[Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, dict]:
        t0 = time.monotonic()
        torch = self._torch

        if (goal_image is not None or goal_pose_xy_theta is not None) \
                and not self._warned_goal:
            _LOG.warning('movla backend is language-only; ignoring image/pose goal')
            self._warned_goal = True

        instruction = lang_instruction or _DEFAULT_INSTRUCTION
        current = current_image.convert('RGB')
        images = self._context_images(current)

        batch = self._build_batch(images, instruction)
        with torch.no_grad():
            chunk = self._policy.predict_chunk(batch)  # (1, H, 3) 生値 m/rad

        # 次回の文脈用に現在フレームをサムネイル化して積む。
        thumb = current.copy()
        thumb.thumbnail((self._context_size, self._context_size))
        self._past.append(thumb)

        embedding = _chunk_to_embedding(chunk[0].float().cpu().numpy())
        return embedding, {
            'inference_ms': (time.monotonic() - t0) * 1000.0,
        }

    def model_info(self) -> ModelInfoDict:
        return ModelInfoDict(
            model_name='NOPLAB/movla',
            model_version=f'movla-stage-a ({self._checkpoint_dir})',
            num_tokens=int(self._expert_cfg.horizon),
            embed_dim=4,
            device=self._device,
            ready=True,
        )

    # ---------------------------------------------------------------- helpers

    def _context_images(self, current: PIL.Image.Image) -> list:
        """学習時の画像構成 (過去サムネイル×N + 現在フレーム) を再現する。

        履歴が足りない間は最古のフレーム (無ければ現在フレーム) で左パディング —
        学習時の ``max(0, cur - j*stride)`` と同じ振る舞い。
        """
        thumbs = list(self._past)
        if not thumbs:
            oldest = current.copy()
            oldest.thumbnail((self._context_size, self._context_size))
            thumbs = [oldest]
        n_ctx = self._past.maxlen
        pad = [thumbs[0]] * (n_ctx - len(thumbs))
        return pad + thumbs + [current]

    def _build_batch(self, images: list, instruction: str):
        from movla_libs.data.schema import NavBatch
        from movla_v1.model.backbone import VLMInputs

        torch = self._torch
        cfg = self._expert_cfg
        h = int(cfg.horizon)
        history = torch.zeros(1, int(cfg.history_len), 4)
        history[:, :, 2] = 1.0  # 停止パディング: (Δx, Δy, cosΔyaw, sinΔyaw)
        return NavBatch(
            vlm_inputs=[VLMInputs(
                images=images,
                instruction=instruction,
                robot_line=self._spec.to_prompt_line(),
                status_line=_status_line(),
                subgoal_line=f'Subgoal: {instruction}',
            )],
            history=history,
            velocity=torch.zeros(1, 2),
            prev_tail=torch.zeros(1, h, 3),
            prev_tail_mask=torch.zeros(1, h, dtype=torch.bool),
            embodiment_vec=torch.from_numpy(self._spec.to_vector()).unsqueeze(0),
            embodiment_ids=[self._embodiment],
            actions=torch.zeros(1, h, 3),
        ).to(self._device)


# Stage A の学習に使われた指示テンプレート (これ以外の自由文は分布外)。
INSTRUCTION_TEMPLATES = (
    'go straight ahead',
    'turn left ahead',
    'turn right ahead',
)
