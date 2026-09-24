"""Remote movla backend based on LFM2.5-VL and a flow matching expert.

Runs the Stage A policy from external/movla and sends waypoint chunks to the
edge over gRPC. The checkpoint directory contains checkpoint.pt and
normalizer.json (fetched by scripts/download_checkpoints.sh movla).

As in OmniVLA-edge Path 3, convert model output (x, y, yaw) to float32
(x, y, cos(yaw), sin(yaw)) for the path-only OmniVLA adapter. x/y stay in meters.

Training inputs: context frames approximate a 0.3 m stride with a ring buffer.
History and velocity come from the benchmark bridge's measured robot state.
Missing initial history is padded with stationary rows as in training.
Previous-tail conditioning is disabled without frame transforms.
Stage A was trained only on straight/left/right instruction templates; image
and pose goals are unsupported. Raspicat is absent from the training data, so
the default turtlebot2 embodiment supplies normalization statistics and spec.
Dockerfile.movla vendors the movla sources under /opt/movla/src.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Optional, Tuple

import numpy as np
import PIL.Image

from .base import ModelInfoDict, VLABackend


_LOG = logging.getLogger(__name__)

_DEFAULT_INSTRUCTION = 'go straight ahead'


def _status_line(cum_yaw_deg: float = 0.0, v_mps: float = 0.0) -> str:
    """Build the status row in the training GnmDatasetBase format."""
    if cum_yaw_deg > 20.0:
        turning = 'turning left'
    elif cum_yaw_deg < -20.0:
        turning = 'turning right'
    else:
        turning = 'going straight'
    return f'Status: {turning} (recent cumulative {cum_yaw_deg:+.0f}deg), v={v_mps:.2f}m/s'


def _chunk_to_embedding(waypoints_xyyaw: np.ndarray) -> np.ndarray:
    """Convert (H, 3) x/y/yaw waypoints to (H, 4) x/y/cos/sin float32.

    Keep x/y in meters for the path-only edge adapter.
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
    """Run the movla Stage A policy and return waypoint chunks."""

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
        # Delay heavyweight imports until backend construction.
        # Keep --help and unit tests usable without model dependencies.
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
        # Use fp32 on CPU; bf16 is the CUDA default.
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
        # Past thumbnails in oldest-first order, up to context_frames.
        self._past = collections.deque(maxlen=int(context_frames))
        self._poses = collections.deque(maxlen=int(expert_cfg.history_len) + 1)
        self._velocity = (0.0, 0.0)
        self._motion_history_m = 0.0
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
        self._poses.clear()
        self._motion_history_m = 0.0

    def set_motion_state(self, pose: list[float], velocity: list[float]) -> None:
        """Accept one measured pose and planar velocity per inference frame."""
        if len(pose) != 3 or len(velocity) != 2:
            raise ValueError('invalid MoVLA motion state shape')
        if not all(math.isfinite(float(x)) for x in [*pose, *velocity]):
            raise ValueError('nonfinite MoVLA motion state')
        self._poses.append(tuple(map(float, pose)))
        self._velocity = tuple(map(float, velocity))
        poses = list(self._poses)
        self._motion_history_m = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(poses, poses[1:]))

    def infer(
        self,
        *,
        current_image: PIL.Image.Image,
        past_image: Optional[PIL.Image.Image] = None,  # noqa: ARG002
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
            chunk = self._policy.predict_chunk(batch)  # Shape (1, H, 3), meters/radians.

        # Thumbnail the current frame for the next inference context.
        thumb = current.copy()
        thumb.thumbnail((self._context_size, self._context_size))
        self._past.append(thumb)

        embedding = _chunk_to_embedding(chunk[0].float().cpu().numpy())
        return embedding, {
            'inference_ms': (time.monotonic() - t0) * 1000.0,
            'motion_history_samples': len(self._poses),
            'motion_velocity': list(self._velocity),
            'motion_history_m': self._motion_history_m,
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
        """Build the training image sequence from past thumbnails and current.

        Pad missing history with the oldest available image, as in training.
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
        history[:, :, 2] = 1.0  # Stationary padding: (dx, dy, cos(dyaw), sin(dyaw)).
        poses = list(self._poses)
        for i in range(1, len(poses)):
            px, py, pa = poses[i - 1]
            x, y, a = poses[i]
            dx, dy = x - px, y - py
            c, s = math.cos(pa), math.sin(pa)
            da = math.atan2(math.sin(a - pa), math.cos(a - pa))
            history[0, -(len(poses) - i)] = torch.tensor(
                [c * dx + s * dy, -s * dx + c * dy, math.cos(da), math.sin(da)])
        cum_yaw = (math.degrees(math.atan2(math.sin(poses[-1][2] - poses[0][2]),
                                               math.cos(poses[-1][2] - poses[0][2])))
                   if poses else 0.0)
        return NavBatch(
            vlm_inputs=[VLMInputs(
                images=images,
                instruction=instruction,
                robot_line=self._spec.to_prompt_line(),
                status_line=_status_line(cum_yaw, self._velocity[0]),
                subgoal_line=f'Subgoal: {instruction}',
            )],
            history=history,
            velocity=torch.tensor([self._velocity]),
            prev_tail=torch.zeros(1, h, 3),
            prev_tail_mask=torch.zeros(1, h, dtype=torch.bool),
            embodiment_vec=torch.from_numpy(self._spec.to_vector()).unsqueeze(0),
            embodiment_ids=[self._embodiment],
            actions=torch.zeros(1, h, 3),
        ).to(self._device)


# Only these instruction templates were used in Stage A training.
INSTRUCTION_TEMPLATES = (
    'go straight ahead',
    'turn left ahead',
    'turn right ahead',
)
