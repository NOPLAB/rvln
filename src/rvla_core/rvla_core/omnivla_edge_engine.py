"""OmniVLAEdgeEngine — the ROS-free OmniVLA-edge policy (model + CLIP + history).

This is the shared compute core of Plan 2B's on-device policy. It owns the
``OmniVLA_edge`` network, CLIP ViT-B/32, the observation ring buffer and all the
pre/post-processing, and turns ``(current RGB frame, goal)`` into a raw action
chunk ``(len_traj_pred, 4)`` packed as ``(x, y, cos, sin)`` in *waypoint-spacing
units* (NOT scaled to metres — callers apply ``metric_waypoint_spacing``).

Two callers share it, which is the whole point of extracting it here:

- **Path 2 (on-edge, standalone):** ``rvla_edge.adapters.omnivla_edge_local``
  wraps this engine and turns the chunk into a ``nav_msgs/Path`` directly on the
  robot. No cloud.
- **Path 3 (remote split — Jetson infers, Raspberry Pi controls):**
  ``rvla_remote.backends.omnivla_edge`` wraps this engine on a Jetson,
  scales the chunk to metres and ships it over gRPC as an ``ActionEmbedding``;
  the Pi runs only the light path-only ``OmniVLAEdgeAdapter`` (no torch).

Keeping the pipeline in one place means the subtle bits that MUST match
``external/OmniVLA/inference/run_omnivla_edge.py`` (ring-buffer stacking, goal
tensors, modality ids, the zero-fill satellite map) live in exactly one file.

The module is deliberately ROS-free (numpy / cv2 / torch / clip only) so the
remote package can import it without pulling rclpy or nav_msgs.

Limitations (v1):
- The upstream ``OmniVLA_edge.forward`` calls ``obs_img.get_device()`` and feeds
  the result to ``.to(device)``, which is GPU-only (CPU tensors report device
  ``-1``). So the engine requires CUDA; ``device='cpu'`` is rejected.
- The ring buffer is per-engine, not per-client. One engine serves one robot
  stream. (Both callers instantiate one engine per node/server, so this holds.)
"""
from __future__ import annotations

import logging
import math
import os
from typing import List, Optional

import cv2
import numpy as np


_LOG = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Trajectory waypoints come out of the model in units of this spacing (metres);
# matches metric_waypoint_spacing in run_omnivla_edge.py.
_METRIC_WAYPOINT_SPACING = 0.1
# Clamp pose-goal range like run_omnivla_edge.py (thres_dist).
_GOAL_DIST_THRESHOLD_M = 30.0

# Modality ids from run_omnivla_edge.run_forward_pass. We only expose the
# single-goal subset reachable from the proto GoalSpec.
_MODALITY_POSE = 4      # pose only
_MODALITY_IMAGE = 6     # image only
_MODALITY_TEXT = 7      # language only
_MODALITY_TEXT_POSE = 8  # language + pose (unused in v1; here for completeness)

# Model hyper-parameters — straight from run_omnivla_edge.py's model_params.
# These define the architecture; they MUST match the omnivla-edge.pth checkpoint.
_MODEL_PARAMS = dict(
    context_size=5,
    len_traj_pred=8,
    learn_angle=True,
    obs_encoder='efficientnet-b0',
    obs_encoding_size=1024,
    late_fusion=False,
    mha_num_attention_heads=4,
    mha_num_attention_layers=4,
    mha_ff_dim_factor=4,
)


def _normalize_chw(image_rgb: np.ndarray, size: int) -> np.ndarray:
    """RGB uint8 HxWx3 -> (3, size, size) ImageNet-normalized float32 (CHW)."""
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f'expected uint8 HxWx3 RGB, got dtype={image_rgb.dtype} shape={image_rgb.shape}'
        )
    img = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))


def _black_chw(size: int) -> np.ndarray:
    """ImageNet-normalized all-black (3, size, size) — the zero-fill satellite/goal."""
    return _normalize_chw(np.zeros((size, size, 3), dtype=np.uint8), size)


def _modality_id_for(mode: str) -> int:
    """Map a proto goal mode string to an OmniVLA-edge modality id."""
    if mode == 'text':
        return _MODALITY_TEXT
    if mode == 'pose':
        return _MODALITY_POSE
    if mode == 'image':
        return _MODALITY_IMAGE
    raise ValueError(f"unknown goal mode {mode!r} (expected 'text'|'pose'|'image')")


def _pose_goal_vector(pose_xy_theta) -> np.ndarray:
    """Build the (4,) goal_pose vector from a robot-relative pose goal.

    The model consumes ``(x_fwd/spacing, y_left/spacing, cos(dtheta),
    sin(dtheta))`` — the same robot-frame (x forward, y left) convention as the
    predicted waypoints (see train_omnivla_dataset.py:357 and the goal-star
    plot in run_omnivla_edge.save_robot_behavior). run_omnivla_edge.py's
    ``(relative_y, -relative_x)`` shuffle is NOT a robot-frame swap: those
    variables live in a compass-aligned frame (rel_x=right, rel_y=forward), and
    the shuffle converts them to exactly this (forward, left) layout. Our input
    is already robot-relative metres (no edge tf in v1), so no swap here. Range
    is clamped to thres_dist first.
    """
    x_fwd, y_left, theta = float(pose_xy_theta[0]), float(
        pose_xy_theta[1]), float(pose_xy_theta[2])
    radius = math.hypot(x_fwd, y_left)
    if radius > _GOAL_DIST_THRESHOLD_M:
        scale = _GOAL_DIST_THRESHOLD_M / radius
        x_fwd *= scale
        y_left *= scale
    return np.array(
        [
            x_fwd / _METRIC_WAYPOINT_SPACING,
            y_left / _METRIC_WAYPOINT_SPACING,
            math.cos(theta),
            math.sin(theta),
        ],
        dtype=np.float32,
    )


def _stack_frames(frames: List[np.ndarray], need: int) -> np.ndarray:
    """Stack the last ``need`` (3, H, W) frames into (1, 3*need, H, W).

    Oldest first, current last. Front-pad with the oldest available frame when
    fewer than ``need`` frames are buffered (matches the run_omnivla_edge.py
    cold-start behaviour of duplicating the current frame). Pure numpy — unit
    testable without torch.
    """
    if not frames:
        raise RuntimeError('no observation frames buffered yet')
    padded = list(frames)
    while len(padded) < need:
        padded.insert(0, padded[0])
    stacked = np.concatenate(padded[-need:], axis=0)  # (3*need, H, W)
    return stacked[None, ...]


class OmniVLAEdgeEngine:
    """The OmniVLA-edge policy forward pass, ROS-free and framework-agnostic.

    Owns the model, CLIP, and the observation ring buffer. Call
    :meth:`infer_chunk` per frame; it returns the raw action chunk in
    waypoint-spacing units. Thread-safety: :meth:`infer_chunk` mutates the ring
    buffer, so serialise calls per engine (both callers tick from a single
    thread / gRPC stream).
    """

    def __init__(
        self,
        *,
        weights_path: str = '/workspace/models/omnivla-edge/omnivla-edge.pth',
        clip_type: str = 'ViT-B/32',
        device: str = 'cuda:0',
    ) -> None:
        import torch
        import clip

        if str(device).startswith('cpu'):
            raise ValueError(
                "OmniVLAEdgeEngine requires CUDA: the vendored OmniVLA_edge "
                "forward pass uses tensor.get_device() which is GPU-only. Pass a "
                "cuda device (e.g. 'cuda:0')."
            )
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available but device=%r requested' % device)

        from .models.omnivla_edge_model import OmniVLA_edge

        self._torch = torch
        self._clip = clip
        self._device = torch.device(device)
        self._context_size = int(_MODEL_PARAMS['context_size'])
        self._len_traj_pred = int(_MODEL_PARAMS['len_traj_pred'])

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f'omnivla-edge weights not found at {weights_path}')

        _LOG.info('loading OmniVLA_edge from %s', weights_path)
        model = OmniVLA_edge(**_MODEL_PARAMS)
        state_dict = torch.load(weights_path, map_location=str(self._device))
        # The released omnivla-edge.pth is a bare state_dict (utils_policy.load_model
        # loads it with strict=True).
        model.load_state_dict(state_dict, strict=True)
        self._model = model.to(self._device).eval()

        _LOG.info('loading CLIP %s', clip_type)
        text_encoder, _preprocess = clip.load(clip_type, device=self._device, jit=False)
        self._text_encoder = text_encoder.to(torch.float32).to(self._device).eval()

        # Ring buffer of the last (context_size+1) ImageNet-normalized 96x96
        # frames (each (3, 96, 96)). Oldest first, current last.
        self._obs_ring: List[np.ndarray] = []

        # Cache CLIP text features keyed by the prompt string — encode_text is
        # the only non-trivial per-goal cost and goals change rarely.
        self._text_cache_key: Optional[str] = None
        self._text_cache_feat = None  # torch.Tensor (1, 512)

        # Reusable zero-fill tensor.
        self._black96_chw = _black_chw(96)  # (3, 96, 96)

    # --------------------------------------------------------------- properties

    @property
    def len_traj_pred(self) -> int:
        return self._len_traj_pred

    @property
    def context_size(self) -> int:
        return self._context_size

    @property
    def metric_waypoint_spacing(self) -> float:
        return _METRIC_WAYPOINT_SPACING

    # ----------------------------------------------------------------- internals

    def reset(self) -> None:
        """Drop the observation history (e.g. after warmup or a goal switch)."""
        self._obs_ring = []

    def _push_frame(self, frame_chw: np.ndarray) -> None:
        self._obs_ring.append(frame_chw)
        if len(self._obs_ring) > self._context_size + 1:
            self._obs_ring.pop(0)

    def _stack_obs(self) -> np.ndarray:
        """-> (1, 3*(context_size+1), 96, 96). Front-pad with the oldest frame."""
        return _stack_frames(list(self._obs_ring), self._context_size + 1)

    def _text_features(self, text: str):
        torch = self._torch
        if text == self._text_cache_key and self._text_cache_feat is not None:
            return self._text_cache_feat
        tokens = self._clip.tokenize(text or 'xxxx', truncate=True).to(self._device)
        with torch.no_grad():
            feat = self._text_encoder.encode_text(tokens).to(torch.float32)
        self._text_cache_key = text
        self._text_cache_feat = feat
        return feat

    # ------------------------------------------------------------- infer_chunk

    def infer_chunk(
        self,
        *,
        cur_image_rgb: np.ndarray,
        goal_mode: str,
        goal_text: str = '',
        goal_pose_xy_theta=None,
        goal_image_rgb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run one observation through OmniVLA-edge.

        Returns the raw action chunk ``(len_traj_pred, 4)`` = ``(x, y, cos, sin)``
        in waypoint-spacing units (float32). Mirrors run_omnivla_edge.py's
        run_forward_pass. Callers scale x/y by :attr:`metric_waypoint_spacing`.
        """
        if cur_image_rgb is None:
            raise ValueError('OmniVLAEdgeEngine.infer_chunk requires cur_image_rgb')

        torch = self._torch

        # 1. Observation history.
        self._push_frame(_normalize_chw(cur_image_rgb, 96))
        obs_np = self._stack_obs()                       # (1, 18, 96, 96)
        obs_images = torch.from_numpy(obs_np).to(torch.float32).to(self._device)
        obs_image_cur = obs_images[:, -3:, :, :]         # last frame, (1, 3, 96, 96)
        cur_large = torch.from_numpy(
            _normalize_chw(cur_image_rgb, 224)[None, ...]
        ).to(torch.float32).to(self._device)             # (1, 3, 224, 224)

        # 2. Goal tensors.
        modality_id = _modality_id_for(goal_mode)

        if goal_pose_xy_theta is not None:
            goal_pose_np = _pose_goal_vector(goal_pose_xy_theta)
        else:
            goal_pose_np = np.zeros(4, dtype=np.float32)
        goal_pose = torch.from_numpy(goal_pose_np[None, ...]).to(torch.float32).to(self._device)

        if goal_mode == 'image' and goal_image_rgb is not None:
            goal_img_np = _normalize_chw(goal_image_rgb, 96)[None, ...]
        else:
            goal_img_np = self._black96_chw[None, ...]
        goal_image = torch.from_numpy(goal_img_np).to(torch.float32).to(self._device)

        # Zero-fill satellite map: cat(current_map, goal_map, obs_image_cur).
        black96 = torch.from_numpy(self._black96_chw[None, ...]).to(torch.float32).to(self._device)
        map_images = torch.cat((black96, black96, obs_image_cur), dim=1)  # (1, 9, 96, 96)

        feat_text = self._text_features(goal_text if goal_mode == 'text' else '')

        modality_id_t = torch.tensor([modality_id], device=self._device)

        # 3. Forward pass (single-frame batch).
        with torch.no_grad():
            action_pred, _dist_pred, _mask = self._model(
                obs_images,
                goal_pose,
                map_images,
                goal_image,
                modality_id_t,
                feat_text,
                cur_large,
            )
        return action_pred[0].float().cpu().numpy()  # (len_traj_pred, 4)
