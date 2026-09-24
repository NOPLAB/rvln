"""OmniVLA-edge remote backend (Plan 2B Path 3 — Jetson infers, Pi controls).

Path 2 runs the OmniVLA-edge policy entirely on the robot. Path 3 splits that
same policy across two boxes: a GPU box (typically a Jetson) runs the heavy
OmniVLA-edge forward pass here and publishes predicted waypoints through ROS 2; the
Raspberry Pi runs only the light path-only :class:`OmniVLAEdgeAdapter`
(``adapter_kind=omnivla``) and turns those waypoints into a ``nav_msgs/Path``.

The forward pass, CLIP and the observation ring buffer live in the shared
:class:`~raspicat_vla_core.omnivla_edge_engine.OmniVLAEdgeEngine` (imported from
the ROS-free core package). This backend is the thin inference layer:
decode the goal, run the engine, scale the raw chunk to **metres** (so the Pi's
path-only adapter, which does not rescale, plots the right geometry), and hand it
back as the ``(NUM_ACTIONS_CHUNK, ACTION_DIM)`` ActionEmbedding payload.

Contrast with :class:`OmniVLABackend` (Path 1), which runs OmniVLA-*original*.
This backend serves the same *edge* policy as Path 2 — the split point differs,
not the model.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np
import PIL.Image

from .base import ModelInfoDict, VLABackend


_LOG = logging.getLogger(__name__)


def _goal_mode_for(
    *, has_lang: bool, has_image_goal: bool, has_pose: bool,
) -> str:
    """Collapse the per-frame goal fields into a single OmniVLA-edge mode.

    The server's ``_proto_goal_to_python`` already yields exactly one populated
    goal field per Observation (the proto carries a single ``GoalSpec``), so the
    priority here only matters for defensive completeness. Language nav is the
    primary/cleanest modality, then image, then pose; default to text (with an
    empty prompt the engine substitutes) when nothing is set.
    """
    if has_lang:
        return 'text'
    if has_image_goal:
        return 'image'
    if has_pose:
        return 'pose'
    return 'text'


class OmniVLAEdgeBackend(VLABackend):
    """Runs the OmniVLA-edge policy on a remote GPU and serves the action chunk."""

    def __init__(
        self,
        *,
        weights_path: str = '/workspace/models/omnivla-edge/omnivla-edge.pth',
        clip_type: str = 'ViT-B/32',
        device: str = 'cuda:0',
    ) -> None:
        # Imported here (not at module top) so the module is importable for
        # --help / arg parsing without torch/clip.
        from raspicat_vla_core.omnivla_edge_engine import OmniVLAEdgeEngine

        self._engine = OmniVLAEdgeEngine(
            weights_path=weights_path, clip_type=clip_type, device=device,
        )
        self._device = str(device)
        self._weights_path = weights_path
        self._spacing = float(self._engine.metric_waypoint_spacing)

    def warmup(self, num_iters: int = 1) -> None:
        # Page weights / build CUDA graphs on a black frame + text goal, then
        # drop the black frames so the real stream starts with a clean history.
        black = np.zeros((224, 224, 3), dtype=np.uint8)
        for _ in range(max(1, num_iters)):
            self._engine.infer_chunk(cur_image_rgb=black, goal_mode='text', goal_text='warmup')
        self._engine.reset()

    def infer(
        self,
        *,
        current_image: PIL.Image.Image,
        past_image: Optional[PIL.Image.Image] = None,  # noqa: ARG002 (history kept in the engine)
        lang_instruction: str,
        goal_image: Optional[PIL.Image.Image],
        goal_pose_xy_theta: Optional[Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, dict]:
        t0 = time.monotonic()

        cur_rgb = np.asarray(current_image.convert('RGB'), dtype=np.uint8)
        goal_rgb = (
            None if goal_image is None
            else np.asarray(goal_image.convert('RGB'), dtype=np.uint8)
        )
        mode = _goal_mode_for(
            has_lang=bool(lang_instruction),
            has_image_goal=goal_image is not None,
            has_pose=goal_pose_xy_theta is not None,
        )

        chunk = self._engine.infer_chunk(
            cur_image_rgb=cur_rgb,
            goal_mode=mode,
            goal_text=lang_instruction,
            goal_pose_xy_theta=goal_pose_xy_theta,
            goal_image_rgb=goal_rgb,
        )  # (len_traj_pred, 4) in waypoint-spacing units

        # Scale x/y to metres so the Pi's path-only adapter can plot them
        # directly (it does not rescale); leave cos/sin untouched.
        out = np.asarray(chunk, dtype=np.float32).copy()
        out[:, 0] *= self._spacing
        out[:, 1] *= self._spacing

        return out, {
            'inference_ms': (time.monotonic() - t0) * 1000.0,
            'modality_id': _modality_id(mode),
        }

    def model_info(self) -> ModelInfoDict:
        return ModelInfoDict(
            model_name='NHirose/omnivla-edge',
            model_version='omnivla-edge-v1',
            num_tokens=int(self._engine.len_traj_pred),
            embed_dim=4,
            device=self._device,
            ready=True,
        )


def _modality_id(mode: str) -> int:
    from raspicat_vla_core.omnivla_edge_engine import _modality_id_for
    return _modality_id_for(mode)
