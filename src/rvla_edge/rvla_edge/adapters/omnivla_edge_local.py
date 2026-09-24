"""OmniVLAEdgeLocalAdapter — Plan 2B Path 2: run OmniVLA-edge ON the robot.

Unlike the cloud-heavy Path 1 (``adapters/omnivla.py``), this adapter runs the
full ``OmniVLA_edge`` policy + CLIP locally on the edge and turns the result into
a ``nav_msgs/Path`` directly — no cloud, no gRPC, no embedding cache. The edge
node runs in *standalone* mode (``is_local`` is True): the action loop drives
:meth:`predict_path` straight from the latest camera frame + goal.

The heavy lifting (model, CLIP, observation history, goal tensors) lives in the
ROS-free :class:`~rvla_core.omnivla_edge_engine.OmniVLAEdgeEngine`, which
Path 3's Jetson backend (``rvla_remote.backends.omnivla_edge``) reuses.
This adapter is the thin ROS layer on top: hold the goal, call the engine, scale
the raw chunk to metres and build the Path.

Goal modalities supported in v1: ``text`` (language nav, modality 7), ``pose``
(modality 4) and ``image`` (modality 6). Single goal at a time (the proto carries
one ``GoalSpec``).

Limitations (v1):
- Requires CUDA (see the engine docstring — the OmniVLA_edge forward pass is
  GPU-only). ``device='cpu'`` is rejected.
- Without tf/odometry on the edge, a ``pose`` goal is interpreted as
  robot-relative metres (x forward, y left). Use ``text`` goals for the cleanest
  behaviour until edge localization is wired up.
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np
from nav_msgs.msg import Path

from .base import EdgeAdapter, EdgeGoal
from ._path_util import trajectory_to_path
# Re-export the ROS-free pipeline pieces from the shared engine so existing
# imports (and unit tests) that reach them here keep working.
from rvla_core.omnivla_edge_engine import (  # noqa: F401
    _METRIC_WAYPOINT_SPACING,
    _MODEL_PARAMS,
    OmniVLAEdgeEngine,
    _black_chw,
    _modality_id_for,
    _normalize_chw,
    _pose_goal_vector,
    _stack_frames,
)


def _trajectory_to_path(
    waypoints: np.ndarray,
    *,
    spacing: float = _METRIC_WAYPOINT_SPACING,
    frame_id: str = 'base_link',
) -> Path:
    """Back-compat shim for existing imports; canonical impl in ``_path_util``."""
    return trajectory_to_path(waypoints, spacing=spacing, frame_id=frame_id)


class OmniVLAEdgeLocalAdapter(EdgeAdapter):
    """Runs the full OmniVLA-edge policy on the robot (Plan 2B Path 2)."""

    def __init__(
        self,
        *,
        weights_path: str = '/workspace/models/omnivla-edge/omnivla-edge.pth',
        clip_type: str = 'ViT-B/32',
        device: str = 'cuda:0',
    ) -> None:
        self._engine = OmniVLAEdgeEngine(
            weights_path=weights_path, clip_type=clip_type, device=device,
        )
        self._goal: Optional[EdgeGoal] = None
        self._lock = threading.Lock()

    @property
    def is_local(self) -> bool:
        # The whole policy runs here — the edge node needs no cloud.
        return True

    # -------------------------------------------------------------- goal intake

    def set_goal(self, goal: EdgeGoal) -> None:
        with self._lock:
            self._goal = goal

    # ------------------------------------------------------------- predict_path

    def predict_path(
        self,
        *,
        embedding: Optional[np.ndarray] = None,      # noqa: ARG002 (ignored: model runs on edge)
        embedding_shape: Optional[Tuple[int, int, int]] = None,  # noqa: ARG002
        cur_image_rgb: Optional[np.ndarray] = None,
        past_image_rgb: Optional[np.ndarray] = None,  # noqa: ARG002 (history kept internally)
        frame_id: str = 'base_link',
    ) -> Path:
        if cur_image_rgb is None:
            raise ValueError('OmniVLAEdgeLocalAdapter requires cur_image_rgb')

        with self._lock:
            goal = self._goal
        if goal is None:
            # No goal yet -> empty Path (the follower safe-stops). Don't pollute
            # the ring buffer until we actually have a goal to chase.
            path = Path()
            path.header.frame_id = frame_id
            return path

        waypoints = self._engine.infer_chunk(
            cur_image_rgb=cur_image_rgb,
            goal_mode=goal.mode,
            goal_text=goal.text,
            goal_pose_xy_theta=goal.pose_xy_theta,
            goal_image_rgb=goal.image_rgb,
        )
        return trajectory_to_path(
            waypoints, spacing=self._engine.metric_waypoint_spacing, frame_id=frame_id,
        )
