"""OmniVLAEdgeAdapter: action chunk -> nav_msgs/Path. Pure math, no model.

Plan 2B Path 1: the cloud already ran the full OmniVLA-original pipeline and
serialized the predicted absolute waypoints with shape
``(NUM_ACTIONS_CHUNK, ACTION_DIM)`` = typically ``(8, 4)`` where the last
dim packs ``(x, y, cos(theta), sin(theta))``, x/y already scaled to metres
(both cloud backends apply metric_waypoint_spacing before serializing).
The edge just builds the Path.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from nav_msgs.msg import Path

from .base import EdgeAdapter
from ._path_util import trajectory_to_path


class OmniVLAEdgeAdapter(EdgeAdapter):
    """Path-only adapter: read (x, y, cos, sin) tuples from the embedding."""

    def predict_path(
        self,
        *,
        embedding: np.ndarray,
        embedding_shape: Tuple[int, int, int],
        cur_image_rgb: Optional[np.ndarray] = None,    # noqa: ARG002 (unused)
        past_image_rgb: Optional[np.ndarray] = None,   # noqa: ARG002 (unused)
        frame_id: str = 'base_link',
    ) -> Path:
        wp = np.asarray(embedding, dtype=np.float32).reshape(embedding_shape[1:])
        # Waypoints arrive already in metres (the cloud scales them), so spacing=1.
        return trajectory_to_path(wp, frame_id=frame_id)
