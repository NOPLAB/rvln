"""EdgeAdapter ABC, implemented by stub / asyncvla / omnivla / omnivla_edge_local adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from nav_msgs.msg import Path


@dataclass
class EdgeGoal:
    """Goal handed to adapters that run the policy on the edge.

    Cloud-heavy adapters (stub / asyncvla / omnivla Path 1) get the goal baked
    into the embedding they receive from the remote backend and ignore this.
    On-edge adapters (``omnivla_edge_local``, Plan 2B Path 2) need the raw goal
    because there is no cloud doing the language/pose conditioning for them.

    ``mode`` is one of ``'pose' | 'text' | 'image'``. Only the field matching
    ``mode`` is meaningful; the others are left at their defaults.
    """

    mode: str
    pose_xy_theta: Optional[Tuple[float, float, float]] = None
    text: str = ''
    image_rgb: Optional[np.ndarray] = None  # RGB uint8 HxWx3


class EdgeAdapter(ABC):
    """Convert a `(B=1, num_tokens, embed_dim)` cloud embedding into a Path.

    Concrete adapters are responsible for producing a `nav_msgs/Path` in the
    robot frame. They may consult the latest RGB images (e.g. AsyncVLA's
    Edge_adapter) or use the embedding alone (Plan 2B Path 1's OmniVLA).
    """

    @abstractmethod
    def predict_path(
        self,
        *,
        embedding: np.ndarray,
        embedding_shape: Tuple[int, int, int],
        cur_image_rgb: Optional[np.ndarray] = None,
        past_image_rgb: Optional[np.ndarray] = None,
        frame_id: str = 'base_link',
    ) -> Path:
        ...

    def set_goal(self, goal: EdgeGoal) -> None:  # noqa: ARG002
        """Receive the latest navigation goal.

        Default is a no-op: cloud-heavy adapters get the goal via the embedding.
        On-edge adapters override this to drive their local policy. Called from
        the edge node's goal subscription, i.e. a different thread than
        ``predict_path`` — implementations must be thread-safe.
        """
        return None

    @property
    def is_local(self) -> bool:
        """True if this adapter runs the full policy on the edge (no cloud).

        When True, the edge node bypasses the embedding cache and gRPC client
        entirely and drives ``predict_path`` directly from the latest camera
        frame + goal (``embedding`` is passed as ``None``). Cloud-heavy adapters
        leave this False and consume the cloud embedding via the cache.
        """
        return False
