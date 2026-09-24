"""Shared action-chunk to Path state and watchdog for mobile transports."""
from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np
from nav_msgs.msg import Path

from rvln_proto.conversions import fp16_bytes_to_float32_list

from .adapters._path_util import trajectory_to_path


def decode_path(
    raw: bytes,
    *,
    num_tokens: int,
    embed_dim: int,
    scaled_to_m: bool,
    waypoint_spacing: float,
    frame_id: str,
) -> Path:
    """Validate the shared fp16 payload and convert its trajectory to a Path."""
    if num_tokens < 1 or embed_dim < 4:
        raise ValueError(f'bad shape: num_tokens={num_tokens} embed_dim={embed_dim}')
    values = fp16_bytes_to_float32_list(raw)
    if len(values) != num_tokens * embed_dim:
        raise ValueError(
            f'values length {len(values)} != num_tokens*embed_dim {num_tokens * embed_dim}'
        )
    waypoints = np.asarray(values, dtype=np.float32).reshape(num_tokens, embed_dim)
    spacing = 1.0 if scaled_to_m else waypoint_spacing
    return trajectory_to_path(waypoints, spacing=spacing, frame_id=frame_id)


class ActionPathBridge:
    """Hold the latest Path and emit one empty Path when input expires."""

    def __init__(self, *, frame_id: str, max_age_sec: float) -> None:
        self._frame_id = frame_id
        self._max_age_sec = max_age_sec
        self._lock = threading.Lock()
        self._pending: Optional[Path] = None
        self._last_rx: Optional[float] = None
        self._goal_id = ''
        self._stopped = True

    @property
    def following(self) -> bool:
        with self._lock:
            return not self._stopped

    def receive(self, path: Path, goal_id: str, now: float) -> Optional[str]:
        """Queue a Path; return the previous goal ID when it changed."""
        with self._lock:
            previous = self._goal_id if goal_id != self._goal_id else None
            self._goal_id = goal_id
            self._pending = path
            self._last_rx = now
            self._stopped = False
            return previous

    def tick(self, now: float) -> Tuple[Optional[Path], bool]:
        """Return a queued Path or a one-shot stop Path, and whether it timed out."""
        with self._lock:
            if (self._last_rx is not None and not self._stopped
                    and now - self._last_rx > self._max_age_sec):
                self._stopped = True
                self._pending = None
                empty = Path()
                empty.header.frame_id = self._frame_id
                return empty, True
            if self._pending is not None:
                pending, self._pending = self._pending, None
                return pending, False
        return None, False
