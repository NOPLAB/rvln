"""Thread-safe camera frame and goal/observation correlation state."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional, Tuple

import numpy as np


class CameraFrameStore:
    """Keep one RGB frame with its monotonic capture time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._image: Optional[np.ndarray] = None
        self._stamp_ns = 0

    def put(self, image: np.ndarray, *, stamp_ns: Optional[int] = None) -> None:
        with self._lock:
            self._image = image
            self._stamp_ns = time.monotonic_ns() if stamp_ns is None else stamp_ns

    def clear(self) -> None:
        with self._lock:
            self._image = None
            self._stamp_ns = 0

    def fresh(self, max_age_ns: int, *, now_ns: Optional[int] = None) -> Optional[np.ndarray]:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            if self._image is None or now_ns - self._stamp_ns > max_age_ns:
                return None
            return self._image.copy()

    def has_fresh(self, max_age_ns: int, *, now_ns: Optional[int] = None) -> bool:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            return self._image is not None and now_ns - self._stamp_ns <= max_age_ns

    def age_ms(self, now_ns: int) -> float:
        with self._lock:
            return (now_ns - self._stamp_ns) / 1e6


class ObservationLedger:
    """Atomically pair goal generations, sent frame IDs, and reply images."""

    def __init__(self, max_frames: int = 8) -> None:
        self._lock = threading.Lock()
        self._goal: Any = None
        self._generation = 0
        self._frame_counter = 0
        self._sent_frames: dict[int, np.ndarray] = {}
        self._max_frames = max_frames

    @property
    def frame_counter(self) -> int:
        with self._lock:
            return self._frame_counter

    @property
    def has_goal(self) -> bool:
        with self._lock:
            return self._goal is not None

    def change_goal(self, goal: Any, invalidate: Callable[[int], None]) -> None:
        """Change goal and invalidate all older replies before new sends commit."""
        with self._lock:
            self._generation += 1
            self._goal = goal
            self._sent_frames.clear()
            invalidate(self._frame_counter)

    def snapshot_goal(self) -> Tuple[Any, int]:
        with self._lock:
            return self._goal, self._generation

    def record_sent(self, generation: int, image: np.ndarray) -> Optional[int]:
        """Allocate a frame ID only if preprocessing used the current goal."""
        with self._lock:
            if generation != self._generation or self._goal is None:
                return None
            self._frame_counter += 1
            frame_id = self._frame_counter
            self._sent_frames[frame_id] = image
            while len(self._sent_frames) > self._max_frames:
                del self._sent_frames[next(iter(self._sent_frames))]
            return frame_id

    def take_reply_frame(self, frame_id: int) -> Optional[np.ndarray]:
        with self._lock:
            image = self._sent_frames.pop(frame_id, None)
            for old in [f for f in self._sent_frames if f < frame_id]:
                del self._sent_frames[old]
            return image

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._goal = None
            self._sent_frames.clear()
