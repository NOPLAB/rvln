"""In-process V4L2 camera capture for the edge node."""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import cv2

from .observation_state import CameraFrameStore


class V4L2CameraCapture:
    """Own the device, capture thread, and conversion to RGB frames."""

    def __init__(self, frames: CameraFrameStore, warn: Callable[[str], None]) -> None:
        self._frames = frames
        self._warn = warn
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def open_device(self, device: str, *, width: int, height: int, fps: float) -> bool:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0.0:
            cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        return True

    def start(self) -> None:
        if self._cap is None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.capture_loop, name='camera-capture', daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self) -> None:
        self.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frames.clear()

    def capture_loop(self) -> None:
        """Read at the driver's pace; freshness checks handle a failed camera."""
        while not self._stop.is_set():
            if self._cap is None:
                return
            ok, frame_bgr = self._cap.read()
            if self._stop.is_set():
                return
            if not ok:
                self._warn('camera read failed; retrying')
                time.sleep(0.1)
                continue
            self._frames.put(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
