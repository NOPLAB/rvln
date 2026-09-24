"""Image preprocessing for VLA edge node."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def resize_and_jpeg(
    image_rgb: np.ndarray,
    target: Tuple[int, int] = (224, 224),
    quality: int = 85,
) -> Tuple[bytes, int, int]:
    """Resize an RGB uint8 image and JPEG-encode it.

    Returns: (jpeg_bytes, width, height)
    """
    if image_rgb.dtype != np.uint8:
        raise ValueError(f'expected uint8 RGB, got dtype={image_rgb.dtype}')
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f'expected HxWx3 RGB, got shape={image_rgb.shape}')

    w, h = target
    resized = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return buf.tobytes(), w, h


def decode_jpeg_to_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes back to an RGB uint8 ndarray."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('failed to decode JPEG')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
