"""fp16 byte conversion helpers for the mobile and web bridges."""
from __future__ import annotations

import numpy as np


def float32_array_to_fp16_bytes(arr: np.ndarray) -> bytes:
    """Convert a contiguous float32 array to little-endian fp16 bytes."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    fp16 = arr.astype('<f2', copy=False)
    return fp16.tobytes()


def fp16_bytes_to_float32_list(raw: bytes) -> list[float]:
    """Convert little-endian fp16 bytes to a Python list of float32 values."""
    fp16 = np.frombuffer(raw, dtype='<f2')
    return fp16.astype(np.float32).tolist()
