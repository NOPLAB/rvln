"""Tests for mobile fp16 conversion helpers."""
import numpy as np
from rvla_proto.conversions import (
    fp16_bytes_to_float32_list,
    float32_array_to_fp16_bytes,
)


def test_fp16_bytes_round_trip():
    arr = np.arange(8 * 1024, dtype=np.float32) / 100.0
    raw = float32_array_to_fp16_bytes(arr)
    assert isinstance(raw, bytes)
    assert len(raw) == 8 * 1024 * 2  # fp16 = 2 bytes
    back = np.array(fp16_bytes_to_float32_list(raw), dtype=np.float32)
    assert back.shape == arr.shape
    # fp16 has ~10 bits of mantissa → relative precision ~5e-4. Tolerance must
    # scale with magnitude (rtol), not just be absolute. atol covers near-zero.
    np.testing.assert_allclose(back, arr, rtol=2e-3, atol=1e-3)
