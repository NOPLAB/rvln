"""Tests for edge image preprocessing."""
import numpy as np
import pytest

from rvla_edge.preprocess import resize_and_jpeg, decode_jpeg_to_rgb


def _make_rgb(h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_resize_and_jpeg_returns_bytes_and_target_size():
    img = _make_rgb(480, 640)
    raw, w, h = resize_and_jpeg(img, target=(224, 224), quality=85)
    assert isinstance(raw, (bytes, bytearray))
    assert (w, h) == (224, 224)
    # JPEG magic
    assert raw[:3] == b'\xff\xd8\xff'


def test_resize_and_jpeg_round_trip_within_jpeg_tolerance():
    img = _make_rgb(300, 400)
    raw, _, _ = resize_and_jpeg(img, target=(224, 224), quality=95)
    decoded = decode_jpeg_to_rgb(raw)
    assert decoded.shape == (224, 224, 3)
    assert decoded.dtype == np.uint8


def test_resize_and_jpeg_rejects_non_uint8():
    img = np.zeros((100, 100, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        resize_and_jpeg(img, target=(224, 224))


def test_resize_and_jpeg_rejects_wrong_channels():
    img = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        resize_and_jpeg(img, target=(224, 224))
