"""Validate metric scene registration and convert Habitat poses without Isaac."""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_scene_record(path: Path, scene: str) -> dict:
    """Require an explicit, metric alignment for each converted MP3D scan."""
    import numpy as np

    record = json.loads(path.read_text(encoding="utf-8"))["scenes"][scene]
    usd = Path(record["usd"]).expanduser()
    if not usd.is_absolute():
        raise ValueError("scene USD path must be absolute")
    usd = usd.resolve()
    if not usd.is_file() or usd.suffix.lower() not in (".usd", ".usda", ".usdc"):
        raise ValueError(f"missing USD scene: {usd}")
    matrix = record["isaac_from_habitat"]
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("isaac_from_habitat must be a 4x4 matrix")
    transform = np.asarray(matrix, dtype=float)
    if not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("invalid scene transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("scene transform must preserve metric distance")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError("scene transform must be a proper rotation")
    dome_intensity = float(record.get("dome_light_intensity", 0.0))
    if not math.isfinite(dome_intensity) or dome_intensity < 0:
        raise ValueError("dome_light_intensity must be finite and nonnegative")
    return {
        "usd": usd,
        "matrix": transform,
        "inverse": np.linalg.inv(transform),
        "dome_light_intensity": dome_intensity,
    }


def transform_point(matrix, position):
    import numpy as np

    return (matrix @ np.asarray([*position, 1.0], dtype=float))[:3]


def initial_yaw(episode: dict, rotation) -> float:
    """Convert Habitat's camera-forward (-Z) direction into Isaac Z-up yaw."""
    import numpy as np

    x, y, z, w = (float(v) for v in episode["start_rotation"])
    forward = np.array(
        [
            -2 * (x * z + w * y),
            -2 * (y * z - w * x),
            -(1 - 2 * (x * x + y * y)),
        ]
    )
    direction = rotation @ forward
    if math.hypot(*direction[:2]) < 1e-5:
        raise ValueError("initial forward vector is vertical after conversion")
    return math.atan2(direction[1], direction[0])
