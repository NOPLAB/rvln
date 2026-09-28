"""Estimate a metric Habitat-to-Isaac scan transform from surveyed landmarks."""
from __future__ import annotations

import json
import math
from pathlib import Path

from bench.artifacts import sha256, write_json


def fit_rigid_transform(pairs: list[dict], max_error_m: float) -> dict:
    """Fit a proper, unit-scale 3D transform and reject poor correspondences."""
    import numpy as np

    if len(pairs) < 3 or not math.isfinite(max_error_m) or max_error_m <= 0:
        raise ValueError('need at least three landmarks and a positive error limit')
    try:
        habitat = np.asarray([pair['habitat'] for pair in pairs], dtype=float)
        isaac = np.asarray([pair['isaac'] for pair in pairs], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('each landmark needs habitat and isaac XYZ coordinates') from error
    if (habitat.shape != (len(pairs), 3) or isaac.shape != habitat.shape
            or not np.isfinite(habitat).all() or not np.isfinite(isaac).all()):
        raise ValueError('landmark coordinates must be finite XYZ vectors')
    source_mean = habitat.mean(axis=0)
    target_mean = isaac.mean(axis=0)
    centered_source = habitat - source_mean
    centered_target = isaac - target_mean
    if np.linalg.matrix_rank(centered_source, tol=1e-6) < 2:
        raise ValueError('Habitat landmarks must not be collinear')
    if np.linalg.matrix_rank(centered_target, tol=1e-6) < 2:
        raise ValueError('Isaac landmarks must not be collinear')
    u, _, vh = np.linalg.svd(centered_source.T @ centered_target)
    correction = np.diag([1.0, 1.0,
                          1.0 if np.linalg.det(u @ vh) >= 0 else -1.0])
    rotation = u @ correction @ vh
    translation = target_mean - source_mean @ rotation
    predicted = habitat @ rotation + translation
    residual = np.linalg.norm(predicted - isaac, axis=1)
    maximum = float(residual.max())
    if maximum > max_error_m:
        raise ValueError(f'landmark residual {maximum:.4f} m exceeds '
                         f'{max_error_m:.4f} m; check coordinates and handedness')
    transform = np.eye(4)
    transform[:3, :3] = rotation.T
    transform[:3, 3] = translation
    return {
        'isaac_from_habitat': transform.tolist(),
        'landmarks': len(pairs),
        'rms_error_m': float(np.sqrt(np.mean(residual ** 2))),
        'max_error_m': maximum,
    }


def calibrate_scene(pairs_path: Path, scene: str, usd: Path, output: Path,
                    max_error_m: float, dome_intensity: float = 0.0) -> dict:
    """Write a one-scene registry only after its source USD and fit validate."""
    usd = usd.expanduser().resolve()
    if not usd.is_file() or usd.suffix.lower() not in ('.usd', '.usda', '.usdc'):
        raise ValueError(f'missing scene USD: {usd}')
    if not scene or any(char in scene for char in '/\\') or scene in ('.', '..'):
        raise ValueError('scene must be a single MP3D scan ID')
    if not math.isfinite(dome_intensity) or dome_intensity < 0:
        raise ValueError('dome intensity must be finite and nonnegative')
    registry = {'schema': 1, 'scenes': {}}
    if output.exists():
        registry = json.loads(output.read_text(encoding='utf-8'))
        if registry.get('schema') != 1 or not isinstance(registry.get('scenes'), dict):
            raise ValueError(f'invalid existing scene registry: {output}')
        if scene in registry['scenes']:
            raise ValueError(f'scene is already registered: {scene}')
    pairs = json.loads(pairs_path.read_text(encoding='utf-8'))['landmarks']
    if not isinstance(pairs, list):
        raise ValueError('landmarks must be an array of coordinate pairs')
    fit = fit_rigid_transform(pairs, max_error_m)
    record = {'usd': str(usd), **fit, 'landmarks_sha256': sha256(pairs_path),
              'dome_light_intensity': dome_intensity}
    registry['scenes'][scene] = record
    write_json(output, registry)
    return registry
