"""Isaac R2R-CE episode loading and scoring; no simulator dependency."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path


def load_episodes(path: Path) -> list[dict]:
    """Read the official R2R_VLNCE_v1-3 split without changing its annotations."""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        episodes = json.load(stream)['episodes']
    ids = set()
    for episode in episodes:
        key = str(episode['episode_id'])
        if key in ids:
            raise ValueError(f'duplicate episode_id: {key}')
        ids.add(key)
        if not episode['instruction']['instruction_text'].strip():
            raise ValueError(f'empty instruction: {key}')
        for field in ('start_position',):
            if len(episode[field]) != 3 or not all(math.isfinite(v) for v in episode[field]):
                raise ValueError(f'invalid {field}: {key}')
        if not episode.get('goals') or len(episode['goals'][0]['position']) != 3:
            raise ValueError(f'invalid goal: {key}')
        if episode['info']['geodesic_distance'] <= 0:
            raise ValueError(f'invalid geodesic distance: {key}')
    return episodes


def scene_name(episode: dict) -> str:
    """Extract the MP3D scan ID from a VLN-CE scene path."""
    scene = Path(episode['scene_id'].replace('\\', '/')).stem
    if not scene or '/' in scene or '\\' in scene or scene in ('.', '..'):
        raise ValueError('invalid scene ID')
    return scene


def distance(a: list[float], b: list[float]) -> float:
    return math.dist(a, b)


def score(episode: dict, positions_habitat: list[list[float]], stopped: bool) -> dict:
    """Score an Isaac trajectory under the R2R-CE 3 m stop convention.

    The source geodesic is retained for provenance. Isaac trajectories use their
    own distance and are labeled as a transfer evaluation, not official VLN-CE.
    """
    if not positions_habitat:
        raise ValueError('empty trajectory')
    goal = episode['goals'][0]['position']
    error = distance(positions_habitat[-1], goal)
    traveled = sum(distance(a, b) for a, b in zip(positions_habitat,
                                                    positions_habitat[1:]))
    shortest = float(episode['info']['geodesic_distance'])
    success = stopped and error <= 3.0
    return {
        'episode_id': str(episode['episode_id']),
        'scene_id': scene_name(episode),
        'navigation_error_m': error,
        'trajectory_length_m': traveled,
        'source_geodesic_m': shortest,
        'stopped': stopped,
        'success': success,
        'spl': shortest / max(shortest, traveled) if success else 0.0,
        'metric_namespace': 'isaac_r2r_transfer',
    }
