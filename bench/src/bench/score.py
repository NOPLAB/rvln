"""Score closed-loop VLN episodes from structured, auditable trace JSONL."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


_INTENTIONAL_STOPS = {'model_stop', 'goal_tolerance'}


def score_episode(episode: dict, tolerance_m: float = 0.30) -> dict:
    """Compute endpoint and intentional-stop success without conflating watchdogs."""
    trace = episode['trace']
    if not trace:
        raise ValueError(f"episode {episode.get('id')} has no pose trace")
    goal_x, goal_y = episode['goal_xy']
    last = trace[-1]
    final_distance = math.hypot(last['x'] - goal_x, last['y'] - goal_y)
    length = sum(
        math.hypot(b['x'] - a['x'], b['y'] - a['y'])
        for a, b in zip(trace, trace[1:])
    )
    endpoint_success = final_distance <= tolerance_m
    stop_reason = episode.get('stop_reason')
    success = endpoint_success and stop_reason in _INTENTIONAL_STOPS
    shortest = episode.get('shortest_m')
    spl = None
    if shortest is not None:
        if shortest <= 0 or not math.isfinite(shortest):
            raise ValueError(f"invalid shortest_m for {episode.get('id')}")
        spl = shortest / max(shortest, length) if success else 0.0
    collisions = episode.get('collisions')
    if collisions is not None and (not isinstance(collisions, int) or collisions < 0):
        raise ValueError('collisions must be nonnegative')
    return {
        'id': episode['id'],
        'model': episode['model'],
        'deployment': episode['deployment'],
        'final_distance_m': final_distance,
        'travel_m': length,
        'elapsed_sec': last['t_sec'] - trace[0]['t_sec'],
        'endpoint_success': endpoint_success,
        'success': success,
        'collision_free_success': success and collisions == 0 if collisions is not None else None,
        'had_collision': collisions > 0 if collisions is not None else None,
        'collisions': collisions,
        'stop_reason': stop_reason,
        'spl': spl,
    }


def _mean_ci(values: list[float], seed: int = 0, n_boot: int = 2000) -> dict:
    if not values:
        return {'mean': None, 'ci95': None}
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choices(values, k=len(values)))
        for _ in range(n_boot)
    )
    return {
        'mean': statistics.mean(values),
        'ci95': [means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]],
    }


def summarize(scored: list[dict]) -> dict:
    """Keep incompatible deployments in separate report groups."""
    groups = {}
    for row in scored:
        groups.setdefault((row['model'], row['deployment']), []).append(row)
    summary = []
    for (model, deployment), rows in sorted(groups.items()):
        result = {
            'model': model, 'deployment': deployment, 'episodes': len(rows),
            'success_rate': _mean_ci([float(r['success']) for r in rows]),
            'endpoint_rate': _mean_ci([float(r['endpoint_success']) for r in rows]),
            'collision_free_rate': (
                _mean_ci([float(r['collision_free_success']) for r in rows])
                if all(r['collision_free_success'] is not None for r in rows) else None),
            'collision_episode_rate': (
                _mean_ci([float(r['had_collision']) for r in rows])
                if all(r['had_collision'] is not None for r in rows) else None),
            'collisions': (sum(r['collisions'] for r in rows)
                           if all(r['collisions'] is not None for r in rows) else None),
        }
        spl = [r['spl'] for r in rows]
        result['spl'] = _mean_ci(spl) if all(v is not None for v in spl) else None
        summary.append(result)
    return {'schema': 1, 'groups': summary, 'episodes': scored}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes_jsonl', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.episodes_jsonl.open(encoding='utf-8') as source:
        scored = [score_episode(json.loads(line)) for line in source if line.strip()]
    if not scored:
        parser.error('input has no episodes')
    result = summarize(scored)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for group in result['groups']:
        print(f"{group['model']}/{group['deployment']}: "
              f"SR={group['success_rate']['mean']:.3f} n={group['episodes']}")


if __name__ == '__main__':
    main()
