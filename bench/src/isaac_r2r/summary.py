"""Aggregate completed Isaac R2R transfer episodes from JSON traces."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from bench.artifacts import sha256

from isaac_r2r.protocol import load_episodes, scene_name


def summarize(paths: list[Path], split: Path | None = None) -> dict:
    if not paths:
        raise ValueError('no episode results')
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    if any(row.get('status') != 'completed' for row in rows):
        raise ValueError('episode result is incomplete')
    ids = [str(row['episode_id']) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate episode result')
    if {row['metric_namespace'] for row in rows} != {'isaac_r2r_transfer'}:
        raise ValueError('mixed benchmark metric namespaces')
    if {row.get('navigation_error_kind', 'euclidean') for row in rows} != {'euclidean'}:
        raise ValueError('mixed navigation error distance definitions')
    hashes = {row['split_sha256'] for row in rows}
    if len(hashes) != 1:
        raise ValueError('mixed R2R-CE split versions')
    policy_hashes = {row.get('policy_split_sha256') for row in rows}
    if len(policy_hashes) != 1:
        raise ValueError('mixed policy instruction split versions')
    policy_ids = {row.get('policy_id') for row in rows}
    if len(policy_ids) != 1:
        raise ValueError('mixed policy model identities')
    if split is not None:
        expected = {str(item['episode_id']): scene_name(item)
                    for item in load_episodes(split)}
        actual = {str(row['episode_id']): row['scene_id'] for row in rows}
        if actual != expected:
            raise ValueError('results do not cover the exact split episodes and scenes')
        if hashes != {sha256(split)}:
            raise ValueError('result split hash does not match the supplied split')
    result = {
        'schema': 1, 'metric_namespace': 'isaac_r2r_transfer',
        'navigation_error_kind': 'euclidean',
        'split_sha256': hashes.pop(), 'episodes': len(rows),
        'scenes': len({row['scene_id'] for row in rows}),
        'success_rate': statistics.mean(bool(row['success']) for row in rows),
        'spl': statistics.mean(float(row['spl']) for row in rows),
        'navigation_error_m': statistics.mean(float(row['navigation_error_m'])
                                              for row in rows),
        'blocked_steps': sum(int(row['blocked_steps']) for row in rows),
    }
    if None not in policy_hashes:
        result['policy_split_sha256'] = policy_hashes.pop()
    if None not in policy_ids:
        result['policy_id'] = policy_ids.pop()
    return result
