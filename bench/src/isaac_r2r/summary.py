"""Aggregate completed Isaac R2R transfer episodes from JSON traces."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError('no episode results')
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    ids = [str(row['episode_id']) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate episode result')
    if {row['metric_namespace'] for row in rows} != {'isaac_r2r_transfer'}:
        raise ValueError('mixed benchmark metric namespaces')
    hashes = {row['split_sha256'] for row in rows}
    if len(hashes) != 1:
        raise ValueError('mixed R2R-CE split versions')
    return {
        'schema': 1, 'metric_namespace': 'isaac_r2r_transfer',
        'split_sha256': hashes.pop(), 'episodes': len(rows),
        'scenes': len({row['scene_id'] for row in rows}),
        'success_rate': statistics.mean(bool(row['success']) for row in rows),
        'spl': statistics.mean(float(row['spl']) for row in rows),
        'navigation_error_m': statistics.mean(float(row['navigation_error_m'])
                                              for row in rows),
        'blocked_steps': sum(int(row['blocked_steps']) for row in rows),
    }
