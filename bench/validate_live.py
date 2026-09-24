"""Audit every real inference and its ROS/Edge handoff in live pilot traces."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SHAPES = {'asyncvla': [8, 1024], 'omnivla': [8, 4],
          'omnivla_edge': [8, 4],
          'navila': [1, 4], 'navida': [1, 4]}
VERSIONS = {'asyncvla': 'asyncvla-step750000',
            'omnivla': 'omnivla-orig-step120000',
            'omnivla_edge': 'omnivla-edge-v1',
            'movla': 'stage_a_v9b',
            'navila': 'models/navila-llama3-8b-8f',
            'navida': 'models/NaVIDA'}


def validate(row: dict) -> dict:
    """Fail on wrong model, malformed inference, broken Path handoff, or missing actuation."""
    from rvln_remote.backends.navida import action_to_embedding as navida_action
    from rvln_remote.backends.navila import action_to_embedding as navila_action

    model = row['model']
    runs = row['inferences']
    if row['errors'] or not runs or row['embeddings'] != len(runs):
        raise ValueError(f'{model}/{row["id"]}: inference errors or count mismatch')
    if row['instruction_sent'] != row['text'] and model != 'movla':
        raise ValueError(f'{model}/{row["id"]}: wrong instruction')
    if not row['model_version'] or not row['contact_samples']:
        raise ValueError(f'{model}/{row["id"]}: missing model or sensor evidence')
    if VERSIONS[model] not in row['model_version']:
        raise ValueError(f'{model}/{row["id"]}: wrong checkpoint version')
    if any((x['shape'] != SHAPES[model] if model in SHAPES
            else len(x['shape']) != 2 or x['shape'][0] < 1 or x['shape'][1] != 4)
           for x in runs):
        raise ValueError(f'{model}/{row["id"]}: unexpected output shape')
    ids = [x['frame_id'] for x in runs]
    if ids != sorted(set(ids)):
        raise ValueError(f'{model}/{row["id"]}: duplicate or reordered frames')
    for inference in runs:
        if inference['round_trip_ms'] <= 0 or inference['server_wall_ms'] <= 0:
            raise ValueError(f'{model}/{row["id"]}: invalid inference timing')
        if model in ('asyncvla', 'omnivla', 'omnivla_edge'):
            if inference['diagnostics'].get('modality_id') != 7:
                raise ValueError(f'{model}/{row["id"]}: wrong language modality')
        if model == 'asyncvla':
            if not math.isfinite(inference['feature_l2']) or inference['feature_l2'] <= 0:
                raise ValueError(f'{model}/{row["id"]}: empty image feature')
        else:
            wp = inference['first_waypoint']
            if not all(math.isfinite(v) for v in wp):
                raise ValueError(f'{model}/{row["id"]}: nonfinite waypoint')
        if model in ('navila', 'navida'):
            raw = inference['diagnostics']['raw_response']
            action = navila_action(raw) if model == 'navila' else navida_action(raw)
            if any(abs(float(a) - b) > 1e-5
                   for a, b in zip(action[0], inference['first_waypoint'])):
                raise ValueError(f'{model}/{row["id"]}: action text conversion mismatch')
        if model == 'movla':
            diagnostic = inference['diagnostics']
            if diagnostic['motion_history_samples'] < 1:
                raise ValueError(f'{model}/{row["id"]}: missing measured pose history')
            if len(diagnostic['motion_velocity']) != 2:
                raise ValueError(f'{model}/{row["id"]}: missing measured velocity')
    if model == 'movla' and not any(
            i['diagnostics']['motion_history_m'] > 0.05 for i in runs):
        raise ValueError(f'{model}/{row["id"]}: measured history remained stationary')
    paths = row['path_samples']
    if row['nonempty_paths'] < 1 or not paths:
        raise ValueError(f'{model}/{row["id"]}: no Edge path')
    if any(p['frame_id'] != 'base_link' for p in paths):
        raise ValueError(f'{model}/{row["id"]}: wrong path frame')
    if model != 'asyncvla':
        if not all(any(math.dist(p['first_xy'], i['first_waypoint'][:2]) < 1e-4
                       for i in runs) for p in paths):
            raise ValueError(f'{model}/{row["id"]}: Edge path differs from model output')
    trace = row['trace']
    commanded = any(abs(p['cmd_v']) > 0.01 or abs(p['cmd_w']) > 0.01 for p in trace)
    moved = any(math.dist((p['x'], p['y']), (trace[0]['x'], trace[0]['y'])) > 0.01
                or abs(p['yaw'] - trace[0]['yaw']) > 0.01 for p in trace)
    if not commanded or not moved:
        raise ValueError(f'{model}/{row["id"]}: Edge command or Gazebo motion missing')
    return {'model': model, 'id': row['id'], 'inferences': len(runs),
            'nonempty_paths': row['nonempty_paths'], 'commanded': commanded,
            'moved': moved, 'model_version': row['model_version']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--skip-errors', action='store_true')
    args = parser.parse_args()
    audited = []
    for path in sorted(args.runs.glob('*.json')):
        if path.name.endswith('.contacts.json'):
            continue
        raw = json.loads(path.read_text(encoding='utf-8'))
        if not all(key in raw for key in ('model', 'id', 'trace', 'inferences')):
            continue
        if args.skip_errors and raw['errors']:
            continue
        audited.append(validate(raw))
    if len(audited) != 23 and not args.allow_partial:
        raise ValueError(f'expected 23 episode audits, got {len(audited)}')
    args.out.write_text(json.dumps({'schema': 1, 'episodes': audited}, indent=2) + '\n',
                        encoding='utf-8')
    print(f'validated {len(audited)} real-model episodes')


if __name__ == '__main__':
    main()
