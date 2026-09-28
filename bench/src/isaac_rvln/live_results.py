"""Finalize a ROS-driven Isaac episode with its PhysX contact evidence."""
from __future__ import annotations

import json
from pathlib import Path


def attach_contacts(result_path: Path, contact_path: Path) -> dict:
    """Merge the finished Isaac PhysX log after the simulator exits."""
    row = json.loads(result_path.read_text(encoding='utf-8'))
    if row.get('pose_source') != 'isaac_ground_truth':
        raise ValueError('episode is not an Isaac ground-truth trace')
    if row.get('contact_log') is not None:
        raise ValueError('episode already has contact data')
    contacts = json.loads(contact_path.read_text(encoding='utf-8'))
    if contacts.get('simulator') != 'isaac_sim' or contacts.get('schema') != 1:
        raise ValueError('contact log is not an Isaac Sim report')
    if contacts.get('receive_errors') or contacts.get('samples', {}).get('physics_steps', 0) < 1:
        raise ValueError('Isaac contact log has no physics evidence or contains errors')
    events = contacts.get('events')
    if not isinstance(events, list) or contacts.get('collisions') != len(events):
        raise ValueError('Isaac contact count differs from events')
    start = row.get('wall_started_at')
    end = row.get('wall_ended_at')
    if start is None or end is None or end <= start:
        raise ValueError('episode has no valid contact measurement window')
    events = [event for event in events
              if start <= event['wall_monotonic_sec'] <= end]
    for event in events:
        event['t_sec'] = event['wall_monotonic_sec'] - start
    row['contact_events'] = events
    row['collisions'] = len(events)
    row['contact_samples'] = contacts['samples']
    row['contact_log'] = contact_path.name
    result_path.write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
    return row
