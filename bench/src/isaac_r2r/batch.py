"""Run and audit every episode in an Isaac R2R-CE transfer split."""
from __future__ import annotations

import json
import subprocess
import sys
from itertools import count
from pathlib import Path

from bench.artifacts import sha256, write_json

from isaac_r2r.cma_inputs import load_cma_tokens
from isaac_r2r.protocol import load_episodes, scene_name
from isaac_r2r.run import initial_yaw, load_scene_record
from isaac_r2r.summary import summarize


def open_attempt_log(result_dir: Path, index: int):
    """Reserve a fresh log name across retries and later batch resumes."""
    for number in count():
        suffix = '' if number == 0 else f'.attempt-{number}'
        try:
            return (result_dir / f'{index:05d}{suffix}.log').open(
                'x', encoding='utf-8')
        except FileExistsError:
            continue


def run_batch(split: Path, scenes: Path, policy_url: str, policy_id: str,
              out_dir: Path, max_steps: int, headless: bool, resume: bool,
              episode_timeout: int, startup_retries: int = 2,
              policy_split: Path | None = None) -> dict:
    """Execute each episode in a fresh Kit process and require full coverage."""
    if not policy_id.strip():
        raise ValueError('policy_id must identify the fixed model and checkpoint')
    if max_steps < 1 or episode_timeout < 1 or startup_retries < 0:
        raise ValueError('max_steps and episode_timeout must be positive; '
                         'startup_retries must be nonnegative')
    episodes = load_episodes(split)
    if not episodes:
        raise ValueError('R2R split contains no episodes')
    tokens_by_id = None
    if policy_split is not None:
        _, tokens_by_id = load_cma_tokens(split, policy_split)
    records = {}
    for episode in episodes:
        name = scene_name(episode)
        if name not in records:
            records[name] = load_scene_record(scenes, name)
        initial_yaw(episode, records[name]['matrix'][:3, :3])
    plan = {
        'split_sha256': sha256(split), 'scenes_sha256': sha256(scenes),
        'usd_sha256': {name: sha256(record['usd'])
                       for name, record in sorted(records.items())},
        'policy_url': policy_url, 'policy_id': policy_id,
        'policy_split_sha256': sha256(policy_split) if policy_split else None,
        'max_steps': max_steps, 'headless': headless,
        'episode_timeout': episode_timeout, 'startup_retries': startup_retries,
        'episode_ids': [str(item['episode_id']) for item in episodes],
    }
    manifest_path = out_dir / 'manifest.json'
    if resume:
        if not manifest_path.is_file():
            raise ValueError('cannot resume without an existing manifest')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['plan'] != plan:
            raise ValueError('resume plan differs from the recorded batch')
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        manifest = {'schema': 1, 'status': 'running', 'completed_episodes': 0,
                    'plan': plan}
        write_json(manifest_path, manifest)
    result_dir = out_dir / 'episodes'
    result_dir.mkdir(exist_ok=True)
    paths = []
    for index, episode in enumerate(episodes):
        episode_id = str(episode['episode_id'])
        scene_id = scene_name(episode)
        path = result_dir / f'{index:05d}.json'
        paths.append(path)
        if path.is_file():
            try:
                row = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                row = {}
            if row.get('status') == 'completed' and \
                    row.get('episode_id') == episode_id and \
                    row.get('split_sha256') == plan['split_sha256'] and \
                    row.get('scene_id') == scene_id and \
                    row.get('usd_sha256') == plan['usd_sha256'][scene_id] and \
                    row.get('policy_id') == policy_id and \
                    row.get('policy_split_sha256') == plan['policy_split_sha256']:
                continue
        command = [sys.executable, '-m', 'bench.cli', 'r2r',
                   '--split', str(split), '--scenes', str(scenes),
                   '--episode', episode_id, '--policy-url', policy_url,
                   '--policy-id', policy_id,
                   '--max-steps', str(max_steps), '--out', str(path)]
        if tokens_by_id is not None:
            command.extend(['--policy-tokens-json', json.dumps(tokens_by_id[episode_id]),
                            '--policy-split-sha256', plan['policy_split_sha256']])
        if headless:
            command.append('--headless')
        for attempt in range(startup_retries + 1):
            with open_attempt_log(result_dir, index) as log:
                try:
                    child = subprocess.run(command, stdout=log,
                                           stderr=subprocess.STDOUT,
                                           timeout=episode_timeout, check=False)
                except subprocess.TimeoutExpired:
                    manifest.update(status='error', failed_episode_id=episode_id,
                                    error='episode timed out')
                    write_json(manifest_path, manifest)
                    raise RuntimeError(f'episode {episode_id} timed out') from None
            try:
                row = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                row = {}
            # Kit can terminate natively after the trace is marked "started".
            # Retry only this incomplete state, never a handled episode error.
            if row.get('status') == 'completed' or row.get('status') == 'error':
                break
            if attempt < startup_retries:
                manifest.setdefault('startup_retry_counts', {})[episode_id] = attempt + 1
                write_json(manifest_path, manifest)
        if child.returncode or row.get('status') != 'completed' or \
                row.get('episode_id') != episode_id or \
                row.get('scene_id') != scene_id or \
                row.get('usd_sha256') != plan['usd_sha256'][scene_id] or \
                row.get('policy_id') != policy_id or \
                row.get('policy_split_sha256') != plan['policy_split_sha256']:
            manifest.update(status='error', failed_episode_id=episode_id,
                            error=f'child exit {child.returncode}: {row.get("error")}')
            write_json(manifest_path, manifest)
            raise RuntimeError(f'episode {episode_id} failed; inspect {path}')
        manifest['completed_episodes'] = index + 1
        manifest['status'] = 'running'
        manifest.pop('failed_episode_id', None)
        manifest.pop('error', None)
        write_json(manifest_path, manifest)
    result = summarize(paths, split)
    result['policy_id'] = policy_id
    write_json(out_dir / 'summary.json', result)
    manifest['completed_episodes'] = len(episodes)
    manifest['status'] = 'completed'
    write_json(manifest_path, manifest)
    return result
