"""R2R-CE transfer commands implemented with Isaac Sim."""
from __future__ import annotations

import argparse
from pathlib import Path

from bench.artifacts import sha256, write_json
from bench.episode import EpisodeRequest, HttpPolicy, run_episode
from isaac_r2r.protocol import load_episodes, scene_name, score
from isaac_r2r.summary import summarize


def run(args) -> dict:
    from isaac_r2r.run import IsaacR2RSimulator, initial_yaw, load_scene_record

    if not args.check and args.out is None:
        raise ValueError('r2r requires --out unless --check is set')
    episodes = load_episodes(args.split)
    episode = next((item for item in episodes
                    if str(item['episode_id']) == args.episode), None)
    if episode is None:
        raise ValueError(f'episode ID not in split: {args.episode}')
    scene = load_scene_record(args.scenes, scene_name(episode))
    initial_yaw(episode, scene['matrix'][:3, :3])
    result = {
        'episode_id': args.episode,
        'scene_id': scene_name(episode),
        'split_sha256': sha256(args.split),
        'usd': str(scene['usd']),
        'usd_sha256': sha256(scene['usd']),
        'metric_namespace': 'isaac_r2r_transfer',
    }
    if args.check:
        result['status'] = 'assets_validated'
        return result
    result['status'] = 'started'
    write_json(args.out, result)
    simulator = None
    try:
        simulator = IsaacR2RSimulator(scene, headless=args.headless)
        request = EpisodeRequest(
            episode_id=str(episode['episode_id']),
            instruction=episode['instruction']['instruction_text'],
            payload=episode,
            allowed_actions=frozenset({'stop', 'forward', 'left', 'right'}),
            terminal_action='stop')
        trace = run_episode(simulator, HttpPolicy(args.policy_url), request,
                            args.max_steps)
        result.update(score(episode, trace['positions'], trace['stopped']))
        result.update(trace)
        result['blocked_steps'] = simulator.blocked_steps
        result['status'] = 'completed'
        write_json(args.out, result)
    except Exception as error:
        result['status'] = 'error'
        result['error'] = f'{type(error).__name__}: {error}'
        write_json(args.out, result)
        raise
    finally:
        if simulator is not None:
            simulator.close()
    return result


def summarize_results(args) -> dict:
    result = summarize(args.results)
    write_json(args.out, result)
    return result


def register(commands: argparse._SubParsersAction) -> None:
    r2r = commands.add_parser('r2r', help='run an R2R-CE transfer episode')
    r2r.add_argument('--split', type=Path, required=True)
    r2r.add_argument('--scenes', type=Path, required=True)
    r2r.add_argument('--episode', required=True)
    r2r.add_argument('--policy-url', default='http://127.0.0.1:8765')
    r2r.add_argument('--out', type=Path)
    r2r.add_argument('--max-steps', type=int, default=500)
    r2r.add_argument('--headless', action='store_true')
    r2r.add_argument('--check', action='store_true')
    r2r.set_defaults(handler=run)

    summary = commands.add_parser('summarize-r2r', help='summarize transfer results')
    summary.add_argument('results', type=Path, nargs='+')
    summary.add_argument('--out', type=Path, required=True)
    summary.set_defaults(handler=summarize_results)
