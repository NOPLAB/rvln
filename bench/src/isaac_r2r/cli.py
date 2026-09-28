"""R2R-CE transfer commands implemented with Isaac Sim."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bench.artifacts import sha256, write_json
from bench.episode import EpisodeRequest, run_episode
from isaac_r2r.policy import R2RHttpPolicy
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
    if args.policy_split and args.policy_tokens_json:
        raise ValueError('choose either --policy-split or batch-supplied tokens')
    policy_tokens = None
    policy_split_hash = None
    if args.policy_split:
        from isaac_r2r.cma_inputs import load_cma_tokens

        _, tokens_by_id = load_cma_tokens(args.split, args.policy_split)
        policy_tokens = tokens_by_id[args.episode]
        policy_split_hash = sha256(args.policy_split)
    elif args.policy_tokens_json:
        policy_tokens = json.loads(args.policy_tokens_json)
        policy_split_hash = args.policy_split_sha256
        if not policy_split_hash:
            raise ValueError('batch-supplied tokens require a policy split hash')
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
    if policy_split_hash is not None:
        result['policy_split_sha256'] = policy_split_hash
    if args.policy_id is not None:
        result['policy_id'] = args.policy_id
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
        policy = R2RHttpPolicy(args.policy_url, policy_tokens, args.policy_id)
        trace = run_episode(simulator, policy, request,
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
    result = summarize(args.results, args.split)
    write_json(args.out, result)
    return result


def run_full_split(args) -> dict:
    from isaac_r2r.batch import run_batch

    return run_batch(args.split, args.scenes, args.policy_url, args.policy_id,
                     args.out_dir, args.max_steps, args.headless, args.resume,
                     args.episode_timeout, args.startup_retries,
                     args.policy_split)


def inventory_scenes(args) -> dict:
    from isaac_r2r.inventory import inventory

    result = inventory(args.split, args.scans_root, args.scenes)
    if args.out is not None:
        write_json(args.out, result)
    return result


def audit_cma_inputs(args) -> dict:
    from isaac_r2r.cma_inputs import audit_cma_split

    result = audit_cma_split(args.split, args.policy_split)
    if args.out is not None:
        write_json(args.out, result)
    return result


def convert_scan_asset(args) -> dict:
    from isaac_r2r.scans import validate_scan

    source, output = validate_scan(args.glb, args.out)
    command = [sys.executable, '-m', 'isaac_r2r.scan_worker',
               '--glb', str(source), '--out', str(output)]
    child = subprocess.run(command, check=False)
    sidecar = output.with_suffix('.json')
    if child.returncode or not output.is_file() or not sidecar.is_file():
        raise ValueError(f'Isaac scan import failed or left no verified USD: {output}')
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    if (metadata.get('source_sha256') != sha256(source)
            or metadata.get('usd') != str(output)
            or metadata.get('meshes', 0) < 1
            or metadata.get('triangles', 0) < 1):
        raise ValueError(f'Isaac scan import metadata is incomplete: {sidecar}')
    return metadata


def calibrate_scan(args) -> dict:
    from isaac_r2r.calibration import calibrate_scene

    return calibrate_scene(args.landmarks, args.scene, args.usd, args.out,
                           args.max_error_m, args.dome_intensity)


def register(commands: argparse._SubParsersAction) -> None:
    r2r = commands.add_parser('r2r', help='run an R2R-CE transfer episode')
    r2r.add_argument('--split', type=Path, required=True)
    r2r.add_argument('--scenes', type=Path, required=True)
    r2r.add_argument('--episode', required=True)
    r2r.add_argument('--policy-url', default='http://127.0.0.1:8765')
    r2r.add_argument('--policy-id', help='require the server to attest this model identity')
    r2r.add_argument('--policy-split', type=Path,
                     help='aligned preprocessed R2R split for pretrained CMA tokens')
    r2r.add_argument('--policy-tokens-json', help=argparse.SUPPRESS)
    r2r.add_argument('--policy-split-sha256', help=argparse.SUPPRESS)
    r2r.add_argument('--out', type=Path)
    r2r.add_argument('--max-steps', type=int, default=500)
    r2r.add_argument('--headless', action='store_true')
    r2r.add_argument('--check', action='store_true')
    r2r.set_defaults(handler=run)

    summary = commands.add_parser('summarize-r2r', help='summarize transfer results')
    summary.add_argument('results', type=Path, nargs='+')
    summary.add_argument('--split', type=Path,
                         help='require exact episode coverage of this split')
    summary.add_argument('--out', type=Path, required=True)
    summary.set_defaults(handler=summarize_results)

    batch = commands.add_parser('r2r-batch',
                                help='run and audit every episode in an R2R split')
    batch.add_argument('--split', type=Path, required=True)
    batch.add_argument('--scenes', type=Path, required=True)
    batch.add_argument('--policy-url', default='http://127.0.0.1:8765')
    batch.add_argument('--policy-id', required=True,
                       help='fixed model/checkpoint identity for the report')
    batch.add_argument('--policy-split', type=Path,
                       help='aligned preprocessed R2R split for pretrained CMA tokens')
    batch.add_argument('--out-dir', type=Path, required=True)
    batch.add_argument('--max-steps', type=int, default=500)
    batch.add_argument('--episode-timeout', type=int, default=3600)
    batch.add_argument('--startup-retries', type=int, default=2,
                       help='retry incomplete native Kit exits before an episode finishes')
    batch.add_argument('--headless', action='store_true')
    batch.add_argument('--resume', action='store_true')
    batch.set_defaults(handler=run_full_split)

    audit = commands.add_parser(
        'inventory-r2r', help='inventory split scenes and local scan assets')
    audit.add_argument('--split', type=Path, required=True)
    audit.add_argument('--scans-root', type=Path, required=True)
    audit.add_argument('--scenes', type=Path, help='optional USD scene registry')
    audit.add_argument('--out', type=Path)
    audit.set_defaults(handler=inventory_scenes)

    cma = commands.add_parser('audit-cma-split',
                              help='verify pretrained CMA token IDs and episode alignment')
    cma.add_argument('--split', type=Path, required=True)
    cma.add_argument('--policy-split', type=Path, required=True)
    cma.add_argument('--out', type=Path)
    cma.set_defaults(handler=audit_cma_inputs)

    scan = commands.add_parser('convert-scan',
                               help='import an MP3D GLB as a collidable Isaac USD')
    scan.add_argument('--glb', type=Path, required=True)
    scan.add_argument('--out', type=Path, required=True)
    scan.set_defaults(handler=convert_scan_asset)

    calibration = commands.add_parser(
        'calibrate-r2r', help='fit a Habitat-to-Isaac transform from landmarks')
    calibration.add_argument('--landmarks', type=Path, required=True)
    calibration.add_argument('--scene', required=True)
    calibration.add_argument('--usd', type=Path, required=True)
    calibration.add_argument('--out', type=Path, required=True)
    calibration.add_argument('--max-error-m', type=float, default=0.05)
    calibration.add_argument('--dome-intensity', type=float, default=0.0,
                             help='verified dome-light intensity for this scene')
    calibration.set_defaults(handler=calibrate_scan)
