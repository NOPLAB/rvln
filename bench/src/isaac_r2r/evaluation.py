"""Asset preflight and R2R orchestration, separate from CLI parsing."""

from __future__ import annotations

import json
import subprocess
import sys

from bench.artifacts import sha256, write_json
from bench.episode import EpisodeRequest, run_episode
from isaac_r2r.protocol import load_episodes, scene_name, score
from isaac_r2r.summary import summarize


def run(args) -> dict:
    from isaac_r2r.policy import R2RHttpPolicy
    from isaac_r2r.scene import initial_yaw, load_scene_record

    if not args.check and args.out is None:
        raise ValueError("r2r requires --out unless --check is set")
    episodes = load_episodes(args.split)
    episode = next((item for item in episodes if str(item["episode_id"]) == args.episode), None)
    if episode is None:
        raise ValueError(f"episode ID not in split: {args.episode}")
    if args.policy_split and args.policy_tokens_json:
        raise ValueError("choose either --policy-split or batch-supplied tokens")
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
            raise ValueError("batch-supplied tokens require a policy split hash")
    scene = load_scene_record(args.scenes, scene_name(episode))
    initial_yaw(episode, scene["matrix"][:3, :3])
    result = {
        "episode_id": args.episode,
        "scene_id": scene_name(episode),
        "split_sha256": sha256(args.split),
        "usd": str(scene["usd"]),
        "usd_sha256": sha256(scene["usd"]),
        "metric_namespace": "isaac_r2r_transfer",
    }
    if policy_split_hash is not None:
        result["policy_split_sha256"] = policy_split_hash
    if args.policy_id is not None:
        result["policy_id"] = args.policy_id
    if args.check:
        result["status"] = "assets_validated"
        return result
    result["status"] = "started"
    write_json(args.out, result)
    simulator = None
    try:
        from isaac_r2r.run import IsaacR2RSimulator

        simulator = IsaacR2RSimulator(scene, headless=args.headless)
        request = EpisodeRequest(
            episode_id=str(episode["episode_id"]),
            instruction=episode["instruction"]["instruction_text"],
            payload=episode,
            allowed_actions=frozenset({"stop", "forward", "left", "right"}),
            terminal_action="stop",
        )
        policy = R2RHttpPolicy(args.policy_url, policy_tokens, args.policy_id)
        trace = run_episode(simulator, policy, request, args.max_steps)
        result.update(score(episode, trace["positions"], trace["stopped"]))
        result.update(trace)
        result["blocked_steps"] = simulator.blocked_steps
        result["status"] = "completed"
        write_json(args.out, result)
    except Exception as error:
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
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

    return run_batch(
        args.split,
        args.scenes,
        args.policy_url,
        args.policy_id,
        args.out_dir,
        args.max_steps,
        args.headless,
        args.resume,
        args.episode_timeout,
        args.startup_retries,
        args.policy_split,
    )


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
    command = [
        sys.executable,
        "-m",
        "isaac_r2r.scan_worker",
        "--glb",
        str(source),
        "--out",
        str(output),
    ]
    child = subprocess.run(command, check=False)
    sidecar = output.with_suffix(".json")
    if child.returncode or not output.is_file() or not sidecar.is_file():
        raise ValueError(f"Isaac scan import failed or left no verified USD: {output}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        metadata.get("source_sha256") != sha256(source)
        or metadata.get("usd") != str(output)
        or metadata.get("meshes", 0) < 1
        or metadata.get("triangles", 0) < 1
    ):
        raise ValueError(f"Isaac scan import metadata is incomplete: {sidecar}")
    return metadata


def calibrate_scan(args) -> dict:
    from isaac_r2r.calibration import calibrate_scene

    return calibrate_scene(
        args.landmarks, args.scene, args.usd, args.out, args.max_error_m, args.dome_intensity
    )
