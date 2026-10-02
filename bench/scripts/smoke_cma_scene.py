"""Run synthetic CMA weights through the real HTTP and Isaac room episode path."""

from __future__ import annotations

import argparse
import gc
import json
import select
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import torch

from isaac_r2r.cma_model import CMAPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dome-intensity", type=float, default=800.0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="published CMA weights; omit for forward-biased synthetic weights",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="run two episodes through the exact-coverage batch path",
    )
    args = parser.parse_args()
    usd = args.usd.resolve(strict=True)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    scenes = out_dir / "scenes.json"
    split = out_dir / "episodes.json"
    result_path = out_dir / "result.json"
    scenes.write_text(
        json.dumps(
            {
                "scenes": {
                    "furnished_room": {
                        "usd": str(usd),
                        "dome_light_intensity": args.dome_intensity,
                        "isaac_from_habitat": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    episode = {
        "episode_id": f"synthetic-cma-room-{uuid.uuid4().hex[:8]}",
        "scene_id": "data/scene_datasets/mp3d/furnished_room.glb",
        "instruction": {
            "instruction_text": "Move through the room.",
            "instruction_tokens": [2, 3, 4],
        },
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, -0.7071067811865475, 0.0, 0.7071067811865476],
        "goals": [{"position": [0.0, 0.0, 3.0]}],
        "info": {"geodesic_distance": 3.0},
    }
    episodes = [episode]
    if args.batch:
        episodes.append(
            {
                **episode,
                "episode_id": f"synthetic-cma-room-{uuid.uuid4().hex[:8]}",
                "instruction": {
                    "instruction_text": "Walk past the furniture.",
                    "instruction_tokens": [2, 3, 4],
                },
            }
        )
    split.write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = (
            args.checkpoint.resolve(strict=True)
            if args.checkpoint
            else Path(temporary) / "synthetic-cma.pth"
        )
        if args.checkpoint is None:
            model = CMAPolicy()
            with torch.no_grad():
                model.action_distribution.linear.weight.zero_()
                model.action_distribution.linear.bias.copy_(torch.tensor([-2.0, 2.0, -2.0, -2.0]))
            torch.save({"state_dict": model.state_dict()}, checkpoint)
            del model
            gc.collect()
        with (out_dir / "policy.log").open("w", encoding="utf-8") as policy_log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "isaac_r2r.policy_server",
                    "--checkpoint",
                    str(checkpoint),
                    "--factory",
                    "isaac_r2r.cma_model:load_model",
                    "--port",
                    "0",
                ],
                stdout=subprocess.PIPE,
                stderr=policy_log,
                text=True,
                bufsize=1,
            )
            try:
                readable, _, _ = select.select([server.stdout], [], [], 45)
                if not readable:
                    raise RuntimeError("CMA policy server did not start within 45 s")
                first_line = server.stdout.readline()
                if not first_line:
                    raise RuntimeError("CMA policy server exited before readiness")
                ready = json.loads(first_line)
                command = [
                    sys.executable,
                    "-m",
                    "bench.cli",
                    "r2r-batch" if args.batch else "r2r",
                    "--split",
                    str(split),
                    "--policy-split",
                    str(split),
                    "--scenes",
                    str(scenes),
                    "--policy-url",
                    f"http://127.0.0.1:{ready['port']}",
                    "--policy-id",
                    ready["policy_id"],
                    "--max-steps",
                    "3",
                ]
                if args.batch:
                    command.extend(
                        [
                            "--out-dir",
                            str(out_dir / "batch"),
                            "--episode-timeout",
                            "180",
                            "--startup-retries",
                            "0",
                        ]
                    )
                else:
                    command.extend(["--episode", episode["episode_id"], "--out", str(result_path)])
                with (out_dir / "isaac.log").open("w", encoding="utf-8") as sim_log:
                    child = subprocess.run(
                        command,
                        stdout=sim_log,
                        stderr=subprocess.STDOUT,
                        timeout=300 if args.batch else 180,
                        check=False,
                    )
                if args.batch:
                    batch_dir = out_dir / "batch"
                    result_path = batch_dir / "summary.json"
                    manifest = json.loads((batch_dir / "manifest.json").read_text())
                    results = [
                        json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted((batch_dir / "episodes").glob("*.json"))
                    ]
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    valid = (
                        manifest["status"] == "completed"
                        and result["episodes"] == len(episodes)
                        and result["policy_id"] == ready["policy_id"]
                    )
                    if args.checkpoint is None:
                        valid = valid and result["blocked_steps"] >= len(episodes)
                else:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    results = [result]
                    valid = result.get("status") == "completed"
                valid = (
                    valid
                    and not child.returncode
                    and len(results) == len(episodes)
                    and all(
                        row.get("status") == "completed"
                        and row.get("policy_id") == ready["policy_id"]
                        for row in results
                    )
                )
                if args.checkpoint is None:
                    valid = valid and all(
                        row.get("actions") == ["forward"] * 3 and row.get("blocked_steps", 0) >= 1
                        for row in results
                    )
                if not valid:
                    raise RuntimeError(
                        f"CMA/Isaac integration failed: exit={child.returncode}, result={result}"
                    )
                print(
                    json.dumps(
                        {
                            "episodes": len(episodes),
                            "checkpoint": str(checkpoint)
                            if args.checkpoint
                            else "synthetic_forward",
                            "actions": [row["actions"] for row in results],
                            "blocked_steps": sum(row["blocked_steps"] for row in results),
                            "policy_id": ready["policy_id"],
                            "result": str(result_path),
                        }
                    ),
                    flush=True,
                )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)


if __name__ == "__main__":
    main()
