"""Run one Isaac/Edge episode against a Slurm hosted inference server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.paths import workspace_root
from isaac_rvln.live_results import attach_contacts


def __getattr__(name: str):
    # Preserve old callable locations without loading ROS for CLI operations.
    if name in ("Episode", "spin_until"):
        from isaac_rvln import live_transport

        return getattr(live_transport, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run(args: argparse.Namespace) -> dict:
    from isaac_rvln.live_transport import run as execute

    return execute(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url")
    parser.add_argument("--model")
    parser.add_argument("--deployment", default="remote_gpu_local_edge")
    parser.add_argument("--episode")
    parser.add_argument("--manifest", type=Path, default=workspace_root() / "episodes/pilot.json")
    parser.add_argument("--world-source", type=Path)
    parser.add_argument("--usd", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--finalize-contacts", type=Path)
    parser.add_argument("--text")
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--tolerance", type=float, default=0.30)
    args = parser.parse_args()
    if args.finalize_contacts is not None:
        result = attach_contacts(args.out, args.finalize_contacts)
        print(
            json.dumps(
                {
                    "model": result["model"],
                    "episode": result["id"],
                    "collisions": result["collisions"],
                }
            ),
            flush=True,
        )
        return
    if not all((args.url, args.model, args.episode, args.world_source, args.usd, args.video)):
        parser.error("run requires --url, --model, --episode, --world-source, --usd, and --video")
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "model": args.model,
                "episode": args.episode,
                "stop_reason": result["stop_reason"],
                "embeddings": result["embeddings"],
                "video_frames": result["video_frames"],
                "collisions": result["collisions"],
                "final": result["trace"][-1] if result["trace"] else None,
                "errors": result["errors"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
