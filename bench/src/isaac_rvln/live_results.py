"""Finalize a ROS-driven Isaac episode with its PhysX contact evidence."""

from __future__ import annotations

import json
from pathlib import Path


def assemble_result(
    args,
    manifest: dict,
    episode: dict,
    node,
    actual_hash: str,
    usd_hash: str,
    stop_reason: str,
    confirmed_stop: bool,
    wall_ended_at: float,
) -> dict:
    """Assemble the audit report without live execution dependencies."""
    return {
        **episode,
        "model": args.model,
        "deployment": args.deployment,
        "oracle": manifest["oracle"],
        "pose_source": "isaac_ground_truth",
        "world_source_sha256": actual_hash,
        "usd_sha256": usd_hash,
        "wall_started_at": node.started_at,
        "wall_ended_at": wall_ended_at,
        "instruction_sent": node.instruction,
        "trace": node.trace,
        "stop_reason": stop_reason,
        "confirmed_stop": confirmed_stop,
        "collisions": None,
        "contact_events": None,
        "contact_samples": None,
        "contact_log": None,
        "observations": node.received,
        "embeddings": node.published,
        "paths": node.path_count,
        "nonempty_paths": node.nonempty_paths,
        "path_samples": node.path_samples,
        "model_version": node.model_version,
        "inferences": node.inferences,
        "errors": node.errors,
        "video": args.video.name,
        "video_frames": node.video_frames,
        "final_odom_xy": node.last_odom_pose,
    }


def attach_contacts(result_path: Path, contact_path: Path) -> dict:
    """Merge the finished Isaac PhysX log after the simulator exits."""
    row = json.loads(result_path.read_text(encoding="utf-8"))
    if row.get("pose_source") != "isaac_ground_truth":
        raise ValueError("episode is not an Isaac ground-truth trace")
    if row.get("contact_log") is not None:
        raise ValueError("episode already has contact data")
    contacts = json.loads(contact_path.read_text(encoding="utf-8"))
    if contacts.get("simulator") != "isaac_sim" or contacts.get("schema") != 1:
        raise ValueError("contact log is not an Isaac Sim report")
    if contacts.get("receive_errors") or contacts.get("samples", {}).get("physics_steps", 0) < 1:
        raise ValueError("Isaac contact log has no physics evidence or contains errors")
    events = contacts.get("events")
    if not isinstance(events, list) or contacts.get("collisions") != len(events):
        raise ValueError("Isaac contact count differs from events")
    start = row.get("wall_started_at")
    end = row.get("wall_ended_at")
    if start is None or end is None or end <= start:
        raise ValueError("episode has no valid contact measurement window")
    events = [event for event in events if start <= event["wall_monotonic_sec"] <= end]
    for event in events:
        event["t_sec"] = event["wall_monotonic_sec"] - start
    row["contact_events"] = events
    row["collisions"] = len(events)
    row["contact_samples"] = contacts["samples"]
    row["contact_log"] = contact_path.name
    result_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return row
