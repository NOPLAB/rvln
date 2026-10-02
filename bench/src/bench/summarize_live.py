"""Summarize ground-truth route traces with linked episode videos."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from bench.paths import workspace_root

from bench.score import score_episode, summarize


ORDER = ("asyncvla", "omnivla", "omnivla_edge", "movla", "navila", "navida")
POSE_SOURCES = {"gazebo_model_states": "gazebo", "isaac_ground_truth": "isaac_sim"}


def percentile(values: list[float], rank: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    position = (len(values) - 1) * rank
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def run(runs: Path) -> dict:
    episodes = []
    oracle = None
    pose_source = None
    latencies = {}
    for model in ORDER:
        for path in sorted(runs.glob(f"{model}-*.json")):
            if path.name.endswith(".contacts.json"):
                continue
            raw = json.loads(path.read_text())
            source = raw.get("pose_source")
            if source not in POSE_SOURCES:
                raise ValueError(f"unknown ground-truth pose source in {path.name}: {source}")
            if pose_source is None:
                pose_source = source
            elif source != pose_source:
                raise ValueError("mixed simulator pose sources need separate reports")
            video = runs / raw.get("video", "")
            if not video.is_file() or video.stat().st_size < 1000 or raw["video_frames"] < 1:
                raise ValueError(f"missing or empty video for {path.name}")
            if not raw["trace"]:
                raise ValueError(f"missing trace for {path.name}")
            if raw.get("collisions") is None or not raw.get("contact_samples"):
                raise ValueError(f"missing validated contact data for {path.name}")
            if raw.get("shortest_m") is None or not raw.get("oracle"):
                raise ValueError(f"missing route oracle for {path.name}")
            if oracle is None:
                oracle = raw["oracle"]
            elif oracle != raw["oracle"]:
                raise ValueError(f"inconsistent route oracle for {path.name}")
            scored = score_episode(raw)
            scored["scene"] = raw["scene"]
            scored["instruction_sent"] = raw["instruction_sent"]
            scored["video"] = video.name
            scored["video_frames"] = raw["video_frames"]
            scored["observations"] = raw["observations"]
            scored["embeddings"] = raw["embeddings"]
            scored["confirmed_stop"] = raw["confirmed_stop"]
            scored["errors"] = raw["errors"]
            scored["contact_samples"] = raw["contact_samples"]
            scored["model_version"] = raw["model_version"]
            scored["closest_distance_m"] = min(
                math.hypot(p["x"] - raw["goal_xy"][0], p["y"] - raw["goal_xy"][1])
                for p in raw["trace"]
            )
            scored["round_trip_p95_ms"] = percentile(
                [i["round_trip_ms"] for i in raw["inferences"]], 0.95
            )
            latencies.setdefault(model, []).extend(i["round_trip_ms"] for i in raw["inferences"])
            episodes.append(scored)
    if not episodes:
        raise ValueError(f"no live episodes in {runs}")
    result = summarize(episodes)
    for group in result["groups"]:
        rows = [r for r in episodes if r["model"] == group["model"]]
        group["median_final_distance_m"] = statistics.median(r["final_distance_m"] for r in rows)
        group["median_travel_m"] = statistics.median(r["travel_m"] for r in rows)
        group["median_elapsed_sec"] = statistics.median(r["elapsed_sec"] for r in rows)
        group["round_trip_p95_ms"] = percentile(latencies.get(group["model"], []), 0.95)
        group["goal_tolerance_stops"] = sum(r["stop_reason"] == "goal_tolerance" for r in rows)
        group["timeouts"] = sum(r["stop_reason"] == "timeout" for r in rows)
        group["track"] = "template_commands" if group["model"] == "movla" else "general_text"
    result["protocol"] = {
        "pose_source": pose_source,
        "simulator": POSE_SOURCES[pose_source],
        "goal_tolerance_m": 0.30,
        "max_duration_sec": 35,
        "video_fps": 2,
        "success_definition": "ground-truth endpoint within tolerance and gated follower stop",
        "collision_rate": "episodes with at least one static obstacle contact event",
        "spl": "SPL@0.30m using measured route and disk-footprint shortest route",
        "oracle": oracle,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=workspace_root() / "runs/contact_2026-09-25")
    parser.add_argument(
        "--out", type=Path, default=workspace_root() / "results/live_contacts_spl_2026-09-25.json"
    )
    args = parser.parse_args()
    result = run(args.runs)
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for group in result["groups"]:
        print(
            f"{group['model']}: SR={group['success_rate']['mean']:.2f} "
            f"n={group['episodes']} median-final={group['median_final_distance_m']:.2f}m"
        )


if __name__ == "__main__":
    main()
