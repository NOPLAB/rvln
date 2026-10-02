"""Inventory R2R-CE scenes before importing licensed scans into Isaac Sim."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from bench.artifacts import sha256

from isaac_r2r.protocol import load_episodes, scene_name


def inventory(split: Path, scans_root: Path, registry: Path | None = None) -> dict:
    """Count split episodes and locate matching MP3D scans and registered USDs."""
    episodes = load_episodes(split)
    if not episodes:
        raise ValueError("R2R-CE split has no episodes")
    counts = Counter(scene_name(episode) for episode in episodes)
    registered = {}
    if registry is not None:
        registered = json.loads(registry.read_text(encoding="utf-8"))["scenes"]
    scenes = []
    for name, count in sorted(counts.items()):
        scan = scans_root / name / f"{name}.glb"
        record = registered.get(name, {})
        usd = Path(record["usd"]).expanduser() if "usd" in record else None
        scenes.append(
            {
                "scene_id": name,
                "episodes": count,
                "scan_glb": str(scan),
                "scan_exists": scan.is_file(),
                "registered_usd": str(usd) if usd is not None else None,
                "usd_exists": bool(usd is not None and usd.is_absolute() and usd.is_file()),
                "transform_registered": "isaac_from_habitat" in record,
            }
        )
    return {
        "schema": 1,
        "split": str(split),
        "split_sha256": sha256(split),
        "episodes": len(episodes),
        "scenes": scenes,
        "scan_count": sum(row["scan_exists"] for row in scenes),
        "usd_count": sum(row["usd_exists"] and row["transform_registered"] for row in scenes),
    }
