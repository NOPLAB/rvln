"""Render RGB-D and one collision-aware step inside a converted USD room."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from bench.episode import EpisodeRequest

from isaac_r2r.run import IsaacR2RSimulator


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--start", nargs=3, type=float, default=(0.0, 0.0, 0.0))
parser.add_argument("--headless", action="store_true")
parser.add_argument("--dome-intensity", type=float, default=0.0)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

matrix = np.eye(4)
scene = {
    "usd": args.usd.resolve(),
    "matrix": matrix,
    "inverse": matrix,
    "dome_light_intensity": args.dome_intensity,
}
episode = EpisodeRequest(
    "imported-room-smoke",
    "Walk forward.",
    {
        "start_position": list(args.start),
        "start_rotation": [0.0, -0.7071067811865475, 0.0, 0.7071067811865476],
    },
    frozenset({"forward", "stop"}),
    "stop",
)
simulator = IsaacR2RSimulator(scene, headless=args.headless)
try:
    start = simulator.reset(episode)
    first = simulator.observe()
    (args.out_dir / "first.jpg").write_bytes(first.jpeg)
    position = simulator.apply("forward")
    second = simulator.observe()
    (args.out_dir / "second.jpg").write_bytes(second.jpeg)
    depth = np.frombuffer(second.depth_f32, dtype="<f4")
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if not first.jpeg.startswith(b"\xff\xd8") or not second.jpeg.startswith(b"\xff\xd8"):
        raise RuntimeError("Isaac did not return JPEG camera frames")
    if len(valid) == 0:
        raise RuntimeError("Isaac did not return valid depth")
    with Image.open(io.BytesIO(second.jpeg)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    mean_rgb = float(rgb.mean())
    variation_rgb = float(rgb.std())
    if mean_rgb < 5 or variation_rgb < 3:
        raise RuntimeError(
            f"Isaac RGB scene is dark or flat: mean={mean_rgb:.2f}, std={variation_rgb:.2f}"
        )
    print(
        json.dumps(
            {
                "start": start,
                "after_forward": position,
                "blocked_steps": simulator.blocked_steps,
                "rgb_mean": mean_rgb,
                "rgb_std": variation_rgb,
                "depth_valid_pixels": len(valid),
                "depth_near_m": float(valid.min()),
                "depth_far_m": float(valid.max()),
            }
        ),
        flush=True,
    )
finally:
    simulator.close()
