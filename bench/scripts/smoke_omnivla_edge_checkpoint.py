"""Run one published OmniVLA-edge image inference on a reserved GPU."""

from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image

from bench.replay import build_backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    backend = build_backend("omnivla_edge", args.checkpoint, args.device, 120000)
    image = Image.fromarray(np.full((224, 224, 3), 127, dtype=np.uint8))
    samples = []
    for _ in range(args.iterations):
        output, reported = backend.infer(
            current_image=image,
            past_image=image,
            lang_instruction="Go straight ahead.",
            goal_image=None,
            goal_pose_xy_theta=None,
        )
        actions = np.asarray(output)
        if actions.shape != (8, 4) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid OmniVLA-edge output: {actions.shape}")
        samples.append(float(reported["inference_ms"]))
    print(
        json.dumps(
            {
                "model_version": backend.model_info().model_version,
                "shape": list(actions.shape),
                "first_waypoint": actions[0].tolist(),
                "inference_ms": samples[-1],
                "inference_ms_samples": samples,
            }
        )
    )


if __name__ == "__main__":
    main()
