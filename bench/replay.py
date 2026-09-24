"""Measure real backend inference on a fixed sequence of recorded camera frames.

Run one process per episode: several backends keep temporal state internally and
there is not yet a shared episode-reset API in the ROS inference contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image


_BACKENDS = ('dummy', 'asyncvla', 'omnivla', 'omnivla_edge',
             'movla', 'navila', 'navida')


def build_backend(name: str, checkpoint: str, device: str, resume_step: int):
    """Import only the requested policy and preserve its production constructor."""
    if name == 'dummy':
        from rvln_remote.backends.dummy import DummyBackend

        return DummyBackend()
    if name == 'asyncvla':
        from rvln_remote.backends.asyncvla import AsyncVLABackend

        return AsyncVLABackend(vla_path=checkpoint, resume_step=resume_step,
                               device=device)
    if name == 'omnivla':
        from rvln_remote.backends.omnivla import OmniVLABackend

        return OmniVLABackend(vla_path=checkpoint, resume_step=resume_step,
                              device=device)
    if name == 'omnivla_edge':
        from rvln_remote.backends.omnivla_edge import OmniVLAEdgeBackend

        return OmniVLAEdgeBackend(weights_path=checkpoint, device=device)
    if name == 'movla':
        from rvln_remote.backends.movla import MovlaBackend

        return MovlaBackend(checkpoint_dir=checkpoint, device=device)
    if name == 'navila':
        from rvln_remote.backends.navila import NaVILABackend

        return NaVILABackend(checkpoint_dir=checkpoint, device=device)
    if name == 'navida':
        from rvln_remote.backends.navida import NaVIDABackend

        return NaVIDABackend(checkpoint_dir=checkpoint, device=device)
    raise ValueError(f'unknown backend {name!r}')


def _gpu_stats(device: str) -> dict:
    if not device.startswith('cuda'):
        return {}
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        'gpu_name': torch.cuda.get_device_name(device),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
    }


def _synchronize(device: str) -> None:
    if device.startswith('cuda'):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)


def run(args: argparse.Namespace) -> dict:
    """Load a checkpoint, infer each frame in order, and return raw measurements."""
    frames = sorted(Path(args.frames).glob('*.jpg'))
    frames += sorted(Path(args.frames).glob('*.jpeg'))
    if not frames:
        raise ValueError(f'no JPEG frames in {args.frames}')
    if args.max_frames:
        frames = frames[:args.max_frames]
    if args.goal_mode == 'text' and not args.text:
        raise ValueError('text goal requires --text')
    if args.goal_mode == 'pose' and args.pose is None:
        raise ValueError('pose goal requires --pose X Y THETA')
    if args.backend in ('navila', 'navida', 'movla') and args.goal_mode != 'text':
        raise ValueError(f'{args.backend} is text-only in this benchmark')

    start = time.monotonic()
    backend = build_backend(args.backend, args.checkpoint, args.device, args.resume_step)
    load_ms = (time.monotonic() - start) * 1000.0
    info = backend.model_info()
    if args.warmup:
        backend.warmup(args.warmup)
    _synchronize(args.device)
    rows = []
    previous = None
    goal_image = None
    if args.goal_mode == 'image':
        if not args.goal_image:
            raise ValueError('image goal requires --goal-image')
        with Image.open(args.goal_image) as source:
            goal_image = source.convert('RGB')
    for index, frame in enumerate(frames):
        with Image.open(frame) as source:
            current = source.convert('RGB')
        _synchronize(args.device)
        start = time.monotonic()
        output, reported = backend.infer(
            current_image=current,
            past_image=previous if previous is not None else current,
            lang_instruction=args.text if args.goal_mode == 'text' else '',
            goal_image=goal_image,
            goal_pose_xy_theta=tuple(args.pose) if args.goal_mode == 'pose' else None,
        )
        _synchronize(args.device)
        wall_ms = (time.monotonic() - start) * 1000.0
        arr = np.asarray(output, dtype=np.float32)
        if arr.ndim != 2 or not arr.size or not np.isfinite(arr).all():
            raise ValueError(f'invalid output at frame {frame}: shape={arr.shape}')
        rows.append({
            'index': index,
            'frame': frame.name,
            'wall_ms': wall_ms,
            'backend_ms': float(reported.get('inference_ms', math.nan)),
            'shape': list(arr.shape),
            'first_token': arr[0].tolist(),
            'sha256_f32': hashlib.sha256(arr.tobytes()).hexdigest(),
        })
        previous = current
    return {
        'schema': 1,
        'backend': args.backend,
        'model_name': info.model_name,
        'model_version': info.model_version,
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'device': args.device,
        'python': platform.python_version(),
        'goal_mode': args.goal_mode,
        'goal_text': args.text if args.goal_mode == 'text' else None,
        'load_ms': load_ms,
        'warmup_iters': args.warmup,
        'gpu': _gpu_stats(args.device),
        'frames': rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=_BACKENDS, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume-step', type=int, default=120000)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--goal-mode', choices=('text', 'pose', 'image'), default='text')
    parser.add_argument('--text', default='go straight ahead')
    parser.add_argument('--pose', type=float, nargs=3)
    parser.add_argument('--goal-image')
    args = parser.parse_args()
    result = run(args)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    values = [row['wall_ms'] for row in result['frames']]
    print(json.dumps({
        'backend': result['backend'], 'frames': len(values),
        'load_ms': result['load_ms'], 'median_ms': float(np.median(values)),
        'p95_ms': float(np.percentile(values, 95)), 'out': str(output),
    }))


if __name__ == '__main__':
    main()
