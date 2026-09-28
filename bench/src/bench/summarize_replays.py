"""Summarize fixed-camera GPU replays without interpreting them as navigation success."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

SCENES = ('corridor', 'junction', 'weave')
BACKENDS = ('asyncvla', 'omnivla', 'omnivla_edge', 'movla', 'navila', 'navida')


def percentile(values: list[float], percent: float) -> float:
    """Compute a linearly interpolated percentile from all observed samples."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError('empty sample')
    index = (len(ordered) - 1) * percent / 100.0
    lower = int(index)
    fraction = index - lower
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(root: Path) -> dict:
    """Validate the paired scene matrix and preserve provenance in the summary."""
    captures = {}
    for scene in SCENES:
        capture = root / f'{scene}_capture' / 'capture.json'
        content = capture.read_bytes()
        rows = json.loads(content)['frames']
        if len(rows) != 20:
            raise ValueError(f'{scene}: expected 20 captured frames, got {len(rows)}')
        captures[scene] = {'sha256': hashlib.sha256(content).hexdigest(),
                           'frame_count': len(rows)}

    models = {}
    for backend in BACKENDS:
        samples = []
        steady = []
        first = []
        loads = []
        peaks = []
        versions = set()
        output_shapes = set()
        for scene in SCENES:
            path = root / scene / f'{backend}.json'
            result = json.loads(path.read_text(encoding='utf-8'))
            if result['backend'] != backend or len(result['frames']) != 20:
                raise ValueError(f'wrong backend or frame count: {path}')
            if [row['frame'] for row in result['frames']] != [
                    f'{i:04d}.jpg' for i in range(20)]:
                raise ValueError(f'unpaired frame sequence: {path}')
            times = [float(row['wall_ms']) for row in result['frames']]
            samples.extend(times)
            steady.extend(times[1:])
            first.append(times[0])
            loads.append(float(result['load_ms']))
            peaks.append(int(result['gpu']['peak_allocated_bytes']))
            versions.add(result['model_version'])
            output_shapes.update(tuple(row['shape']) for row in result['frames'])
        if len(versions) != 1 or len(output_shapes) != 1:
            raise ValueError(f'{backend}: model version or output shape varied across scenes')
        models[backend] = {
            'model_version': next(iter(versions)),
            'output_shape': list(next(iter(output_shapes))),
            'frames': len(samples),
            'load_median_ms': statistics.median(loads),
            'first_frame_median_ms': statistics.median(first),
            'inference_p50_ms': percentile(samples, 50),
            'inference_p95_ms': percentile(samples, 95),
            'steady_p50_ms': percentile(steady, 50),
            'steady_p95_ms': percentile(steady, 95),
            'peak_allocated_gib': max(peaks) / 2**30,
        }
    return {'schema': 1, 'workload': 'stationary Gazebo camera, open-loop replay',
            'scenes': captures, 'models': models}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('bench/runs'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for backend, row in result['models'].items():
        print(f'{backend:14} {row["frames"]:2} '
              f'{row["steady_p50_ms"]:7.1f} {row["steady_p95_ms"]:7.1f} '
              f'{row["peak_allocated_gib"]:5.1f}')


if __name__ == '__main__':
    main()
