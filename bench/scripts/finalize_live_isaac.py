"""Attach PhysX contacts and score one synthetic Isaac live-pipeline pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.score import score_episode
from isaac_rvln.live_results import attach_contacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--episode-file', default='stub-c01.json')
    parser.add_argument('--expect-model', default='stub')
    parser.add_argument('--expect-deployment', default='isaac_gpu_edge_local_stub')
    parser.add_argument('--benchmark-kind', default='synthetic_stub_integration')
    args = parser.parse_args()
    out = args.out_dir
    episode = attach_contacts(out / args.episode_file, out / 'contacts.json')
    if (episode['model'] != args.expect_model or
            episode['deployment'] != args.expect_deployment):
        raise SystemExit('episode model or deployment does not match the requested run')
    if episode['errors'] or not episode['trace'] or not episode['inferences']:
        raise SystemExit('live episode has errors or missing pose/inference trace')
    if episode['embeddings'] < 1 or episode['nonempty_paths'] < 1:
        raise SystemExit('Edge produced no embedding or path')
    video = out / episode['video']
    if not video.is_file() or video.stat().st_size < 1000:
        raise SystemExit('live episode video is missing')
    score = score_episode(episode)
    report = {'schema': 1, 'benchmark_kind': args.benchmark_kind,
              'pose_source': episode['pose_source'], 'score': score,
              'observations': episode['observations'],
              'embeddings': episode['embeddings'],
              'nonempty_paths': episode['nonempty_paths'],
              'video_frames': episode['video_frames']}
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n',
                                      encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
