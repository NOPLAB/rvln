"""Join complete camera/trajectory episode clips with readable result cards."""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import cv2
import numpy as np


ORDER = ('asyncvla', 'omnivla', 'omnivla_edge', 'movla', 'navila', 'navida')
SCENES = {'c': 'CORRIDOR', 'j': 'JUNCTION', 'w': 'WEAVE'}
SIZE = (960, 480)
FPS = 2.0


def put(canvas: np.ndarray, lines: list[str], y: int = 125) -> None:
    for line in lines:
        cv2.putText(canvas, line, (62, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9 if y <= 125 else 0.65, (240, 240, 240), 2)
        y += 50


def card(writer, lines: list[str], seconds: float = 3.0) -> None:
    frame = np.full((SIZE[1], SIZE[0], 3), (34, 30, 27), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (960, 18), (0, 185, 255), -1)
    put(frame, lines)
    for _ in range(round(seconds * FPS)):
        writer.write(frame)


def result(row: dict) -> tuple[bool, float]:
    goal = row['goal_xy']
    end = row['trace'][-1] if row['trace'] else None
    distance = math.hypot(end['x'] - goal[0], end['y'] - goal[1]) if end else math.inf
    return distance <= 0.30 and row['stop_reason'] == 'goal_tolerance', distance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*'mp4v'),
                             FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError(f'cannot write {args.out}')
    clips = 0
    try:
        card(writer, ['RASPI CAT VLN - CLOSED LOOP PILOT',
                      'Gazebo + Edge: local workstation',
                      'Remote inference: Slurm GPU on pve1ubuntu',
                      'Full episode footage, camera and overhead trace'], 4)
        for model in ORDER:
            rows = []
            for json_path in sorted(args.runs.glob(f'{model}-*.json')):
                row = json.loads(json_path.read_text())
                video = args.runs / row.get('video', '')
                if (row.get('pose_source') != 'gazebo_model_states'
                        or not video.is_file() or row.get('video_frames', 0) < 1):
                    continue
                rows.append((row, video))
            if not rows:
                continue
            successes = sum(result(row)[0] for row, _ in rows)
            card(writer, [model.upper(),
                          f'{len(rows)} routes   {successes} reached and stopped',
                          'Goal tolerance: 0.30 m   Time limit: 35 s'], 3)
            for row, video in rows:
                success, distance = result(row)
                label = 'ARRIVED' if success else 'NOT ARRIVED'
                from score import score_episode
                scored = score_episode(row)
                spl = scored['spl']
                metric = (f'Contacts: {scored["collisions"]}    SPL (0.30m): {spl:.3f}'
                          if spl is not None and scored['collisions'] is not None
                          else 'Contact/SPL data unavailable')
                instruction = textwrap.wrap(row['instruction_sent'], width=44)
                card(writer, [f'{model.upper()} / {row["id"].upper()} '
                              f'/ {SCENES[row["id"][0]]}',
                              *instruction[:2],
                              f'Outcome: {label}    Final distance: {distance:.2f} m',
                              metric], 2)
                capture = cv2.VideoCapture(str(video))
                if not capture.isOpened():
                    raise RuntimeError(f'cannot read {video}')
                frames = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if (frame.shape[1], frame.shape[0]) != SIZE:
                        frame = cv2.resize(frame, SIZE)
                    writer.write(frame)
                    frames += 1
                capture.release()
                if frames != row['video_frames']:
                    raise RuntimeError(f'frame count mismatch in {video}: '
                                       f'{frames} != {row["video_frames"]}')
                clips += 1
        card(writer, [f'END / {clips} COMPLETE ROUTE VIDEOS',
                      'Contact rate and SPL use Gazebo contacts and path oracle.',
                      'Small simulation pilot: 3-4 episodes per model.'], 5)
    finally:
        writer.release()
    print(json.dumps({'clips': clips, 'out': str(args.out)}))


if __name__ == '__main__':
    main()
