# VLN closed-loop Gazebo pilot and complete run video

The subsequent 23-run [contact and SPL rerun](VLN_CONTACT_SPL_2026-09-25.md)
includes obstacle contacts, a footprint-aware shortest-route oracle, a new
combined video, and per-model inference-integrity checks. Results below are
the earlier baseline runs.

Date: 2026-09-25 (JST). This is a small Raspicat simulation pilot, not a
held-out model ranking. The local workstation ran Gazebo Classic, the D435-like
handrail camera, the ROS Edge adapter and the path follower. `pve1ubuntu` ran
the real checkpoint inference servers in one-GPU Slurm jobs 2508–2513 on an
NVIDIA GeForce RTX 5090. Each job was cancelled after its model's episodes;
the Slurm queue was empty at the end. The committed checkpoint SHA-256 manifest
is [here](../../bench/results/checkpoints_2026-09-24.json).

## Watch the runs

The [edited full-run video](../../bench/runs/vln_all_models_2026-09-25.mp4) joins
all 23 complete clips in model order. It has a title card for each model and
episode, shows the instruction and scored outcome, and preserves every captured
frame of the runs. Duration: 12 min 36 s; 960 × 480 at 2 fps. The left view
is the robot camera; the right view shows its Gazebo ground-truth trajectory,
current position, goal, distance, and command. Camera samples were captured at
the 2 Hz observation rate, so this is complete sampled footage, not a 30 fps
recording of every Gazebo render frame. The individual MP4s and raw JSON traces
are under ignored `bench/runs/` and are local workstation artifacts.

| Model | Corridor | Left junction | Right junction | Weave |
| --- | --- | --- | --- | --- |
| AsyncVLA | [c01](../../bench/runs/asyncvla-c01.mp4) | [j01](../../bench/runs/asyncvla-j01.mp4) | [j02](../../bench/runs/asyncvla-j02.mp4) | [w01](../../bench/runs/asyncvla-w01.mp4) |
| OmniVLA-original | [c01](../../bench/runs/omnivla-c01.mp4) | [j01](../../bench/runs/omnivla-j01.mp4) | [j02](../../bench/runs/omnivla-j02.mp4) | [w01](../../bench/runs/omnivla-w01.mp4) |
| OmniVLA-edge | [c01](../../bench/runs/omnivla_edge-c01.mp4) | [j01](../../bench/runs/omnivla_edge-j01.mp4) | [j02](../../bench/runs/omnivla_edge-j02.mp4) | [w01](../../bench/runs/omnivla_edge-w01.mp4) |
| movla Stage A templates | [c01](../../bench/runs/movla-c01.mp4) | [j01](../../bench/runs/movla-j01.mp4) | [j02](../../bench/runs/movla-j02.mp4) | Outside template track |
| NaVILA | [c01](../../bench/runs/navila-c01.mp4) | [j01](../../bench/runs/navila-j01.mp4) | [j02](../../bench/runs/navila-j02.mp4) | [w01](../../bench/runs/navila-w01.mp4) |
| NaVIDA | [c01](../../bench/runs/navida-c01.mp4) | [j01](../../bench/runs/navida-j01.mp4) | [j02](../../bench/runs/navida-j02.mp4) | [w01](../../bench/runs/navida-w01.mp4) |

## Protocol and result

The four general-language tasks used exactly the same pilot manifest entries
`c01`, `j01`, `j02`, and `w01` for each of five models. movla used its three
trained command templates (`go straight ahead`, `turn left ahead`, `turn right
ahead`) on the corresponding three geometries; it is a separate track. Each
run restarted Gazebo and Edge, rebuilt the remote backend to clear history,
and used the same start pose, 0.4 m/s speed cap, 2 Hz observations and 20 Hz
follower. The time limit was 35 s. Arrival means the Gazebo **model pose** ended
within 0.30 m of the goal and the follower's zero command was confirmed after
the goal-tolerance trigger. This is **goal-triggered arrival**, not an
autonomous model stop action. Wheel odometry is recorded separately and is not
used for scoring.

| Model and track | Reached / runs | SR | Median final distance | Median travel | Observation-to-embedding p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AsyncVLA, general | 1 / 4 | 25% | 1.38 m | 2.27 m | 147 ms |
| OmniVLA-original, general | 0 / 4 | 0% | 1.20 m | 1.80 m | 102 ms |
| OmniVLA-edge, general | 2 / 4 | 50% | 0.79 m | 2.07 m | 52 ms |
| NaVILA, general | 1 / 4 | 25% | 13.54 m | 12.58 m | 400 ms |
| NaVIDA, general | 0 / 4 | 0% | 2.01 m | 0.01 m | 360 ms |
| movla Stage A, templates | 0 / 3 | 0% | 2.36 m | 2.52 m | 87 ms |

Per-episode scores, distances, stops, frame counts, latency samples and video
names are in [the scored JSON](../../bench/results/live_pilot_2026-09-25.json).
The individual raw traces also retain every sampled pose and command. Notable
failures in the video: OmniVLA-edge reached the right junction but turned toward
the wrong side on the left instruction; movla overshot the left destination;
NaVILA left the intended area in the junction and weave scenes; NaVIDA mostly
rotated near its start. The OmniVLA-edge weave run recorded one HTTP timeout.

Only one run was made per model and task. A previous unrecorded AsyncVLA
corridor trial succeeded, while one discarded trial overshot; this is evidence
of run variance, not an additional scored episode. Layouts, lighting and
simulator seeds were not varied, and the 60 held-out paired episodes are still
needed for a comparison with useful uncertainty. One NaVILA callback caused
an actual run duration of 37.5 s against the nominal 35 s timer. Collision
contacts and a footprint-aware shortest-path oracle are not validated, so
collision-free SR and SPL are unavailable. These results do not establish
physical robot performance.

## Reproduce and inspect

`bench/run_live_matrix.py` controls the workstation's Docker Gazebo/Edge
container and one remote Slurm backend at a time. `bench/live_episode.py`
records the camera, Gazebo model pose, command, remote round trips, and stop
reason. `bench/summarize_live.py` scores only traces whose pose source is
`gazebo_model_states` and checks every individual video. `bench/make_live_video.py`
creates the joined MP4 and verifies every input clip's frame count. The earlier
fixed-image inference replay and checkpoint details are in
[the replay report](VLN_BENCHMARK_REPLAY_2026-09-24.md).
