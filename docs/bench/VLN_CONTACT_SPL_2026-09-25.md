# Gazebo VLN contact and SPL rerun

This rerun uses the six checkpoint backends in the local closed-loop Gazebo
pilot. Gazebo and Edge run on the workstation; each remote inference backend
runs in a separate Slurm GPU reservation on `pve1ubuntu`. The camera image and
language goal pass through the normal ROS observation and Edge path topics.

## Metric definitions

- **Contact episode rate:** fraction of episodes with at least one contact
  onset against a named scene obstacle. Gazebo contact sensors cover the base,
  handrail camera assembly, caster, and both wheels. Ground and robot self
  contacts are filtered. Repeated messages during a continuous contact are
  one event; a gap over 0.5 s starts another event.
- **Arrival success:** Gazebo ground-truth endpoint lies within 0.30 m of the
  goal and the local follower confirms a zero command after the goal tolerance
  trigger. This is a goal-triggered stop, not a model stop prediction.
- **SPL@0.30m:** zero for an unsuccessful run; otherwise shortest route length
  divided by the larger of shortest route and measured traveled length. The
  shortest route reaches any free point within the 0.30 m goal disk. `Theta*`
  uses a 0.0125 m grid and a 0.34 m circular robot footprint against verified
  world collision boxes. This is a footprint approximation and differs from a
  Habitat navigation mesh SPL.

The `pilot.json` oracle records each world SHA-256 and route length. The run
checks the current world hash before the robot moves. Contact logging must
receive all seven sensor streams or the run is invalid. Raw episode JSON keeps
all contact events, inference timing and diagnostics, ROS path samples, commands,
Gazebo pose samples, and the corresponding complete 2 Hz camera/trajectory MP4.

## Inference integrity checks

`bench/validate_live.py` checks checkpoint version presence, all inference
shapes and frame IDs, finite outputs, instruction transmission, action-text
decoding for NaVILA and NaVIDA, remote waypoints against local ROS Paths,
local motor commands, and resulting Gazebo motion. AsyncVLA is checked for
nonzero visual feature output and nonempty local Edge paths. MoVLA receives
measured pose history and odometry velocity; its embodiment uses turtlebot2
statistics because Raspicat was not in its training set. Its three template
instructions form a separate track. Passing these checks establishes that the
program exercised the intended model and control chain for these runs. It does
not establish generalization or equivalent behavior on physical hardware.

## Results and video

All 23 accepted episodes have seven live contact streams, a world-hash-checked
shortest route, nonempty camera video, and no recorded inference or ROS errors.
The strict [inference audit](../../bench/results/live_inference_audit_2026-09-25.json)
passed all 23 episodes. Individual route details and bootstrap intervals are
in the [scored JSON](../../bench/results/live_contacts_spl_2026-09-25.json).

| Model and track | Arrived / runs | Contact episodes | Contact rate | SPL@0.30m |
| --- | ---: | ---: | ---: | ---: |
| AsyncVLA, general text | 1 / 4 | 3 / 4 | 75% | 0.244 |
| OmniVLA-original, general text | 0 / 4 | 4 / 4 | 100% | 0.000 |
| OmniVLA-edge, general text | 2 / 4 | 2 / 4 | 50% | 0.426 |
| NaVILA, general text | 1 / 4 | 0 / 4 | 0% | 0.236 |
| NaVIDA, general text | 0 / 4 | 0 / 4 | 0% | 0.000 |
| MoVLA Stage A, command templates | 0 / 3 | 3 / 3 | 100% | 0.000 |

There were 12 obstacle-contact events in total. NaVILA's zero contact rate
does not mean safe navigation: two junction routes and the weave route went far
outside the intended route area. NaVIDA's zero contact rate reflects mostly
turning near the start. MoVLA used actual measured motion history; its three
template tasks are not directly comparable to general-text instructions.

The [edited 23-run video](../../bench/runs/contact_2026-09-25/vln_all_models_contacts_spl.mp4)
is 12 min 39.5 s at 2 fps and shows every captured episode frame, with readable
cards for the instruction, arrival, contact count, and SPL. The left half shows
the robot camera and the right half shows the Gazebo trajectory, goal, distance,
and command. These are all **2 Hz sampled frames**, rather than every simulator
render frame. Videos and raw traces are ignored local artifacts in
`bench/runs/contact_2026-09-25/`.

| Model | Corridor | Left junction | Right junction | Weave |
| --- | --- | --- | --- | --- |
| AsyncVLA | [c01](../../bench/runs/contact_2026-09-25/asyncvla-c01.mp4) | [j01](../../bench/runs/contact_2026-09-25/asyncvla-j01.mp4) | [j02](../../bench/runs/contact_2026-09-25/asyncvla-j02.mp4) | [w01](../../bench/runs/contact_2026-09-25/asyncvla-w01.mp4) |
| OmniVLA-original | [c01](../../bench/runs/contact_2026-09-25/omnivla-c01.mp4) | [j01](../../bench/runs/contact_2026-09-25/omnivla-j01.mp4) | [j02](../../bench/runs/contact_2026-09-25/omnivla-j02.mp4) | [w01](../../bench/runs/contact_2026-09-25/omnivla-w01.mp4) |
| OmniVLA-edge | [c01](../../bench/runs/contact_2026-09-25/omnivla_edge-c01.mp4) | [j01](../../bench/runs/contact_2026-09-25/omnivla_edge-j01.mp4) | [j02](../../bench/runs/contact_2026-09-25/omnivla_edge-j02.mp4) | [w01](../../bench/runs/contact_2026-09-25/omnivla_edge-w01.mp4) |
| NaVILA | [c01](../../bench/runs/contact_2026-09-25/navila-c01.mp4) | [j01](../../bench/runs/contact_2026-09-25/navila-j01.mp4) | [j02](../../bench/runs/contact_2026-09-25/navila-j02.mp4) | [w01](../../bench/runs/contact_2026-09-25/navila-w01.mp4) |
| NaVIDA | [c01](../../bench/runs/contact_2026-09-25/navida-c01.mp4) | [j01](../../bench/runs/contact_2026-09-25/navida-j01.mp4) | [j02](../../bench/runs/contact_2026-09-25/navida-j02.mp4) | [w01](../../bench/runs/contact_2026-09-25/navida-w01.mp4) |
| MoVLA | [c01](../../bench/runs/contact_2026-09-25/movla-c01.mp4) | [j01](../../bench/runs/contact_2026-09-25/movla-j01.mp4) | [j02](../../bench/runs/contact_2026-09-25/movla-j02.mp4) | Outside template track |

This is one episode per model and route, so the rates are descriptive pilot
figures with wide uncertainty. Gazebo contact geometry and the circular
footprint oracle are approximations. The test does not establish performance
on a physical robot or a held-out navigation benchmark.
