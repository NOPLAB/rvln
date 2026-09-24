# Raspicat VLN benchmark plan

Status: gate-0/early gate-1 pilot completed on 2026-09-24; route evaluation
remains open. See `VLN_BENCHMARK_REPLAY_2026-09-24.md` for measurements and limits.
No navigation-quality ranking is claimed by this document.

## Objective and comparison groups

Measure (1) navigation quality on the same Raspicat camera episodes, (2) inference
and control latency, and (3) failure behavior. Keep model quality separate from
where a model runs. The current remote backends are `asyncvla`, `omnivla`,
`omnivla_edge`, `movla`, `navila`, and `navida`. `omnivla_edge_local` and the
mobile ONNX app are deployment variants of OmniVLA-edge, not additional models.
`dummy` is a transport/control sanity baseline, not a VLN candidate.

| Group | Runs | Eligible tasks |
| --- | --- | --- |
| General language | AsyncVLA, OmniVLA-original, OmniVLA-edge remote/local, NaVILA, NaVIDA | Text instruction routes; verify each checkpoint and adapter before inclusion |
| Template commands | All available backends, including movla Stage A | `go straight ahead`, `turn left ahead`, `turn right ahead`; displacement and heading control |
| Other goals | AsyncVLA, OmniVLA-original, OmniVLA-edge | Pose and image goals, reported separately from language routes |
| Deployment | OmniVLA-edge remote, local, and mobile ONNX where hardware/assets exist | Same recorded inputs and eligible closed-loop routes; compare latency and parity |

`movla` currently ignores image/pose goals and was trained on three text templates.
NaVILA and NaVIDA currently accept text only. Mark unsupported cells `N/A`; do not
score them as failed episodes. Published R2R/RxR scores use different simulators,
splits, embodiments, and checkpoints, so they cannot be a Raspicat ranking.
The research shortlist in `docs/memo/vln-model-survey-2026.md` (InternVLA-N1,
OmniNav, StreamVLN, Uni-NaVid, and others) enters the same protocol only after
checkpoint inference and a ROS path/action adapter pass gate 0. Models without
reproducible weights stay in a literature-only table.

## Prerequisites (gate 0)

1. Freeze repo commit, checkpoint SHA-256, model code revision, Docker digest,
   CUDA/PyTorch version, camera calibration, random seed, and ROS parameters.
   Verify each checkpoint loads and returns a valid metric Path. Missing assets
   are `not run`, not zero quality.
2. Keep Gazebo and the edge/follower local. Reserve one GPU for each remote run
   with Slurm on `pve1ubuntu`; record job ID, GPU type, memory, utilization, and
   assigned device. Run one model at a time on a GPU and warm it before timing.
3. Verify the D435-like Gazebo color stream, odometry, robot spawn, collision
   contacts, and command stop. The current empty world verifies plumbing only;
   it cannot measure language navigation.
4. Add an explicit episode ID/reset. The current remote node retains a previous
   image, and NaVILA/NaVIDA retain history across calls. Until reset exists,
   restart the inference process between episodes. Prevent a previous goal's
   embedding or Path from entering the next episode.
5. Instrument stop provenance: model stop, goal tolerance, stale observation,
   stale embedding, watchdog, collision, manual stop, and timeout. An empty Path
   caused by a stale remote response must never count as intentional success.

## Scenario set (gate 1)

Create three local Gazebo worlds using the same robot/camera/follower: (A) straight
corridor with a destination, (B) T junction with distinct visual landmarks,
(C) multi-turn route with furniture and safe clearance. Include unseen object
placements and lighting variants. Supply every episode as a versioned manifest:
world hash, start pose, destination region, reference instruction, goal modality,
random seed, shortest collision-free route length, max duration (initially 120 s),
and collision geometry. Generate the reference route on a footprint-inflated
occupancy map; inspect sample routes before publishing SPL.

Pilot with 12 episodes per model, balanced across worlds and turns. For the main
text-route comparison, use 60 paired episodes (20 per world) in a held-out
layout set. Re-run 20 stratified episodes with three seeds to estimate run
variance; expand the episode count if the paired uncertainty is too wide to
distinguish candidates. Keep prompt text, start pose, simulator seed, camera
stream, follower gains, speed limits, and time budget fixed per comparison.
Start the common-schedule track at the current edge defaults: 2 Hz observation
publishing, 224 x 224 JPEG input, 20 Hz follower, and the same 0.4 m/s and
1.0 rad/s speed caps. Record actual frame delivery and dropped observations;
do not pretend a slow backend kept pace with 2 Hz. Run a second native-schedule
track for each deployment's intended rates, labeled separately.
Record the model-specific preprocessing and history window; changing those
changes the tested policy. Use separate subsets for the three movla templates,
paraphrases, pose goals, and image goals.

## Measurements

| Layer | Required measurements |
| --- | --- |
| Backend replay | Cold load time, warm-up time, `infer()` latency p50/p95/p99, GPU peak memory and utilization, output validity, determinism, model errors |
| Transport/edge | Image capture-to-observation, observation-to-embedding, embedding-to-Path, Path-to-`cmd_vel`, and image-to-command p50/p95/p99; dropped/stale frames and timeout frequency |
| Closed-loop route | Final success within 0.30 m **and** a confirmed commanded stop, completion time, final distance, path length, collision count, intervention count, and time spent stopped/stale |
| Task diagnostics | Wrong turn, missed landmark, early stop, passed-goal stop, oscillation, route deviation, command-template displacement/heading error |

Report success rate (SR) and collision-free SR. Report SPL only after the
shortest-path oracle is validated; compute per successful episode as
`shortest_length / max(shortest_length, traveled_length)` and zero otherwise.
Report endpoint success and stop success separately. Do not compare a safety
watchdog stop with an intentional navigation stop. For latency, use monotonic
timestamps at every process boundary; synchronize clocks or measure round trip
on one host. Exclude load/warm-up from warm inference statistics but report
them separately. Include median, tail percentiles, paired bootstrap 95% CIs,
and per-episode rows, not only aggregate means.

## Execution sequence

1. **Offline replay:** Save a fixed camera/goal sequence and send identical
   observations to each eligible backend. Validate output shapes, coordinate
   frame, metric scale, stop conversion, episode reset, and latency. This does
   not establish navigation success.
2. **Open-loop command tests:** Place the robot in fixed views and evaluate
   straight/left/right commands. Check motion direction and distance against
   odometry with motors routed through the same follower.
3. **Closed-loop simulation:** Run paired text episodes, one model/job at a
   time. Reset world, robot, goal, caches, and model history for every episode.
   Stop at terminal criteria and save ROS bag, Gazebo pose/contact log, model
   metadata, and per-frame timestamps.
4. **Robustness:** Repeat a defined subset with dim lighting, blur/occlusion,
   compression, dropped frames, and injected remote delay. Report each shift
   separately from nominal performance.
5. **Physical pilot (later):** After simulation gates pass, run low-speed,
   supervised routes with an independent stop path. Report physical results
   separately from Gazebo; simulation success alone is not hardware acceptance.

## Deliverables and decision gate

Implement `bench/episodes/*.yaml`, a local episode runner/reset service, remote
replay/timing driver, and a scorer that emits `runs/<run-id>/manifest.json`,
`episodes.jsonl`, `summary.json`, plots, and failure clips. Capture all raw
timestamps and stop reasons so scores can be recomputed. A report table needs
one row per model **and deployment mode** with eligibility, checkpoint hash,
SR, collision-free SR, SPL (if valid), p95 image-to-command latency, GPU memory,
and failure counts. Choose a candidate only after comparing paired routes and
tail latency, then review representative failures before physical testing.

The [Habitat SPL definition](https://aihabitat.org/docs/habitat-lab/habitat.config.default_structured_configs.SPLMeasurementConfig.html)
is the reference for path efficiency. Habitat also defines navigation success
using a stop action near the goal; this workspace needs the explicit stop-reason
instrumentation above before claiming an equivalent score.
