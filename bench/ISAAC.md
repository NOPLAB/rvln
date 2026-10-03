# Isaac Sim benchmark

`bench/src/bench` owns only simulator- and dataset-independent interfaces,
execution, artifact helpers, and the `rvln-bench` plugin host. It discovers
implementations through package entry points and does not import either adapter.
`bench/src/isaac_r2r` owns R2R-CE split loading, scoring, result aggregation,
Habitat coordinate and action adapters, discrete Isaac rendering and movement,
and evaluation. The discrete episode adapter is separate from usim's continuous
ROS-driven robot lifecycle.
`bench/src/isaac_rvln` configures the generic usim Isaac/ROS bridge for
Raspicat and delegates SDF conversion to usim. RVLN owns the three pilot worlds. New Isaac commands enter
through `rvln-bench`; the historical Gazebo tools remain Python modules under
`bench/src/bench/legacy_gazebo`.
The `rvln` command directly registers usim's execution handler with Raspicat
defaults; usim owns configuration validation, ROS loading, contacts and cleanup.

Benchmark resources resolve under `RVLN_BENCH_ROOT`, defaulting to this source
checkout's `bench/`. They never resolve through `USIM_ROOT`. For source-only CPU
checks, the existing benchmark interpreter also works with
`PYTHONPATH=<rvln>/bench/src;<rvln>/external/usim/src` on Windows (`:` on Linux). The RVLN plugin host is `bench.cli`.

## Install

Use Python 3.12 and [Isaac Sim 6.1's pip package](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html).
From the repository root:

```bash
cd bench
# Requires the usim submodule: git submodule update --init external/usim
uv lock
uv sync --extra isaac --group dev --frozen
uv run --extra isaac rvln-bench --help
```

For standalone USD conversion, run `uv sync --extra usd`. Switching between
`usd` and `isaac` extras updates the same `.venv`; they require incompatible
NumPy versions, so sync the desired extra again before running its commands.

`pyproject.toml` selects NVIDIA's Python index. Review NVIDIA's terms and set
`OMNI_KIT_ACCEPT_EULA=YES` before launching Isaac. The ROS 2 bridge also needs
`rclpy`, `geometry_msgs`, `nav_msgs`, `sensor_msgs`, and `std_srvs` importable in
the Isaac Python environment. Isaac Sim 6.1 uses Python 3.12; custom ROS 2
interfaces must be built for that Python version. Do not source a Python 3.10
ROS installation into the Isaac process. The RVLN bridge can use Isaac Sim 6.1's
bundled Humble modules. On Linux, add
`.venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/humble/lib`
to `LD_LIBRARY_PATH` before starting Python so the bundled `rclpy` extension
can load its ROS libraries. See [NVIDIA's ROS installation guide](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_ros.html).

## R2R-CE transfer track

Obtain the published `R2R_VLNCE_v1-3` split from the [VLN-CE authors](https://jacobkrantz.github.io/vlnce/data)
and matching Matterport3D scenes under their terms. Scene assets are not in the
repository.

The [Matterport3D access instructions](https://github.com/niessner/Matterport/blob/master/README.md#data)
require signing the Terms of Use with an institutional email address and sending
the form to the dataset maintainers. After access is granted, obtain the
`habitat` GLBs using the official download script; the [VLN-CE instructions](https://github.com/jacobkrantz/VLN-CE/blob/master/README.md#scenes-matterport3d)
expect `mp3d/<scan-id>/<scan-id>.glb`. The local `val_unseen` split needs 11
matching scans. Do not substitute a different scene dataset for these IDs.

The [VLN-CE authors also publish a pretrained CMA_PM_DA_Aug policy](https://github.com/jacobkrantz/VLN-CE/blob/master/vlnce_baselines/config/r2r_baselines/README.md#pretrained-models).
Their [license notice](https://github.com/jacobkrantz/VLN-CE/blob/master/README.md#license)
classifies trained models as MP3D-derived data subject to the Matterport3D
Terms of Use and CC BY-NC-SA 3.0 US. The public checkpoint has now been
retrieved and exercised in an integration test. Its release does not replace
the required MP3D access or establish rights for an official evaluation. The
released Habitat evaluation code uses RGB plus depth; Isaac emits both and the
`/act` CMA adapter is implemented.
The [published model configuration](https://github.com/jacobkrantz/VLN-CE/blob/729d141b2ee10628061ada74dd3a5b9f70faeba5/vlnce_baselines/config/default.py)
initializes the instruction encoder from the
`R2R_VLNCE_v1-3_preprocessed` embeddings and the depth encoder from a
PointGoal checkpoint. These are separate from the minimal split inventoried
here. The [official trainer's checkpoint writer](https://github.com/jacobkrantz/VLN-CE/blob/729d141b2ee10628061ada74dd3a5b9f70faeba5/vlnce_baselines/common/base_il_trainer.py)
saves the complete policy `state_dict`, including frozen encoder parameters.
A Habitat-free inference port needs only the released CMA checkpoint and
preprocessed instruction tokens at runtime: the complete published checkpoint
now loads strictly, including the frozen encoder weights. The separate
embedding and PointGoal files are training inputs. The CMA adapter preserves
its recurrent state and previous action across steps and resets both per episode.
The minimal `val_unseen` split is not a CMA token source: 1,624 of its 1,839
episodes contain an instruction token ID outside the published 2,504-word
embedding table. The official preprocessed ZIP has now been downloaded to the
ignored shared `runs/datasets/R2R_VLNCE_v1-3_preprocessed.zip` (SHA-256
`3171f5bed90c81d7eb4db9554955b6229ec1fe96d808c9340582ba33eafdb4de`).
Its `val_unseen.json.gz` has SHA-256
`1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3`.
The `audit-cma-split` command matched all 1,839 episode IDs, scenes,
instructions, starts and goals against the minimal scoring split; it checked
367,800 tokens with maximum ID 2500 below the 2,504-word limit. The
machine-readable audit and both split hashes are at
`runs/validation/official_cma_split_audit.json` on shared storage. Repeat with:

```bash
uv run --extra isaac rvln-bench audit-cma-split \
  --split /data/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz \
  --policy-split /data/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz
```

`isaac_r2r.cma_inputs.prepare_rgbd` converts the Isaac observation to the
published CMA input shapes (224 x 224 RGB and 256 x 256 depth). It clips metric
depth to the published 0-10 m range and normalizes to [0, 1]. This preparation
is tested. The Habitat-free CMA inference graph now loads the public trained
checkpoint strictly and has run in Isaac. Official R2R-CE scores still require
the licensed MP3D scans, preprocessed instruction split, and calibration.
The R2R adapter passes audited tokens only to `/act` and records the policy
split SHA-256 with each episode. A scripted two-episode Isaac batch checked
28 RGB-D requests with different tokens per episode, and a single-episode
run exercised the direct `--policy-split` path. These are transport tests;
neither used the pretrained CMA weights.

Convert each licensed scan to USD:

```bash
uv run --extra isaac rvln-bench convert-scan \
  --glb /data/mp3d/<scan-id>/<scan-id>.glb \
  --out /data/mp3d-usd/<scan-id>.usd
```

`convert-scan` uses Isaac Sim 6.1's glTF asset importer, preserves visual
materials, rotates a Y-up import to Z-up when needed, and applies static
triangle-mesh collision to imported meshes. It writes a sidecar JSON with the
source GLB hash, mesh and triangle counts, and the number of expanded instances.
The parent command checks that the worker produced both files and matching
metadata, because Kit shutdown can return exit code 0 after a Python import
error. It rejects empty or invalid mesh geometry. Imported glTF instances are
expanded before collision authoring so their visible meshes also collide.
Inspect geometry, texture resolution, physics
raycasts, and known Habitat poses in Isaac before registering a converted scan;
the import does not establish the Habitat-to-Isaac transform automatically.
The command requires a new output path and leaves source scans unchanged.
For an asset-free importer check, run `uv run --extra isaac python scripts/smoke_scan.py`
from `bench/`. It writes a small generated GLB and its converted USD under
`runs/validation/scan-import/`, then reopens the USD and checks mesh collision.

For each converted scan, record at least three non-collinear points that can be
identified in both Habitat and Isaac. Spread them across the scan; do not use
episode goals as substitutes for surveyed correspondences. Store the measured
coordinates in a JSON file:

```json
{
  "landmarks": [
    {"habitat": [0, 0, 0], "isaac": [4, -2, 1]},
    {"habitat": [2, 0, 0], "isaac": [6, -2, 1]},
    {"habitat": [0, 3, 0], "isaac": [4, -2, 4]},
    {"habitat": [1, 1, 2], "isaac": [5, -4, 2]}
  ]
}
```

Then fit a proper, unit-scale transform. The command rejects collinear points,
reflections, and any landmark residual above 5 cm by default. Reuse the output
registry path to add further scene IDs; it refuses to replace a registered ID.
These example coordinates only illustrate the file format, not an MP3D
calibration.

```bash
uv run --extra isaac rvln-bench calibrate-r2r \
  --landmarks /data/landmarks/<scan-id>.json --scene <scan-id> \
  --usd /data/mp3d-usd/<scan-id>.usd --out /data/scene_registry.json
```

The resulting registry uses this structure:

```json
{
  "scenes": {
    "<MP3D scan ID>": {
      "usd": "/absolute/path/to/converted/scan.usd",
      "isaac_from_habitat": [
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
      ]
    }
  }
}
```

This matrix only illustrates an axis change. Measure each conversion and check
known positions and headings before scoring. Validate an episode without
starting Isaac, then run it:

```bash
uv run --extra isaac rvln-bench inventory-r2r \
  --split /data/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz \
  --scans-root /data/mp3d --scenes /data/scene_registry.json \
  --out runs/isaac_r2r/inventory.json
uv run --extra isaac rvln-bench r2r \
  --split /data/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz \
  --scenes /data/scene_registry.json --episode EXISTING_EPISODE_ID --check
uv run --extra isaac rvln-bench r2r \
  --split /data/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz \
  --policy-split /data/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz \
  --scenes /data/scene_registry.json --episode EXISTING_EPISODE_ID \
  --policy-url http://127.0.0.1:8765 --max-steps 500 \
  --out runs/isaac_r2r/episode-1.json
uv run --extra isaac rvln-bench summarize-r2r \
  runs/isaac_r2r/episode-*.json \
  --split /data/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz \
  --out runs/isaac_r2r/summary.json
```

For a complete split, run the sequential batch command. Supply a stable
`--policy-id` such as the policy checkpoint SHA-256. It launches a fresh Isaac
process per episode, records child logs and a manifest, and writes `summary.json`
only after every episode in the split has completed and passed the exact
episode/scene/hash audit. An interrupted batch can resume with `--resume` only
when the split, scene registry, converted USD hashes, policy identity, endpoint,
and run options match its original manifest. Trace and manifest JSON files are
replaced atomically; a resumed attempt gets a new log file and preserves prior
diagnostics. A native Kit exit that leaves an
episode trace at `started` is retried at most twice by default; each attempt
has a separate log. A handled episode error or timeout still stops the batch.

```bash
uv run --extra isaac rvln-bench r2r-batch \
  --split /data/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz \
  --policy-split /data/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz \
  --scenes /data/scene_registry.json \
  --policy-url http://127.0.0.1:8765 \
  --policy-id sha256:ACTUAL_CHECKPOINT_HASH \
  --out-dir runs/isaac_r2r/val_unseen
```

`inventory-r2r` counts episodes per scene and reports which source GLBs and
registered USDs exist. It checks file presence; use `r2r --check` to validate
each registered transform before launching a scene.

The policy receives `POST /reset` with `episode_id` and must echo the ID.
When `--policy-id` is supplied, the server must also return its independently
computed `policy_id` on reset; a mismatch stops the episode. Use the SHA-256
of the loaded checkpoint for the ID in a learned-policy service.
`POST /act` receives `episode_id`, `frame_id`, `instruction`, `jpeg_base64`,
`depth_f32_base64`, `depth_width`, and `depth_height`, and must echo both IDs
plus `action`: `forward`, `left`, `right`, or `stop`. Depth is a row-major,
little-endian float32 array in metres at the same 256 x 256 resolution as RGB.
When `--policy-split` is supplied, `/act` also contains that episode's audited
`instruction_tokens`; its split hash is recorded in the trace and summary.
History is reset per episode. The goal is never sent to the policy. Isaac uses
square RGB-D with a 90-degree horizontal field of view, 0.25 m forward,
15-degree turns, and PhysX raycasts before translation. The lateral raycast
footprint uses the [Habitat 0.1.7 agent's 0.1 m radius](https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/config/default.py);
its three sampled heights and kinematic translation remain an approximation of
Habitat collision and sliding. The [published R2R
task settings](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/config/vlnce_task.yaml)
use 224 x 224 RGB and 256 x 256 depth; the CMA input adapter resizes and
normalizes the Isaac observations accordingly. The existing RVLN embedding
backend exposes `/infer`; the R2R-CE track uses a separate discrete `/act`
policy server.

`python -m isaac_r2r.policy_server --checkpoint CHECKPOINT.pth \
--factory isaac_r2r.cma_model:load_model --port 8765` starts the localhost HTTP
bridge with the Habitat-free CMA inference graph. A different factory may
return an object with `reset() -> state` and
`act(rgb, depth, tokens, previous_action, state) -> (action_id, next_state)`.
The bridge hashes the checkpoint for `policy_id`, enforces ordered frames and
fixed instructions within an episode, resets recurrent state per episode,
converts RGB-D to CMA input shapes, and maps the published action order
`0=stop, 1=forward, 2=left, 3=right`. `cma_model` implements the published
ResNet50 RGB encoder, Habitat-style depth ResNet50, bidirectional instruction
LSTM, cross-modal attention, two recurrent GRUs, and greedy action head. It
loads the complete checkpoint state dict with `strict=True` and
an allowlisted pickle loader. The published trainer also stores a Habitat
`Config` object. The loader maps only that object and YACS `CfgNode` to inert
dictionary holders, permits PyTorch's tensor reconstruction globals, and
rejects other globals. A checkpoint with an unexpected serialization type
still needs review before use; strict weight loading remains mandatory.
The [pinned VLN-CE CMA source](https://github.com/jacobkrantz/VLN-CE/blob/729d141b2ee10628061ada74dd3a5b9f70faeba5/vlnce_baselines/models/cma_policy.py)
uses Habitat 0.1.7 model utilities and is not directly importable in Isaac
Sim 6's Python 3.12 environment. That environment has PyTorch 2.11 and
torchvision, but not Habitat. The synthetic GPU smoke created a 147,833,607-byte
checkpoint containing a Habitat-style configuration, strictly reloaded 521
state keys, performed two GPU inference
steps with a two-layer recurrent state, reset it, and produced a valid action
through the HTTP bridge. This synthetic check proved the wiring before the
released weights were obtained.
Run `python scripts/smoke_cma_model.py` in the Isaac environment to repeat it.
`scripts/compare_cma_reference.py` also compares this graph against the
published VLN-CE source at `729d141b2ee10628061ada74dd3a5b9f70faeba5`
and Habitat-Lab at `d6ed1c0a0e786f16f261de2beafe347f4186d0d8`. It uses
module shims for training-only imports, the same synthetic state dict, and two
inference steps. All 521 key names and shapes match. With TF32 disabled, the
CPU and GPU maximum action-logit differences were below `3e-7`, and recurrent
state differences were below `3e-6`. The initially different spatial embedding
layout was corrected using this comparison. This is source-level numerical
parity for synthetic weights, not a check of the released checkpoint or
real-scene benchmark behavior. The inference backend disables TF32 as well.
With the two public source trees checked out at the commits above, run:

```bash
python scripts/compare_cma_reference.py \
  --vlnce runs/vendor/VLN-CE --habitat runs/vendor/habitat-lab-0.1.7
```

For a reproducible integration check with the generated corridor USD, run
`uv run --extra isaac python scripts/smoke_r2r.py` from `bench/`. The script
starts a local scripted policy, checks each JPEG observation, saves the first
frame under `runs/validation/`, and exits nonzero unless the episode completes.
Pass `--headless` only on hosts where Kit's headless renderer advances.
`python scripts/smoke_r2r.py --batch --cma-tokens` additionally checks the
preprocessed-token transport and CMA RGB-D input shapes over two episodes.

The tested Slurm cluster has the `rvln-isaac6-ubuntu22` image, NVIDIA driver
library mounts, and a node-local Python 3.12 environment prepared on
`pve2ubuntu`. With the project staged on the shared mount, reproduce the
two-episode GPU check from `pve1ubuntu`:

```bash
srun -p main -w pve2ubuntu --gres=gpu:1 --time=00:15:00 \
  bash /mnt/workspace/nop/rvln/bench/scripts/isaac6_container.sh \
  /bin/bash /mnt/workspace/nop/rvln/bench/scripts/isaac6_xvfb.sh \
  /var/tmp/rvln-isaac6-container/venv/bin/python \
  /mnt/workspace/nop/rvln/bench/scripts/smoke_r2r.py --batch --cma-tokens
```

The container launcher is specific to those mounted paths and the prepared
node-local image. The project, USD, and result files stay on shared storage.

On the tested Slurm RTX 5070 Ti node, the renderer did not advance in Kit's
`--headless` mode, even with Vulkan available. An Ubuntu 22.04 container with
Xvfb produced RGB and depth frames. For that setup, launch Xvfb inside the
container, set `DISPLAY`, and omit `--headless` from the `rvln-bench` command.
Keep Xvfb running until the command exits. This is a measured requirement of
the tested environment, not a general requirement of Isaac Sim.

Scores carry `isaac_r2r_transfer`: they use the R2R-CE source geodesic and
require stop at less than 3 m, but their endpoint error is Euclidean distance.
The [Habitat 0.1.7 measure](https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py)
uses navigable geodesic distance to the goal, and its SPL numerator is the
simulator's start-to-goal geodesic. Isaac also has a different renderer, camera,
motion and collision model. These transfer scores are not official Habitat
VLN-CE scores and cannot be combined with local
Gazebo SPL@0.30m. Report evaluated episode and scene counts for subsets.

## RVLN pilot worlds and ROS bridge

Generate the owned SDF worlds with `uv run --project bench python -m bench.scenes`
from the repository
root. Import the pinned `raspicat_description` revision listed in
`raspicat.repos`. Its checked-in expanded URDF lacks collision and inertia
elements, so prepare an Isaac-specific copy. From `bench/`, convert each world
and start the bridge with the prepared robot:

```bash
uv run --extra usd rvln-bench convert-world \
  --world worlds/corridor.world --out assets/corridor.usd
uv run --extra isaac rvln-bench prepare-robot \
  --description-root ../src/raspicat_description \
  --out assets/raspicat_isaac.urdf
uv run --extra isaac rvln-bench rvln --world assets/corridor.usd \
  --robot-urdf assets/raspicat_isaac.urdf --headless \
  --contact-out runs/rvln-contacts.json
```

The default wheel radius and separation match the pinned Raspicat description
(0.0762 m and 0.27918 m). Override them only for a different URDF.
The bridge subscribes to `/cmd_vel`, offers `/motor_power`, and publishes RGB,
depth, `/clock`, `/tf`, `/odom`, and `/sim/ground_truth_pose`. `/odom` pose and
velocity come from the simulated articulation. The existing Gazebo contact
logger has an Isaac PhysX replacement: `--contact-out` writes named obstacle
contact onsets, excluding ground and self contacts. The bridge also publishes
ROS TF. The common live scorer and video combiner accept `isaac_ground_truth`
traces and reject mixed simulator reports. The legacy live episode runner and
matrix orchestration still depend on Gazebo topics and have not been migrated,
so their results remain separate.
On the Ubuntu GPU node, `scripts/Dockerfile.ros_edge` builds a ROS Humble image
for the owned `rvln_msgs` and `rvln_edge` packages. Build those packages into a
node-local `/runtime` with `scripts/build_ros_edge.sh`; Isaac stays in the
separate uv-managed Python 3.12 container. For cross-container ROS traffic,
`scripts/fastdds_udp.xml` selects UDPv4 and the two launchers set
`FASTRTPS_DEFAULT_PROFILES_FILE`. Both containers use host networking and the
same `ROS_DOMAIN_ID`; their IPC namespaces remain private. A CPU-only check is
recorded in `runs/validation/ros_udp_pair_1/`; a later Isaac GPU check also
confirmed cross-container odometry. The full live episode below now exercises
the same communication path together with inference and actuation.
Isaac 6.1 imported the original wheel cylinders with intermittent ground
contact and no useful traction. The prepared URDF uses radius-matched spherical
collision proxies for the two drive wheels and gives the chassis underside
clearance; wheel visuals remain the original meshes. This is an explicit
contact-model approximation, not a validated reproduction of Gazebo dynamics.

`--physics-only` runs the articulation and ROS publishers without camera
rendering for a limited smoke test; add `--max-seconds 3` to end it. It does not
validate the visual benchmark.

## Validation status

The Isaac adapter is locked to Isaac Sim 6.1.0.0 and Python 3.12. On the Slurm
hosts, `uv sync --extra isaac --frozen --python 3.12` installed 172 packages,
including Isaac Sim 6.1.0.0, in a temporary node-local environment while the
project and assets remained on shared storage. The latest Python 3.12 Isaac
environment passes 50 tests, with one skipped because Gazebo ROS messages are
absent, plus three subtests. The RTX 5070 Ti
node runs Ubuntu 26.04 and driver 610.57.04; [Isaac Sim 6.1 lists Ubuntu
22.04/24.04 and a tested 595.58.03 driver](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html).

On 2026-09-28, an Ubuntu 22.04 container on that node produced 224 x 224 RGB
from an ordinary USD Camera prim through Isaac Sim's `CameraSensor`. The
`RtxCamera`-created prim returned empty RGB payloads in the same probe. With
the USD camera and a Replicator capture step, one scripted synthetic corridor
episode completed with 11 JPEG policy observations, 10 forward actions, a stop,
success `true`, navigation error 0.15 m, and SPL 0.82. Its result is at
`runs/validation/synthetic_result.json` on the shared Slurm workspace.
This checks image delivery, action transport, and scoring, but uses a scripted
policy and a synthetic scene. It is not a published R2R-CE evaluation.

The Isaac RVLN bridge also ran for 21 physics steps and published 21 camera
frames in five seconds. A separate ROS 2 subscriber received 16 RGB, 15 depth,
and 33 each of odometry, clock, and TF messages during an eight-second run;
RGB messages were 480 x 640 x 3 bytes and depth messages 480 x 640 x 4 bytes.
The Isaac PhysX contact path was exercised by `scripts/smoke_isaac_contacts.py`
on the same Slurm GPU node. With a URDF freshly generated by `prepare-robot`,
ROS `/cmd_vel` moved the robot from about x=-0.03 m to x=2.43 m; the report at
`runs/validation/rvln_contact_generated_1/` contains one `red_marker` onset,
4 contact points, 4,464 odometry samples, and no ground or self-contact events.
This is a synthetic corridor integration check, not an RVLN policy score.
The same generated robot also ran in the converted junction and weave worlds.
`runs/validation/rvln_contact_junction_2/` recorded one `end_wall` onset at
x=1.71 m; `runs/validation/rvln_contact_weave_1/` recorded one `left_block`
onset at x=0.56 m. All three pilot worlds now have GPU-backed ROS command,
odometry, and obstacle-contact proof. These runs do not exercise the Edge
follower or Remote inference models.
At this stage, the Linux Python suite passed 50 tests, with one skipped and three subtests.
The ROS Humble Edge image built both owned packages and imported their Python
modules. Two independent ROS containers then exchanged a `/rvln/dds_probe`
message using the UDP-only profile; the log is at
`runs/validation/ros_udp_pair_1/`.
The real Edge lifecycle and follower were also exercised in a ROS Humble
container. A synthetic RGB image
and one latched text goal produced a 14,597-byte JPEG observation, then a
synthetic embedding led to a 10-waypoint path and a 0.4 m/s forward command.
The report is `runs/validation/edge_observation_path_3/result.json`. This
checks the Edge ROS contract only; no Isaac camera or robot participated.
The next probe used Isaac's camera and motor service and checked that the
resulting `/cmd_vel` moved the robot.
The Slurm job `2961` passed after job `2959` received Isaac `/odom` in a
separate ROS Humble container. The real-camera run recorded 81 frames, 17
received commands, one motor-enable request, and three driven physics steps;
its Edge observation contained a 2,311-byte JPEG and produced a 10-waypoint
path with a 0.4 m/s forward command. The robot moved only a few centimetres,
so this is a communication and actuation check, not a route benchmark.

`isaac_rvln.live_episode` adapts the old live ROS evaluation to Isaac's
`/sim/ground_truth_pose`. After Isaac exits, `isaac_rvln.live_results` attaches
PhysX contact events within the episode's wall-clock measurement window before
the common scorer evaluates the trace. `scripts/smoke_live_isaac.sh` runs a
synthetic embedding server through this complete live path; its result must
stay separate from real-model and official R2R-CE scores.
Slurm job `2968` completed this pilot and its post-processing without manual
intervention. The `c01` run captured 47 observations and embeddings, 229
nonempty Edge paths, 46 video frames, and 1.179 m of measured travel over
25.40 s. It timed out 1.192 m from the goal with zero collisions, SR 0 and
SPL 0. `runs/validation/live_isaac_stub_2968/summary.json` and the sidecar
video/contact/trace files contain the auditable result. The recorded
deployment is `isaac_gpu_edge_local_stub`: the simulator ran on the GPU and
the synthetic embedding server ran locally on the CPU. These numbers are not
a trained-policy benchmark. An earlier attempt had one 8-second HTTP timeout;
the synthetic server no longer writes a synchronous shared-storage log per
request, and the completed run recorded no inference errors.
The existing `omnivla-edge.pth` checkpoint also passed a Slurm GPU forward
smoke (`2974`, 8×4 finite waypoint output). A pve1 Slurm inference server and
pve2 Isaac 6.1 + ROS Edge job then completed one corridor episode (`2977`).
The server was bound to the pve1 Tailscale address and accepted only the pve2
Tailscale client. The result at
`runs/validation/live_isaac_omnivla_2977/summary.json` is explicitly
`synthetic_scene_real_model_integration`: 25 observations, 25 real-model
embeddings, 122 nonempty predicted paths, 24 recorded video frames, 1.139 m
travel, 1.312 m final goal distance, zero PhysX contact events, timeout, SR 0,
SPL 0. The episode JSON has no transport or inference errors. The scene and
instruction are synthetic, so this is not an official R2R-CE measurement.
`scripts/slurm_omnivla_edge_server.sbatch` and
`scripts/slurm_live_isaac_omnivla.sbatch` reproduce the two-node split. The
first model run used an existing pve1 Python 3.12 environment read-only plus
`uv pip install --target /var/tmp/rvln-infer-py312 regex`.
The locked `omnivla-edge-inference` extra now builds a dedicated Python 3.12
environment with `UV_PROJECT_ENVIRONMENT=/var/tmp/rvln-bench-infer-venv uv sync
--project bench --extra omnivla-edge-inference --no-dev --frozen --python 3.12`.
Current `rvln_core` and `rvln_remote` Python sources are staged under the
shared `rvln/src/` tree; the model and CLIP weights are staged under ignored
`bench/runs/datasets/omnivla-edge/` with SHA-256 values
`f93b056f124380899fce3e087a5c2b41ce4d8c07e211daad52b6ebf02e80d04d`
and `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
Direct GPU forward checks passed on both pve1 (one inference, 302 ms) and
pve2 (three inference calls: 446, 15, and 13 ms) using only this locked
environment, current sources, and staged weights. An earlier pve2 call took
65 s during transient shared-host I/O trouble, so these timings are smoke
observations, not a stable latency benchmark. A repeat closed-loop Isaac run
using the locked environment then produced
`runs/validation/live_isaac_omnivla_uv_20260929/summary.json` on pve2:
47 observations and embeddings, 229 nonempty paths, 46 video frames,
1.095 m travel, 1.308 m final goal distance, zero contact events,
zero episode errors, timeout, SR 0, SPL 0. The model and Isaac ran on the
same 5070 Ti with the ROS Edge container; Slurm was drained for an unrelated
NTP health check, so this used the previously authorized direct execution.
This remains a synthetic-scene integration result, not R2R-CE.
The official v1-3 `val_unseen` split was downloaded separately and passed the
adapter's parser: 1,839 episodes across 11 MP3D scans, with split SHA-256
`d173d8028537f30ab652dc5d24ead737e2b6010b6a4599f974351685710d18e8`.
The same split is now staged on the shared Slurm workspace at
`bench/runs/datasets/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz`. The
Linux Isaac environment independently parsed it and reproduced that hash and
episode count. Its inventory at
`bench/runs/validation/official_val_unseen_inventory.json` found none of the
11 required GLBs or registered USDs. The public CMA policy and aligned
preprocessed tokens are staged; real R2R-CE scoring still requires the matching
licensed Matterport3D scans converted to collidable USD and measured coordinate
transforms.

On the local Windows RTX 4060 Ti, `uv sync --extra isaac --frozen --python 3.12`
installed Isaac Sim 6.1.0.0 in a separate ignored environment. The generated
GLB importer smoke completed: one Y-up mesh became a Z-up, metre-scale USD
with one static triangle-mesh collider. Early synthetic R2R episodes completed
with valid JPEG and depth payloads, but later image inspection found intermittent
horizontal color stripes. Their start was also only 2.35 m from the goal,
inside the task's 3 m success radius, so those earlier SR 1.0 / SPL 0.82
scores did not show that movement was necessary. The smoke rejects synthetic
sky-color corruption and saves the offending frame.
The renderer now resets temporal accumulation and captures four Replicator
subframes after each camera move, then reads 512 x 512 RGB and depth directly
and resizes them to 256 x 256. The revised synthetic episode starts 3.8 m
from its goal; 13 forward actions move 3.25 m before stop, resulting in 0.55 m
navigation error, success, and SPL 1.0. A second episode uses the same policy
and trajectory but a different goal: it ends 3.5 m away and fails with SPL 0.
The two-episode visible GPU batch checked 28 distinct RGB-D frames, produced
SR 0.5 / SPL 0.5, and wrote an exact-coverage summary with both outcomes.
Both episodes also exercised audited per-episode CMA token transport and RGB-D
input preparation, without loading model weights.
The same updated batch completed on the Slurm RTX 5070 Ti with Isaac Sim 6.1
in the Ubuntu 22.04 container and Xvfb. Its shared-storage summary reports two
episodes, one scene, SR 0.5, SPL 0.5, zero blocked steps, and a completed
manifest. All 28 RGB-D requests arrived; the Linux Python suite passed 37
tests with one skip and three subtests. The run artifacts are under
`/mnt/workspace/nop/rvln/bench/runs/validation/synthetic_batch_91e925855a3941f68a39103bf5892937/`.
The tracked launcher reproduced SR 0.5 / SPL 0.5 in a second Slurm run under
`synthetic_batch_c332464fb4524981bfa8678bca54dfb2/`. The generated GLB
importer smoke also completed on that GPU node: one Y-up mesh was converted to
USD with one collider. These checks still do not validate MP3D scene import or
coordinate calibration.
The new stateful HTTP policy bridge then completed the same two-episode GPU
batch with 28 ordered requests and a checkpoint-derived identity. Its manifest
is `completed` and its summary reports SR 0.5 / SPL 0.5 at
`synthetic_batch_f9f3e5b7bb704ab2acae822b6448738b/`. This run still used
a scripted backend, not CMA weights.
As an additional importer stress test, a [CC0 furnished-room GLB](https://3dassets.dev/assets/bedroom-and-living-room-furniture-furnis-0989213f-starter-scene)
was run through the shared-storage Isaac 6.1 GPU workflow. The original GLB's
required `KHR_mesh_quantization` extension imported as 140 empty meshes despite
the converter reporting success. Geometry validation now rejects it with CLI
exit code 2 and no output USD. A one-off `gltf-transform dequantize` preprocessing
produced a GLB that converted successfully: 161 nonempty USD meshes, 86,460
vertices, 39,772 triangles, 21 expanded instances, and collision on every mesh.
The source GLB's triangle count differs, so these counts do not establish exact
mesh fidelity. The converted room needed an added dome light to make RGB visible;
the scene registry accepts `dome_light_intensity`, and `calibrate-r2r` has a
`--dome-intensity` option for a measured value. With intensity 800, a rendered
256 x 256 JPEG visibly showed the furniture, had mean RGB 153.32 and standard
deviation 74.76, and yielded 39,541 valid depth pixels from 1.00 to 4.54 m.
A forward command was blocked by furniture collision. The reproducible probe is
`scripts/smoke_imported_scene.py`; it rejects dark or flat frames. This CC0 room
checks the importer and renderer only. It cannot replace any MP3D scan in the
official R2R-CE split or establish a comparable score.
The full CMA server and `rvln-bench r2r` path also ran together in this room.
`scripts/smoke_cma_scene.py` made a temporary synthetic checkpoint, forced its
action head to choose forward, and ran three RGB-D policy requests through a
separate policy process and Isaac process. The completed trace at
`runs/validation/cma_room_integrated_2/result.json` records three forward
actions, three furniture-blocked steps, no stop, and success 0 / SPL 0. This
checks checkpoint loading, scene rendering, policy transport, recurrent steps,
collision, and result persistence in one run. Its weights and episode are
synthetic, so the zero score is not an R2R-CE performance measurement.
With `--batch`, the same script kept one CMA server alive across two episodes
and launched separate Isaac processes through `rvln-bench r2r-batch`. The
Slurm GPU run at `runs/validation/cma_room_batch_1/` finished with a completed
manifest, two episode traces and an exact-coverage summary: six forward actions,
six furniture-blocked steps, SR 0 and SPL 0. This exercises per-episode policy
reset, process restart, provenance hashes, and batch aggregation with the
actual CMA graph and RGB-D observations. The head was still forced to forward
with synthetic weights; these values are integration results only.
The [official VLN-CE baseline README](https://github.com/jacobkrantz/VLN-CE/blob/master/vlnce_baselines/config/r2r_baselines/README.md)
also publishes the `CMA_PM_DA_Aug` checkpoint. It was downloaded to the
ignored shared `runs/datasets/cma_pm_da_aug.pth` (147,827,650 bytes, SHA-256
`06278fdbb52ad8c453b404dacb079fb353252056314b1c567a381df77e6827a3`).
The Habitat-free loader accepted its legacy Python `set` and defaulted one
non-learned BatchNorm counter absent from the file; all 520 saved state entries
matched shape and GPU inference ran. With this published checkpoint,
`scripts/smoke_cma_scene.py --checkpoint runs/datasets/cma_pm_da_aug.pth --batch`
completed two episodes in the CC0 furnished room. The manifest at
`runs/validation/cma_published_room_batch_1/batch/` reports exact coverage,
six RGB-D policy actions (all right turns), SR 0 and SPL 0. The room and
instruction tokens are synthetic and outside the model's official evaluation
distribution, so these numbers are an integration check only.
After matching the raycast footprint to the published Habitat agent radius,
the two-episode GPU regression passed again with 28 RGB-D policy requests,
zero blocked steps in the corridor, and scripted SR 0.5 / SPL 0.5. The results
remain synthetic and do not validate navigation inside MP3D.
One earlier two-episode attempt exited natively during Kit startup with
Windows code `0xc000070a`, before camera capture. The batch now has a bounded
retry for incomplete native exits; its recovery path passed a focused test,
while the latest GPU batch completed without needing a retry. This remains a
scripted synthetic check, and the local renderer has not been validated on
MP3D scenes. A headless R2R
retry on this host produced empty camera frames, so use the visible renderer
for this smoke. The local Python 3.12 suite passes 38 tests with one skipped.
A five-second headless RVLN bridge run completed
11 physics steps and 11 camera frames after retrying an initially empty camera
capture. These checks validate the local integration, not a real MP3D scene or
model policy. The local launch needed an inaccessible `Unity\bin` entry removed
from `PATH`; this was an environment-specific DLL search error.

The following results are historical Isaac Sim 5.0 checks and do not validate
the 6.1 visual benchmark. The local Windows environment installed 5.0.0 through `uv`. The
standalone USD converter produced and reopened collidable USD for all three
pilot worlds; each has four collision prims. Robot preparation produced a
19-link, 18-joint URDF with collision and inertia on the physical links. CPU
tests cover shared execution, split scoring, pilot SDF parsing, robot
preparation, and wheel kinematics (20 tests, one skipped because Gazebo ROS
messages are unavailable). Isaac imported the URDF without missing meshes.
With velocity drives configured, a direct physics probe moved the robot 0.26 m
in the corridor over 180 steps. The `--physics-only` RVLN command completed
324 physics steps while spinning its internal ROS node. Cross-process ROS topic
and service discovery did not work in this Windows environment, even between
two standalone ROS processes; external ROS integration is therefore unverified.

Full Isaac startup still logs a native RTX access violation shortly after
`app ready`. The process continues when crash reporting is disabled, but both
R2R and RVLN cameras returned empty frames after 30 render/capture attempts.
The synthetic R2R episode therefore recorded `status: error` before calling a
local policy, and no visual benchmark or official R2R-CE score has completed.
The cause of the renderer failure has not been established. This machine has
an RTX 4060 Ti, below [Isaac Sim 5.0's minimum RTX 4080 GPU](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html).
Results are written with `status: started`, `error`, or `completed` before
Isaac shutdown, and aggregation accepts only completed episodes. Kit shutdown
can return process exit code 0 even after a Python error, so inspect the result
status (or RVLN's printed status) rather than relying on the exit code here.
The R2R-CE transfer track also needs separately licensed scene and episode
assets.

On the Slurm host, `uv sync --extra isaac --frozen --python 3.11` completed in
the shared workspace. The RTX 5070 Ti node runs Ubuntu 26.04, which lacks
`libGLU.so.1` and the `libxml2.so.2` ABI required by Isaac Sim 5.0; their
Ubuntu packages and the matching ICU runtime were extracted under a user-owned
shared directory for this run. Isaac then started and entered `observe()`, but
segfaulted on the first rendered step. PathTracing gave the same result as the
default renderer. The synthetic result remained `status: started`, and the
aggregator correctly rejected it as incomplete. The RTX 5090 Slurm node was
unavailable at that time due to a 1 MiB RealMemory mismatch. Slurm registration
has since been repaired.
