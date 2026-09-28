# Isaac Sim benchmark

`bench/src/bench` owns only simulator- and dataset-independent interfaces,
execution, artifact helpers, and the `rvln-bench` plugin host. It discovers
implementations through package entry points and does not import either adapter.
`bench/src/isaac_r2r` owns R2R-CE split loading, scoring, result aggregation,
the Isaac renderer, and discrete movement.
`bench/src/isaac_rvln` provides the Isaac ROS 2 robot bridge and converts the
three owned Gazebo pilot SDF worlds to collidable USD. New Isaac commands enter
through `rvln-bench`; the historical Gazebo tools remain Python modules under
`bench/src/bench/legacy_gazebo`.

## Install

Use Python 3.12 and [Isaac Sim 6.1's pip package](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html).
From the repository root:

```bash
cd bench
uv sync --extra isaac
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
repository. Register each scan in a JSON file, with an absolute collidable USD
path and its measured rigid transform from Habitat coordinates:

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
  --scenes /data/scene_registry.json --episode 1 --check
uv run --extra isaac rvln-bench r2r \
  --split /data/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz \
  --scenes /data/scene_registry.json --episode 1 \
  --policy-url http://127.0.0.1:8765 --max-steps 500 \
  --out runs/isaac_r2r/episode-1.json --headless
uv run --extra isaac rvln-bench summarize-r2r \
  runs/isaac_r2r/episode-*.json --out runs/isaac_r2r/summary.json
```

`inventory-r2r` counts episodes per scene and reports which source GLBs and
registered USDs exist. It checks file presence; use `r2r --check` to validate
each registered transform before launching a scene.

The policy receives `POST /reset` with `episode_id` and must echo the ID.
`POST /act` receives `episode_id`, `frame_id`, `instruction`, and `jpeg_base64`,
and must echo both IDs plus `action`: `forward`, `left`, `right`, or `stop`.
History is reset per episode. The goal is never sent to the policy. Isaac uses
224 x 224 RGB, 0.25 m forward, 15-degree turns, and PhysX raycasts before translation.
The existing RVLN embedding backend exposes `/infer`, so it needs a policy
adapter that produces these discrete `/act` actions for this track.

For a reproducible integration check with the generated corridor USD, run
`uv run --extra isaac python scripts/smoke_r2r.py` from `bench/`. The script
starts a local scripted policy, checks each JPEG observation, saves the first
frame under `runs/validation/`, and exits nonzero unless the episode completes.
Pass `--headless` only on hosts where Kit's headless renderer advances.

On the tested Slurm RTX 5070 Ti node, the renderer did not advance in Kit's
`--headless` mode, even with Vulkan available. An Ubuntu 22.04 container with
Xvfb produced RGB and depth frames. For that setup, launch Xvfb inside the
container, set `DISPLAY`, and omit `--headless` from the `rvln-bench` command.
Keep Xvfb running until the command exits. This is a measured requirement of
the tested environment, not a general requirement of Isaac Sim.

Scores carry `isaac_r2r_transfer`: they use the R2R-CE source geodesic and 3 m
stop threshold but a different renderer, camera, motion and collision model.
They are not official Habitat VLN-CE scores and cannot be combined with local
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
  --robot-urdf assets/raspicat_isaac.urdf --headless
```

The default wheel radius and separation match the pinned Raspicat description
(0.0762 m and 0.27918 m). Override them only for a different URDF.
The bridge subscribes to `/cmd_vel`, offers `/motor_power`, and publishes RGB,
depth, `/clock`, `/tf`, `/odom`, and `/sim/ground_truth_pose`. `/odom` pose and
velocity come from the simulated articulation. The existing Gazebo contact
logger, ROS TF, live episode runner, and matrix orchestration have
not been migrated. Keep the legacy Gazebo path for those measurements until an
Isaac run demonstrates equivalent topics and scoring.

`--physics-only` runs the articulation and ROS publishers without camera
rendering for a limited smoke test; add `--max-seconds 3` to end it. It does not
validate the visual benchmark.

## Validation status

The Isaac adapter is locked to Isaac Sim 6.1.0.0 and Python 3.12. On the Slurm
hosts, `uv sync --extra isaac --frozen --python 3.12` installed 172 packages,
including Isaac Sim 6.1.0.0, in a temporary node-local environment while the
project and assets remained on shared storage. CPU tests passed (20 cases,
one skipped locally because Gazebo ROS messages are absent). The RTX 5070 Ti
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
The official v1-3 `val_unseen` split was downloaded separately and passed the
adapter's parser: 1,839 episodes across 11 MP3D scans, with split SHA-256
`d173d8028537f30ab652dc5d24ead737e2b6010b6a4599f974351685710d18e8`.
The inventory found none of those scans or registered USDs in the tested data
directory. Real R2R-CE scoring still requires the matching licensed Matterport3D
scans converted to collidable USD, measured coordinate transforms, and an
R2R-compatible policy.

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
