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

Use Python 3.11 and [Isaac Sim 5.0's pip package](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html).
From the repository root:

```bash
cd bench
uv sync --extra isaac
uv run --extra isaac rvln-bench --help
```

For standalone USD conversion, run `uv sync --extra usd`. Switching between
`usd` and `isaac` extras updates the same `.venv`; sync the desired extra again
before running its commands.

`pyproject.toml` selects NVIDIA's Python index. Review NVIDIA's terms and set
`OMNI_KIT_ACCEPT_EULA=YES` before launching Isaac. The ROS 2 bridge also needs
`rclpy`, `geometry_msgs`, `nav_msgs`, `sensor_msgs`, and `std_srvs` importable in
the Isaac Python environment. Isaac Sim 5.0 bundles Python 3.11-compatible ROS 2
Humble libraries and the bridge enables them; do not source a Python 3.10 ROS
installation into the Isaac process. See [NVIDIA's ROS installation guide](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_ros.html).

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
uv run --extra isaac rvln-bench r2r --split /data/val_unseen.json.gz \
  --scenes /data/scene_registry.json --episode 1 --check
uv run --extra isaac rvln-bench r2r --split /data/val_unseen.json.gz \
  --scenes /data/scene_registry.json --episode 1 \
  --policy-url http://127.0.0.1:8765 --max-steps 500 \
  --out runs/isaac_r2r/episode-1.json --headless
uv run --extra isaac rvln-bench summarize-r2r \
  runs/isaac_r2r/episode-*.json --out runs/isaac_r2r/summary.json
```

The policy receives `POST /reset` with `episode_id` and must echo the ID.
`POST /act` receives `episode_id`, `frame_id`, `instruction`, and `jpeg_base64`,
and must echo both IDs plus `action`: `forward`, `left`, `right`, or `stop`.
History is reset per episode. The goal is never sent to the policy. Isaac uses
224 x 224 RGB, 0.25 m forward, 15-degree turns, and PhysX raycasts before translation.
The existing RVLN embedding backend exposes `/infer`, so it needs a policy
adapter that produces these discrete `/act` actions for this track.

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

## Validation status

The local Windows environment installed Isaac Sim 5.0.0 through `uv`. The
standalone USD converter produced and reopened collidable USD for all three
pilot worlds. CPU tests cover shared execution, split scoring, pilot SDF
parsing, robot preparation, and wheel kinematics (19 tests, one skipped because
Gazebo ROS messages are unavailable). Isaac startup crashed inside
`rtx.scenedb.plugin.dll` before loading benchmark code, so rendering,
simulation, ROS bridge behavior, and policy evaluation remain unverified.
The R2R-CE transfer track also needs separately licensed scene and episode
assets. No official R2R-CE score has been produced.
