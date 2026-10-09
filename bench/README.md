# VLN benchmark harness

The Isaac Sim benchmark is under `bench/src/`: `bench` owns dataset-independent
contracts, execution, and the `rvln-bench` plugin host. `isaac_r2r` owns its
R2R protocol and scoring; `isaac_rvln` owns Raspicat preparation and RVLN protocol adapters. Generic
physics, world conversion, and ROS bridges are imported from `usim`.
The `rvln` command uses usim's CLI handler directly with Raspicat defaults;
configuration validation and engine lifecycle are owned by usim. There is no
benchmark-specific continuous simulator or ROS bridge wrapper. R2R's discrete
camera and action adapter stays in `isaac_r2r` because it implements the
evaluation protocol rather than a ROS-driven robot.

The package stays `rvln-bench`, with its own `rvln_bench.plugins` discovery.
`bench/pyproject.toml` installs usim editable from the `external/usim`
submodule (`git submodule update --init external/usim`). For source-only CPU
checks on Windows, the existing `bench/.venv/Scripts/python.exe` also works with
`PYTHONPATH=<rvln>/bench/src;<rvln>/external/usim/src`.
Benchmark resources use `RVLN_BENCH_ROOT` (default: this checkout's `bench/`),
never `USIM_ROOT`. Installed wheels require `RVLN_BENCH_ROOT`. See
[ISAAC.md](ISAAC.md) for setup,
assets, commands, and current runtime limitations. The Gazebo workflow below
remains the legacy path until the Isaac ROS bridge is validated end to end.

`docs/bench/VLN_BENCHMARK_PLAN.md` defines the evaluation protocol. The current
scripts cover camera capture, fixed-image GPU replay, a one-run live
Remote/Edge bridge, and a route scorer. Results and limitations are in
`docs/bench/VLN_BENCHMARK_REPLAY_2026-09-24.md`.

## Capture (local Gazebo and Edge)

Generate the three worlds with `uv run --project bench python -m bench.scenes`.
On a local ROS 2
Humble host or simulator container, launch `rvln_bringup sim.launch.py` with
`world:=/workspace/bench/worlds/corridor.world`, `adapter_kind:=omnivla`, and
`gui:=false`. Verify `/model_states` and `/camera/color/image_raw`, then run:

```bash
PYTHONPATH=/workspace/bench/src python3 -m bench.legacy_gazebo.capture \
  --out /workspace/bench/runs/corridor_capture \
  --count 20 --interval 0.5
```

The Raspicat simulator starts with motor power off. For a *live driving* test,
call `ros2 service call /motor_power std_srvs/srv/SetBool '{data: true}'` after
the robot spawns, then turn it off after the test. Fixed-camera replay does not
need motor power.

## GPU replay (pve1ubuntu)

Sync the owned `bench/` and `src/rvln_*` sources with `rsync` through WSL to
`/mnt/workspace/nop/raspicat_vla`. Keep the capture JPEGs under
`bench/runs/<scene>_capture/`. Checkpoint roots are listed in
`scripts/submit_replays.sh`. Use pinned upstream sources and apply the two patches in
`bench/patches/` to the AsyncVLA and NaVILA checkouts. The AsyncVLA patch
avoids training-only imports. The NaVILA patch makes FlashAttention optional
for the released SigLIP checkpoint. These patches change the upstream working
trees and should be recorded with each run.

```bash
git -C external/AsyncVLA apply ../../bench/patches/asyncvla-inference-only.patch
git -C bench/upstream/NaVILA apply --unidiff-zero \
  ../../patches/navila-optional-intern-flash.patch
```

```bash
# On pve1ubuntu, from /mnt/workspace/nop/raspicat_vla:
bash bench/scripts/submit_replays.sh corridor 'go straight ahead'
bash bench/scripts/submit_replays.sh junction 'turn left ahead'
bash bench/scripts/submit_replays.sh weave 'go straight ahead'
```

Each call submits six one-GPU Slurm jobs. Once complete, copy their JSON files
back under `bench/runs/` and regenerate the versioned aggregate:

```bash
uv run --project bench python -m bench.summarize_replays \
  --out bench/results/replay_2026-09-24.json
```

`bench/runs/` is ignored because it contains raw frames and per-run logs.
`bench/results/` holds the small summary and SHA-256 checkpoint manifest.
Do not use replay outputs as route success or SPL.

## Live Remote/Edge connectivity check

`scripts/slurm_live.sbatch` starts one GPU backend as a temporary HTTP service on the
GPU host's Tailnet address. `bench.live_bridge` stays with Gazebo and Edge on the
local machine, listens to `/rvln/observation`, and publishes matched replies
to `/rvln/remote_embedding`. For example:

```bash
# pve1ubuntu (GPU allocated by Slurm):
sbatch bench/scripts/slurm_live.sbatch --backend omnivla_edge \
  --checkpoint models/omnivla-edge/omnivla-edge.pth \
  --bind <pve1ubuntu-tailnet-ip> --port 8765

# Local simulator, with rvln_bringup using adapter_kind:=omnivla:
PYTHONPATH=/workspace/bench/src python3 -m bench.live_bridge \
  --url http://<pve1ubuntu-tailnet-ip>:8765 --duration 40
```

This bridge currently supports text goals and one episode per server process.
Stop the Slurm job after the test. Route benchmarking needs a reset-aware
runner, stop provenance, contacts, and a checked shortest-path oracle.

## Real NaVIDA executor

`bench.live_navida` is the real-robot executor recovered from the October 1–5
NaVIDA tests. It is separate from the Gazebo episode runner. It consumes the
first metric action token and then executes the remaining actions from
`diagnostics.raw_response` in order. Fused `/odom` pose and angular velocity
close the motion loop; a completed primitive is not executed again just because
its response remains available.

The executor checks motor-power acknowledgements and measured cessation before
using a new camera exposure. Edge preserves the camera message's acquisition
timestamp in `/rvln/observation`; publication time must not replace it. Use a
ROS camera source with valid image header timestamps for this executor. The
in-process V4L2 path has no ROS acquisition timestamp and is not an input path
for these post-stop NaVIDA checks.

On the ROS 2 Humble edge host, with the remote NaVIDA server already running:

```bash
PYTHONPATH="$PWD/bench/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m bench.live_navida \
  --url http://<inference-host>:8765 --odom-topic /odom \
  --max-v 0.1 --max-w 0.35
```

This node owns `/cmd_vel` and `/rvln/follower_stop`; do not run it alongside the
generic path follower. Goal changes and forced stops invalidate in-flight
responses. Keep motor-power and ESTOP operation under the existing robot
operating procedure; this command does not change ESTOP state. When stopping an
executor inside a container, signal the actual `bench.live_navida` process,
not only the shell that launched it.

The remote-specific restored Slurm script is covered by the existing
`bench/scripts/slurm_live.sbatch`: select `--backend navida`, the same checkpoint,
and set `RVLN_BENCH_PYTHON` to the prepared NaVIDA environment. GPU inference
still requires a Slurm allocation.

Run the recovered CPU regression checks on a ROS 2 Humble host or test container:

```bash
bash bench/scripts/run_navida_qa.sh
```

The checks use synthetic odometry clocks, an event-driven local HTTP server,
and a motor-service fake. They build the ROS interfaces and do not launch
hardware or a GPU model. `ROS_DOMAIN_ID=77` and `ROS_LOCALHOST_ONLY=1` isolate the
test graph. Real camera-height and image-geometry experiment outputs stay under
`bench/runs/validation/`, outside versioned application source.
