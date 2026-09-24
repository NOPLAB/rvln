# VLN benchmark harness

`docs/VLN_BENCHMARK_PLAN.md` defines the evaluation protocol. The current
scripts cover camera capture, fixed-image GPU replay, a one-run live
Remote/Edge bridge, and a route scorer. Results and limitations are in
`docs/VLN_BENCHMARK_REPLAY_2026-09-24.md`.

## Capture (local Gazebo and Edge)

Generate the three worlds with `python bench/scenes.py`. On a local ROS 2
Humble host or simulator container, launch `rvln_bringup sim.launch.py` with
`world:=/workspace/bench/worlds/corridor.world`, `adapter_kind:=omnivla`, and
`gui:=false`. Verify `/model_states` and `/camera/color/image_raw`, then run:

```bash
python3 /workspace/bench/capture.py --out /workspace/bench/runs/corridor_capture \
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
`submit_replays.sh`. Use pinned upstream sources and apply the two patches in
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
bash bench/submit_replays.sh corridor 'go straight ahead'
bash bench/submit_replays.sh junction 'turn left ahead'
bash bench/submit_replays.sh weave 'go straight ahead'
```

Each call submits six one-GPU Slurm jobs. Once complete, copy their JSON files
back under `bench/runs/` and regenerate the versioned aggregate:

```bash
python bench/summarize_replays.py --out bench/results/replay_2026-09-24.json
```

`bench/runs/` is ignored because it contains raw frames and per-run logs.
`bench/results/` holds the small summary and SHA-256 checkpoint manifest.
Do not use replay outputs as route success or SPL.

## Live Remote/Edge connectivity check

`slurm_live.sbatch` starts one GPU backend as a temporary HTTP service on the
GPU host's Tailnet address. `live_bridge.py` stays with Gazebo and Edge on the
local machine, listens to `/rvln/observation`, and publishes matched replies
to `/rvln/remote_embedding`. For example:

```bash
# pve1ubuntu (GPU allocated by Slurm):
sbatch bench/slurm_live.sbatch --backend omnivla_edge \
  --checkpoint models/omnivla-edge/omnivla-edge.pth \
  --bind <pve1ubuntu-tailnet-ip> --port 8765

# Local simulator, with rvln_bringup using adapter_kind:=omnivla:
python3 /workspace/bench/live_bridge.py \
  --url http://<pve1ubuntu-tailnet-ip>:8765 --duration 40
```

This bridge currently supports text goals and one episode per server process.
Stop the Slurm job after the test. Route benchmarking needs a reset-aware
runner, stop provenance, contacts, and a checked shortest-path oracle.
