# raspicat-vla

[![ci-lint](https://github.com/NOPLAB/raspicat-vla/actions/workflows/ci-lint.yml/badge.svg?branch=main)](https://github.com/NOPLAB/raspicat-vla/actions/workflows/ci-lint.yml)
[![ci-flutter](https://github.com/NOPLAB/raspicat-vla/actions/workflows/ci-flutter.yml/badge.svg?branch=main)](https://github.com/NOPLAB/raspicat-vla/actions/workflows/ci-flutter.yml)
[![ci-web](https://github.com/NOPLAB/raspicat-vla/actions/workflows/ci-web.yml/badge.svg?branch=main)](https://github.com/NOPLAB/raspicat-vla/actions/workflows/ci-web.yml)

ROS2 Humble nodes for running Vision-Language-Action (VLA) navigation models on
the Raspberry Pi Cat (rt-net `raspicat`).

The repository defines a model-agnostic edge / remote split: the lightweight
edge runs on the robot, a remote workstation hosts the heavy VLA policy, and
ROS 2 topics carry compressed observations and action embeddings. The
same interface supports multiple backends — the dummy backend (for CI / MVP),
[AsyncVLA](https://asyncvla.github.io/),
[OmniVLA](https://omnivla-nav.github.io/), and `movla` (the in-house LFM2.5-VL
Stage A policy from [`external/movla`](https://github.com/NOPLAB/movla)). For
OmniVLA-edge the policy can also run fully on-robot with no cloud
(`--mode edge-local`).

A separate effort ports OmniVLA-edge off the workstation entirely:

- `app/inference/` — Flutter smartphone app: on-device ONNX inference; the phone
  streams action chunks to the Pi over its own gRPC interface
  (`proto/edge_action.proto`). See [`docs/design/mobile_port_spec.md`](docs/design/mobile_port_spec.md).
- `web/` — browser sibling of the mobile port (Next.js static export,
  onnxruntime-web / WebGPU); chunks go to the Pi over WebSocket. See
  [`docs/design/web_port_spec.md`](docs/design/web_port_spec.md).

Independently, `app/logger/` is a standalone Flutter data-logging app for VLA
fine-tuning: it captures camera / IMU / GNSS / audio to raw per-session logs and
does **no** inference (offline conversion + prompt labeling happen downstream).
See [`docs/design/logger_app_spec.md`](docs/design/logger_app_spec.md).

## Workspace layout

This repository is itself a colcon workspace.

```
src/raspicat_vla_msgs/      # ROS2 messages, services, actions (model-agnostic)
src/raspicat_vla_proto/     # mobile gRPC stubs and fp16 helpers
src/raspicat_vla_core/      # ROS-free OmniVLA-edge inference core (shared by edge & remote)
src/raspicat_vla_remote/    # ROS 2 inference node and model backends
src/raspicat_vla_edge/      # Edge ROS2 nodes (lifecycle, adapters, path follower,
                            #   phone/browser action receivers)
src/raspicat_vla_bringup/   # Launch composition
```

Not built by colcon:

```
app/inference/              # Flutter on-device inference port (Dart, app/inference/README.md)
app/logger/                 # Flutter VLA data-logger app (Dart, app/logger/README.md)
web/                        # Browser port (pnpm toolchain, see web/README.md)
docker/                     # Dockerfiles + compose topology for every run mode
external/                   # Research submodules: AsyncVLA, OmniVLA, MBRA, movla,
                            #   raspicat-sim-docker (reference code)
models/                     # VLA weights, gitignored — scripts/download_*.sh
```

The rt-net ROS2 source packages (`raspicat_ros`, `raspicat_description`,
`raspicat_sim`, `raspicat_slam_navigation`) are managed via vcstool, not
submodules — see `raspicat.repos`.

## Communication interfaces

The edge publishes `raspicat_vla_msgs/msg/Observation` on
`/raspicat_vla/observation`. The remote inference node publishes
`raspicat_vla_msgs/msg/ActionEmbedding` on
`/raspicat_vla/remote_embedding`. Use the same `ROS_DOMAIN_ID` on both PCs,
allow DDS discovery between them, and build matching message definitions.
Both topics use best-effort QoS with depth one so slow inference receives the
newest available frame. The `movla` image runs ROS 2 Jazzy on Ubuntu 24.04;
other runtime images use Humble. Its generated message definitions must match
the edge package.

`proto/edge_action.proto` is the independent phone → Pi interface for the
mobile port (`EdgeActionService.StreamActions`; the phone is the client, the
Pi the server).

`scripts/gen_proto.sh` regenerates the mobile Python stubs (into
`src/raspicat_vla_proto/raspicat_vla_proto/`) and the Dart stubs
for `edge_action.proto` (into `app/inference/lib/src/grpc/gen/`, committed).

## Build

First-time setup fetches the rt-net source packages into `src/` via vcstool,
then resolves their transitive ROS dependencies via rosdep:

```bash
source /opt/ros/humble/setup.bash
vcs import src < raspicat.repos              # one-time / on raspicat.repos changes
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

To bump the pinned rt-net versions, edit `raspicat.repos` and re-run
`vcs import src < raspicat.repos`.

## Running

`scripts/vla.sh` is the primary entry point for build/run/test. A run is
`vla.sh run MODEL --mode MODE`; the container topology behind each mode is
declared in `docker/compose.yaml` (one compose profile per mode). Run
`scripts/vla.sh` with no arguments for the authoritative MODEL / MODE / flag
list; the full operator guide lives at [`docs/USAGE.md`](docs/USAGE.md). A
quick orientation:

```bash
# Inference PC
ROS_DOMAIN_ID=42 scripts/vla.sh run omnivla --mode remote --gpu
# Robot PC, on the same reachable LAN
ROS_DOMAIN_ID=42 scripts/vla.sh run omnivla --mode edge --camera edge
```

* `scripts/vla.sh build TARGET` — build one of the images
  (`asyncvla`, `omnivla`, `movla`, `real`, `sim`, `test`, plus `*-jetson` for ARM64).
* `--mode remote {--cpu|--gpu}` — host the ROS 2 inference node here.
* `--mode edge` ? on-robot edge stack; match `ROS_DOMAIN_ID` on both PCs.
* `--mode cmd_vel` — all-in-one on this host, no robot: remote + edge in two
  containers, follower on a non-motor topic (`/cmd_vel_vla`).
* `--mode sim` — Gazebo + edge.
* `--mode edge-local` — OmniVLA-edge policy standalone on the robot, no cloud
  (needs CUDA + `models/omnivla-edge/omnivla-edge.pth`).
* `run omnivla_edge_mobile --mode cmd_vel` — Pi side of the mobile port: the
  phone infers, this host receives action chunks and follows.
* `scripts/vla.sh test [PYTEST_ARGS...]` — pytest in the CPU test image.

For a no-Docker single-host bring-up there is
`ros2 launch raspicat_vla_bringup local_stack.launch.py backend:=dummy|asyncvla|omnivla|omnivla_edge`.

With a stack running, `scripts/control.sh` drives it from the host (motor
power + VLA goals).

Both `asyncvla` and `omnivla` backends work on CPU but are slow; GPU is
strongly recommended for anything beyond wiring smoke tests. See
[`docs/USAGE.md`](docs/USAGE.md) §5.6 for CPU-specific caveats.
