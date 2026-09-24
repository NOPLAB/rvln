# rvla

[![ci-lint](https://github.com/NOPLAB/rvla/actions/workflows/ci-lint.yml/badge.svg?branch=main)](https://github.com/NOPLAB/rvla/actions/workflows/ci-lint.yml)
[![ci-flutter](https://github.com/NOPLAB/rvla/actions/workflows/ci-flutter.yml/badge.svg?branch=main)](https://github.com/NOPLAB/rvla/actions/workflows/ci-flutter.yml)
[![ci-web](https://github.com/NOPLAB/rvla/actions/workflows/ci-web.yml/badge.svg?branch=main)](https://github.com/NOPLAB/rvla/actions/workflows/ci-web.yml)

ROS 2 Humble VLA navigation for Raspberry Pi Cat. Nodes connect through ROS 2
topics and can run either natively on the host or in Docker.

## How the nodes connect

```text
Camera ── Image ──────────┐
Goal ── GoalSpec ─────────┼─> vla_edge_node ── Observation ─> vla_inference_node
                          │          ^                              │
                          │          └──── ActionEmbedding ─────────┘
                          │          │
                          │          └── Path ─> path_follower_node ─> Twist
Phone ── gRPC ─> edge_action_grpc ── Path ─────────────┘
Browser ── WebSocket ─> edge_action_ws ── Path ──────┘
```

Choose one node to produce `Path` for each deployment: `vla_edge_node` for the
standard ROS 2 pipeline, `edge_action_grpc` for a phone, or `edge_action_ws`
for a browser. With local OmniVLA-edge inference, `vla_edge_node` runs the
model itself and does not need the remote inference node.

### Node inputs and outputs

These are the default topic names; node parameters can change them.

| Node (executable) | Input | Output | Purpose |
| --- | --- | --- | --- |
| `vla_edge_node` (`rvla_edge vla_edge_node`) | `/camera/image_raw` (`sensor_msgs/msg/Image`), `/rvla/goal` (`rvla_msgs/msg/GoalSpec`), `/rvla/remote_embedding` (`rvla_msgs/msg/ActionEmbedding`) | `/rvla/observation` (`rvla_msgs/msg/Observation`), `/rvla/predicted_path` (`nav_msgs/msg/Path`), `/rvla/status` (`diagnostic_msgs/msg/DiagnosticArray`), and `/rvla/embedding` (`ActionEmbedding`, when debug publishing is enabled) | Lifecycle node that combines images and goals for inference, then turns the result into a path in the robot frame. Local inference does not use Observation or remote embedding topics. |
| `vla_inference_node` (`ros2 run rvla_remote vla_inference_node`) | `/rvla/observation` (`Observation`) | `/rvla/remote_embedding` (`ActionEmbedding`) | Runs the selected backend and returns a result correlated by `frame_id`. Select it with `backend:=` in the launch file or `--backend` with `ros2 run`. |
| `path_follower_node` (`rvla_edge path_follower_node`) | `/rvla/predicted_path` (`nav_msgs/msg/Path`) | `/cmd_vel` (`geometry_msgs/msg/Twist`) | Converts a `base_link` path into velocity commands. Stops on an empty or stale path. The output topic is configurable. |
| `edge_action_grpc` (`rvla_edge edge_action_grpc_server`) | `ActionChunk` via `EdgeActionService.StreamActions` (default `0.0.0.0:50061`) | `/rvla/predicted_path` (`Path`) and `ControlAck` to the sender | Receives phone inference results and publishes an empty path if chunks stop arriving. |
| `edge_action_ws` (`rvla_edge edge_action_ws_server`) | Action chunks via WebSocket (default `0.0.0.0:8765`) | `/rvla/predicted_path` (`Path`) and an ack to the sender | Receives browser inference results and publishes an empty path if chunks stop arriving. |

`vla_edge_node` reads the camera directly when `camera_device` is set, for
example to `/dev/video0`; in that mode it does not subscribe to an image
topic. When `image_topic` ends in `/compressed`, it subscribes to
`sensor_msgs/msg/CompressedImage`. Compressed images are suitable for cameras
on another host. The edge node subscribes to `/rvla/goal` with
`TRANSIENT_LOCAL` durability. Use the same durability when publishing a
one-shot goal. Observation and remote embedding use best-effort QoS with depth 1.

`GoalSpec` represents a pose, text, or image goal. `Observation` carries a
JPEG image and goal; `ActionEmbedding` carries the inference result and its
`frame_id`. See [`src/rvla_msgs/`](src/rvla_msgs/) for the
message definitions. `SetGoal.srv` and `NavigateVLA.action` are also defined,
but the current nodes receive goals through a topic. The phone protocol is
defined in [`proto/edge_action.proto`](proto/edge_action.proto).

## Native: run on ROS 2

These commands assume Ubuntu 22.04 and ROS 2 Humble. Run them from the
repository root. Real hardware and simulation also require the rt-net packages
listed in `raspicat.repos`. The `dummy` backend needs no model weights.

```bash
source /opt/ros/humble/setup.bash
vcs import src < raspicat.repos
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install 'grpcio>=1.50' 'grpcio-tools>=1.50' 'Pillow' 'opencv-python' 'websockets>=10'
scripts/gen_proto.sh
colcon build --symlink-install
source install/setup.bash
```

`scripts/gen_proto.sh` generates the Python gRPC stubs for the Pi. It also
generates Flutter stubs if the Dart plugin is installed. Real model backends
need their Python dependencies and checkpoints; the Dockerfiles provide
dependency examples.

### Minimal connection check (one host, no motor output)

Run each command in a separate terminal. Source
`/opt/ros/humble/setup.bash` and `install/setup.bash` in each terminal,
and use the same `ROS_DOMAIN_ID`.

```bash
# Terminal 1: inference node with no model weights
ROS_DOMAIN_ID=42 ros2 launch rvla_remote inference.launch.py backend:=dummy

# Terminal 2: edge and follower; publish commands to a non-motor topic
ROS_DOMAIN_ID=42 ros2 launch rvla_edge edge_only.launch.py \
  adapter_kind:=stub with_follower:=true cmd_vel_topic:=/cmd_vel_vla

# Terminal 3: publish a synthetic camera image and pose goal
ROS_DOMAIN_ID=42 python3 tools/publish_fake_image.py

# Terminal 4: inspect the result
ROS_DOMAIN_ID=42 ros2 topic echo /cmd_vel_vla
```

The edge is a lifecycle node. This launch file automatically configures and
activates it. Use `ros2 lifecycle get /vla_edge_node` to inspect its state and
`ros2 topic list -t` to inspect topics. `local_stack.launch.py backend:=dummy`
also starts the stack in one command, but its follower publishes to `/cmd_vel`.

### Native split-host deployment

Run `ros2 launch rvla_remote inference.launch.py backend:=dummy` on the
inference PC and the `rvla_edge edge_only.launch.py` command above
on the robot. Both hosts need the same `ROS_DOMAIN_ID` and message definitions,
plus a network that permits DDS discovery. For a real model, match the remote
`backend`, `vla_path`, and `device` launch arguments with the edge `adapter_kind`:
`asyncvla` ↔ `asyncvla`, or `omnivla` / `omnivla_edge` ↔ `omnivla`.
Set the weights path to a path that exists on the host.

For example, run `ros2 launch rvla_remote inference.launch.py
backend:=omnivla vla_path:=/path/to/omnivla-original device:=cuda:0` on the
inference PC, with `adapter_kind:=omnivla` on the robot. The launch file also
accepts `resume_step`, `observation_topic`, and `embedding_topic`. It uses
the model's usual resume step for AsyncVLA and OmniVLA when none is supplied.

Rebuild the workspace after updating to install the remote launch file and
executable.

For direct camera capture, pass `camera_kind:=v4l2
camera_device:=/dev/video0` to the edge launch file. To use an existing ROS
camera topic, pass `image_topic:=...`. To send velocity commands to the robot,
set `with_follower:=true cmd_vel_topic:=/cmd_vel`. Local OmniVLA-edge runs with
`ros2 launch rvla_bringup omnivla_edge_local.launch.py
weights_path:=/path/to/omnivla-edge.pth` and requires a CUDA-capable edge host.

To receive paths from a phone, run
`ros2 launch rvla_bringup mobile_cmd_vel.launch.py`. For a browser,
run `ros2 launch rvla_bringup phone_ws.launch.py`. Both publish to
`/cmd_vel_vla` by default and do not start edge or remote inference nodes.

## Docker: run the same ROS 2 interfaces

Use Docker Compose on Linux. `scripts/vla.sh` selects the image and Compose
profile. Topic names and message types are the same as in the table above.

```bash
# Inference PC
scripts/vla.sh build omnivla
ROS_DOMAIN_ID=42 scripts/vla.sh run omnivla --mode remote --gpu

# Robot PC (another host, same ROS_DOMAIN_ID)
scripts/vla.sh build real
ROS_DOMAIN_ID=42 scripts/vla.sh run omnivla --mode edge --camera edge
```

Put model weights under `models/` (excluded from Git). `--mode cmd_vel`
starts remote and edge on one host and publishes to `/cmd_vel_vla`.
`--mode sim` starts Gazebo, `--mode edge-local` runs OmniVLA-edge on the
robot, and `run omnivla_edge_mobile --mode cmd_vel` receives phone actions
over gRPC. For CPU tests, run `scripts/vla.sh build test` followed by
`scripts/vla.sh test`. Run `scripts/vla.sh --help` for supported models,
modes, and options.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/rvla_msgs/` | ROS 2 message definitions |
| `src/rvla_edge/` | Edge, path follower, and phone / browser receiver nodes |
| `src/rvla_remote/` | Inference node and backends (`dummy`, `asyncvla`, `omnivla`, `omnivla_edge`, `movla`) |
| `src/rvla_core/` | ROS-independent OmniVLA-edge inference core |
| `src/rvla_proto/` | Python mobile gRPC stubs and conversion code |
| `src/rvla_bringup/` | Launch files for different deployments |
| `app/inference/`, `web/` | Phone and browser clients that infer and send paths |
| `app/logger/` | Standalone Flutter training-data logger |

For current Docker options, run `scripts/vla.sh --help`. For client setup,
see [`app/inference/README.md`](app/inference/README.md) and
[`web/README.md`](web/README.md).
