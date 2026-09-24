#!/usr/bin/env bash
# scripts/vla.sh — thin driver for the rvla Docker stacks.
#
# The container topology of every mode lives in docker/compose.yaml (one
# compose profile per mode; see its header for the service map). This script
# only parses the CLI, picks the profile + structural overlays
# (docker/compose.*.yaml), exports the VLA_* variables, and runs
# `docker compose up`.
#
# Subcommands:
#   build TARGET            asyncvla | omnivla | movla | test | real | sim | --all
#   run MODEL --mode MODE [OPTS]
#                           MODEL = asyncvla | omnivla | omnivla_edge | movla
#                                   | omnivla_edge_mobile (phone infers; Pi receives)
#                           MODE  = remote {--cpu|--gpu}
#                                   edge
#                                   cmd_vel {--cpu|--gpu}   (remote+edge, no motors)
#                                   sim
#                                   edge-local              (omnivla_edge only)
#                           edge/cmd_vel/edge-local also take
#                                   --camera edge|realsense|/dev/videoN
#
# Run `vla.sh --help` for the full reference.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Phone -> Pi EdgeActionService port (proto/edge_action.proto); distinct from
# Mobile receiver keeps its own gRPC port.
EDGE_ACTION_PORT="${EDGE_ACTION_PORT:-50061}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
HOST_ARCH="$(uname -m)"

# Jetson (L4T/aarch64) needs the ARM remote images + the nvidia container
# runtime for GPU (compose.jetson.yaml), not x86's `gpus: all`
# (compose.gpu.yaml). Auto-detected from the host arch; force with
# RVLA_JETSON=1 (or =0 to disable, e.g. cross-build on an aarch64 host).
is_jetson() {
    case "${RVLA_JETSON:-}" in
        1) return 0 ;;
        0) return 1 ;;
    esac
    [[ $HOST_ARCH == aarch64 || $HOST_ARCH == arm64 ]]
}

# Image / Dockerfile / model knob registries. Bash 4 associative arrays.
declare -A IMAGES=(
    [asyncvla]="rvla-asyncvla"
    [omnivla]="rvla-omnivla"
    # omnivla_edge (Path 3) reuses the OmniVLA remote image (it adds CLIP +
    # efficientnet); the remote backend just loads a different checkpoint.
    [omnivla_edge]="rvla-omnivla"
    # movla (external/movla): in-house LFM2.5-VL Stage A policy. No Jetson
    # variant yet (needs an aarch64 torch-2.12 wheel story first).
    [movla]="rvla-movla"
    [asyncvla-jetson]="rvla-asyncvla-jetson"
    [omnivla-jetson]="rvla-omnivla-jetson"
    [omnivla_edge-jetson]="rvla-omnivla-jetson"
    [test]="rvla-test"
    [real]="rvla-real"
    [sim]="rvla-sim"
)
declare -A DOCKERFILES=(
    [asyncvla]="docker/Dockerfile.asyncvla"
    [omnivla]="docker/Dockerfile.omnivla"
    [movla]="docker/Dockerfile.movla"
    [asyncvla-jetson]="docker/Dockerfile.asyncvla.jetson"
    [omnivla-jetson]="docker/Dockerfile.omnivla.jetson"
    [test]="docker/Dockerfile.test"
    [real]="docker/Dockerfile.real"
    [sim]="docker/Dockerfile.sim"
)
declare -A RESUME_STEP=(
    [asyncvla]=750000
    [omnivla]=120000
    [omnivla_edge]=0      # unused: omnivla-edge.pth is a bare state_dict
    [movla]=0             # unused: checkpoint.pt carries its own expert_cfg
)
declare -A WEIGHTS_DIR=(
    [asyncvla]="/workspace/models/AsyncVLA_release"
    [omnivla]="/workspace/models/omnivla-original"
    # For omnivla_edge --vla-path is the .pth weights file, not a checkpoint dir.
    [omnivla_edge]="/workspace/models/omnivla-edge/omnivla-edge.pth"
    # movla: dir with checkpoint.pt + normalizer.json (scripts/download_movla_checkpoint.sh)
    [movla]="/workspace/models/movla/stage_a_v2"
)

usage() {
    cat <<'EOF'
Usage: vla.sh COMMAND [ARGS]

Commands:
  build TARGET            Build a Docker image
    TARGET = asyncvla | omnivla | movla | test | real | sim | --all
             asyncvla-jetson | omnivla-jetson   (ARM64 / Jetson AGX Orin)
  run MODEL --mode MODE [OPTS]   Run a configuration
    MODEL = asyncvla | omnivla | omnivla_edge | omnivla_edge_mobile | movla
    MODE (selected with --mode MODE):
      remote {--cpu|--gpu}          Host the remote ROS 2 inference node here.
                                    Uses Dockerfile.<MODEL>.
      edge                          Edge stack here. ROS 2 discovers the remote
                                    node on the same ROS_DOMAIN_ID.
                                    Uses Dockerfile.real.

    Camera (edge | cmd_vel | edge-local):
      --camera edge|realsense|/dev/videoN
                                    Give the edge its camera frames directly.
                                      edge        v4l2 webcam, default device
                                                  (/dev/video0) — a preset.
                                      /dev/videoN v4l2 webcam, explicit device.
                                                  v4l2 devices are grabbed
                                                  IN-PROCESS by the edge node
                                                  (no driver node, no DDS image
                                                  hop); passed through via
                                                  compose.camera-v4l2.yaml.
                                      realsense   Intel RealSense (separate
                                                  realsense2_camera node); the
                                                  container runs privileged with
                                                  /dev bind-mounted for USB
                                                  (compose.camera-realsense.yaml).
                                    Omit to feed frames some other way (a camera
                                    launched outside, publish_fake_image.py, sim).
      --image-topic TOPIC           Where the edge reads frames when NOT grabbing
                                    a device in-process (edge|cmd_vel|edge-local|
                                    sim). A topic ending in /compressed is
                                    subscribed as CompressedImage (JPEG) — use it
                                    whenever the camera sits on another host
                                    (raw Image does not survive WiFi), e.g.
                                    --image-topic /camera/image_raw/compressed
      cmd_vel {--cpu|--gpu}         All-in-one on THIS host, no real robot: one
                                    command starts BOTH the remote server (bound
                                    to 127.0.0.1) and the edge stack, in two
                                    containers (compose profile "cmd_vel"; both
                                    logs stream in the foreground). The follower
                                    publishes to a non-motor topic (/cmd_vel_vla),
                                    so the whole pipeline runs and cmd_vel is
                                    observable (ros2 topic echo /cmd_vel_vla)
                                    without driving the robot's motors. Feed
                                    frames with tools/publish_fake_image.py or a
                                    real camera. Add --drive-motors to publish to
                                    the real /cmd_vel topic instead (motors WILL
                                    be driven).
      sim                           Edge + Gazebo simulation with ROS 2 remote.
                                    Uses Dockerfile.sim. Plan 3 wip.
      edge-local                    Plan 2B Path 2 (omnivla_edge ONLY): run the
                                    OmniVLA-edge policy ON the edge, standalone —
                                    no cloud, just edge node + follower
                                    (omnivla_edge_local.launch.py). Requires CUDA
                                    and models/omnivla-edge/omnivla-edge.pth.
                                    Uses Dockerfile.real; GPU via
                                    compose.gpu.yaml (x86) or
                                    compose.jetson.yaml (Jetson/L4T).

    omnivla_edge modes (Plan 2B Path 3 — remote split, "Jetson infers, Pi
    controls"): --mode remote runs the OmniVLA-edge policy on this GPU box (the
    omnivla image + omnivla-edge.pth); the Pi side runs --mode edge/sim with the
    light path-only adapter (adapter_kind=omnivla, no torch). Path 2's
    --mode edge-local runs the whole thing on one CUDA box instead.

    movla (external/movla — in-house LFM2.5-VL Stage A policy): same remote
    split as omnivla_edge Path 3 — the movla image runs the policy (remote |
    cmd_vel here; CPU works, GPU faster) and streams metre-scaled waypoints;
    the edge side uses the path-only 'omnivla' adapter automatically. Weights:
    models/movla/<run>/ via scripts/download_movla_checkpoint.sh (default run
    stage_a_v2). The Stage A policy is language-only; goals should use the
    trained instruction templates ("go straight ahead" / "turn left ahead" /
    "turn right ahead").

    omnivla_edge_mobile (mobile port — "phone infers, Pi controls"): the
    smartphone app (app/) does camera capture + on-device OmniVLA-edge
    inference and streams action chunks here over gRPC
    (proto/edge_action.proto). Only --mode cmd_vel is supported: it starts the
    EdgeActionService receiver (edge_action_grpc_node) + path follower in ONE
    container — no VLA server, no --cpu/--gpu, no --camera (the phone is the
    camera). --host BIND[:PORT] overrides the listen address (default
    0.0.0.0:$EDGE_ACTION_PORT). Point the app at this host's IP. As with the
    other models, the follower publishes /cmd_vel_vla unless --drive-motors.
    Requires scripts/gen_proto.sh to have generated the edge_action stubs.
  test [PYTEST_ARGS...]   Run pytest in rvla-test (CPU). Auto-builds
                          the image if missing. Pass extra args to pytest:
                            vla.sh test                        # full suite
                            vla.sh test -k checkpoint          # filter
                            vla.sh test src/rvla_edge/test  # subset
  help, -h, --help        Show this help

Examples:
  vla.sh build asyncvla
  vla.sh build --all
  ROS_DOMAIN_ID=42 vla.sh run asyncvla --mode remote --gpu  # workstation
  ROS_DOMAIN_ID=42 vla.sh run asyncvla --mode edge          # robot
  vla.sh run omnivla  --mode edge --camera edge             # v4l2 /dev/video0
  vla.sh run omnivla  --mode edge --camera /dev/cam1        # v4l2 explicit device
  vla.sh run omnivla  --mode edge --camera realsense        # Intel RealSense
  vla.sh run omnivla  --mode cmd_vel --gpu                 # remote+edge here, no motors
  vla.sh run omnivla  --mode sim
  vla.sh run omnivla_edge --mode edge-local               # Path 2, standalone on-edge policy (GPU)
  vla.sh run omnivla_edge --mode remote --gpu             # Path 3, OmniVLA-edge server (Jetson)
  vla.sh run omnivla_edge --mode edge                      # Path 3, Pi edge -> Jetson node
  vla.sh run movla --mode cmd_vel --cpu                   # in-house movla policy, all local
  vla.sh run movla --mode remote --gpu                    # movla server on a GPU box
  vla.sh run omnivla_edge_mobile --mode cmd_vel           # phone infers -> this host follows (no motors)
  vla.sh run omnivla_edge_mobile --mode cmd_vel --host :50062  # custom listen port
  vla.sh test                                              # full pytest suite
  vla.sh test -k omnivla                                   # filter by name

Jetson AGX Orin (ARM64):
  On an aarch64 host this script auto-selects the *-jetson remote images and
  swaps compose.gpu.yaml for compose.jetson.yaml (nvidia runtime). Build + run
  on the device:
    vla.sh build omnivla-jetson
    vla.sh run omnivla --mode remote --gpu           # uses rvla-omnivla-jetson
  Match the image to your JetPack via Docker build args (see the Dockerfile
  header), e.g.:
    docker build -f docker/Dockerfile.omnivla.jetson \
      --build-arg L4T_BASE=nvcr.io/nvidia/l4t-jetpack:r36.4.0 \
      --build-arg TORCH_VERSION=2.8.0 -t rvla-omnivla-jetson .
  Force/disable Jetson mode with RVLA_JETSON=1 / =0.

Environment overrides:
  ROS_DOMAIN_ID        ROS 2 discovery domain; set identically on both PCs
  EDGE_ACTION_PORT     phone->Pi EdgeActionService port (default 50061)
  HF_CACHE_DIR         HuggingFace cache mount (default $HOME/.cache/huggingface)
  RVLA_JETSON  1 = force Jetson images + nvidia runtime; 0 = force x86
  ROS_DOMAIN_ID        forwarded into every ROS container to isolate DDS
                       discovery (unset => ROS default 0). Under sudo pass it
                       through: sudo ROS_DOMAIN_ID=N ./scripts/vla.sh ...
EOF
}

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
err()  { printf '\033[1;31m!!\033[0m  %s\n' "$*" >&2; }

# ------------------------------------------------------------ compose driver
# Everything docker-related below funnels into `docker compose` against
# docker/compose.yaml; the mode-specific run_* functions only add overlays
# (compose_add) and export VLA_* variables consumed by the compose files.
COMPOSE_FILES=(-f "$REPO_ROOT/docker/compose.yaml")

compose_add() { COMPOSE_FILES+=(-f "$REPO_ROOT/docker/$1"); }

# The CUDA overlay matching this host: x86 container toolkit vs Jetson runtime.
compose_add_gpu() {
    if is_jetson; then compose_add compose.jetson.yaml; else compose_add compose.gpu.yaml; fi
}

# Variables every compose invocation needs, mode-independent.
export VLA_REPO_ROOT="$REPO_ROOT"
export VLA_HF_CACHE="$HF_CACHE_DIR"
# Host user:group. Every ROS container runs as this so files it writes under
# the /workspace bind-mount stay host-owned, not root.
VLA_UID_GID="$(id -u):$(id -g)"
export VLA_UID_GID

# Foreground `docker compose up` of one profile, torn down on exit (compose
# `up` does not remove stopped containers by itself). --abort-on-container-exit
# makes any container's death (e.g. the edge in cmd_vel mode) stop the whole
# profile instead of leaving the server running headless.
compose_up() {
    local profile=$1; shift
    # shellcheck disable=SC2064
    trap "docker compose ${COMPOSE_FILES[*]} --profile '$profile' down -t 5 >/dev/null 2>&1 || true" EXIT INT TERM
    docker compose "${COMPOSE_FILES[@]}" --profile "$profile" up \
        --abort-on-container-exit "$@"
}

# Resolve a --camera value to "KIND DEVICE" (space-separated) on stdout.
#   edge        -> v4l2 /dev/video0     (preset for the robot's default webcam)
#   /dev/*      -> v4l2 <path>          (explicit v4l2 device, e.g. /dev/cam1)
#   realsense   -> realsense <empty>    (Intel RealSense, driven over USB — no
#                                        device node path; realsense2_camera
#                                        enumerates it itself)
# Returns non-zero for anything else so the caller can reject typos.
resolve_camera() {
    case $1 in
        realsense) printf 'realsense \n' ;;
        edge)      printf 'v4l2 /dev/video0\n' ;;
        /dev/*)    printf 'v4l2 %s\n' "$1" ;;
        *)         return 1 ;;
    esac
}

# Add the camera passthrough overlay for the resolved camera kind and export
# its variables (device path + owning gid — see the overlay headers for why
# the gid is needed). No camera => no-op.
compose_add_camera() {
    local kind=$1 dev=$2
    case $kind in
        v4l2)
            [[ -n $dev ]] || return 0
            compose_add compose.camera-v4l2.yaml
            export VLA_CAMERA_DEVICE="$dev"
            if [[ -e $dev ]]; then
                VLA_CAMERA_GID="$(stat -c '%g' "$dev")"
                export VLA_CAMERA_GID
            fi
            ;;
        realsense)
            compose_add compose.camera-realsense.yaml
            local vgid=""
            vgid=$(getent group video 2>/dev/null | cut -d: -f3)
            [[ -z $vgid && -e /dev/video0 ]] && vgid=$(stat -c '%g' /dev/video0)
            [[ -n $vgid ]] && export VLA_CAMERA_GID="$vgid"
            ;;
    esac
    return 0
}

# Append camera_kind:=/camera_device:= to the launch argv array named by $1,
# skipping any that are empty — ROS2 rejects a bare `foo:=` with no value, and
# the launch files default both to '' (no camera node). $2 = kind, $3 = device.
_append_camera_launch_args() {
    local -n _arr=$1
    local kind=$2 dev=$3
    [[ -n $kind ]] && _arr+=("camera_kind:=${kind}")
    [[ -n $dev ]] && _arr+=("camera_device:=${dev}")
    # Explicit success: a trailing `[[ ... ]] &&` that fails (no camera => both
    # branches skipped) would make this function return 1 and, under `set -e`,
    # abort the whole script right after the caller's log line.
    return 0
}

# Append image_topic:=... to the launch argv array named by $1 when the user
# passed --image-topic (stored in RUN_IMAGE_TOPIC by cmd_run).
_append_image_topic_arg() {
    local -n _arr=$1
    [[ -n ${RUN_IMAGE_TOPIC:-} ]] && _arr+=("image_topic:=${RUN_IMAGE_TOPIC}")
    return 0
}

# split_hostport HOST[:PORT] DEFAULT_HOST DEFAULT_PORT -> "HOST PORT" on stdout.
# Empty HOST (e.g. ":8080") falls back to DEFAULT_HOST. Missing :PORT -> DEFAULT_PORT.
split_hostport() {
    local raw=$1 default_host=$2 default_port=$3
    local host port
    if [[ $raw == *:* ]]; then
        host=${raw%:*}
        port=${raw##*:}
    else
        host=$raw
        port=$default_port
    fi
    [[ -z $host ]] && host=$default_host
    if ! [[ $port =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
        err "invalid port in '$raw'"
        return 1
    fi
    printf '%s %s\n' "$host" "$port"
}

build_one() {
    local target=$1
    local dfile_rel="${DOCKERFILES[$target]:-}"
    local image="${IMAGES[$target]:-}"
    [[ -n $dfile_rel && -n $image ]] || { err "unknown build target: $target"; return 1; }
    local dfile="${REPO_ROOT}/${dfile_rel}"
    [[ -f $dfile ]] || { err "Dockerfile not found: $dfile"; return 1; }
    if [[ $target == sim ]] && is_jetson; then
        warn "'sim' uses osrf/ros:humble-desktop-full, which has no arm64 image;"
        warn "this build will fail on Jetson with 'exec format error'. Use a x86 host for sim."
    fi
    log "building ${image} from ${dfile_rel}"
    docker build -f "$dfile" -t "$image" "$REPO_ROOT"
}

cmd_build() {
    local target=${1:-}
    case $target in
        --all)
            local rc=0 all_targets
            if is_jetson; then
                # On Jetson: build the ARM remote variants (the x86 ones have no
                # aarch64 cu121 torch wheel). Skip sim — its base image
                # (osrf/ros:humble-desktop-full) has no arm64 build, and a Gazebo
                # GUI workstation isn't a Jetson use case.
                all_targets=(asyncvla-jetson omnivla-jetson test real)
                warn "Jetson: skipping 'sim' from --all (osrf desktop-full has no arm64 image)."
            else
                all_targets=(asyncvla omnivla movla test real sim)
            fi
            for t in "${all_targets[@]}"; do
                build_one "$t" || rc=1
            done
            return $rc
            ;;
        asyncvla|omnivla|movla|asyncvla-jetson|omnivla-jetson|test|real|sim)
            build_one "$target"
            ;;
        '')
            err "build: missing target"
            usage
            return 1
            ;;
        *)
            err "build: unknown target '$target'"
            usage
            return 1
            ;;
    esac
}

# Export the `remote` service's variables (image, backend knobs, bind address)
# and add the GPU overlay when requested. Shared by run_remote and run_cmd_vel.
export_remote_env() {
    local model=$1 device=$2
    local image="${IMAGES[$model]}"
    # On Jetson the backend name (--backend / weights / resume-step) is unchanged;
    # only the container image (ARM build) and the GPU wiring differ.
    is_jetson && image="${IMAGES[${model}-jetson]}"
    export VLA_REMOTE_IMAGE="$image"
    export VLA_ROS_DISTRO=humble
    [[ $model == movla ]] && export VLA_ROS_DISTRO=jazzy
    export VLA_BACKEND="$model"
    export VLA_WEIGHTS="${WEIGHTS_DIR[$model]}"
    export VLA_RESUME_STEP="${RESUME_STEP[$model]}"
    if [[ $device == gpu ]]; then
        compose_add_gpu
        export VLA_DEVICE="cuda:0"
    else
        export VLA_DEVICE="cpu"
    fi
}

# Export the `edge` service's image + overlay, falling back to the test image
# when rvla-real isn't built. $1 = model (for the fallback warnings).
export_edge_image() {
    local model=$1
    local image="${IMAGES[real]}"
    if docker image inspect "$image" >/dev/null 2>&1; then
        export VLA_EDGE_IMAGE="$image"
        export VLA_EDGE_OVERLAY="/opt/real_ws/install/setup.bash"
    else
        warn "image ${image} not built; falling back to ${IMAGES[test]} (no rt-net packages)."
        warn "run \`vla.sh build real\` for the full image with raspicat_ros + Edge_adapter deps."
        if [[ $model == asyncvla ]]; then
            warn "AsyncVLA edge needs torch + MBRA on PYTHONPATH; the test image lacks them."
        fi
        export VLA_EDGE_IMAGE="${IMAGES[test]}"
        export VLA_EDGE_OVERLAY=""
    fi
}

# The edge-side adapter_kind for a remote MODEL. omnivla_edge (Path 3) runs the
# policy on the remote box, so the Pi uses the light path-only 'omnivla' adapter;
# every other model's edge adapter matches the model name.
edge_adapter_for() {
    local model=$1
    # omnivla_edge (Path 3) and movla both run the policy on the remote box and
    # stream metre-scaled (x, y, cos, sin) waypoints, so the Pi uses the light
    # path-only 'omnivla' adapter; every other model's edge adapter matches the
    # model name.
    case $model in
        omnivla_edge|movla) printf 'omnivla\n' ;;
        *)                  printf '%s\n' "$model" ;;
    esac
}

run_remote() {
    local model=$1 device=$2
    export_remote_env "$model" "$device"
    log "${model} remote ROS 2 node on ${VLA_DEVICE}; domain ${ROS_DOMAIN_ID:-0}"
    compose_up remote
}

run_edge() {
    local model=$1 camera_kind=${2:-} camera_device=${3:-}
    local adapter_kind
    adapter_kind=$(edge_adapter_for "$model")
    export_edge_image "$model"
    compose_add_camera "$camera_kind" "$camera_device"
    local launch=(
        rvla_edge edge_only.launch.py
        "adapter_kind:=${adapter_kind}"
        with_follower:=true
    )
    _append_camera_launch_args launch "$camera_kind" "$camera_device"
    _append_image_topic_arg launch
    export VLA_EDGE_LAUNCH="${launch[*]}"
    log "${model} edge (real); ROS domain ${ROS_DOMAIN_ID:-0}${camera_kind:+; camera=${camera_kind}${camera_device:+ ${camera_device}}}"
    compose_up edge
}

# cmd_vel mode: one command, two containers (compose profile "cmd_vel"), no
# real robot. The remote and edge nodes discover each other via ROS 2;
# the follower publishes to a non-motor topic so the full pipeline runs
# and cmd_vel is observable without driving the robot's motors. Both logs
# stream in the foreground; when either container exits (or Ctrl-C), compose
# stops the other and the EXIT trap tears the profile down.
run_cmd_vel() {
    local model=$1 device=$2 camera_kind=${3:-} camera_device=${4:-} cmd_vel_topic=${5:-/cmd_vel_vla}
    local adapter_kind
    adapter_kind=$(edge_adapter_for "$model")

    if [[ $cmd_vel_topic == /cmd_vel ]]; then
        log "cmd_vel: launching ${model} remote server + edge on this host (MOTORS DRIVEN via /cmd_vel)"
    else
        log "cmd_vel: launching ${model} remote server + edge on this host (motors NOT driven)"
    fi
    export_remote_env "$model" "$device"
    export_edge_image "$model"
    compose_add_camera "$camera_kind" "$camera_device"
    local launch=(
        rvla_edge edge_only.launch.py
        "adapter_kind:=${adapter_kind}"
        "cmd_vel_topic:=${cmd_vel_topic}"
        with_follower:=true
    )
    _append_camera_launch_args launch "$camera_kind" "$camera_device"
    _append_image_topic_arg launch
    export VLA_EDGE_LAUNCH="${launch[*]}"
    log "cmd_vel: ROS domain ${ROS_DOMAIN_ID:-0}; follower publishes ${cmd_vel_topic}${camera_kind:+; camera=${camera_kind}${camera_device:+ ${camera_device}}}"
    compose_up cmd_vel
}

# omnivla_edge_mobile cmd_vel: the smartphone runs the whole OmniVLA-edge
# policy on-device and streams action chunks over gRPC (proto/edge_action.proto,
# phone = client); this host only receives them (edge_action_grpc_node) and
# follows (path_follower_node). ONE container (compose profile "mobile"), no
# VLA server, no camera — the phone is the camera. The follower publishes to
# cmd_vel_topic (/cmd_vel_vla unless --drive-motors), same safety story as the
# other cmd_vel runs.
run_mobile_cmd_vel() {
    local bind_host=$1 bind_port=$2 cmd_vel_topic=$3
    if [[ $cmd_vel_topic == /cmd_vel ]]; then
        log "mobile cmd_vel: EdgeActionService @ ${bind_host}:${bind_port} (MOTORS DRIVEN via /cmd_vel)"
    else
        log "mobile cmd_vel: EdgeActionService @ ${bind_host}:${bind_port}; follower publishes ${cmd_vel_topic} (motors NOT driven)"
    fi
    log "point the smartphone app at this host's IP, port ${bind_port}"
    export_edge_image omnivla_edge_mobile
    local launch=(
        rvla_bringup mobile_cmd_vel.launch.py
        "listen_host:=${bind_host}"
        "listen_port:=${bind_port}"
        "cmd_vel_topic:=${cmd_vel_topic}"
    )
    export VLA_EDGE_LAUNCH="${launch[*]}"
    compose_up mobile
}

run_sim() {
    local model=$1
    local adapter_kind
    adapter_kind=$(edge_adapter_for "$model")
    if ! docker image inspect "${IMAGES[sim]}" >/dev/null 2>&1; then
        warn "image ${IMAGES[sim]} not built; falling back to ${IMAGES[test]} (no Gazebo)."
        warn "run \`vla.sh build sim\` for the full sim image with Gazebo + raspicat_sim."
        # Fallback: the plain edge service (edge_only, no Gazebo) in the test image.
        export VLA_EDGE_IMAGE="${IMAGES[test]}"
        export VLA_EDGE_OVERLAY=""
        local launch=(
            rvla_edge edge_only.launch.py
            "adapter_kind:=${adapter_kind}"
            with_follower:=true
        )
        export VLA_EDGE_LAUNCH="${launch[*]}"
        log "${model} edge (sim-fallback, image=${IMAGES[test]}); ROS domain ${ROS_DOMAIN_ID:-0}"
        compose_up edge
        return
    fi

    # Forward DISPLAY so gzclient renders on the host (headless works without).
    [[ -n ${DISPLAY:-} ]] && compose_add compose.sim-display.yaml

    # GPU passthrough for OpenGL (see compose.gpu.yaml's sim section). Skip
    # with a warning if the nvidia runtime is missing — the run still comes up
    # on software GL, just slowly enough that spawn_entity may time out.
    if docker info 2>/dev/null | grep -q ' nvidia'; then
        compose_add compose.gpu.yaml
    else
        warn "nvidia container runtime not found; sim falls back to software GL."
        warn "Gazebo camera rendering will be slow and spawn_entity may time out."
        warn "Install nvidia-container-toolkit + 'nvidia-ctk runtime configure --runtime=docker'."
    fi

    # Synthesize an /etc/passwd entry for the host UID inside the container so
    # gzclient stops spamming "Error getting username: no matching password
    # record". We can't hand it the host's /etc/passwd as-is because Gazebo
    # would then try HOME=/home/<user> which doesn't exist in the image; the
    # synthesized entry points HOME at /tmp instead.
    local uid gid passwd_dir
    uid=$(id -u); gid=$(id -g)
    passwd_dir=$(mktemp -d)
    cat /etc/passwd > "$passwd_dir/passwd"
    grep -q "^[^:]*:[^:]*:${uid}:" "$passwd_dir/passwd" || \
        echo "raspicat:x:${uid}:${gid}:raspicat:/tmp:/bin/bash" >> "$passwd_dir/passwd"
    cat /etc/group > "$passwd_dir/group"
    grep -q "^[^:]*:[^:]*:${gid}:" "$passwd_dir/group" || \
        echo "raspicat:x:${gid}:" >> "$passwd_dir/group"
    export VLA_SIM_PASSWD="$passwd_dir/passwd"
    export VLA_SIM_GROUP="$passwd_dir/group"

    local launch=(
        rvla_bringup sim.launch.py
        "adapter_kind:=${adapter_kind}"
    )
    export VLA_SIM_LAUNCH="${launch[*]}"
    log "${model} sim (image=${IMAGES[sim]}); ROS domain ${ROS_DOMAIN_ID:-0}"
    compose_up sim
}

# Plan 2B Path 2: the OmniVLA-edge policy runs entirely on the edge. The edge
# node operates standalone — no cloud, no gRPC, no embedding cache — so this is
# a single-container run of omnivla_edge_local.launch.py (edge node + follower).
# Needs CUDA (the vendored OmniVLA_edge forward pass is GPU-only) and the
# omnivla-edge weights at models/omnivla-edge/omnivla-edge.pth.
run_edge_local() {
    local camera_kind=${1:-} camera_device=${2:-}
    if ! docker image inspect "${IMAGES[real]}" >/dev/null 2>&1; then
        err "image ${IMAGES[real]} not built; run \`vla.sh build real\` first."
        return 1
    fi
    warn "Path 2 runs the OmniVLA-edge policy on-device and REQUIRES CUDA."
    warn "Dockerfile.real ships CPU torch; on a GPU host, rebuild it with a CUDA torch wheel."
    warn "Needs weights at models/omnivla-edge/omnivla-edge.pth (scripts/download_omnivla_edge_checkpoints.sh)."
    mkdir -p "${HOME}/.cache/clip"
    export VLA_CLIP_CACHE="${HOME}/.cache/clip"
    compose_add_gpu
    compose_add_camera "$camera_kind" "$camera_device"
    local launch=(
        rvla_bringup omnivla_edge_local.launch.py
        "device:=cuda:0"
    )
    _append_camera_launch_args launch "$camera_kind" "$camera_device"
    _append_image_topic_arg launch
    export VLA_EDGE_LOCAL_LAUNCH="${launch[*]}"
    log "omnivla_edge edge-local (image=${IMAGES[real]}); standalone edge + follower${camera_kind:+; camera=${camera_kind}${camera_device:+ ${camera_device}}}"
    compose_up edge-local
}

cmd_run() {
    local model=${1:-}
    case $model in
        asyncvla|omnivla|omnivla_edge|omnivla_edge_mobile|movla) ;;
        '')
            err "run: missing model (asyncvla|omnivla|omnivla_edge|omnivla_edge_mobile|movla)"; usage; return 1 ;;
        *)
            err "run: unknown model '$model'"; usage; return 1 ;;
    esac
    shift

    local mode='' host='' device='' camera='' drive_motors='' image_topic=''
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                [[ $# -ge 2 ]] || { err "--mode requires an argument (remote|edge|cmd_vel|sim|edge-local)"; return 1; }
                case $2 in
                    remote)     mode=remote ;;
                    edge)       mode=edge ;;
                    cmd_vel)    mode=cmd_vel ;;
                    sim)        mode=sim ;;
                    edge-local) mode=edge_local ;;
                    *) err "run: unknown mode '$2' (remote|edge|cmd_vel|sim|edge-local)"; return 1 ;;
                esac
                shift 2 ;;
            --host)
                [[ $# -ge 2 ]] || { err "--host requires an argument"; return 1; }
                host=$2; shift 2 ;;
            --camera)
                [[ $# -ge 2 ]] || { err "--camera requires an argument (edge|realsense|/dev/videoN)"; return 1; }
                camera=$2; shift 2 ;;
            --image-topic)
                [[ $# -ge 2 ]] || { err "--image-topic requires a topic name"; return 1; }
                image_topic=$2; shift 2 ;;
            --cpu)    device=cpu; shift ;;
            --gpu)    device=gpu; shift ;;
            --drive-motors) drive_motors=1; shift ;;
            -h|--help) usage; return 0 ;;
            *) err "run: unknown option '$1'"; usage; return 1 ;;
        esac
    done

    # omnivla_edge_mobile: the phone does capture + inference, so this host is
    # only the chunk receiver — cmd_vel is the sole mode, and the camera /
    # image-topic knobs are meaningless here.
    if [[ $model == omnivla_edge_mobile ]]; then
        if [[ $mode != cmd_vel ]]; then
            err "omnivla_edge_mobile only supports --mode cmd_vel (the phone is the edge)"; return 1
        fi
        if [[ -n $camera || -n $image_topic ]]; then
            err "--camera/--image-topic are invalid for omnivla_edge_mobile (the phone is the camera)"; return 1
        fi
        [[ -n $device ]] && warn "--cpu/--gpu are ignored for omnivla_edge_mobile (no local VLA server; the phone infers)"
        local pair bind_host bind_port
        pair=$(split_hostport "${host:-0.0.0.0}" "0.0.0.0" "$EDGE_ACTION_PORT") || return 1
        read -r bind_host bind_port <<<"$pair"
        local cmd_vel_topic=/cmd_vel_vla
        if [[ -n $drive_motors ]]; then
            cmd_vel_topic=/cmd_vel
            warn "--drive-motors: follower publishes to /cmd_vel — the robot's motors WILL be driven"
        fi
        run_mobile_cmd_vel "$bind_host" "$bind_port" "$cmd_vel_topic"
        return
    fi

    # --camera drives a camera node on the edge and needs the device exposed to
    # the container, so it only applies to the edge-side modes. remote hosts no
    # camera; sim gets its frames from Gazebo's virtual RealSense.
    local camera_kind='' camera_device=''
    if [[ -n $camera ]]; then
        case $mode in
            edge|cmd_vel|edge_local) ;;
            *) err "--camera is only valid for --mode edge|cmd_vel|edge-local (not '$mode')"; return 1 ;;
        esac
        local resolved
        resolved=$(resolve_camera "$camera") || {
            err "--camera: unknown value '$camera' (want edge|realsense|/dev/videoN)"; return 1
        }
        read -r camera_kind camera_device <<<"$resolved"
        # Only a v4l2 device is a host file we can pre-check; RealSense enumerates
        # over USB at run time.
        if [[ $camera_kind == v4l2 && ! -e $camera_device ]]; then
            warn "camera device ${camera_device} not present on host; passing it through anyway (edge will fail to open it if still absent at run time)"
        fi
    fi

    # --image-topic overrides where the edge reads frames from (only meaningful
    # for the edge-side modes; a */compressed topic is subscribed as JPEG).
    if [[ -n $image_topic ]]; then
        case $mode in
            edge|cmd_vel|edge_local|sim) ;;
            *) err "--image-topic is only valid for --mode edge|cmd_vel|edge-local|sim (not '$mode')"; return 1 ;;
        esac
        if [[ -n $camera ]]; then
            warn "--image-topic is ignored while --camera grabs the device in-process"
        fi
    fi
    RUN_IMAGE_TOPIC=$image_topic

    # --drive-motors flips cmd_vel mode from the safe /cmd_vel_vla preview topic to
    # the real /cmd_vel motor topic; it is meaningless for any other mode.
    if [[ -n $drive_motors && $mode != cmd_vel ]]; then
        err "--drive-motors is only valid for --mode cmd_vel (not '$mode')"; return 1
    fi

    # --mode edge-local (Path 2, on-edge standalone policy) is only meaningful for
    # omnivla_edge. omnivla_edge additionally supports --mode remote (Path 3 server
    # on a GPU box / Jetson) and --mode edge/sim (the Pi-side edge, path-only adapter).
    if [[ $model != omnivla_edge && $mode == edge_local ]]; then
        err "--mode edge-local is only valid for model omnivla_edge"; return 1
    fi

    case $mode in
        edge_local)
            [[ -n $host ]] && warn "--host is ignored for --mode edge-local (standalone, no cloud)"
            run_edge_local "$camera_kind" "$camera_device"
            ;;
        remote)
            if [[ -z $device ]]; then
                err "--mode remote requires --cpu or --gpu"; return 1
            fi
            [[ -n $host ]] && warn "--host is ignored for ROS 2 remote mode; use ROS_DOMAIN_ID and DDS discovery"
            run_remote "$model" "$device"
            ;;
        cmd_vel)
            if [[ -z $device ]]; then
                err "--mode cmd_vel requires --cpu or --gpu (for the local remote server)"; return 1
            fi
            [[ -n $host ]] && warn "--host is ignored for --mode cmd_vel (ROS 2 discovery)"
            local cmd_vel_topic=/cmd_vel_vla
            if [[ -n $drive_motors ]]; then
                cmd_vel_topic=/cmd_vel
                warn "--drive-motors: follower publishes to /cmd_vel — the robot's motors WILL be driven"
            fi
            run_cmd_vel "$model" "$device" "$camera_kind" "$camera_device" "$cmd_vel_topic"
            ;;
        edge|sim)
            [[ -n $host ]] && warn "--host is ignored for ROS 2 edge/sim mode; use ROS_DOMAIN_ID and DDS discovery"
            if [[ $mode == edge ]]; then
                run_edge "$model" "$camera_kind" "$camera_device"
            else
                run_sim "$model"
            fi
            ;;
        '')
            err "run: missing --mode (remote|edge|cmd_vel|sim|edge-local)"; usage; return 1 ;;
    esac
}

cmd_test() {
    local image="${IMAGES[test]}"
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        warn "image ${image} not built; building first..."
        build_one test || return 1
    fi

    # Build the default test set: explicit files (not directories) because
    # ROS2's launch_testing pytest plugin claims directories and silently
    # drops every test in them ("collected 0 items / 1 skipped"). Smoke tests
    # guarded by ASYNCVLA_E2E / OMNIVLA_E2E env vars are included but skip
    # cleanly without GPU.
    local default_paths
    mapfile -t default_paths < <(
        find "$REPO_ROOT/src" -path '*/test/test_*.py' -not -path '*/__pycache__/*' \
            | sort \
            | sed "s|^$REPO_ROOT/||"
    )

    local args
    if [[ $# -eq 0 ]]; then
        args=(-v "${default_paths[@]}")
    else
        # Decide whether the user supplied any test path. Pure-flag invocations
        # (`vla.sh test -k name`, `--lf`, `-x`) need default_paths prepended
        # so pytest doesn't fall back to cwd discovery (which would walk
        # external/ and crash on missing transitive deps).
        local has_path=false a
        for a in "$@"; do
            [[ -e $a || -e $REPO_ROOT/$a ]] && { has_path=true; break; }
        done
        if $has_path; then
            args=("$@")
        else
            args=("${default_paths[@]}" "$@")
        fi
    fi

    log "pytest in ${image} (${#args[@]} args)"
    docker compose "${COMPOSE_FILES[@]}" run --rm test python3 -m pytest "${args[@]}"
}

case ${1:-} in
    -h|--help|help|'') usage ;;
    build) shift; cmd_build "$@" ;;
    run)   shift; cmd_run "$@" ;;
    test)  shift; cmd_test "$@" ;;
    *) err "unknown command: '$1'"; usage; exit 1 ;;
esac
