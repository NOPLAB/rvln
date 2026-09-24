#!/usr/bin/env bash
# control.sh — drive a running rvln stack: motor power + VLA goals.
#
# Works against whichever vla.sh mode is up — edge, cmd_vel, sim, or edge-local
# — because all of them run the same edge node, which subscribes to the goal
# topic and (where a robot/raspimouse is present) offers the /motor_power
# service. There is nothing to control in a bare `--mode remote` box: that
# container is a headless gRPC server with no ROS node, no goal subscriber, and
# no motors. Point this at the host running the *edge* side instead.
#
# Thin host wrapper around scripts/control.py: it runs the Python helper inside
# the running edge container (where rclpy + the rvln_msgs overlay live),
# sourcing the ROS overlays first. /workspace is bind-mounted into the container,
# so the helper is reached at /workspace/scripts/control.py.
#
# Usage:
#   scripts/control.sh motor on|off
#   scripts/control.sh goal pose X Y [THETA] [FRAME]    # FRAME default: odom
#   scripts/control.sh goal text "go down the hallway"
#   scripts/control.sh goal image /workspace/path/to.jpg
#   scripts/control.sh stop                             # motor off (coast to halt)
#   scripts/control.sh status
#   scripts/control.sh logs [-f] [server|edge]          # tail container output
#
# `logs` is a host-side helper (a `docker logs` shortcut), not a ROS command: it
# surfaces model-load progress + runtime output. In cmd_vel / detached-remote
# modes the VLA model loads in the remote *server* container; in edge-local it
# loads in the *edge* container. With no target it prefers the server when one is
# up (that's where the big checkpoint + CLIP load happens), else the edge. Pass
# -f/--follow to stream. It runs on the host regardless of the direct-run path
# below, since `docker logs` is meaningless from inside a container.
#
# First run is typically:  scripts/control.sh motor on
#                          scripts/control.sh goal pose 2 0
#
# Container selection: by default we probe the edge-capable images in order
# (real, sim, test) and use the first running container. Override with
# RVLN_CONTAINER=<name-or-id> to pin one explicitly (RASPICAT_SIM_CONTAINER
# is still honoured for backward compatibility). If a ROS environment with
# rvln_msgs is already on PATH (e.g. you're inside the container), the
# helper runs directly instead of via docker exec.
#
# ROS_DOMAIN_ID: if set in the host environment it is forwarded into the
# container so the helper talks on the same DDS domain as the edge node. In the
# direct-run path it is already inherited from the host env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Edge-capable images, most-specific first (shared by `logs` and the exec path).
CANDIDATE_IMAGES=(
    "${RVLN_REAL_IMAGE:-rvln-real}"
    "${RVLN_SIM_IMAGE:-rvln-sim}"
    "${RVLN_TEST_IMAGE:-rvln-test}"
)

# First running edge container across the candidate images (honours the
# RVLN_CONTAINER / RASPICAT_SIM_CONTAINER overrides).
find_edge_cid() {
    local cid="${RVLN_CONTAINER:-${RASPICAT_SIM_CONTAINER:-}}"
    if [[ -z $cid ]]; then
        local img
        for img in "${CANDIDATE_IMAGES[@]}"; do
            cid="$(docker ps -q --filter "ancestor=${img}" | head -1)"
            [[ -n $cid ]] && break
        done
    fi
    printf '%s' "$cid"
}

# `logs` is a host-side `docker logs` shortcut — handle it before the direct-run
# path, since it must run on the host even if rclpy happens to be importable.
if [[ ${1:-} == logs ]]; then
    shift
    follow=()
    target=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -f|--follow) follow=(-f); shift ;;
            server|edge) target="$1"; shift ;;
            *) echo "usage: control.sh logs [-f|--follow] [server|edge]" >&2; exit 2 ;;
        esac
    done

    # compose names the server container rvln-remote-<n> (service
    # "remote" in docker/compose.yaml); rvln-cmdvel-server-<pid> is the
    # pre-compose name, kept for stacks started by an older vla.sh.
    server_cid="$(docker ps -q --filter name=rvln-remote --filter name=rvln-cmdvel-server | head -1)"
    edge_cid="$(find_edge_cid)"

    case "$target" in
        server) cid="$server_cid"; label="remote server" ;;
        edge)   cid="$edge_cid";   label="edge" ;;
        *)      if [[ -n $server_cid ]]; then cid="$server_cid"; label="remote server"
                else cid="$edge_cid"; label="edge"; fi ;;
    esac

    if [[ -z ${cid:-} ]]; then
        echo "error: no ${target:-rvln} container running to show logs for." >&2
        echo "       start a stack first, e.g.: scripts/vla.sh run omnivla_edge --mode cmd_vel --gpu" >&2
        exit 1
    fi
    echo "==> ${label} logs (${cid}); Ctrl-C to stop" >&2
    exec docker logs "${follow[@]}" "$cid"
fi

# Already inside a ROS env with our messages? Run directly.
if python3 -c 'import rclpy, rvln_msgs.msg' >/dev/null 2>&1; then
    exec python3 "${REPO_ROOT}/scripts/control.py" "$@"
fi

# The edge node — the thing that owns the goal topic + motor service — runs in
# one of CANDIDATE_IMAGES (defined above); the remote server images
# (omnivla/asyncvla) run no ROS node, so they are intentionally omitted.
cid="$(find_edge_cid)"
if [[ -z $cid ]]; then
    echo "error: no running rvln edge container found." >&2
    echo "       looked for images: ${CANDIDATE_IMAGES[*]}" >&2
    echo "       start the edge side first, e.g.:" >&2
    echo "         scripts/vla.sh run omnivla --mode cmd_vel --gpu" >&2
    echo "         scripts/vla.sh run omnivla --mode edge --host HOST:PORT" >&2
    echo "         scripts/vla.sh run omnivla --mode sim  --host HOST:PORT" >&2
    echo "       (or set RVLN_CONTAINER=<name-or-id>)" >&2
    exit 1
fi

# Build a safely-quoted arg list for the inner shell.
quoted=""
for a in "$@"; do
    quoted+=" $(printf '%q' "$a")"
done

# Forward ROS_DOMAIN_ID into the container when it is set on the host, so the
# helper joins the same DDS domain as the running edge node.
exec_env=()
if [[ -n ${ROS_DOMAIN_ID:-} ]]; then
    exec_env+=(-e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")
fi

exec docker exec "${exec_env[@]}" "$cid" bash -lc "
    source /opt/ros/humble/setup.bash
    [ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash
    exec python3 /workspace/scripts/control.py${quoted}
"
