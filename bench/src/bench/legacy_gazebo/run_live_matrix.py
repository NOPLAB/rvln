"""Orchestrate local Gazebo/Edge and one Slurm GPU backend at a time.

Run on the Windows workstation with Docker Desktop and WSL SSH configured.
Benchmark and RVLN roots, SSH transport, and runtime container are explicit.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

from bench.paths import workspace_root


CHECKPOINTS = {
    "asyncvla": ("models/AsyncVLA_release", 750000),
    "omnivla": ("models/omnivla-original", 120000),
    "omnivla_edge": ("models/omnivla-edge/omnivla-edge.pth", 120000),
    "movla": ("models/movla/stage_a_v9b", 120000),
    "navila": ("models/navila-llama3-8b-8f", 120000),
    "navida": ("models/NaVIDA", 120000),
}
EPISODES = ("c01", "j01", "j02", "w01")
TEMPLATES = {"c01": "go straight ahead", "j01": "turn left ahead", "j02": "turn right ahead"}


def command(args: list[str], *, timeout: float = 120, check: bool = True):
    result = subprocess.run(
        args, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout
    )
    if check and result.returncode:
        raise RuntimeError(
            f"{args[:4]} exit={result.returncode}: {result.stdout[-1200:]} {result.stderr[-1200:]}"
        )
    return result


def remote(config, statement: str, *, timeout: float = 120):
    return command(
        [
            *shlex.split(config.ssh_command),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            config.ssh_host,
            statement,
        ],
        timeout=timeout,
    )


def docker(*args: str, timeout: float = 120, check: bool = True):
    return command(["docker", *args], timeout=timeout, check=check)


def wait_server(config, job: str) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        health = docker(
            "exec",
            config.container,
            "python3",
            "-c",
            f"import json; from urllib.request import urlopen; print(json.load("
            f"urlopen({(config.server + '/health')!r}, timeout=2))"
            f'["slurm_job_id"])',
            timeout=8,
            check=False,
        )
        if health.returncode == 0 and health.stdout.strip() == job:
            return

    status = remote(
        config,
        f'squeue -j {job} -o "%i %t %R"; '
        f"tail -n 20 {shlex.quote(config.remote_bench_root)}/runs/slurm-{job}.err",
    )
    raise RuntimeError(f"server unavailable for job {job}: {status.stdout}")


def start_sim(config, model: str, scene: str) -> None:
    docker("stop", config.container, timeout=25, check=False)
    docker("start", config.container)
    adapter = "asyncvla" if model == "asyncvla" else "omnivla"
    script = (
        "source /opt/ros/humble/setup.bash; "
        "source /opt/sim_ws/install/setup.bash; "
        f"source {shlex.quote(config.container_rvln_root)}/install/setup.bash; "
        f"export RVLN_BENCH_ROOT={shlex.quote(config.container_bench_root)}; "
        f"export PYTHONPATH={shlex.quote(config.container_bench_root)}/src:"
        f"{shlex.quote(config.container_rvln_root)}/external/usim/src:${{PYTHONPATH:-}}; "
        "export RVLN_BENCH_CONTACTS=1; "
        f"bash {shlex.quote(config.container_bench_root)}/scripts/isaac6_xvfb.sh "
        f"ros2 launch rvln_bringup sim.launch.py adapter_kind:={adapter} "
        f"gui:=false rviz:=false "
        f"world:={shlex.quote(config.container_bench_root + '/worlds/' + scene + '.world')} "
        "> /tmp/rvln-sim.log 2>&1"
    )
    docker("exec", "-d", config.container, "bash", "-lc", script)


def run_episode(config, model: str, episode: str) -> dict:
    scene = {"c": "corridor", "j": "junction", "w": "weave"}[episode[0]]
    start_sim(config, model, scene)
    output = f"{config.container_bench_root}/runs/{config.run_name}/{model}-{episode}.json"
    video = f"{config.container_bench_root}/runs/{config.run_name}/{model}-{episode}.mp4"
    script = (
        "source /opt/ros/humble/setup.bash; "
        "source /opt/sim_ws/install/setup.bash; "
        f"source {shlex.quote(config.container_rvln_root)}/install/setup.bash; "
        f"export RVLN_BENCH_ROOT={shlex.quote(config.container_bench_root)}; "
        f"export PYTHONPATH={shlex.quote(config.container_bench_root)}/src:"
        f"{shlex.quote(config.container_rvln_root)}/external/usim/src:${{PYTHONPATH:-}}; "
        f"mkdir -p {shlex.quote(config.container_bench_root + '/runs/' + config.run_name)}; "
        f"python3 -m bench.legacy_gazebo.live_episode --url {shlex.quote(config.server)} "
        f"--model {model} --episode {episode} --out {shlex.quote(output)} "
        f"--video {shlex.quote(video)} --duration 35"
    )
    if model == "movla":
        script += f' --text "{TEMPLATES[episode]}"'
    try:
        result = docker("exec", config.container, "bash", "-lc", script, timeout=150)
        print(result.stdout.strip(), flush=True)
        local = config.bench_root / "runs" / config.run_name / f"{model}-{episode}.json"
        docker("cp", f"{config.container}:{output}", str(local))
        docker("cp", f"{config.container}:{video}", str(local.with_suffix(".mp4")))
        docker(
            "cp",
            f"{config.container}:{output[:-5]}.contacts.json",
            str(local.with_suffix(".contacts.json")),
        )
        docker(
            "cp",
            f"{config.container}:{output[:-5]}.contacts.log",
            str(local.with_suffix(".contacts.log")),
        )
        return json.loads(local.read_text())
    except Exception:
        log = docker(
            "exec", config.container, "tail", "-n", "50", "/tmp/rvln-sim.log", check=False
        )
        print(log.stdout[-3000:], flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=CHECKPOINTS, default=list(CHECKPOINTS))
    parser.add_argument("--episodes", nargs="+", choices=EPISODES, default=list(EPISODES))
    parser.add_argument("--bench-root", type=Path, default=workspace_root())
    parser.add_argument(
        "--ssh-command",
        default=os.environ.get("RVLN_BENCH_SSH_COMMAND", "ssh"),
        help='command prefix, e.g. "wsl.exe -d DISTRO -- ssh"',
    )
    parser.add_argument("--ssh-host", default=os.environ.get("RVLN_BENCH_SSH_HOST"))
    parser.add_argument("--remote-bench-root", default=os.environ.get("RVLN_BENCH_REMOTE_ROOT"))
    parser.add_argument("--remote-rvln-root", default=os.environ.get("RVLN_REMOTE_ROOT"))
    parser.add_argument("--container", default=os.environ.get("RVLN_BENCH_CONTAINER"))
    parser.add_argument(
        "--container-bench-root",
        default=os.environ.get("RVLN_BENCH_CONTAINER_ROOT", "/workspace/bench"),
    )
    parser.add_argument(
        "--container-rvln-root", default=os.environ.get("RVLN_CONTAINER_ROOT", "/workspace")
    )
    parser.add_argument("--server", default=os.environ.get("RVLN_BACKEND_URL"))
    parser.add_argument("--bind", default=os.environ.get("RVLN_INFER_BIND"))
    parser.add_argument("--run-name", default=os.environ.get("RVLN_BENCH_RUN_NAME", "live_matrix"))
    parser.add_argument("--slurm-output", default=os.environ.get("RVLN_BENCH_SLURM_OUTPUT"))
    parser.add_argument("--slurm-error", default=os.environ.get("RVLN_BENCH_SLURM_ERROR"))
    parser.add_argument(
        "--movla-checkpoint", default=os.environ.get("RVLN_BENCH_MOVLA_CHECKPOINT")
    )
    config = parser.parse_args()
    for name in ("ssh_host", "remote_bench_root", "remote_rvln_root", "container", "server"):
        if not getattr(config, name):
            parser.error(f"--{name.replace('_', '-')} or its environment override is required")
    server = urlsplit(config.server)
    if server.scheme not in ("http", "https") or not server.hostname:
        parser.error("--server must be an HTTP URL")
    bind = config.bind or server.hostname
    port = server.port or (443 if server.scheme == "https" else 80)
    q = shlex.quote
    (config.bench_root / "runs" / config.run_name).mkdir(parents=True, exist_ok=True)
    for model in config.models:
        checkpoint, step = CHECKPOINTS[model]
        checkpoint = (
            config.movla_checkpoint
            if model == "movla" and config.movla_checkpoint
            else config.remote_rvln_root + "/" + checkpoint
        )
        output = config.slurm_output or config.remote_bench_root + "/runs/slurm-%j.out"
        error = config.slurm_error or config.remote_bench_root + "/runs/slurm-%j.err"
        statement = (
            f"mkdir -p {q(config.remote_bench_root + '/runs')}; "
            f"export RVLN_BENCH_ROOT={q(config.remote_bench_root)} RVLN_BENCH_WORKSPACE={q(config.remote_rvln_root)}; "
            f"sbatch --parsable --chdir={q(config.remote_bench_root)} "
            f"--output={q(output)} --error={q(error)} "
            f"{q(config.remote_bench_root + '/scripts/slurm_live.sbatch')} "
            f"--backend {model} --checkpoint {q(checkpoint)} "
            f"--resume-step {step} --bind {q(bind)} --port {port}"
        )
        job = remote(config, statement).stdout.strip().splitlines()[-1].strip().split(";")[0]
        if not job.isdecimal():
            raise RuntimeError(f"invalid Slurm job ID: {job}")
        print(f"{model}: Slurm job {job}", flush=True)
        try:
            docker("start", config.container, check=False)
            wait_server(config, job)
            selected = [
                episode for episode in config.episodes if model != "movla" or episode in TEMPLATES
            ]
            for episode in selected:
                row = run_episode(config, model, episode)
                final = row["trace"][-1] if row["trace"] else None
                print(
                    f"{model}/{episode}: {row['stop_reason']} "
                    f"pose={final} errors={row['errors'][:2]}",
                    flush=True,
                )
        finally:
            try:
                remote(config, f"scancel {job}", timeout=60)
            finally:
                docker("stop", config.container, timeout=25, check=False)


if __name__ == "__main__":
    main()
