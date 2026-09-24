"""Orchestrate local Gazebo/Edge and one Slurm GPU backend at a time.

Run on the Windows workstation with Docker Desktop and WSL SSH configured.
The output under bench/runs is deliberately ignored by Git.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/mnt/workspace/nop/raspicat_vla'
SERVER = 'http://100.76.158.87:8765'
CONTAINER = 'rvln-sim-check'
RUN_NAME = 'contact_2026-09-25'
CHECKPOINTS = {
    'asyncvla': ('models/AsyncVLA_release', 750000),
    'omnivla': ('models/omnivla-original', 120000),
    'omnivla_edge': ('models/omnivla-edge/omnivla-edge.pth', 120000),
    'movla': ('/mnt/workspace/nop/vla/runs/stage_a_v9b', 120000),
    'navila': ('models/navila-llama3-8b-8f', 120000),
    'navida': ('models/NaVIDA', 120000),
}
EPISODES = ('c01', 'j01', 'j02', 'w01')
TEMPLATES = {'c01': 'go straight ahead', 'j01': 'turn left ahead',
             'j02': 'turn right ahead'}


def command(args: list[str], *, timeout: float = 120, check: bool = True):
    result = subprocess.run(args, text=True, encoding='utf-8', errors='replace',
                            capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'{args[:4]} exit={result.returncode}: '
                           f'{result.stdout[-1200:]} {result.stderr[-1200:]}')
    return result


def remote(statement: str, *, timeout: float = 120):
    return command(['wsl.exe', '-d', 'FedoraLinux-43', '--', 'ssh',
                    '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                    'nop@pve1ubuntu', statement], timeout=timeout)


def docker(*args: str, timeout: float = 120, check: bool = True):
    return command(['docker', *args], timeout=timeout, check=check)


def wait_server(job: str) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        health = docker('exec', CONTAINER, 'python3', '-c',
                        f'import json,urllib.request; print(json.load('
                        f'urllib.request.urlopen("{SERVER}/health", timeout=2))'
                        f'["slurm_job_id"])',
                        timeout=8, check=False)
        if health.returncode == 0 and health.stdout.strip() == job:
            return
        time.sleep(3)
    status = remote(f'squeue -j {job} -o "%i %t %R"; '
                    f'tail -n 20 {REMOTE}/bench/runs/slurm-{job}.err')
    raise RuntimeError(f'server unavailable for job {job}: {status.stdout}')


def start_sim(model: str, scene: str) -> None:
    docker('stop', CONTAINER, timeout=25, check=False)
    docker('start', CONTAINER)
    docker('exec', CONTAINER, 'bash', '-lc',
           'rm -f /tmp/.X99-lock /tmp/.X11-unix/X99; '
           'nohup Xvfb :99 -screen 0 1280x720x24 >/tmp/xvfb.log 2>&1 &')
    adapter = 'asyncvla' if model == 'asyncvla' else 'omnivla'
    script = (
        'source /opt/ros/humble/setup.bash; '
        'source /opt/sim_ws/install/setup.bash; '
        'source /workspace/install/setup.bash; '
        'export DISPLAY=:99 RVLN_BENCH_CONTACTS=1; '
        f'ros2 launch rvln_bringup sim.launch.py adapter_kind:={adapter} '
        f'gui:=false rviz:=false world:=/workspace/bench/worlds/{scene}.world '
        '> /tmp/rvln-sim.log 2>&1'
    )
    docker('exec', '-d', CONTAINER, 'bash', '-lc', script)


def run_episode(model: str, episode: str) -> dict:
    scene = {'c': 'corridor', 'j': 'junction', 'w': 'weave'}[episode[0]]
    start_sim(model, scene)
    output = f'/workspace/bench/runs/{RUN_NAME}/{model}-{episode}.json'
    video = f'/workspace/bench/runs/{RUN_NAME}/{model}-{episode}.mp4'
    script = (
        'source /opt/ros/humble/setup.bash; '
        'source /opt/sim_ws/install/setup.bash; '
        'source /workspace/install/setup.bash; '
        f'python3 /workspace/bench/live_episode.py --url {SERVER} '
        f'--model {model} --episode {episode} --out {output} '
        f'--video {video} --duration 35'
    )
    if model == 'movla':
        script += f' --text "{TEMPLATES[episode]}"'
    try:
        result = docker('exec', CONTAINER, 'bash', '-lc', script, timeout=150)
        print(result.stdout.strip(), flush=True)
        local = ROOT / 'bench' / 'runs' / RUN_NAME / f'{model}-{episode}.json'
        docker('cp', f'{CONTAINER}:{output}', str(local))
        docker('cp', f'{CONTAINER}:{video}',
               str(local.with_suffix('.mp4')))
        docker('cp', f'{CONTAINER}:{output[:-5]}.contacts.json',
               str(local.with_suffix('.contacts.json')))
        docker('cp', f'{CONTAINER}:{output[:-5]}.contacts.log',
               str(local.with_suffix('.contacts.log')))
        return json.loads(local.read_text())
    except Exception:
        log = docker('exec', CONTAINER, 'tail', '-n', '50', '/tmp/rvln-sim.log',
                     check=False)
        print(log.stdout[-3000:], flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=CHECKPOINTS,
                        default=list(CHECKPOINTS))
    parser.add_argument('--episodes', nargs='+', choices=EPISODES, default=list(EPISODES))
    args = parser.parse_args()
    (ROOT / 'bench' / 'runs' / RUN_NAME).mkdir(parents=True, exist_ok=True)
    for model in args.models:
        checkpoint, step = CHECKPOINTS[model]
        statement = (f'cd {REMOTE}; sbatch --parsable bench/slurm_live.sbatch '
                     f'--backend {model} --checkpoint {checkpoint} '
                     f'--resume-step {step} --bind 100.76.158.87 --port 8765')
        job = remote(statement).stdout.strip().splitlines()[-1].strip()
        print(f'{model}: Slurm job {job}', flush=True)
        try:
            docker('start', CONTAINER, check=False)
            wait_server(job)
            selected = [episode for episode in args.episodes
                        if model != 'movla' or episode in TEMPLATES]
            for episode in selected:
                row = run_episode(model, episode)
                final = row['trace'][-1] if row['trace'] else None
                print(f'{model}/{episode}: {row["stop_reason"]} '
                      f'pose={final} errors={row["errors"][:2]}', flush=True)
        finally:
            remote(f'scancel {job}', timeout=60)
            docker('stop', CONTAINER, timeout=25, check=False)


if __name__ == '__main__':
    main()
