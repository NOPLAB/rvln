"""Exercise portable shell argv/environment contracts without a Docker daemon."""

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.legacy_gazebo import run_live_matrix
from bench.validate_live import validate


RVLN = Path(__file__).resolve().parents[2]


def shell(script, args, tmp_path, overrides=None):
    tools = tmp_path / "bin"
    tools.mkdir(exist_ok=True)
    log = tmp_path / "argv.bin"
    for name in ("docker", "sbatch", "record-python", "nvidia-smi"):
        tool = tools / name
        tool.write_text(
            "#!/usr/bin/env bash\n"
            'printf "%s\\0" "$0" "$USIM_ROOT" "${RVLN_ROOT:-}" "${PYTHONPATH:-}" "$@" >> "$ARGV_LOG"\n'
            'printf "\\n" >> "$ARGV_LOG"\n'
            'if [[ ${FAKE_FAIL_BASE:-0} == 1 && " $* " == *"/Dockerfile.gazebo "* ]]; then exit 7; fi\n',
            encoding="utf-8",
        )
        tool.chmod(0o755)
    env = os.environ.copy()
    env.pop("USIM_ROOT", None)
    env.update(
        {
            "PATH": str(tools) + os.pathsep + env["PATH"],
            "ARGV_LOG": str(log),
            "RVLN_ROOT": str(tmp_path / "policy checkout"),
            "RVLN_JETSON": "0",
        }
    )
    env.update(overrides or {})
    result = subprocess.run(
        [shutil.which("bash"), str(script), *args],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = (
        [
            [field.decode() for field in row.rstrip(b"\0").split(b"\0")]
            for row in log.read_bytes().splitlines()
        ]
        if log.exists()
        else []
    )
    return result, rows


def test_sim_build_orders_distinct_contexts(tmp_path):
    result, rows = shell(RVLN / "scripts/vla.sh", ["build", "sim"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(rows) == 2
    base, overlay = (row[4:] for row in rows)
    assert base[:2] == ["build", "-f"]
    assert "--build-arg" not in base
    # Git Bash spells the checkout differently (/tmp, /c/...); compare its repo-relative tail.
    assert base[-1].replace("\\", "/").endswith("/external/usim")
    assert base[base.index("-f") + 1].endswith("/external/usim/docker/Dockerfile.gazebo")
    assert base[base.index("-t") + 1] == "usim-gazebo:local"
    assert overlay[:3] == ["build", "--build-arg", "USIM_GAZEBO_IMAGE=usim-gazebo:local"]
    assert overlay[overlay.index("-t") + 1] == "rvln-sim"
    assert overlay[-1] != base[-1]


def test_sim_build_stops_on_base_failure(tmp_path):
    result, rows = shell(
        RVLN / "scripts/vla.sh", ["build", "sim"], tmp_path, {"FAKE_FAIL_BASE": "1"}
    )
    assert result.returncode == 7
    assert len(rows) == 1


def test_other_build_target_unchanged(tmp_path):
    result, rows = shell(RVLN / "scripts/vla.sh", ["build", "test"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(rows) == 1
    argv = rows[0][4:]
    assert argv[0] == "build" and "--build-arg" not in argv
    assert argv[argv.index("-t") + 1] == "rvln-test"


def test_matrix_generates_portable_remote_and_world_arguments(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        run_live_matrix,
        "command",
        lambda argv, **kw: seen.append(argv) or SimpleNamespace(returncode=0),
    )
    config = SimpleNamespace(
        ssh_command="wsl.exe -d Linux -- ssh",
        ssh_host="user@host",
        container="generic-sim",
        container_rvln_root="/policy source",
        container_bench_root="/benchmark source",
    )
    run_live_matrix.remote(config, "squeue -j 123")
    assert seen[-1][:6] == ["wsl.exe", "-d", "Linux", "--", "ssh", "-o"]
    assert seen[-1][-2:] == ["user@host", "squeue -j 123"]
    run_live_matrix.start_sim(config, "omnivla", "corridor")
    argv = seen[-1]
    assert argv[:4] == ["docker", "exec", "-d", "generic-sim"]
    command = argv[-1]
    assert "RVLN_BENCH_ROOT='/benchmark source'" in command
    assert "world:='/benchmark source/worlds/corridor.world'" in command
    assert "'/policy source'/install/setup.bash" in command


def test_validate_non_language_backend_does_not_import_rvln():
    with pytest.raises(ValueError):
        validate({"model": "omnivla", "id": "1", "errors": ["invalid"], "inferences": []})
