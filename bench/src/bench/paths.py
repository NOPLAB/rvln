"""RVLN benchmark resources, independent of the generic simulator checkout."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    """Use RVLN_BENCH_ROOT or this source checkout's bench directory.

    Wheels exclude resources and require RVLN_BENCH_ROOT. Never use USIM_ROOT.
    """
    override = os.environ.get("RVLN_BENCH_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "episodes").is_dir():
        return root
    raise ValueError("set RVLN_BENCH_ROOT to the RVLN benchmark resource directory")
