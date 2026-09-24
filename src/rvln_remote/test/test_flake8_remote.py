"""Lint this ROS package with ament_flake8 and the repository config.

The _remote suffix keeps test module names unique in the combined pytest run.
"""
from pathlib import Path

import pytest

ament_flake8 = pytest.importorskip('ament_flake8.main')

_PKG_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PKG_DIR.parents[1]


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    rc, errors = ament_flake8.main_with_errors(
        argv=['--config', str(_REPO_ROOT / '.flake8'), str(_PKG_DIR)])
    assert rc == 0, f'flake8 found {len(errors)} style violations'
