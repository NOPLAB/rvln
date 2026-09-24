"""Lint package docstrings with ament_pep257.

ament_pep257 does not read .pydocstyle, so keep the ignore list here in sync.
The _proto suffix avoids module name collisions in the combined pytest run.
"""
from pathlib import Path

import pytest

ament_pep257 = pytest.importorskip('ament_pep257.main')

_PKG_DIR = Path(__file__).resolve().parents[1]
# Exclude generated code, vendored code, and build artifacts.
_EXCLUDES = sorted(
    str(p)
    for pattern in ('**/*_pb2*.py', '**/omnivla_edge_model.py')
    for p in _PKG_DIR.glob(pattern)
) + [str(_PKG_DIR / 'build')]


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    argv = [str(_PKG_DIR), '--exclude', *_EXCLUDES,
            '--add-ignore', 'D213', 'D400', 'D401', 'D403', 'D406', 'D407', 'D413', 'D415']
    rc = ament_pep257.main(argv=argv)
    assert rc == 0, 'pep257 found style violations'
