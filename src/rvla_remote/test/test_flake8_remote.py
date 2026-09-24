"""ament_flake8 でパッケージ全体を lint する(設定はリポジトリ直下の .flake8)。

ファイル名の _remote は vla.sh test が全パッケージを 1 回の pytest で回すため
(同名モジュールのインポート衝突回避でリポジトリ内一意にする)。
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
    assert rc == 0, f'flake8 が {len(errors)} 件のスタイル違反を検出'
