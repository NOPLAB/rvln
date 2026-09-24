"""ament_pep257 でパッケージの docstring を lint する。

ament_pep257 は設定ファイル (.pydocstyle) を読まないため、同じ ignore リストを
ここで明示的に渡す。変更時は .pydocstyle / CMakeLists.txt 側と同期させること。
ファイル名の _core はリポジトリ内一意にするため(vla.sh test の一括 pytest 対策)。
"""
from pathlib import Path

import pytest

ament_pep257 = pytest.importorskip('ament_pep257.main')

_PKG_DIR = Path(__file__).resolve().parents[1]
# 生成コード (*_pb2*.py)・vendored コード (omnivla_edge_model.py)・build 産物は対象外。
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
    assert rc == 0, 'pep257 がスタイル違反を検出'
