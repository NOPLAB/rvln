"""Reject Japanese text in owned ROS sources, including non-Python files."""

from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
JAPANESE = re.compile(
    '[\u3000-\u303f\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff'
    '\uf900-\ufaff\uff01-\uff9f\U0001b000-\U0001b12f\U00020000-\U0002fa1f]'
)


def _in_src(path):
    try:
        path.resolve().relative_to(SRC_ROOT)
    except ValueError:
        return False
    return True


def check_japanese(physical_line, filename):
    """Flake8 local plugin: report Japanese text in a Python source line."""
    if not _in_src(Path(filename)):
        return
    match = JAPANESE.search(physical_line)
    if match:
        yield match.start(), 'RJV100 Japanese text is not allowed under src/'


def main(paths):
    """Check all text sources, including ROS message, XML, and CMake files."""
    failed = False
    for arg in paths or [SRC_ROOT]:
        path = Path(arg)
        files = path.rglob('*') if path.is_dir() else [path]
        for file in files:
            if not file.is_file() or not _in_src(file):
                continue
            try:
                lines = file.read_text(encoding='utf-8').splitlines()
            except (UnicodeError, OSError):
                continue
            for number, line in enumerate(lines, 1):
                match = JAPANESE.search(line)
                if match:
                    print(f'{file}:{number}:{match.start() + 1}: RJV100 Japanese text is not '
                          'allowed under src/')
                    failed = True
    return int(failed)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
