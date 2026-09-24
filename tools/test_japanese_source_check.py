"""Focused tests for the repository Japanese-source check."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from japanese_source_check import SRC_ROOT, check_japanese, main


class JapaneseSourceCheckTest(unittest.TestCase):
    """Exercise both Flake8 and non-Python entry points."""

    def test_flake8_reports_only_src(self):
        source = 'message = "\u65e5\u672c\u8a9e"\n'
        self.assertEqual(len(list(check_japanese(source, SRC_ROOT / 'test.py'))), 1)
        self.assertEqual(list(check_japanese(source, Path(__file__))), [])
        self.assertEqual(list(check_japanese('message = "English"\n',
                                             SRC_ROOT / 'test.py')), [])

    def test_cli_checks_non_python_files(self):
        with tempfile.TemporaryDirectory(dir=SRC_ROOT) as directory:
            source = Path(directory) / 'test.cmake'
            source.write_text('# \u30c6\u30b9\u30c8\n', encoding='utf-8')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([source]), 1)
            self.assertIn('RJV100', output.getvalue())
            source.write_text('# English\n', encoding='utf-8')
            self.assertEqual(main([source]), 0)


if __name__ == '__main__':
    unittest.main()
