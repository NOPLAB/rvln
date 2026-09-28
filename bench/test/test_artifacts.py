"""Result writes must preserve a resumable manifest on replacement failure."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench.artifacts import write_json


class ArtifactTest(unittest.TestCase):
    def test_failed_replace_preserves_previous_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            write_json(path, {'status': 'running', 'completed_episodes': 1})
            with patch('bench.artifacts.os.replace', side_effect=OSError('replace failed')):
                with self.assertRaisesRegex(OSError, 'replace failed'):
                    write_json(path, {'status': 'running', 'completed_episodes': 2})
            self.assertEqual(json.loads(path.read_text())['completed_episodes'], 1)
            self.assertEqual(list(Path(directory).glob('.manifest.json.*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
