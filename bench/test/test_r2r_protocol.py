"""Focused tests for the portable R2R-CE episode contract."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from isaac_r2r.protocol import load_episodes, score
from isaac_r2r.summary import summarize


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.episode = {
            'episode_id': '7', 'scene_id': 'mp3d/abc/abc.glb',
            'instruction': {'instruction_text': 'Go to the door.'},
            'start_position': [0.0, 0.0, 0.0],
            'goals': [{'position': [4.0, 0.0, 0.0], 'radius': 3.0}],
            'info': {'geodesic_distance': 4.0},
        }

    def test_gzip_split_and_stop_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'val_unseen.json.gz'
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                json.dump({'episodes': [self.episode]}, stream)
            episode = load_episodes(path)[0]
        path = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        self.assertEqual(score(episode, path, True)['spl'], 1.0)
        self.assertFalse(score(episode, path, False)['success'])

    def test_duplicate_episode_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'val.json'
            path.write_text(json.dumps({'episodes': [self.episode, self.episode]}))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_episodes(path)

    def test_aggregate_rejects_split_mix(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, split in enumerate(('a', 'b')):
                path = Path(directory) / f'{index}.json'
                path.write_text(json.dumps({
                    'episode_id': str(index), 'metric_namespace': 'isaac_r2r_transfer',
                    'split_sha256': split, 'scene_id': 'abc', 'success': False,
                    'spl': 0.0, 'navigation_error_m': 5.0, 'blocked_steps': 0,
                }))
                paths.append(path)
            with self.assertRaisesRegex(ValueError, 'mixed R2R-CE split'):
                summarize(paths)


if __name__ == '__main__':
    unittest.main()
