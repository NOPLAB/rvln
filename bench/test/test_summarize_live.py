"""Check that live reports identify and separate their simulator pose source."""
import json
import tempfile
import unittest
from pathlib import Path

from bench.summarize_live import run


def episode(source: str, episode_id: str) -> dict:
    return {
        'id': episode_id, 'model': 'asyncvla', 'scene': 'corridor',
        'deployment': 'remote_gpu_local_edge', 'pose_source': source,
        'goal_xy': [1.0, 0.0], 'shortest_m': 1.0,
        'oracle': {'world_sha256': {'corridor': 'example'}},
        'trace': [{'t_sec': 0.0, 'x': 0.0, 'y': 0.0},
                  {'t_sec': 2.0, 'x': 1.0, 'y': 0.0}],
        'stop_reason': 'goal_tolerance', 'collisions': 0,
        'contact_samples': {'physics_steps': 120},
        'video': f'asyncvla-{episode_id}.mp4', 'video_frames': 1,
        'instruction_sent': 'Go forward.', 'observations': 1, 'embeddings': 1,
        'confirmed_stop': True, 'errors': [], 'model_version': 'sample',
        'inferences': [{'round_trip_ms': 10.0}],
    }


class SummarizeLiveTest(unittest.TestCase):
    def write_episode(self, root: Path, source: str, episode_id: str) -> None:
        row = episode(source, episode_id)
        (root / row['video']).write_bytes(b'0' * 1001)
        (root / f'asyncvla-{episode_id}.json').write_text(json.dumps(row))

    def test_accepts_isaac_ground_truth_with_contact_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_episode(root, 'isaac_ground_truth', 'c01')
            (root / 'asyncvla-c01.contacts.json').write_text('{"collisions": 0}')
            report = run(root)
            self.assertEqual(report['protocol']['simulator'], 'isaac_sim')
            self.assertEqual(report['groups'][0]['success_rate']['mean'], 1.0)
            self.assertEqual(report['groups'][0]['collision_episode_rate']['mean'], 0.0)

    def test_rejects_mixed_simulator_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_episode(root, 'isaac_ground_truth', 'c01')
            self.write_episode(root, 'gazebo_model_states', 'c02')
            with self.assertRaisesRegex(ValueError, 'mixed simulator'):
                run(root)


if __name__ == '__main__':
    unittest.main()
