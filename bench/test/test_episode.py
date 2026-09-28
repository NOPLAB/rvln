"""Verify common orchestration without importing either simulator."""
import unittest

from bench.episode import EpisodeRequest, EpisodeSimulator, Observation, Policy, run_episode


class FakeSimulator(EpisodeSimulator):
    def __init__(self):
        self.x = 0.0

    def reset(self, episode):
        self.x = 0.0
        return [0.0, 0.0, 0.0]

    def observe(self):
        return Observation(b'jpeg')

    def apply(self, action):
        self.x += 1.0
        return [self.x, 0.0, 0.0]

    def close(self):
        pass


class FakePolicy(Policy):
    def __init__(self):
        self.reset_id = None

    def reset(self, episode_id):
        self.reset_id = episode_id

    def act(self, episode_id, frame_id, instruction, observation):
        assert self.reset_id == episode_id
        assert observation.jpeg == b'jpeg'
        return 'stop' if frame_id == 1 else 'forward'


class EpisodeTest(unittest.TestCase):
    def test_reset_then_forward_then_model_stop(self):
        result = run_episode(FakeSimulator(), FakePolicy(),
                             EpisodeRequest('a', 'Go forward.', None,
                                            frozenset({'forward', 'stop'}), 'stop'), 5)
        self.assertEqual(result['actions'], ['forward', 'stop'])
        self.assertEqual(result['positions'][-1], [1.0, 0.0, 0.0])
        self.assertTrue(result['stopped'])


if __name__ == '__main__':
    unittest.main()
