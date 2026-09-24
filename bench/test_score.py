"""Checks for the benchmark's success and path-efficiency definitions."""
import unittest

from score import score_episode, summarize


def episode(reason='model_stop', end=1.0, collisions=0, shortest=1.0):
    return {
        'id': 'e1', 'model': 'sample', 'deployment': 'remote',
        'goal_xy': [1.0, 0.0], 'stop_reason': reason,
        'collisions': collisions, 'shortest_m': shortest,
        'trace': [{'t_sec': 0.0, 'x': 0.0, 'y': 0.0},
                  {'t_sec': 2.0, 'x': end, 'y': 0.0}],
    }


class ScoreTests(unittest.TestCase):
    def test_intentional_stop_near_goal_scores_success(self):
        result = score_episode(episode())
        self.assertTrue(result['success'])
        self.assertEqual(result['spl'], 1.0)

    def test_watchdog_stop_at_goal_is_not_model_success(self):
        result = score_episode(episode(reason='stale_embedding'))
        self.assertTrue(result['endpoint_success'])
        self.assertFalse(result['success'])
        self.assertEqual(result['spl'], 0.0)

    def test_collision_free_success_is_separate(self):
        result = score_episode(episode(collisions=1))
        self.assertTrue(result['success'])
        self.assertFalse(result['collision_free_success'])

    def test_spl_unavailable_without_checked_oracle(self):
        result = score_episode(episode(shortest=None))
        self.assertIsNone(result['spl'])
        self.assertIsNone(summarize([result])['groups'][0]['spl'])

    def test_missing_collision_log_is_not_counted_as_collision_free(self):
        candidate = episode()
        del candidate['collisions']
        result = score_episode(candidate)
        self.assertIsNone(result['collision_free_success'])
        self.assertIsNone(summarize([result])['groups'][0]['collision_free_rate'])


if __name__ == '__main__':
    unittest.main()
