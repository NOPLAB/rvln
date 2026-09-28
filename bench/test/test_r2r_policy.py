"""R2R policy transport must send audited tokens without hidden goals."""
import unittest
from unittest.mock import patch

from bench.episode import HttpPolicy, Observation
from isaac_r2r.policy import R2RHttpPolicy


class R2RPolicyTest(unittest.TestCase):
    def test_action_carries_preprocessed_tokens_and_keeps_reset_scoped(self):
        policy = R2RHttpPolicy('http://127.0.0.1:8765', [2, 3, 0])
        with patch.object(HttpPolicy, '_request', side_effect=[
            {'episode_id': 'one'},
            {'episode_id': 'one', 'frame_id': 0, 'action': 'left'},
        ]) as request:
            policy.reset('one')
            action = policy.act('one', 0, 'Turn left.', Observation(b'jpeg'))
        self.assertEqual(action, 'left')
        reset_payload = request.call_args_list[0].args[1]
        action_payload = request.call_args_list[1].args[1]
        self.assertNotIn('instruction_tokens', reset_payload)
        self.assertEqual(action_payload['instruction_tokens'], [2, 3, 0])
        self.assertNotIn('goal', action_payload)

    def test_rejects_incompatible_minimal_split_tokens(self):
        with self.assertRaisesRegex(ValueError, 'preprocessed'):
            R2RHttpPolicy('http://127.0.0.1:8765', [2707])

    def test_reset_rejects_different_model_identity(self):
        policy = R2RHttpPolicy('http://127.0.0.1:8765', policy_id='sha256:expected')
        with patch.object(HttpPolicy, '_request', return_value={
            'episode_id': 'one', 'policy_id': 'sha256:other',
        }):
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                policy.reset('one')


if __name__ == '__main__':
    unittest.main()
