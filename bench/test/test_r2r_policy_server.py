"""Test the real HTTP policy boundary and its per-episode state rules."""
import io
import struct
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from bench.episode import Observation

from isaac_r2r.policy import R2RHttpPolicy
from isaac_r2r.policy_server import PolicySession, make_handler


class Model:
    def __init__(self):
        self.calls = []
        self.resets = 0

    def reset(self):
        self.resets += 1
        return 0

    def act(self, rgb, depth, tokens, previous_action, state):
        self.calls.append((rgb.shape, depth.shape, tokens.copy(), previous_action, state))
        return (1 if state < 2 else 0), state + 1


def observation():
    buffer = io.BytesIO()
    Image.new('RGB', (256, 256), (30, 40, 50)).save(buffer, format='JPEG')
    depth = struct.pack('<f', 2.0) * (256 * 256)
    return Observation(buffer.getvalue(), depth, 256, 256)


class PolicyServerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        checkpoint = Path(self.directory.name) / 'model.pth'
        checkpoint.write_bytes(b'fixed-test-checkpoint')
        self.model = Model()
        self.session = PolicySession(self.model, checkpoint)

    def tearDown(self):
        self.directory.cleanup()

    def test_state_and_previous_action_reset_between_episodes(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.session))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            policy = R2RHttpPolicy(f'http://127.0.0.1:{server.server_port}',
                                   [2, 3], self.session.policy_id)
            policy.reset('one')
            actions = [policy.act('one', frame, 'Move.', observation())
                       for frame in range(3)]
            self.assertEqual(actions, ['forward', 'forward', 'stop'])
            with self.assertRaises(Exception):
                policy.act('one', 3, 'Move.', observation())
            policy.reset('two')
            self.assertEqual(policy.act('two', 0, 'Turn.', observation()), 'forward')
            self.assertEqual(self.model.resets, 2)
            self.assertEqual([call[3:] for call in self.model.calls],
                             [(None, 0), (1, 1), (1, 2), (None, 0)])
            self.assertTrue(all(call[:3] == ((224, 224, 3), (256, 256, 1), [2, 3])
                                for call in self.model.calls))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_rejects_changed_instruction_and_duplicate_frame(self):
        self.session.reset({'episode_id': 'one'})
        obs = observation()
        import base64
        payload = {'episode_id': 'one', 'frame_id': 0, 'instruction': 'Go.',
                   'instruction_tokens': [2, 3],
                   'jpeg_base64': base64.b64encode(obs.jpeg).decode(),
                   'depth_f32_base64': base64.b64encode(obs.depth_f32).decode(),
                   'depth_width': 256, 'depth_height': 256}
        self.session.act(payload)
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.session.act(payload)
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.session.act({**payload, 'frame_id': 1,
                              'instruction_tokens': [2, 4]})
        self.assertEqual(len(self.model.calls), 1)

    def test_rejects_invalid_model_action_without_advance(self):
        class InvalidModel(Model):
            def act(self, *args):
                return 4, 99

        self.session.model = InvalidModel()
        self.session.reset({'episode_id': 'one'})
        obs = observation()
        import base64
        payload = {'episode_id': 'one', 'frame_id': 0, 'instruction': 'Go.',
                   'instruction_tokens': [2, 3],
                   'jpeg_base64': base64.b64encode(obs.jpeg).decode(),
                   'depth_f32_base64': base64.b64encode(obs.depth_f32).decode(),
                   'depth_width': 256, 'depth_height': 256}
        with self.assertRaisesRegex(ValueError, 'invalid action ID'):
            self.session.act(payload)
        self.assertEqual(self.session.next_frame, 0)


if __name__ == '__main__':
    unittest.main()
