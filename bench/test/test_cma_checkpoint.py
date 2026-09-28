"""Ensure CMA checkpoint loading rejects objects outside tensor/config data."""
import pickle
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from isaac_r2r.cma_model import _load_state_dict
except ImportError:
    torch = None


class UnexpectedObject:
    pass


@unittest.skipIf(torch is None, 'torchvision or PyTorch is not installed')
class CMACheckpointTest(unittest.TestCase):
    def test_defaults_only_legacy_batchnorm_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'legacy.pth'
            torch.save({'state_dict': {}}, checkpoint)
            state = _load_state_dict(checkpoint)
            self.assertEqual(list(state),
                             ['net.rgb_encoder.cnn.1.num_batches_tracked'])
            self.assertEqual(int(state[list(state)[0]]), 0)

    def test_rejects_unexpected_global_without_executing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'unexpected.pth'
            torch.save({'state_dict': {}, 'extra': UnexpectedObject()}, checkpoint)
            with self.assertRaisesRegex(pickle.UnpicklingError,
                                        'unsupported CMA checkpoint global'):
                _load_state_dict(checkpoint)


if __name__ == '__main__':
    unittest.main()
