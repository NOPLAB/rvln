"""Contract checks for NaVIDA's short action conversion."""
import math

import numpy as np
import pytest

from rvln_remote.backends.navida import _sample_history, action_to_embedding


def test_navida_first_chunk_action_is_metric():
    path = action_to_embedding('<answer>forward 25 cm, turn left 15 degree</answer>')
    np.testing.assert_allclose(path, [[0.25, 0.0, 1.0, 0.0]])
    assert path.dtype == np.float32


def test_navida_turn_and_stop():
    path = action_to_embedding('turn right 15 degree')
    np.testing.assert_allclose(path[0, 2:], [math.cos(math.pi / 12),
                                              -math.sin(math.pi / 12)], atol=1e-6)
    np.testing.assert_allclose(action_to_embedding('stop'), [[0.0, 0.0, 1.0, 0.0]])


@pytest.mark.parametrize('response', [
    'forward 200 cm', 'turn left 90 degree', 'left 15', 'forward -25 cm',
    '<answer>go around the obstacle</answer>',
])
def test_navida_rejects_untrusted_motion(response):
    with pytest.raises(ValueError):
        action_to_embedding(response)


def test_navida_history_keeps_ends():
    assert _sample_history(list(range(10)), 8)[0] == 0
    assert _sample_history(list(range(10)), 8)[-1] == 9
