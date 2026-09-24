"""Pure contract tests for NaVILA text actions and video sampling."""
import math

import numpy as np
import PIL.Image
import pytest

from rvln_remote.backends.navila import action_to_embedding, sample_and_pad_images


def test_navila_actions_use_metric_robot_frame():
    np.testing.assert_allclose(
        action_to_embedding('The next action is move forward 25 cm.'),
        [[0.25, 0.0, 1.0, 0.0]], atol=1e-6,
    )
    left = action_to_embedding('The next action is turn left 30 degree.')
    right = action_to_embedding('The next action is turn right 15 degrees.')
    assert left.shape == right.shape == (1, 4)
    np.testing.assert_allclose(left[0, 2:],
                               [math.cos(math.pi / 6), math.sin(math.pi / 6)], atol=1e-6)
    np.testing.assert_allclose(right[0, 2:],
                               [math.cos(math.pi / 12), -math.sin(math.pi / 12)], atol=1e-6)
    np.testing.assert_array_equal(action_to_embedding('The next action is stop.'),
                                  [[0.0, 0.0, 1.0, 0.0]])


@pytest.mark.parametrize('response', [
    'turn left 30 degrees', 'The next action is move backward 25 cm.',
    'The next action is move forward 200 cm.',
    'The next action is turn right 90 degrees.',
    'Do not stop.',
])
def test_navila_rejects_unsupported_action(response):
    with pytest.raises(ValueError):
        action_to_embedding(response)


def test_navila_padding_retains_current_frame():
    current = PIL.Image.new('RGB', (64, 64), 'red')
    frames = sample_and_pad_images([current], 8)
    assert len(frames) == 8
    assert frames[-1] is current
    assert frames[0].getpixel((0, 0)) == (0, 0, 0)
