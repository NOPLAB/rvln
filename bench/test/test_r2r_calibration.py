"""Validate coordinate registration before running real R2R episodes."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from isaac_r2r.calibration import calibrate_scene, fit_rigid_transform
from isaac_r2r.run import load_scene_record, transform_point


POINTS = [[0, 0, 0], [2, 0, 0], [0, 3, 0], [1, 1, 2]]
ROTATION = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
TRANSLATION = np.array([4.0, -2.0, 1.0])


def pairs_for(rotation):
    return [{'habitat': point,
             'isaac': (rotation @ point + TRANSLATION).tolist()}
            for point in POINTS]


class CalibrationTest(unittest.TestCase):
    def test_fit_round_trips_through_scene_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            usd = root / 'scan.usd'
            usd.write_bytes(b'usd fixture')
            pairs = root / 'landmarks.json'
            pairs.write_text(json.dumps({'landmarks': pairs_for(ROTATION)}))
            registry = root / 'scenes.json'

            result = calibrate_scene(pairs, 'abc', usd, registry, 0.05, 800.0)
            loaded = load_scene_record(registry, 'abc')

            self.assertEqual(result['scenes']['abc']['landmarks'], 4)
            self.assertEqual(loaded['dome_light_intensity'], 800.0)
            self.assertLess(result['scenes']['abc']['max_error_m'], 1e-10)
            for source, expected in zip(POINTS, pairs_for(ROTATION)):
                np.testing.assert_allclose(
                    transform_point(loaded['matrix'], source), expected['isaac'],
                    atol=1e-10)
            calibrate_scene(pairs, 'def', usd, registry, 0.05)
            self.assertEqual(set(json.loads(registry.read_text())['scenes']),
                             {'abc', 'def'})
            with self.assertRaisesRegex(ValueError, 'already registered'):
                calibrate_scene(pairs, 'abc', usd, registry, 0.05)

    def test_reflection_and_collinear_landmarks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'residual'):
            fit_rigid_transform(pairs_for(np.diag([-1, 1, 1])), 0.05)
        with self.assertRaisesRegex(ValueError, 'collinear'):
            fit_rigid_transform([
                {'habitat': [value, 0, 0], 'isaac': [value, 0, 0]}
                for value in (0, 1, 2)], 0.05)

    def test_nonfinite_light_intensity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            usd = root / 'scan.usd'
            usd.write_bytes(b'usd fixture')
            pairs = root / 'landmarks.json'
            pairs.write_text(json.dumps({'landmarks': pairs_for(ROTATION)}))
            with self.assertRaisesRegex(ValueError, 'intensity'):
                calibrate_scene(pairs, 'abc', usd, root / 'scenes.json',
                                0.05, float('nan'))


if __name__ == '__main__':
    unittest.main()
