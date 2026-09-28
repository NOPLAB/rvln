"""Check the Raspicat differential drive command contract independently of Isaac."""
import unittest

from isaac_rvln.sim import wheel_velocities


class DriveTest(unittest.TestCase):
    def test_forward_and_turn(self):
        self.assertEqual(wheel_velocities(0.2, 0.0, 0.05, 0.2), (4.0, 4.0))
        self.assertEqual(wheel_velocities(0.0, 1.0, 0.05, 0.2), (-2.0, 2.0))

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):
            wheel_velocities(0.0, 0.0, 0.0, 0.2)


if __name__ == '__main__':
    unittest.main()
