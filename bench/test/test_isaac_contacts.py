"""Verify Isaac obstacle contacts preserve the pilot collision definition."""
import unittest

from isaac_rvln.contacts import ObstacleContactLog


class ContactLogTest(unittest.TestCase):
    def test_filters_ground_self_and_merges_continuous_obstacle_contact(self):
        now = [10.0]
        log = ObstacleContactLog({'left_wall'}, lambda: now[0])
        robot = '/World/Raspicat/base/collision'
        wall = '/World/Environment/left_wall/Body'
        ground = '/World/Environment/Ground/Body'
        log.tick()
        log.record(robot, ground, robot, ground, 0.1, 2)
        log.record(robot, '/World/Raspicat/wheel', robot,
                   '/World/Raspicat/wheel', 0.1, 2)
        log.record(robot, wall, robot, wall, 0.1, 3)
        now[0] = 10.2
        log.record(wall, robot, wall, robot, 0.2, 3)
        now[0] = 10.8
        log.record(robot, wall, robot, wall, 0.8, 1)
        result = log.report()
        self.assertEqual(result['collisions'], 2)
        self.assertEqual(result['samples'], {'physics_steps': 1})
        self.assertEqual([row['obstacle'] for row in result['events']],
                         ['left_wall', 'left_wall'])
        self.assertEqual(result['events'][1]['contact_points'], 1)


if __name__ == '__main__':
    unittest.main()
