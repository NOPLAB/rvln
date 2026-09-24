"""Tests for filtering and de-duplicating Gazebo contact events."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import rclpy
from gazebo_msgs.msg import ContactState, ContactsState

from contact_logger import ContactLogger


def sample(other: str) -> ContactsState:
    message = ContactsState()
    state = ContactState()
    state.collision1_name = 'raspicat::base_footprint::base_link_collision'
    state.collision2_name = other
    message.states.append(state)
    return message


class ContactLoggerTests(unittest.TestCase):
    def test_ground_and_self_contacts_are_excluded_and_obstacle_is_deduplicated(self):
        rclpy.init()
        with tempfile.TemporaryDirectory() as directory:
            node = ContactLogger({'left_wall'}, Path(directory) / 'ready')
            try:
                node._on_contacts('base', sample('ground_plane::link::collision'))
                node._on_contacts('base', sample('raspicat::left_wheel::collision'))
                self.assertEqual(node.events, [])
                wall = sample('left_wall::body::box')
                node._on_contacts('base', wall)
                node._on_contacts('left_wheel', wall)
                self.assertEqual(len(node.events), 1)
                self.assertEqual(node.events[0]['obstacle'], 'left_wall')
            finally:
                node.destroy_node()
                rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()
