"""Collect Gazebo bumper contacts from every robot collision body."""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import rclpy
from gazebo_msgs.msg import ContactsState
from rclpy.node import Node


SENSORS = ('base', 'mount', 'camera', 'caster', 'caster_wheel',
           'left_wheel', 'right_wheel')


class ContactLogger(Node):
    """Count obstacle contact onsets after filtering ground and self contacts."""

    def __init__(self, obstacles: set[str], ready: Path) -> None:
        super().__init__('rvln_bench_contact_logger')
        self.obstacles = obstacles
        self.ready = ready
        self.seen = set()
        self.events = []
        self.last_contact = {}
        self.samples = {sensor: 0 for sensor in SENSORS}
        self.contact_subscriptions = []
        for sensor in SENSORS:
            self.contact_subscriptions.append(self.create_subscription(
                ContactsState, f'/bench/contacts/{sensor}',
                lambda msg, name=sensor: self._on_contacts(name, msg), 100))

    def _on_contacts(self, sensor: str, msg: ContactsState) -> None:
        self.samples[sensor] += 1
        self.seen.add(sensor)
        if len(self.seen) == len(SENSORS) and not self.ready.exists():
            self.ready.write_text('ready\n', encoding='utf-8')
        now = time.monotonic()
        for state in msg.states:
            names = (state.collision1_name, state.collision2_name)
            robot = [name for name in names if name.startswith('raspicat::')]
            other = [name for name in names if not name.startswith('raspicat::')]
            if not robot or not other:
                continue
            obstacle = other[0].split('::', 1)[0]
            if obstacle not in self.obstacles:
                continue
            key = obstacle
            previous = self.last_contact.get(key)
            if previous is None or now - previous > 0.5:
                self.events.append({
                    'sensor': sensor, 'obstacle': obstacle,
                    'robot_collision': robot[0], 'scene_collision': other[0],
                    'sim_sec': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    'wall_monotonic_sec': now,
                })
            self.last_contact[key] = now


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--obstacles', required=True,
                        help='comma-separated static Gazebo model names')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ready', type=Path, required=True)
    args = parser.parse_args()
    args.ready.unlink(missing_ok=True)
    obstacles = set(args.obstacles.split(','))
    rclpy.init()
    node = ContactLogger(obstacles, args.ready)
    receive_errors = 0
    stopping = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        while rclpy.ok() and not stopping:
            try:
                rclpy.spin_once(node, timeout_sec=0.1)
            except RuntimeError as exc:
                if stopping:
                    break
                receive_errors += 1
                print(f'contact receive error {receive_errors}: {exc}', flush=True)
                if receive_errors >= 10:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        payload = {'schema': 1, 'obstacles': sorted(obstacles),
                   'sensors': list(SENSORS), 'samples': node.samples,
                   'receive_errors': receive_errors,
                   'events': node.events, 'collisions': len(node.events)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(json.dumps({'collisions': len(payload['events']),
                      'samples': payload['samples']}), flush=True)


if __name__ == '__main__':
    main()
