"""Count Isaac PhysX obstacle-contact onsets for the RVLN pilot worlds."""
from __future__ import annotations

import json
import time
from pathlib import Path


class ObstacleContactLog:
    """Filter ground/self contacts and merge continuous contact per obstacle."""

    def __init__(self, obstacles: set[str], clock=time.monotonic):
        if not obstacles or 'Ground' in obstacles:
            raise ValueError('contact logger needs named non-ground obstacles')
        self.obstacles = obstacles
        self.clock = clock
        self.events = []
        self.last_contact = {}
        self.physics_steps = 0

    def tick(self) -> None:
        self.physics_steps += 1

    def record(self, actor0: str, actor1: str, collider0: str, collider1: str,
               sim_sec: float, points: int) -> None:
        robot = '/World/Raspicat/'
        scene = '/World/Environment/'
        if actor0.startswith(robot) and actor1.startswith(scene):
            robot_collision, scene_collision = collider0, collider1
        elif actor1.startswith(robot) and actor0.startswith(scene):
            robot_collision, scene_collision = collider1, collider0
        else:
            return
        if not robot_collision.startswith(robot) or not scene_collision.startswith(scene):
            return
        obstacle = scene_collision[len(scene):].split('/', 1)[0]
        if obstacle not in self.obstacles:
            return
        now = self.clock()
        previous = self.last_contact.get(obstacle)
        if previous is None or now - previous > 0.5:
            self.events.append({
                'sensor': robot_collision.rsplit('/', 1)[-1],
                'obstacle': obstacle,
                'robot_collision': robot_collision,
                'scene_collision': scene_collision,
                'sim_sec': float(sim_sec),
                'wall_monotonic_sec': now,
                'contact_points': int(points),
            })
        self.last_contact[obstacle] = now

    def report(self) -> dict:
        return {'schema': 1, 'simulator': 'isaac_sim',
                'obstacles': sorted(self.obstacles),
                'samples': {'physics_steps': self.physics_steps},
                'receive_errors': 0,
                'events': self.events, 'collisions': len(self.events)}

    def save(self, path: Path) -> dict:
        payload = self.report()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        return payload
