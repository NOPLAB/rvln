"""Shortest collision-free path to each 0.30 m goal region on a disk footprint."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from scenes import SCENES


ROBOT_RADIUS_M = 0.34
GOAL_TOLERANCE_M = 0.30
BOUNDS = (-1.0, 5.0, -3.0, 3.0)
NEIGHBORS = ((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
             if dx or dy)
NEIGHBORS = tuple(NEIGHBORS)


def scene_boxes(scene: str) -> list[tuple[str, float, float, float, float]]:
    """Return static obstacle collision rectangles from the versioned scene source."""
    return [(name, x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2)
            for name, x, y, _z, sx, sy, _sz, _rgba in SCENES[scene]]


def verify_world(scene: str, world_file: Path) -> str:
    """Refuse to score if generated Gazebo collision boxes differ from the oracle."""
    root = ET.parse(world_file).getroot()
    observed = {}
    for model in root.findall('.//model'):
        box = model.find('./link/collision/geometry/box/size')
        if box is None:
            continue
        x, y, _z, *_ = map(float, model.findtext('pose').split())
        sx, sy, _sz = map(float, box.text.split())
        observed[model.attrib['name']] = (x - sx / 2, x + sx / 2,
                                          y - sy / 2, y + sy / 2)
    expected = {name: (x0, x1, y0, y1)
                for name, x0, x1, y0, y1 in scene_boxes(scene)}
    if observed.keys() != expected.keys() or any(
            any(abs(a - b) > 1e-9 for a, b in zip(observed[name], expected[name]))
            for name in expected):
        raise ValueError(f'world collision geometry differs from {scene} oracle')
    return hashlib.sha256(world_file.read_bytes()).hexdigest()


def shortest_path(scene: str, start: tuple[float, float], goal: tuple[float, float],
                  resolution: float = 0.0125) -> tuple[float, list[tuple[float, float]]]:
    """Use any-angle Theta* and exact disk-to-rectangle clearance checks."""
    xmin, xmax, ymin, ymax = BOUNDS
    nx = round((xmax - xmin) / resolution)
    ny = round((ymax - ymin) / resolution)
    boxes = scene_boxes(scene)
    radius_sq = ROBOT_RADIUS_M ** 2

    def xy(node: tuple[int, int]) -> tuple[float, float]:
        return xmin + node[0] * resolution, ymin + node[1] * resolution

    def index(point: tuple[float, float]) -> tuple[int, int]:
        return round((point[0] - xmin) / resolution), round((point[1] - ymin) / resolution)

    def clear_point(x: float, y: float) -> bool:
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False
        for _name, x0, x1, y0, y1 in boxes:
            dx = max(x0 - x, 0.0, x - x1)
            dy = max(y0 - y, 0.0, y - y1)
            if dx * dx + dy * dy <= radius_sq:
                return False
        return True

    @lru_cache(maxsize=None)
    def free(node: tuple[int, int]) -> bool:
        if not (0 <= node[0] <= nx and 0 <= node[1] <= ny):
            return False
        return clear_point(*xy(node))

    @lru_cache(maxsize=300000)
    def visible(a: tuple[int, int], b: tuple[int, int]) -> bool:
        ax, ay = xy(a)
        bx, by = xy(b)
        steps = max(1, math.ceil(math.hypot(bx - ax, by - ay) / (resolution / 3)))
        return all(clear_point(ax + (bx - ax) * i / steps,
                               ay + (by - ay) * i / steps)
                   for i in range(steps + 1))

    def distance(a: tuple[int, int], b: tuple[int, int]) -> float:
        x0, y0 = xy(a)
        x1, y1 = xy(b)
        return math.hypot(x1 - x0, y1 - y0)

    def heuristic(node: tuple[int, int]) -> float:
        x, y = xy(node)
        return max(0.0, math.hypot(x - goal[0], y - goal[1]) - GOAL_TOLERANCE_M)

    source = index(start)
    if not free(source):
        raise ValueError(f'{scene} start is not collision-free')
    best = {source: 0.0}
    parent = {source: source}
    queue = [(heuristic(source), 0.0, source)]
    while queue:
        _estimate, cost, node = heapq.heappop(queue)
        if cost > best[node] + 1e-9:
            continue
        x, y = xy(node)
        if math.hypot(x - goal[0], y - goal[1]) <= GOAL_TOLERANCE_M and free(node):
            route = [node]
            while route[-1] != source:
                route.append(parent[route[-1]])
            route.reverse()
            return cost, [xy(point) for point in route]
        for dx, dy in NEIGHBORS:
            neighbor = (node[0] + dx, node[1] + dy)
            if not free(neighbor) or not visible(node, neighbor):
                continue
            ancestor = parent[node]
            if visible(ancestor, neighbor):
                new_cost = best[ancestor] + distance(ancestor, neighbor)
                new_parent = ancestor
            else:
                new_cost = cost + distance(node, neighbor)
                new_parent = node
            if new_cost + 1e-9 < best.get(neighbor, math.inf):
                best[neighbor] = new_cost
                parent[neighbor] = new_parent
                heapq.heappush(queue, (new_cost + heuristic(neighbor),
                                       new_cost, neighbor))
    raise ValueError(f'no collision-free route to {goal} in {scene}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('bench/episodes/pilot.json'))
    parser.add_argument('--worlds', type=Path, default=Path('bench/worlds'))
    parser.add_argument('--resolution', type=float, default=0.0125)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text(encoding='utf-8'))
    hashes = {scene: verify_world(scene, args.worlds / f'{scene}.world')
              for scene in SCENES}
    routes = {}
    for episode in data['episodes']:
        key = (episode['scene'], *episode['goal_xy'])
        if key not in routes:
            length, points = shortest_path(episode['scene'],
                                           tuple(episode['start_xyyaw'][:2]),
                                           tuple(episode['goal_xy']), args.resolution)
            routes[key] = {'shortest_m': length, 'route_xy': points}
        episode['shortest_m'] = routes[key]['shortest_m']
    data['oracle'] = {'method': 'Theta* disk-to-goal-region',
                      'robot_radius_m': ROBOT_RADIUS_M,
                      'goal_tolerance_m': GOAL_TOLERANCE_M,
                      'grid_resolution_m': args.resolution,
                      'world_sha256': hashes}
    args.out.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    for key, route in routes.items():
        print(key, round(route['shortest_m'], 4), route['route_xy'])


if __name__ == '__main__':
    main()
