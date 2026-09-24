"""Tests for the footprint-aware Gazebo shortest-route oracle."""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from route_oracle import shortest_path, verify_world


ROOT = Path(__file__).resolve().parents[1]


class RouteOracleTests(unittest.TestCase):
    def test_world_geometry_matches_oracle_source(self):
        for scene in ('corridor', 'junction', 'weave'):
            digest = verify_world(scene, ROOT / 'bench/worlds' / f'{scene}.world')
            self.assertEqual(len(digest), 64)

    def test_corridor_goal_region_shortest_is_direct(self):
        length, route = shortest_path('corridor', (0.0, 0.0), (2.35, 0.0))
        self.assertAlmostEqual(length, 2.05, places=3)
        self.assertLessEqual(math.dist(route[-1], (2.35, 0.0)), 0.30 + 1e-8)

    def test_weave_converges_with_grid_refinement(self):
        coarse, _ = shortest_path('weave', (0.0, 0.0), (2.8, 0.0), 0.025)
        fine, route = shortest_path('weave', (0.0, 0.0), (2.8, 0.0), 0.0125)
        self.assertLess(abs(coarse - fine), 0.02)
        self.assertGreater(fine, 2.8)
        self.assertLessEqual(math.dist(route[-1], (2.8, 0.0)), 0.30 + 1e-8)

    def test_manifest_matches_recomputed_routes(self):
        manifest = json.loads((ROOT / 'bench/episodes/pilot.json').read_text())
        computed = {}
        for episode in manifest['episodes']:
            key = (episode['scene'], *episode['goal_xy'])
            if key not in computed:
                computed[key], _ = shortest_path(
                    episode['scene'], tuple(episode['start_xyyaw'][:2]),
                    tuple(episode['goal_xy']))
            self.assertAlmostEqual(episode['shortest_m'], computed[key], places=6)


if __name__ == '__main__':
    unittest.main()
