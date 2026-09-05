from dataclasses import replace
import math
import unittest

from psi_survival.config import ExperimentConfig
from psi_survival.initialization import generate_initial_world
from psi_survival.rng import keyed_seed
from psi_survival.state import Terrain
from psi_survival.terrain import TerrainMap, traverse_route


class ConfigRngTerrainTests(unittest.TestCase):
    def test_preregistered_defaults_and_roundtrip(self):
        config = ExperimentConfig()
        config.validate()
        self.assertEqual(tuple(range(1001, 1011)), config.preregistration.pilot_seeds)
        self.assertEqual(2001, config.preregistration.evaluation_seed_start)
        self.assertEqual(900001, config.preregistration.demonstration_seed)
        self.assertEqual(180.0, config.preregistration.target_ci_half_width_seconds)
        self.assertEqual(3_600, config.playback.ticks)
        self.assertEqual(20, config.playback.fps)
        self.assertEqual(180.0, config.playback.seconds)
        self.assertEqual(config.digest(), ExperimentConfig.from_dict(config.to_dict()).digest())

    def test_keyed_draws_do_not_depend_on_call_order(self):
        keys = [("scan", tick, agent) for tick in range(5) for agent in ("a", "b")]
        forward = {key: keyed_seed(19, *key) for key in keys}
        reverse = {key: keyed_seed(19, *key) for key in reversed(keys)}
        self.assertEqual(forward, reverse)

    def test_map_roads_override_seeded_puddles(self):
        config = ExperimentConfig()
        terrain = TerrainMap.generate(config.terrain, 5)
        for line in config.terrain.road_lines:
            self.assertTrue(all(terrain.cells[line][x] is Terrain.ROAD for x in range(config.terrain.width)))
            self.assertTrue(all(terrain.cells[y][line] is Terrain.ROAD for y in range(config.terrain.height)))
        self.assertTrue(any(cell is Terrain.PUDDLE for row in terrain.cells for cell in row))

    def test_route_estimate_matches_integrated_route_for_center_goal(self):
        config = ExperimentConfig()
        terrain = TerrainMap.generate(config.terrain, 7)
        for start in ((1.0, 1.0), (1.2, 1.0), (1.0, 1.8)):
            for metric in ("time", "energy"):
                route = terrain.route(start, (51.0, 43.0), metric, stop_distance=1.0)
                expected = terrain.estimate_route_cost(start, (51.0, 43.0), metric, stop_distance=1.0)
                actual = route.travel_time if metric == "time" else route.movement_energy
                self.assertAlmostEqual(expected, actual, places=10)

    def test_movement_and_passive_depletion_are_integrated(self):
        config = ExperimentConfig()
        terrain = TerrainMap.generate(config.terrain, 11)
        route = terrain.route((1.0, 1.0), (2.0, 1.0), "time")
        result = traverse_route(route, (0.9, 0.95, 0.95), 1.0, config.physiology)
        self.assertAlmostEqual(0.95 - config.physiology.water_decay, result.body[1])
        self.assertAlmostEqual(0.95 - config.physiology.integrity_decay, result.body[2])
        self.assertLess(result.body[0], 0.9 - config.physiology.energy_decay)

    def test_death_crossing_stops_movement_exactly_once(self):
        base = ExperimentConfig()
        physiology = replace(base.physiology, initial_energy=0.0005)
        config = replace(base, physiology=physiology)
        terrain = TerrainMap.generate(config.terrain, 11)
        route = terrain.route((1.0, 1.0), (3.0, 1.0), "time")
        result = traverse_route(route, (0.0005, 0.95, 0.95), 1.0, config.physiology)
        self.assertTrue(result.died)
        self.assertGreaterEqual(result.death_offset, 0.0)
        self.assertEqual(0.0, result.body[0])

    def test_initial_world_inventory_and_separation(self):
        config = ExperimentConfig()
        initial = generate_initial_world(config)
        self.assertEqual(12, len(initial.world.agents))
        self.assertEqual(24, len(initial.world.resources))
        types = [resource.type.value for resource in initial.world.resources.values()]
        self.assertEqual(8, types.count("food"))
        self.assertEqual(8, types.count("well"))
        self.assertEqual(4, types.count("healing"))
        self.assertEqual(4, types.count("mixed_harmful"))
        positions = [agent.position for agent in initial.world.agents.values()]
        for index, first in enumerate(positions):
            for second in positions[index + 1:]:
                self.assertGreaterEqual(math.dist(first, second), 2.0)


if __name__ == "__main__":
    unittest.main()
