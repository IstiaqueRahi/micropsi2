from dataclasses import replace
from types import SimpleNamespace
import unittest

from psi_survival.beliefs import initialize_resource_beliefs
from psi_survival.config import ExperimentConfig
from psi_survival.engine import SimulationEngine
from psi_survival.experiment import focused_conditions, pilot_sample_size, primary_analysis, study_analysis
from psi_survival.predictions import start_episode
from psi_survival.state import Commitment, Need, Strategy
from psi_survival.terrain import Route


class EngineExperimentTests(unittest.TestCase):
    def _short_config(self, seed=47, agents=2, ticks=5, trace="summary"):
        base = ExperimentConfig()
        return replace(
            base,
            world_seed=seed,
            agent_count=agents,
            trace_level=trace,
            playback=replace(base.playback, ticks=ticks),
        )

    def test_focused_matrix_has_the_preregistered_18_cells(self):
        conditions = focused_conditions()
        self.assertEqual(18, len(conditions))
        self.assertEqual(10, sum(item.family == "primary_and_ablations" for item in conditions))
        self.assertEqual(4, sum(item.family == "resource_sensitivity" for item in conditions))
        self.assertEqual(4, sum(item.family == "population_sensitivity" for item in conditions))

    def test_container_reordering_does_not_change_seeded_run(self):
        config = self._short_config(ticks=8)
        first = SimulationEngine(config)
        second = SimulationEngine(config)
        second.world.agents = dict(reversed(list(second.world.agents.items())))
        second.world.resources = dict(reversed(list(second.world.resources.items())))
        result_first = first.run()
        result_second = second.run()
        self.assertEqual(result_first.summary, result_second.summary)
        for agent_id in result_first.agents:
            self.assertEqual(result_first.agents[agent_id]["final_body"], result_second.agents[agent_id]["final_body"])

    def test_proportional_contention_conserves_stock(self):
        config = self._short_config(ticks=1)
        engine = SimulationEngine(config)
        resource = next(iter(engine.world.resources.values()))
        resource.stock = 0.25
        agents = list(engine.world.agents.values())
        for agent in agents:
            agent.position = resource.position
            episode = start_episode(agent, Strategy.FORAGE, "consume", resource.id, 1)
            agent.commitment = Commitment(Strategy.FORAGE, resource.id, resource.position, 0.0, Need.ENERGY, "fastest", episode_id=episode.id)
            agent.resource_beliefs[resource.id].type_distribution = {resource.type: 1.0}
        engine._consumption_phase({resource.id: [(agents[0], 0.25, 0.25), (agents[1], 0.25, 0.25)]}, 1)
        self.assertEqual(0.0, resource.stock)
        allocations = [event.details["allocated"] for event in engine.world.events if event.event_type == "consumption"]
        self.assertEqual([0.125, 0.125], allocations)
        self.assertAlmostEqual(0.25, sum(allocations))
        self.assertTrue(all(agent.resource_beliefs[resource.id].last_observed_stock == 0.0 for agent in agents))

    def test_harmful_death_creates_one_corpse_and_no_resurrection(self):
        config = self._short_config(ticks=1, agents=1)
        engine = SimulationEngine(config)
        resource = next(item for item in engine.world.resources.values() if item.type.value == "mixed_harmful")
        agent = next(iter(engine.world.agents.values()))
        agent.position = resource.position
        agent.integrity = 0.05
        episode = start_episode(agent, Strategy.FORAGE, "consume", resource.id, 1)
        agent.commitment = Commitment(Strategy.FORAGE, resource.id, resource.position, 0.0, Need.ENERGY, "fastest", episode_id=episode.id)
        agent.resource_beliefs[resource.id].type_distribution = {resource.type: 1.0}
        engine._consumption_phase({resource.id: [(agent, 0.25, resource.stock)]}, 1)
        self.assertFalse(agent.alive)
        corpses = [item for item in engine.world.resources.values() if item.is_corpse]
        self.assertEqual(1, len(corpses))
        self.assertEqual(0.0, agent.integrity)
        self.assertEqual(1, sum(event.event_type == "death" for event in engine.world.events))

    def test_wait_duration_starts_after_arrival_interaction(self):
        config = self._short_config(ticks=6, agents=1)
        engine = SimulationEngine(config)
        resource = next(item for item in engine.world.resources.values() if item.type.value == "food")
        agent = next(iter(engine.world.agents.values()))
        agent.position = resource.position
        episode = start_episode(agent, Strategy.WAIT, "wait then consume", resource.id, 1)
        agent.commitment = Commitment(
            Strategy.WAIT,
            resource.id,
            resource.position,
            0.0,
            Need.ENERGY,
            "fastest",
            wait_seconds=5,
            episode_id=episode.id,
        )
        agent.resource_beliefs[resource.id].type_distribution = {resource.type: 1.0}
        for tick in range(1, 6):
            engine.world.time = float(tick - 1)
            requests = engine._inspection_and_requests({agent.id: None}, set(), tick)  # type: ignore[arg-type]
            self.assertEqual({}, requests)
        engine.world.time = 5.0
        requests = engine._inspection_and_requests({agent.id: None}, set(), 6)  # type: ignore[arg-type]
        self.assertIn(resource.id, requests)

    def test_overlapping_attacks_resolve_one_duel_and_one_death(self):
        config = self._short_config(seed=53, agents=3, ticks=1)
        engine = SimulationEngine(config)
        a, b, c = sorted(engine.world.agents.values(), key=lambda item: item.id)
        for agent in (a, b, c):
            agent.position = (10.0, 10.0)

        def attack_decision(attacker, target):
            episode = start_episode(attacker, Strategy.HUNT, "attack", target.id, 1)
            attacker.strategy = Strategy.HUNT
            attacker.commitment = Commitment(
                Strategy.HUNT,
                target.id,
                target.position,
                0.0,
                Need.ENERGY,
                "fastest",
                episode_id=episode.id,
            )
            template = SimpleNamespace(strategy=Strategy.HUNT, target_id=target.id)
            plan = SimpleNamespace(template=template, route=Route(attacker.position, target.position, "time", ()))
            return SimpleNamespace(plan=plan)

        decisions = {
            a.id: attack_decision(a, b),
            c.id: attack_decision(c, b),
        }
        matched = engine._combat_phase(decisions, 1)
        self.assertEqual(2, len(matched))
        self.assertEqual(2, len(engine.world.living_agents))
        self.assertEqual(1, sum(item.is_corpse for item in engine.world.resources.values()))
        self.assertEqual(1, sum(event.event_type == "duel" for event in engine.world.events))
        self.assertEqual(1, sum(event.event_type == "attack_deferred" for event in engine.world.events))
        self.assertEqual(1, sum(event.event_type == "death" for event in engine.world.events))

    def test_primary_sizing_and_analysis_use_seed_pairs(self):
        config = ExperimentConfig()
        rows = []
        for offset, seed in enumerate(config.preregistration.pilot_seeds):
            rows.extend([
                {"condition_id": "primary-dynamic-combat-on", "seed": seed, "restricted_mean_survival_time": 2000.0 + 10 * offset},
                {"condition_id": "primary-fixed-combat-on", "seed": seed, "restricted_mean_survival_time": 1900.0},
            ])
        matrix = {
            "seed_blocks": list(config.preregistration.pilot_seeds),
            "base_configuration": config.to_dict(),
            "rows": rows,
        }
        sizing = pilot_sample_size(matrix, config)
        self.assertGreaterEqual(sizing["frozen_evaluation_n"], 30)
        self.assertEqual(18 * sizing["frozen_evaluation_n"], sizing["evaluation_runs"])
        analysis = primary_analysis(matrix)
        self.assertEqual(10, analysis["n_pairs"])
        self.assertEqual(145.0, analysis["mean_difference_seconds"])

        changed = replace(config, regeneration_multiplier=4.0)
        with self.assertRaisesRegex(ValueError, "Pilot configuration differs"):
            pilot_sample_size(matrix, changed)

    def test_study_analysis_reports_predeclared_secondary_pairs(self):
        seeds = (2001, 2002, 2003)
        rows = []
        secondary = {
            "resource_quantity_acquired": 1.0,
            "failed_journeys": 0,
            "inspections": 1,
            "securing_scans": 2,
            "motive_switches": 3,
            "strategy_switches": 4,
            "attack_requests": 0,
            "resolved_duels": 0,
            "unclassified_fraction": 0.1,
            "prediction_expected": 2,
            "prediction_unexpected": 1,
        }
        for condition_index, condition in enumerate(focused_conditions()):
            for seed in seeds:
                rows.append({
                    "condition_id": condition.id,
                    "seed": seed,
                    "restricted_mean_survival_time": 1000.0 + condition_index,
                    "survivor_fraction": 0.5,
                    "secondary_metrics": secondary,
                })
        analysis = study_analysis({"seed_blocks": list(seeds), "rows": rows})
        self.assertEqual(3, analysis["primary"]["n_pairs"])
        self.assertEqual(16, len(analysis["secondary"]))
        self.assertIn("dynamic_minus_fixed_regen_low", analysis["secondary"])


if __name__ == "__main__":
    unittest.main()
