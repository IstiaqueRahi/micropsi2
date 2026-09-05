from dataclasses import replace
import math
import unittest

from psi_survival.beliefs import AgentBelief, initialize_resource_beliefs, refresh_resource_estimates
from psi_survival.combat import AttackRequest, match_attacks, win_probability
from psi_survival.config import ExperimentConfig
from psi_survival.emotions import compute_drives, execute_operator
from psi_survival.initialization import generate_initial_world
from psi_survival.planner import CandidateTemplate, choose_plan, evaluate_template
from psi_survival.state import Commitment, ControlValues, Need, ResourceBelief, ResourceType, Strategy


class PlannerCombatTests(unittest.TestCase):
    def _agent_context(self, seed=23):
        config = ExperimentConfig(world_seed=seed)
        initial = generate_initial_world(config)
        agent = initial.world.agents["agent-01"]
        initialize_resource_beliefs(agent, initial.world.resources, config)
        refresh_resource_estimates(agent, 0.0, config)
        return config, initial, agent

    def test_all_outcome_trees_are_normalized(self):
        config, initial, agent = self._agent_context()
        drives = compute_drives(agent, config)
        execute_operator(agent, drives, 0, 0, config)
        decision = choose_plan(agent, drives, initial.terrain, config, 1, 0.0)
        for row in decision.considered:
            self.assertAlmostEqual(1.0, row["outcome_probability_sum"])

    def test_inspection_values_next_opportunity_consumption(self):
        config, initial, agent = self._agent_context()
        belief = ResourceBelief(
            "test-resource",
            agent.position,
            {
                ResourceType.FOOD: 0.5,
                ResourceType.WELL: 0.5,
            },
            estimated_stock=1.0,
            confidence=1.0,
        )
        template = CandidateTemplate(
            Strategy.INSPECT,
            Need.ENERGY,
            belief.resource_id,
            belief.position,
            "inspect then consume",
            1.0,
            resource_belief=belief,
        )
        route = initial.terrain.route(agent.position, belief.position, "time", stop_distance=1.0)
        plan = evaluate_template(agent, template, route, "fastest", initial.terrain, config)
        food_leaves = [leaf for leaf in plan.outcomes if leaf.details.get("resource_type") == "food" and leaf.details.get("resource_available")]
        well_leaves = [leaf for leaf in plan.outcomes if leaf.details.get("resource_type") == "well"]
        self.assertTrue(food_leaves and well_leaves)
        self.assertGreater(food_leaves[0].body[0], well_leaves[0].body[0])
        self.assertEqual(2.0, food_leaves[0].details["consume_time"])

    def test_revealed_type_changes_inspection_to_forage(self):
        config, initial, agent = self._agent_context(seed=26)
        agent.resource_beliefs = {
            "unknown": ResourceBelief(
                "unknown",
                agent.position,
                {ResourceType.FOOD: 0.5, ResourceType.WELL: 0.5},
                estimated_stock=1.0,
                confidence=1.0,
            )
        }
        drives = compute_drives(agent, config)
        before = choose_plan(agent, drives, initial.terrain, config, 1, 0.0)
        self.assertEqual(Strategy.INSPECT, before.plan.template.strategy)
        agent.commitment = None
        belief = agent.resource_beliefs["unknown"]
        belief.type_distribution = {ResourceType.WELL: 1.0}
        belief.observed_effect = (0.0, 0.8, 0.0)
        after = choose_plan(agent, drives, initial.terrain, config, 2, 1.0)
        self.assertEqual(Strategy.FORAGE, after.plan.template.strategy)

    def test_wait_replanning_preserves_the_active_episode(self):
        config, initial, agent = self._agent_context(seed=27)
        agent.position = (10.0, 10.0)
        agent.ruling_need = Need.ENERGY
        belief = ResourceBelief(
            "food",
            agent.position,
            {ResourceType.FOOD: 1.0},
            last_observed_stock=0.0,
            last_observed_time=1.0,
            estimated_stock=0.0,
            confidence=1.0,
            observed_effect=(0.8, 0.0, 0.0),
        )
        agent.resource_beliefs = {belief.resource_id: belief}
        agent.strategy = Strategy.WAIT
        agent.commitment = Commitment(
            Strategy.WAIT,
            belief.resource_id,
            belief.position,
            0.0,
            Need.ENERGY,
            "fastest",
            wait_seconds=5,
            wait_until_time=6.0,
            episode_id="active-wait",
            created_tick=1,
        )
        decision = choose_plan(agent, compute_drives(agent, config), initial.terrain, config, 3, 2.0)
        self.assertEqual("current commitment", decision.plan.template.mandatory_reason)
        self.assertFalse(decision.commitment_changed)
        self.assertEqual(4, decision.plan.template.wait_seconds)

    def test_hunt_tree_contains_loss_and_delayed_corpse_benefit(self):
        config, initial, agent = self._agent_context()
        opponent = AgentBelief("opponent", agent.position, 0.0, estimated_body=(0.5, 0.5, 0.5))
        template = CandidateTemplate(
            Strategy.HUNT,
            Need.ENERGY,
            opponent.agent_id,
            opponent.last_position,
            "hunt sequence",
            1.0,
            agent_belief=opponent,
        )
        route = initial.terrain.route(agent.position, opponent.last_position, "time", stop_distance=1.0)
        plan = evaluate_template(agent, template, route, "fastest", initial.terrain, config)
        labels = {leaf.label for leaf in plan.outcomes}
        self.assertIn("duel-loss", labels)
        win = next(leaf for leaf in plan.outcomes if leaf.label.startswith("duel-win"))
        no_corpse = 0.9 - config.physiology.energy_decay * 2
        self.assertGreater(win.body[0], no_corpse - config.physiology.energy_decay * 58)

    def test_hunting_can_win_the_common_utility_comparison(self):
        config, initial, agent = self._agent_context(seed=25)
        agent.position = (1.0, 1.0)
        agent.energy = 0.02
        agent.water = 0.02
        agent.integrity = 1.0
        agent.ruling_need = Need.WATER
        agent.competence_by_strategy[Strategy.HUNT] = 1.0
        agent.resource_beliefs = {
            "far-food": ResourceBelief(
                "far-food", (63.0, 63.0), {ResourceType.FOOD: 1.0},
                estimated_stock=1.0, confidence=1.0,
            ),
            "far-well": ResourceBelief(
                "far-well", (62.0, 63.0), {ResourceType.WELL: 1.0},
                estimated_stock=1.0, confidence=1.0,
            ),
            "far-healing": ResourceBelief(
                "far-healing", (63.0, 62.0), {ResourceType.HEALING: 1.0},
                estimated_stock=1.0, confidence=1.0,
            ),
        }
        agent.agent_beliefs = {
            "prey": AgentBelief("prey", agent.position, 0.0, estimated_body=(0.5, 0.5, 0.5))
        }
        drives = compute_drives(agent, config)
        decision = choose_plan(agent, drives, initial.terrain, config, 1, 0.0)
        self.assertEqual(Strategy.HUNT, decision.plan.template.strategy)
        self.assertTrue(any(row["strategy"] != "hunt" for row in decision.considered))

    def test_matching_is_independent_of_request_order(self):
        config = ExperimentConfig(world_seed=91)
        requests = [
            AttackRequest("a", "b"), AttackRequest("b", "a"),
            AttackRequest("b", "c"), AttackRequest("c", "d"),
        ]
        first = match_attacks(requests, 7, config)
        second = match_attacks(list(reversed(requests)), 7, config)
        first_pairs = {(pair.first_id, pair.second_id) for pair in first[0]}
        second_pairs = {(pair.first_id, pair.second_id) for pair in second[0]}
        self.assertEqual(first_pairs, second_pairs)
        self.assertTrue(all(len({pair.first_id, pair.second_id}) == 2 for pair in first[0]))
        participants = [item for pair in first[0] for item in (pair.first_id, pair.second_id)]
        self.assertEqual(len(participants), len(set(participants)))

    def test_equal_strength_probability_is_half(self):
        config = ExperimentConfig()
        self.assertEqual(0.5, win_probability((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), config))

    def test_selection_control_changes_near_threshold_motive_switch(self):
        config, initial, agent = self._agent_context(seed=24)
        # Energy is stronger than water by less than the S=1 margin but more
        # than the S=0 margin.
        agent.ruling_need = Need.WATER
        agent.energy = 0.55
        agent.water = 0.70
        agent.integrity = 0.95
        drives = compute_drives(agent, config)
        agent.controls = ControlValues(0.5, 0.5, 0.0, 0.5, 0.5, 0.5)
        low = choose_plan(agent, drives, initial.terrain, config, 1, 0.0)
        self.assertEqual(Need.ENERGY, low.ruling_need_after)
        agent.ruling_need = Need.WATER
        agent.commitment = None
        agent.controls = ControlValues(0.5, 0.5, 1.0, 0.5, 0.5, 0.5)
        high = choose_plan(agent, drives, initial.terrain, config, 1, 0.0)
        self.assertEqual(Need.WATER, high.ruling_need_after)


if __name__ == "__main__":
    unittest.main()
