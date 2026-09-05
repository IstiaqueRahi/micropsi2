from dataclasses import replace
import unittest

from psi_survival.beliefs import (
    initialize_resource_beliefs,
    observe_resource_presence,
    refresh_resource_estimates,
)
from psi_survival.classifier import classify
from psi_survival.config import ExperimentConfig
from psi_survival.emotions import compute_drives, execute_operator
from psi_survival.initialization import generate_initial_world
from psi_survival.observations import perceive
from psi_survival.predictions import (
    complete_episode,
    consume_cognitive_outcomes,
    create_prediction,
    resolve_prediction,
    start_episode,
)
from psi_survival.state import ControlValues, Strategy, WorldState


class EmotionBeliefPredictionTests(unittest.TestCase):
    def test_cai_worked_example(self):
        result = classify({
            "activation": 0.3441,
            "resolution": 0.6532,
            "securing_threshold": 0.5762,
            "selection_threshold": 0.2504,
            "pleasure": 0.1783,
        })
        expected = {"Angry": 0.4808, "Fear": 0.2426, "Happy": 0.0919, "Sad": 0.7214}
        for emotion, value in expected.items():
            self.assertAlmostEqual(value, result.scores[emotion], delta=0.001)
        self.assertEqual("Sad", result.literal_argmax)

    def test_operator_persistent_state_is_not_reset(self):
        config = ExperimentConfig()
        initial = generate_initial_world(config)
        agent = initial.world.agents["agent-01"]
        drives = compute_drives(agent, config)
        first = execute_operator(agent, drives, 0, 1, config)
        second = execute_operator(agent, drives, 0, 0, config)
        self.assertEqual(1.0, first.raw["base_age"])
        self.assertEqual(2.0, second.raw["base_age"])
        self.assertGreater(first.raw["base_unexpectedness"], 0.0)
        self.assertEqual(first.raw["base_unexpectedness"], second.raw["base_unexpectedness"])

    def test_fixed_controller_preserves_classifier_inputs(self):
        dynamic = ExperimentConfig(controller_condition="dynamic")
        fixed = ExperimentConfig(controller_condition="fixed")
        initial_dynamic = generate_initial_world(dynamic)
        initial_fixed = generate_initial_world(fixed)
        a = initial_dynamic.world.agents["agent-01"]
        b = initial_fixed.world.agents["agent-01"]
        ra = execute_operator(a, compute_drives(a, dynamic), 0, 0, dynamic)
        rb = execute_operator(b, compute_drives(b, fixed), 0, 0, fixed)
        self.assertEqual(ra.classifier_inputs, rb.classifier_inputs)
        self.assertEqual(0.5, rb.behavior_controls.resolution)
        self.assertEqual(0.5, rb.behavior_controls.selection_threshold)
        self.assertEqual(0.5, rb.behavior_controls.securing_rate)

    def test_hidden_resource_state_is_absent_from_perception(self):
        config = ExperimentConfig(world_seed=17)
        initial = generate_initial_world(config)
        agent = initial.world.agents["agent-01"]
        agent.previous_controls = ControlValues(0.5, 1.0, 0.5, 1.0, 0.0, 0.5)
        observation = perceive(agent, initial.world, [], config)
        for item in observation.resources:
            self.assertFalse(hasattr(item, "type"))
            self.assertFalse(hasattr(item, "stock"))

    def test_resolution_control_changes_normal_perception(self):
        config = ExperimentConfig(world_seed=18)
        initial = generate_initial_world(config)
        focal = initial.world.agents["agent-01"]
        other = initial.world.agents["agent-02"]
        focal.position = (1.0, 1.0)
        other.position = (9.0, 1.0)
        focal.previous_controls = ControlValues(0.5, 0.0, 0.5, 0.0, 1.0, 0.5)
        low = perceive(focal, initial.world, [], config)
        focal.previous_controls = ControlValues(0.5, 1.0, 0.5, 0.0, 1.0, 0.5)
        high = perceive(focal, initial.world, [], config)
        self.assertNotIn(other.id, {item.id for item in low.agents})
        self.assertIn(other.id, {item.id for item in high.agents})

    def test_securing_control_changes_extra_scan(self):
        config = ExperimentConfig(world_seed=19)
        initial = generate_initial_world(config)
        focal = initial.world.agents["agent-01"]
        other = initial.world.agents["agent-02"]
        focal.position = (1.0, 1.0)
        other.position = (9.0, 1.0)
        focal.previous_controls = ControlValues(0.5, 0.0, 0.5, 0.0, 1.0, 0.5)
        no_scan = perceive(focal, initial.world, [], config)
        focal.previous_controls = ControlValues(0.5, 0.0, 0.5, 1.0, 0.0, 0.5)
        scan = perceive(focal, initial.world, [], config)
        self.assertFalse(no_scan.securing_scan)
        self.assertTrue(scan.securing_scan)
        self.assertNotIn(other.id, {item.id for item in no_scan.agents})
        self.assertIn(other.id, {item.id for item in scan.agents})

    def test_resource_predictor_uses_elapsed_time_not_true_stock(self):
        config = ExperimentConfig()
        initial = generate_initial_world(config)
        agent = initial.world.agents["agent-01"]
        initialize_resource_beliefs(agent, initial.world.resources, config)
        belief = next(iter(agent.resource_beliefs.values()))
        belief.last_observed_stock = 0.0
        belief.last_observed_time = 0.0
        refresh_resource_estimates(agent, 30.0, config)
        self.assertAlmostEqual(0.25, belief.estimated_stock)
        self.assertAlmostEqual(2 ** -0.3606737602, belief.confidence, places=6)

    def test_newly_perceived_corpse_uses_public_fixed_capacity(self):
        config = replace(
            ExperimentConfig(),
            resources=replace(ExperimentConfig().resources, capacity=3.0),
        )
        agent = generate_initial_world(config).world.agents["agent-01"]
        observe_resource_presence(agent, "corpse-new", (2.0, 3.0), True, 7.0, config)
        belief = agent.resource_beliefs["corpse-new"]
        self.assertEqual(0.5, belief.estimated_stock)
        refresh_resource_estimates(agent, 8.0, config)
        self.assertEqual(0.5, belief.estimated_stock)

    def test_predictions_and_competence_are_counted_once(self):
        config = ExperimentConfig()
        agent = generate_initial_world(config).world.agents["agent-01"]
        episode = start_episode(agent, Strategy.INSPECT, "reveal", "resource-01", 1)
        prediction = create_prediction(agent, episode.id, "resource_type", "food", "reveal type", 1)
        resolve_prediction(agent, prediction.id, "well", 2)
        complete_episode(agent, episode.id, 2, True)
        first = consume_cognitive_outcomes(agent)
        second = consume_cognitive_outcomes(agent)
        self.assertEqual((0, 1), first[:2])
        self.assertEqual((0, 0), second[:2])
        self.assertAlmostEqual(0.525, agent.competence_by_strategy[Strategy.INSPECT])


if __name__ == "__main__":
    unittest.main()
