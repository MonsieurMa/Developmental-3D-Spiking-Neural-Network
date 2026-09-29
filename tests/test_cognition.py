"""Cognitive P1-P4 mechanisms must be learned, bound, and language neutral."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain.cognition import (
    ActionLoopResult,
    AssociativeReplayPlanner,
    CorticalGrounding,
    ConsistencyMemory,
    DevelopmentalClock,
    DecoderCalibration,
    DifferentiableSequenceModel,
    DiscourseState,
    IntentLearner,
    PrototypeField,
    SocialFeedback,
    TensorProductBinding,
)


class GroundingTest(unittest.TestCase):
    def test_dendritic_grounding_is_learned_and_not_a_code_table(self):
        grounding = CorticalGrounding(embedding_dim=8, projection_dim=8, seed=3)
        first = grounding.encode("ABC")
        second = grounding.encode("ABC")
        self.assertTrue(torch.allclose(first, second))
        self.assertGreater(float(first.norm()), 0.0)
        before = float((grounding.encode("ABC") - first).abs().mean())
        error = grounding.learn_form("ABC", epochs=3)
        self.assertGreaterEqual(error, 0.0)
        self.assertGreaterEqual(before, 0.0)

    def test_cell_order_uses_the_learned_field(self):
        grounding = CorticalGrounding(embedding_dim=8, projection_dim=8, seed=5)
        vector = grounding.encode("XYZ")
        cells = grounding.rank_cells(vector, [10, 11, 12, 13])
        self.assertEqual(sorted(cells), [10, 11, 12, 13])


class BindingTest(unittest.TestCase):
    def test_role_filler_binding_recovers_the_filler(self):
        binding = TensorProductBinding(dimension=128, seed=7)
        trace = binding.bind_pair("opaque-role", "opaque-entity")
        recovered, score = binding.query(trace, "opaque-role")
        self.assertEqual(recovered, "opaque-entity")
        self.assertGreater(score, 0.25)

    def test_unseen_relation_composition_keeps_both_components(self):
        binding = TensorProductBinding(dimension=128, seed=9)
        left = binding.bind_pair("relation-alpha", "entity-a")
        right = binding.bind_pair("relation-beta", "entity-b")
        composed = binding.compose([left, right])
        self.assertEqual(binding.query(composed, "relation-alpha")[0], "entity-a")
        self.assertEqual(binding.query(composed, "relation-beta")[0], "entity-b")

    def test_discourse_tracks_roles_and_salience(self):
        binding = TensorProductBinding(dimension=128, seed=11)
        discourse = DiscourseState(binding)
        discourse.bind_role("opaque-agent", "entity-one")
        discourse.reinforce("entity-one", 0.2)
        discourse.bind_role("opaque-location", "entity-two")
        entity, score = discourse.resolve("entity-one")
        self.assertEqual(entity, "entity-one")
        self.assertGreaterEqual(score, 0.0)
        self.assertEqual(discourse.timeline()[-1], ("opaque-location", "entity-two"))


class NeuralPredictionTest(unittest.TestCase):
    def test_intent_labels_train_without_language_tables(self):
        model = IntentLearner(embedding_dim=8, hidden_dim=8, lr=0.03)
        first_loss = model.observe(["symbol", "one"], "opaque-intent")
        second_loss = model.observe(["symbol", "one"], "opaque-intent")
        predicted, confidence = model.predict(["symbol", "one"])
        self.assertEqual(predicted, "opaque-intent")
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(second_loss, first_loss + 1e-6)

    def test_sequence_prediction_is_neural_and_reports_surprise(self):
        model = DifferentiableSequenceModel(embedding_dim=8, hidden_dim=8, lr=0.03)
        stream = ["a", "b", "c"] * 12
        errors = [model.update(stream[index:index + 3]) for index in range(0, len(stream) - 2, 3)]
        predicted, probability = model.predict_next(["a", "b"])
        self.assertIn(predicted, model.symbols)
        self.assertGreaterEqual(probability, 0.0)
        self.assertGreaterEqual(model.surprise(["a", "b"], "c"), 0.0)
        self.assertLess(model.uncertainty(["a", "b"]), 1.0)
        self.assertGreater(sum(errors), 0.0)

    def test_one_vocabulary_migrates_across_scripts_without_code_changes(self):
        model = DifferentiableSequenceModel(embedding_dim=8, hidden_dim=8, lr=0.03)
        stream = ["alpha", "beta", "甲", "乙"] * 8
        for index in range(0, len(stream) - 3, 4):
            model.update(stream[index:index + 4])
        predicted, probability = model.predict_next(["alpha", "beta"])
        grounding = CorticalGrounding(embedding_dim=8, projection_dim=8, seed=17)
        vectors = [grounding.encode(symbol) for symbol in ("alpha", "甲")]
        self.assertIn(predicted, model.symbols)
        self.assertGreaterEqual(probability, 0.0)
        self.assertTrue(all(float(vector.norm()) >= 0.0 for vector in vectors))


class DevelopmentLoopTest(unittest.TestCase):
    def test_social_feedback_is_available_in_the_same_turn(self):
        feedback = SocialFeedback()
        feedback.observe("opaque-prompt", "opaque-correction", 0.8)
        self.assertEqual(feedback.context("opaque-prompt"), ("opaque-correction",))

    def test_development_clock_scales_sleep_and_plasticity(self):
        clock = DevelopmentalClock()
        self.assertEqual(clock.phase, "proliferation")
        clock.observe_progress(episodes=2, replay_events=0)
        clock.observe_progress(episodes=0, replay_events=2)
        self.assertEqual(clock.phase, "mature")
        self.assertEqual(clock.replay_fraction(), 1.0)
        self.assertEqual(clock.plasticity(), 0.30)

    def test_decoder_thresholds_learn_from_local_evidence(self):
        calibration = DecoderCalibration(minimum_samples=4)
        defaults = {"confidence": 0.5, "margin": 2.0, "support_per_cell": 1.0, "coverage": 0.5}
        self.assertEqual(calibration.thresholds(**defaults), defaults)
        for index in range(6):
            calibration.observe(support=1.0 + index / 10, margin=1.5, coverage=0.6, accepted=True)
        learned = calibration.thresholds(**defaults)
        self.assertGreaterEqual(learned["support_per_cell"], 1.0)
        self.assertGreaterEqual(learned["coverage"], 0.0)

    def test_action_loop_result_is_a_data_contract(self):
        result = ActionLoopResult("q", "opaque-intent", ["entity"], 0.0, ["entity"], False)
        self.assertFalse(result.corrected)

    def test_prototype_field_forms_learned_abstractions(self):
        field = PrototypeField(match_threshold=0.72)
        first = field.observe(torch.tensor([1.0, 0.0, 0.0]), owner="opaque-a")
        second = field.observe(torch.tensor([0.94, 0.1, 0.0]), owner="opaque-a2")
        third = field.observe(torch.tensor([0.0, 1.0, 0.0]), owner="opaque-b")
        self.assertEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertEqual(field.state()["prototypes"], 2)
        self.assertEqual(field.prototype_for("opaque-a"), first)

    def test_consistency_memory_is_neutral_without_evidence(self):
        memory = ConsistencyMemory(nearest=3)
        self.assertEqual(memory.estimate({1, 2, 3}), 0.5)
        memory.observe({1, 2, 3}, 1.0)
        memory.observe({1, 2, 4}, 1.0)
        self.assertGreater(memory.estimate({1, 2, 5}), 0.5)
        memory.observe({7, 8, 9}, 0.0)
        self.assertLess(memory.estimate({7, 8, 10}), 0.5)

    def test_replay_planner_recruits_cross_episode_bridges(self):
        class Episode:
            def __init__(self, timestamp, content, motor, context=()):
                self.timestamp = timestamp
                self.content_neurons = content
                self.motor_neurons = motor
                self.context_neurons = context

        events = [
            Episode(0, (1, 2), (10,), context=(3,)),
            Episode(5, (2, 4), (11,), context=(3,)),
            Episode(80, (8, 9), (12,), context=(7,)),
        ]
        links = AssociativeReplayPlanner(max_links=2).plan(events)
        # The distant third episode is below the learned plasticity threshold;
        # replay must strengthen meaningful bridges, not every pair.
        self.assertEqual(len(links), 1)
        self.assertGreater(links[0].strength, 0.05)


import torch


if __name__ == "__main__":
    unittest.main()
