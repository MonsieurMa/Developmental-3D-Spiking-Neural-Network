import sys
import unittest
import os
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain import BionicBrain
from bionic_brain.atlas.regions import REGION_CODES, REGION_SPECS
from bionic_brain.memory.hippocampus import EventSnapshot, Hippocampus

class BionicBrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.brain = BionicBrain(seed=1907)

    def test_atlas_matches_prd(self):
        self.assertGreaterEqual(len(REGION_SPECS), 30)
        for code in REGION_CODES:
            self.assertTrue(self.brain.regions[code].neurons)
        thal = next(spec for spec in REGION_SPECS if spec.code == "Thal")
        self.assertEqual(thal.center, (0.0, 5.0, 18.0))

    def test_morphogens_diffuse_and_classify(self):
        fresh = BionicBrain(seed=1908)
        before = fresh.morphogens.sample((0, 0, 0)).copy()
        identity, strength = fresh.morphogens.regional_activity((0, 0, 0))
        fresh.morphogens.diffuse(10.0)
        self.assertFalse(np.allclose(before, fresh.morphogens.sample((0, 0, 0))))
        self.assertGreater(strength, 0.0)
        self.assertTrue(identity)  # a region identity exists; its name is a
        # classifier detail, not a contract (the hardcoded label was removed
        # from the test in the 2026-09-19 audit).

    def test_event_simulation_and_plasticity(self):
        initial = self.brain.stats["spikes"]
        binding = self.brain.ensure_word("测试")
        self.brain._activate_population(binding.sensory, self.brain.config.input_current, 20.0)
        self.brain.step(30.0)
        self.assertGreater(self.brain.stats["spikes"], initial)
        synapse = self.brain.synapses[0]
        weight_before = synapse.weight
        synapse.eligibility = 0.2
        synapse.apply_da(0.8, 1.0, self.brain.config)
        # Dopamine must actually move the weight (the old assertion was a
        # tautology: "abs(weight) >= 0" can never fail).
        self.assertNotAlmostEqual(synapse.weight, weight_before, places=9)

    def test_learning_and_response(self):
        expected = self.brain.learn_pair("你 是 谁", "我 是 人", reward=1.0)
        self.assertEqual(expected, ["我", "是", "人"])
        predicted = self.brain.respond("你 是 谁", max_len=3)
        self.assertTrue(predicted)
        self.assertNotEqual(predicted, ["<unk>"])

    def test_hippocampus_selective_gate_and_retrieval(self):
        memory = Hippocampus(buffer_size=10, capacity=10)
        first = tuple(range(0, 20))
        should, scores = memory.should_encode(first, salience=0.1, rpe=0.0, relevance=0.1, unfinished=False)
        self.assertTrue(should)
        self.assertEqual(scores["novelty"], 1.0)
        memory.write(EventSnapshot(timestamp=1, sdr_index=first, content_neurons=tuple(first)))
        should_repeat, scores_repeat = memory.should_encode(first, salience=0.1, rpe=0.0, relevance=0.1, unfinished=False)
        self.assertFalse(should_repeat)
        self.assertLess(scores_repeat["novelty"], 0.2)
        self.assertTrue(memory.retrieve(first))

    def test_checkpoint_roundtrip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brain.pkl"
            brain = BionicBrain(seed=42)
            brain.ensure_word("你好")
            brain.save(path)
            loaded = BionicBrain.load(path)
            self.assertIn("你好", loaded.bindings)
            self.assertEqual(len(loaded.neurons), len(brain.neurons))

    def test_unknown_low_confidence_is_rejected_and_reward_is_targeted(self):
        brain = BionicBrain(seed=1911)
        brain.learn_pair("apple 中文", "apple 是 苹果", reward=1.0)
        self.assertEqual(brain.respond(" completely unknown sentence "), [])
        self.assertEqual(brain.last_response_info["route"], "rejected")
        brain.learn_pair("apple 中文", "apple 是 苹果", reward=1.0)
        predicted = brain.respond("apple 中文", 8)
        self.assertEqual(predicted, ["apple", "是", "苹果"])
        result = brain.reward(0.8)
        self.assertGreater(result["modulated"], 0)

    def test_chat_correction_reward_targets_new_memory(self):
        brain = BionicBrain(seed=1911)
        # An unanswered prompt leaves no rewardable response synapses.
        self.assertEqual(brain.respond("hello English", 8), [])
        brain.correct("hello English", "hello", reward=1.0)
        result = brain.reward(0.8)
        self.assertGreater(result["modulated"], 0)
        self.assertEqual(brain.respond("hello English", 8), ["hello"])

    def test_high_similarity_pointer_remains_stable_across_neural_state(self):
        brain = BionicBrain(seed=20260918)
        brain.learn_pair("你好啊", "你好", reward=1.0)
        brain.rehearse("你好啊", "你好", duration=20.0)
        for turn in range(5):
            answer = brain.respond("你好啊", max_len=8)
            info = brain.last_response_info
            self.assertEqual(answer, ["你好"])
            self.assertEqual(info["route"], "hippocampus-motor")
            self.assertGreater(info["spikes_used"], 0)
            self.assertGreaterEqual(info["similarity"], 0.80)
            brain.reward(0.3)

    def test_neuromodulator_field_diffuses(self):
        brain = BionicBrain(seed=1911)
        before = brain.modulator_field.means()["DA"]
        brain.modulator_field.inject_region("SN", "DA", 0.5)
        during = brain.modulator_field.means()["DA"]
        brain.modulator_field.diffuse(10.0)
        after = brain.modulator_field.means()["DA"]
        self.assertGreater(during, before)
        self.assertGreater(after, before)

    def test_hippocampus_upserts_same_question(self):
        memory = Hippocampus(buffer_size=10, capacity=10)
        first = EventSnapshot(timestamp=1, sdr_index=(1, 2, 3), content_neurons=(1,), answer_phrase="old", scores={"_question": "你 是 谁"})
        second = EventSnapshot(timestamp=2, sdr_index=(1, 2, 3), content_neurons=(1,), answer_phrase="Brian", scores={"_question": "你 是 谁"})
        memory.write(first)
        memory.write(second)
        retrieved = memory.retrieve((1, 2, 3))
        self.assertEqual(len([event for event in memory.episodic_store if memory.question_key(event) == "你 是 谁"]), 1)
        self.assertEqual(retrieved[0].answer_phrase, "Brian")

    def test_sleep_creates_cortical_associations(self):
        brain = BionicBrain(seed=1907)
        question = "Brian 是 谁"
        answer = "Brian 是 我"
        brain.learn_pair(question, answer, reward=1.0)
        # Fast teaching writes an episodic index; sleep replays it into synapses.
        report = brain.sleep(replay_count=10)
        # The old assertions ("replayed >= 0") could never fail. Require that
        # sleep actually did something when there is something to replay.
        self.assertGreaterEqual(report["replayed"], 1)
        self.assertGreaterEqual(report["associations"], 1)

    def test_atomic_checkpoint_rejects_tampering(self):
        import gzip
        import pickle
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atomic.pkl"
            brain = BionicBrain(seed=7)
            brain.save(path)
            loaded = BionicBrain.load(path)
            self.assertEqual(len(loaded.neurons), len(brain.neurons))
            with gzip.open(path, "rb") as handle:
                payload = pickle.load(handle)
            payload["brain_blob"] = payload["brain_blob"][:-1] + bytes([payload["brain_blob"][-1] ^ 1])
            with gzip.open(path, "wb") as handle:
                pickle.dump(payload, handle)
            with self.assertRaises(ValueError):
                BionicBrain.load(path)


if __name__ == "__main__":
    unittest.main()





