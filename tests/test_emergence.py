"""Regression tests for the recall, retention, generalization and generation work."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain import BionicBrain
from bionic_brain.config.defaults import BionicConfig
from bionic_brain.memory.hippocampus import EventSnapshot, Hippocampus


def small_brain(seed: int = 20260918) -> BionicBrain:
    return BionicBrain(BionicConfig(seed_profile="minimal"), seed=seed)


class CortexRecallTest(unittest.TestCase):
    def test_recall_runs_the_network_without_the_hippocampus(self):
        brain = small_brain()
        brain.learn_pair("你 是 谁", "我 是 Brian", reward=1.0)
        brain.hippocampus.retrieve = lambda *args, **kwargs: []
        answer = brain.respond("你 是 谁", max_len=8)
        info = brain.last_response_info
        self.assertEqual(" ".join(answer), "我 是 Brian")
        self.assertEqual(info["route"], "population")
        self.assertGreater(info["timing_ms"]["motor_step"], 0.0)
        self.assertFalse(info["memory_pointer"])

    def test_unknown_input_is_still_rejected(self):
        brain = small_brain()
        brain.learn_pair("你 是 谁", "我 是 Brian", reward=1.0)
        self.assertEqual(brain.respond("完全 没有 教 过 的 句子", max_len=8), [])
        self.assertEqual(brain.last_response_info["route"], "rejected")


class PlasticityFloorTest(unittest.TestCase):
    def test_association_weights_stay_above_the_decode_floor(self):
        brain = small_brain()
        brain.learn_pair("你 是 谁", "我 是 Brian", reward=1.0)
        first_lesson = list(brain.last_learning_synapses)
        self.assertTrue(first_lesson)

        for index in range(25):
            brain.learn_pair(f"填充 问题 {index}", f"填充 答案 {index}", reward=1.0)

        floor = brain.config.learned_weight * 0.55
        weakest = min(brain.synapses[syn_id].weight for syn_id in first_lesson)
        self.assertGreaterEqual(weakest, floor)
        self.assertEqual(" ".join(brain.respond("你 是 谁", max_len=8)), "我 是 Brian")

    def test_homeostatic_decay_has_an_engram_floor(self):
        brain = small_brain()
        brain.learn_pair("你 是 谁", "我 是 Brian", reward=1.0)
        ids = list(brain.last_learning_synapses)
        brain._scale_synapses(0.05)
        floor = brain.config.learned_weight * getattr(brain.config, "association_floor", 0.85)
        self.assertGreaterEqual(min(brain.synapses[i].weight for i in ids), floor - 1e-9)

    def test_binding_cells_are_not_killed_by_the_apoptosis_sweep(self):
        brain = small_brain()
        brain.learn_pair("你 是 谁", "我 是 Brian", reward=1.0)
        engram = brain.bindings["你"].sensory
        for neuron_id in engram:
            brain.neurons[neuron_id].energy = 0.0
        brain.lifecycle()
        self.assertTrue(all(brain.neurons[neuron_id].alive for neuron_id in engram))


class RetentionTest(unittest.TestCase):
    def test_retrieval_reads_the_long_term_store_not_only_the_buffer(self):
        memory = Hippocampus(buffer_size=3, capacity=100)
        first = EventSnapshot(
            timestamp=1.0, sdr_index=(1, 2, 3), content_neurons=(1,),
            answer_phrase="old", scores={"_question": "旧 问题"},
        )
        memory.write(first)
        for index in range(12):
            memory.write(EventSnapshot(
                timestamp=10.0 + index, sdr_index=(100 + index,), content_neurons=(1,),
                answer_phrase=f"新 {index}", scores={"_question": f"新 问题 {index}"},
            ))
        recalled = memory.retrieve((1, 2, 3), top_k=1)
        self.assertTrue(recalled)
        self.assertEqual(recalled[0].answer_phrase, "old")
        self.assertLessEqual(len(memory.episodic_buffer), 3)
        self.assertEqual(memory.retained()["store"], 13)

    def test_consolidation_keeps_the_index(self):
        memory = Hippocampus(buffer_size=10, capacity=10)
        memory.write(EventSnapshot(
            timestamp=1.0, sdr_index=(1, 2), content_neurons=(1,),
            answer_phrase="答案", scores={"_question": "问题"},
        ))
        memory.consolidate(5)
        self.assertTrue(memory.retrieve((1, 2), top_k=1))

    def test_learned_pairs_survive_interference(self):
        brain = small_brain()
        brain.learn_pair("你 是 谁", "我 是 Brian", reward=1.0)
        for index in range(40):
            brain.learn_pair(f"无关 问题 {index}", f"无关 答案 {index}", reward=1.0)
        self.assertEqual(" ".join(brain.respond("你 是 谁", max_len=8)), "我 是 Brian")


class GeneralizationTest(unittest.TestCase):
    def test_repeated_coactivation_shares_sensory_cells(self):
        brain = small_brain()
        for _ in range(3):
            brain.learn_pair("苹果 的 英文", "apple", reward=1.0)
            brain.learn_pair("苹果 和 apple 是 一个 意思", "对", reward=1.0)
        # Without a language rule, an untrained Chinese run is one opaque token.
        shared = set(brain.bindings["apple"].sensory) & set(brain.bindings["苹果"].sensory)
        self.assertTrue(shared, "co-activated tokens should share sensory cells")
        self.assertLessEqual(brain.shared_cells.get("apple", 0), brain.config.sdr_share_budget)

    def test_paraphrase_recall_through_the_cortex(self):
        brain = small_brain()
        for _ in range(3):
            brain.learn_pair("苹果 的 英文", "apple", reward=1.0)
            brain.learn_pair("苹果 的 英文 是 什么", "apple", reward=1.0)
            brain.learn_pair("苹果 英文 说 法", "apple", reward=1.0)
        answer = brain.respond("苹果 英文 怎么 说", max_len=4)
        self.assertEqual(" ".join(answer), "apple")


class GenerationTest(unittest.TestCase):
    def test_sequence_generation_chains_lessons(self):
        brain = small_brain()
        brain.learn_pair("你 是 谁", "我 是 Brian", reward=1.0)
        brain.learn_pair("你 是 谁 我 是 Brian 你 会 什么", "我 会 学习", reward=1.0)
        produced = brain.respond_sequence("你 是 谁")
        text = "".join(produced)
        self.assertIn("Brian", text)
        self.assertIn("学习", text)
        self.assertGreaterEqual(len(produced), 6)
        self.assertTrue(brain.last_generation_info["stop_reason"])

    def test_generation_stops_by_itself_without_a_length_limit(self):
        brain = small_brain()
        for _ in range(3):
            brain.observe_stream("小明 在 房间 里 看 书。")
        produced = brain.respond_sequence("小明 在 房间")
        info = brain.last_generation_info
        # The network stops on its own well before the safety cap.
        self.assertLess(len(produced), info["safety_cap"])
        self.assertIn(
            info["stop_reason"],
            {"model-end-marker", "no-supported-plan", "repetition", "expectation-collapsed", "fatigue"},
        )


if __name__ == "__main__":
    unittest.main()
