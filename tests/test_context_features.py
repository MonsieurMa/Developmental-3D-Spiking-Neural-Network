"""Learned context features: semantic geometry and accommodation (splitting)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain import BionicBrain
from bionic_brain.config.defaults import BionicConfig
from bionic_brain.language.dialogue import core_dialogue_tokens
from bionic_brain.language.features import ContextFeatureSpace, TokenStatistics

ANIMALS = [
    "小猫在房间里睡觉", "小狗在院子里跑", "小兔在草地上跳",
    "小猫喜欢吃小鱼", "小狗喜欢吃骨头", "小兔喜欢吃萝卜",
    "小猫是动物", "动物需要吃东西", "动物会跑会跳", "小狗和小猫一起玩",
]
FURNITURE = [
    "桌子在房间里摆放", "椅子在客厅里摆放", "柜子在卧室里摆放",
    "桌子是家具", "椅子是家具", "家具放在房间里",
]


class ContextFeatureSpaceTest(unittest.TestCase):
    def setUp(self):
        self.space = ContextFeatureSpace(n_features=64)
        for line in ANIMALS + FURNITURE:
            self.space.observe(line)

    def test_geometry_separates_domains_and_links_within_them(self):
        # One assertion set for the geometry contract: within-domain tokens are
        # closer than cross-domain ones, in both domains.
        within = self.space.similarity("猫", "狗")
        cross = self.space.similarity("猫", "桌")
        self.assertGreater(within, cross)
        self.assertLess(cross, 0.1)
        self.assertGreater(self.space.similarity("桌", "椅"), self.space.similarity("桌", "狗"))
        neighbours = {token for token, _ in self.space.neighbours("桌", 5)}
        self.assertTrue(neighbours & {"椅", "柜", "房", "间", "放"})

    def test_accommodation_never_shatters_a_coherent_domain(self):
        # Splitting is relative: a domain whose members resemble each other must
        # survive as one cluster (the earlier absolute threshold shattered it).
        used_before = self.space.state()["used_features"]
        self.space.observe(["小猫", "小狗", "小兔", "动物"])
        self.assertLessEqual(self.space.state()["used_features"], used_before + 1)
        self.assertGreater(self.space.similarity("猫", "狗"), 0.0)

    def test_repeated_material_stops_reshaping(self):
        used_after_first = self.space.state()["used_features"]
        # Predictable input carries no gain, so it must not spawn features.
        for _ in range(3):
            for line in ANIMALS + FURNITURE:
                self.space.observe(line, gain=0.0)
        self.assertEqual(self.space.state()["used_features"], used_after_first)


class LearnedDialogueBoundaryTest(unittest.TestCase):
    def test_final_particle_is_learned_not_named(self):
        statistics = TokenStatistics()
        for index in range(20):
            statistics.observe([f"内容{index}", "吗"], sentence_final=True)
        statistics.observe(["陈述"], sentence_final=True)
        self.assertLess(statistics.weight("吗"), 0.40)
        self.assertGreaterEqual(statistics.final_affinity("吗"), 0.55)
        self.assertTrue(statistics.is_final_particle("吗"))

    def test_core_cue_removes_structural_punctuation_and_learned_particle(self):
        statistics = TokenStatistics()
        for index in range(20):
            statistics.observe([f"内容{index}", "吗"], sentence_final=True)
        statistics.observe(["陈述"], sentence_final=True)
        self.assertEqual(
            core_dialogue_tokens(["内容", "，", "吗"], statistics, observe=False),
            ["内容"],
        )


class BrainContextFeatureTest(unittest.TestCase):
    def test_observe_stream_learns_separated_geometry(self):
        config = BionicConfig(seed_profile="minimal", context_features=64)
        brain = BionicBrain(config, seed=7)
        brain.subword_tokenizer.max_unit_symbols = 4
        brain.train_subword_tokenizer(ANIMALS + FURNITURE)
        for line in ANIMALS + FURNITURE:
            brain.observe_stream(line)
        space = brain.context_features
        self.assertGreater(space.similarity("小猫", "小狗"), space.similarity("小猫", "桌子"))
        self.assertLess(space.similarity("小猫", "桌子"), 0.5)
        self.assertGreater(space.state()["used_features"], 1)
        # Accommodation must have fired at least once on this mixed corpus:
        # the animal and furniture sentences share bridge tokens (房间/子), so
        # without splitting they would collapse into one feature.
        self.assertGreaterEqual(space.state()["splits"], 1)

    def test_explicit_teaching_is_unaffected_by_feature_learning(self):
        config = BionicConfig(seed_profile="minimal", context_features=64)
        brain = BionicBrain(config, seed=7)
        brain.learn_pair("你 是 谁", "我 是 Brian", reward=1.0)
        self.assertEqual(" ".join(brain.respond("你 是 谁", max_len=8)), "我 是 Brian")


if __name__ == "__main__":
    unittest.main()
