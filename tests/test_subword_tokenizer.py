"""Learned subword units: no regex, no language rule, same code both languages."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain import BionicBrain
from bionic_brain.config.defaults import BionicConfig
from bionic_brain.learning.sequence import LearnedSegmenter
from bionic_brain.language.subwords import SubwordTokenizer

CHINESE = [
    "小猫在房间里睡觉", "小狗在院子里跑", "小兔在草地上跳", "小猫喜欢吃小鱼",
    "小狗喜欢吃骨头", "小兔喜欢吃萝卜", "小猫是动物", "小狗是动物", "小兔是动物",
    "桌子在房间里摆放", "椅子在客厅里摆放", "桌子是家具", "椅子是家具", "柜子是家具",
]
ENGLISH = [
    "the cat is an animal", "the dog is an animal", "the rabbit is an animal",
    "the cat sleeps in the room", "the dog runs in the yard", "the cat likes fish",
    "the dog likes bones", "animals need food", "the table is furniture",
    "the chair is furniture",
]


class SubwordTokenizerTest(unittest.TestCase):
    def test_learns_multi_character_units_from_chinese(self):
        tokenizer = SubwordTokenizer(max_merges=40, min_pair_count=2, max_unit_symbols=2)
        tokenizer.learn(CHINESE)
        units = tokenizer.learned_units
        self.assertTrue(units, "no merges were learned")
        # 小猫/小狗/小兔 recur, so the learner must join them without being told
        # what a word is.
        self.assertTrue({"小猫", "小狗", "小兔"} & units)
        encoded = tokenizer.encode("小猫在房间里睡觉")
        self.assertLess(len(encoded), len("小猫在房间里睡觉"))

    def test_learns_latin_units_with_the_same_code(self):
        tokenizer = SubwordTokenizer(max_merges=60, min_pair_count=2, max_unit_symbols=5)
        tokenizer.learn(ENGLISH)
        encoded = tokenizer.encode("the cat is an animal")
        # The same learner produces multi-character Latin units; which exact
        # merge wins depends on frequency, so assert the structure, not a word.
        self.assertTrue(any(len(unit) > 1 for unit in encoded))
        self.assertLess(len(encoded), len("the cat is an animal"))

    def test_respects_the_structural_unit_cap(self):
        tokenizer = SubwordTokenizer(max_merges=50, min_pair_count=1, max_unit_symbols=2)
        tokenizer.learn(CHINESE)
        self.assertTrue(all(len(unit) <= 2 for unit in tokenizer.learned_units))

    def test_encoding_keeps_every_character(self):
        tokenizer = SubwordTokenizer(max_merges=30, min_pair_count=2, max_unit_symbols=3)
        tokenizer.learn(CHINESE)
        for line in CHINESE[:4]:
            joined = "".join(tokenizer.encode(line))
            self.assertEqual(joined, line)

    def test_brain_runtime_uses_the_learned_units(self):
        brain = BionicBrain(BionicConfig(seed_profile="minimal"), develop=False)
        stats = brain.train_subword_tokenizer(CHINESE)
        self.assertGreater(stats["merges"], 0)
        encoded = brain._tokenize(CHINESE[6])
        self.assertIn("小猫", encoded)
        self.assertEqual("".join(encoded), CHINESE[6])
        restored = SubwordTokenizer.from_state(brain.subword_tokenizer.state())
        self.assertEqual(restored.encode(CHINESE[6]), encoded)

    def test_narrative_segmenter_has_no_built_in_lexicon(self):
        tokenizer = SubwordTokenizer(max_merges=40, min_pair_count=2, max_unit_symbols=3)
        tokenizer.learn(CHINESE)
        segmenter = LearnedSegmenter(tokenizer)
        self.assertEqual(segmenter.segment(CHINESE[6]), tokenizer.encode(CHINESE[6]))
        fallback = LearnedSegmenter()
        # With no learned units, alphanumeric runs stay intact and punctuation
        # remains a structural boundary.
        self.assertEqual(fallback.segment("Brian."), ["Brian", "."])


if __name__ == "__main__":
    unittest.main()
