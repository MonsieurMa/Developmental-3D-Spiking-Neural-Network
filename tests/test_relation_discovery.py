"""Relation markers must be discovered from distribution, not from a word list."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain.language.relations import RelationDiscoverer

CHINESE = [
    "小猫 在 房间 里 睡觉", "小狗 在 院子 里 跑", "小兔 在 草地 上 跳",
    "小猫 在 沙发 上 睡", "小猫 喜欢 吃 小鱼", "小狗 喜欢 吃 骨头",
    "小兔 喜欢 吃 萝卜", "小猫 是 动物", "小兔 是 动物", "小狗 是 动物",
    "动物 需要 吃 东西", "动物 会 跑 会 跳", "小猫 把 球 给 小狗",
    "小狗 把 骨头 给 小猫", "小兔 把 萝卜 给 小猫",
    "桌子 在 房间 里 摆放", "椅子 在 客厅 里 摆放", "桌子 是 家具",
    "椅子 是 家具", "柜子 是 家具", "家具 放在 房间 里",
]

ENGLISH = [
    "the cat is an animal", "the dog is an animal", "the rabbit is an animal",
    "the cat sleeps in the room", "the dog runs in the yard",
    "the rabbit jumps in the field", "the cat likes fish", "the dog likes bones",
    "animals need food", "the table is furniture", "the chair is furniture",
    "the cat gives the ball to the dog", "the dog gives the bone to the cat",
    "the cat is in the room", "the dog is in the yard",
]


class RelationDiscoveryTest(unittest.TestCase):
    def test_chinese_relations_are_discovered_without_a_lexicon(self):
        discoverer = RelationDiscoverer()
        for line in CHINESE:
            discoverer.observe(line.replace(" ", ""))
        top = {token for token, _ in discoverer.discover(6)}
        # 是 / 在 / 把 are the structural markers of this corpus; nothing told
        # the discoverer they exist.
        self.assertTrue({"是", "在", "把"} & top)
        self.assertGreaterEqual(len({"是", "在", "把"} & top), 2)

    def test_english_relations_are_discovered_by_the_same_code(self):
        discoverer = RelationDiscoverer()
        for line in ENGLISH:
            discoverer.observe(line.split())
        top = {token for token, _ in discoverer.discover(4)}
        self.assertIn("is", top)
        self.assertIn("in", top)
        # A universal determiner must not be treated as a relation marker.
        self.assertGreater(discoverer.score("is"), discoverer.score("the"))

    def test_scores_are_relative_not_absolute(self):
        discoverer = RelationDiscoverer()
        for line in CHINESE:
            discoverer.observe(line.replace(" ", ""))
        self.assertGreater(discoverer.score("在"), discoverer.score("猫"))
        self.assertEqual(discoverer.score("从未出现过"), 0.0)

    def test_learned_marker_set_covers_the_structural_markers_without_fragments(self):
        discoverer = RelationDiscoverer()
        for line in CHINESE:
            discoverer.observe(line.replace(" ", ""))
        markers = discoverer.marker_set(relative_floor=0.35)
        self.assertTrue({"是", "在", "把"}.issubset(markers))
        self.assertFalse(markers & {"子", "里"})


if __name__ == "__main__":
    unittest.main()
