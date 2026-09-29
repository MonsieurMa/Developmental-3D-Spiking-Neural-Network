"""Learned structural segmentation only; prediction lives in neural modules."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from ..language.subwords import SubwordTokenizer


class LearnedSegmenter:
    """Wrap corpus-learned BPE; untrained text remains an opaque surface run.

    ``add_words`` receives units already learned by the network, not a
    hand-written lexicon.  Punctuation is an atomic structural boundary.
    """

    def __init__(
        self,
        tokenizer: SubwordTokenizer | None = None,
        words: Iterable[str] = (),
        max_word_length: int = 8,
    ) -> None:
        self.tokenizer = tokenizer or SubwordTokenizer()
        self.learned_units: set[str] = set()
        for word in words:
            self.add_words([word])
        self.max_word_length = int(max_word_length)
        self.unknown_counts: Counter[str] = Counter()

    def add_words(self, words: Iterable[str]) -> None:
        for word in words:
            clean = str(word).strip()
            if clean and all(char.isalnum() for char in clean):
                self.learned_units.add(clean)

    def segment(self, text: str) -> list[str]:
        if self.tokenizer.merges:
            units = self.tokenizer.encode(text)
            for unit in units:
                if len(unit) == 1 and unit not in self.learned_units:
                    self.unknown_counts[unit] += 1
            return units

        result: list[str] = []
        for word in text.split():
            run: list[str] = []
            for char in word:
                if char.isalnum():
                    run.append(char)
                    continue
                if run:
                    result.append("".join(run))
                    run = []
                if char.strip():
                    result.append(char)
            if run:
                result.append("".join(run))
        return result

    def frequent_unknowns(self, minimum_count: int = 5) -> list[str]:
        return [
            word for word, count in self.unknown_counts.items()
            if count >= minimum_count
        ]
