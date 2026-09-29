"""Learned subword units (BPE-style merges) - no word list, no language rule.

The regex tokeniser encodes two language decisions by hand: Latin runs stay whole
while Chinese is split per character (H11). That single choice creates the
# Example tokens below were removed; runtime behavior is corpus-learned.
then behaves like a relation marker.

This module learns the units instead. It starts from raw characters (any
script), counts adjacent symbol pairs over the corpus, and repeatedly merges the
# Example tokens below were removed; runtime behavior is corpus-learned.
the/ing from English text. Nothing about either language is specified.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .relations import RelationDiscoverer

WORD_SPLIT = re.compile(r"\s+")
END_OF_WORD = "</w>"


@dataclass
class SubwordTokenizer:
    """Frequency-driven byte-pair-encoding over whatever characters appear."""

    max_merges: int = 256
    min_pair_count: int = 2
    # Example tokens below were removed; runtime behavior is corpus-learned.
    # Measured in symbols, so it is language independent.
    max_unit_symbols: int = 3
    # Co-learning, still with no word list: the first merges give the relation
    # discoverer enough units to work with; later merges that would glue a
    # *learned* relational marker to an argument are rejected.
    relation_guidance: bool = True
    relation_bootstrap_merges: int = 64
    relation_refresh_interval: int = 128
    guided_markers: set[str] = field(default_factory=set)
    merges: list[tuple[str, str]] = field(default_factory=list)
    merge_rank: dict[tuple[str, str], int] = field(default_factory=dict)
    learned_units: set[str] = field(default_factory=set)
    observed_words: int = 0

    # ---------------- learning ----------------
    @staticmethod
    def _segments(word: str) -> list[tuple[str, bool]]:
        """Split into alphanumeric runs and structural punctuation symbols.

        The split uses Unicode's alphanumeric property only.  It carries no
        script or word list; punctuation becomes an atomic boundary, so a merge
        # Example tokens below were removed; runtime behavior is corpus-learned.
        """

        segments: list[tuple[str, bool]] = []
        run: list[str] = []
        for char in word:
            if char.isalnum():
                run.append(char)
                continue
            if run:
                segments.append(("".join(run), True))
                run = []
            if char.strip():
                segments.append((char, False))
        if run:
            segments.append(("".join(run), True))
        return segments

    def _word_symbols(self, segment: str, *, is_word: bool = True) -> list[str]:
        symbols = [char for char in segment if char.strip()]
        if is_word:
            symbols.append(END_OF_WORD)
        return symbols

    def learn(self, lines) -> None:
        """Learn merges from raw text lines (characters are the only assumption)."""

        training_lines = [str(line) for line in lines]
        self.guided_markers = set()
        vocab: Counter[tuple[str, ...]] = Counter()
        for line in training_lines:
            for word in WORD_SPLIT.split(str(line)):
                if not word.strip():
                    continue
                for segment, is_word in self._segments(word):
                    vocab[tuple(self._word_symbols(segment, is_word=is_word))] += 1
                self.observed_words += 1
        for _ in range(max(0, self.max_merges)):
            pair_counts: Counter[tuple[str, str]] = Counter()
            for symbols, frequency in vocab.items():
                for index in range(len(symbols) - 1):
                    pair = (symbols[index], symbols[index + 1])
                    candidate = pair[0] + pair[1]
                    if any(marker in candidate for marker in self.guided_markers):
                        continue
                    if len([c for c in (pair[0] + pair[1]) if c]) > self.max_unit_symbols:
                        # Skip this pair, keep learning from the others.
                        continue
                    pair_counts[pair] += frequency
            if not pair_counts:
                break
            (best_pair, best_count) = pair_counts.most_common(1)[0]
            if best_count < self.min_pair_count:
                break
            self.merges.append(best_pair)
            self.merge_rank[best_pair] = len(self.merges)
            self.learned_units.add(best_pair[0] + best_pair[1])
            if (
                self.relation_guidance
                and len(self.merges) >= self.relation_bootstrap_merges
                and (len(self.merges) - self.relation_bootstrap_merges) % self.relation_refresh_interval == 0
            ):
                self._refresh_relation_guidance(training_lines)
            merged_vocab: Counter[tuple[str, ...]] = Counter()
            for symbols, frequency in vocab.items():
                merged_vocab[tuple(self._merge_symbols(list(symbols), best_pair))] += frequency
            vocab = merged_vocab

    def _refresh_relation_guidance(self, training_lines: list[str]) -> None:
        """Re-discover markers and use them as structural merge boundaries."""

        discoverer = RelationDiscoverer()
        for line in training_lines:
            discoverer.observe(self.encode(line))
        self.guided_markers = discoverer.marker_set()

    @staticmethod
    def _merge_symbols(symbols: list[str], pair: tuple[str, str]) -> list[str]:
        out: list[str] = []
        index = 0
        while index < len(symbols):
            if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
                out.append(symbols[index] + symbols[index + 1])
                index += 2
            else:
                out.append(symbols[index])
                index += 1
        return out

    # ---------------- inference ----------------
    def encode(self, text: str) -> list[str]:
        """Split text into learned units (longest applicable merge wins)."""

        units: list[str] = []
        for word in WORD_SPLIT.split(str(text)):
            if not word.strip():
                continue
            for segment, is_word in self._segments(word):
                symbols = self._word_symbols(segment, is_word=is_word)
                if is_word and len(symbols) > 1 and not any(
                    (symbols[index], symbols[index + 1]) in self.merge_rank
                    for index in range(len(symbols) - 1)
                ):
                    # No corpus evidence yet: preserve the opaque alphanumeric
                    # run instead of imposing a per-character language rule.
                    units.append(segment)
                    continue
                while len(symbols) > 1:
                    best_rank: int | None = None
                    best_index = -1
                    for index in range(len(symbols) - 1):
                        rank = self.merge_rank.get((symbols[index], symbols[index + 1]))
                        if rank is None:
                            continue
                        if best_rank is None or rank < best_rank:
                            best_rank, best_index = rank, index
                    if best_index < 0:
                        break
                    symbols = (
                        symbols[:best_index]
                        + [symbols[best_index] + symbols[best_index + 1]]
                        + symbols[best_index + 2:]
                    )
                for index, symbol in enumerate(symbols):
                    token = symbol[:-len(END_OF_WORD)] if symbol.endswith(END_OF_WORD) else symbol
                    # With no learned merge, an alphanumeric run is one opaque
                    # surface unit; per-character splitting is a corpus decision,
                    # never a script rule.
                    if token and (len(symbols) > 1 or is_word or not token.isalnum()):
                        units.append(token)
                    elif token and index == 0:
                        units.append(token)
        return units

    def state(self) -> dict:
        """Full learned state (distinct from the compact diagnostic state)."""

        return {
            "max_merges": self.max_merges,
            "min_pair_count": self.min_pair_count,
            "max_unit_symbols": self.max_unit_symbols,
            "merges": [list(pair) for pair in self.merges],
            "observed_words": self.observed_words,
            "guided_markers": sorted(self.guided_markers),
        }

    @classmethod
    def from_state(cls, state: dict) -> "SubwordTokenizer":
        tokenizer = cls(
            max_merges=int(state.get("max_merges", 256)),
            min_pair_count=int(state.get("min_pair_count", 2)),
            max_unit_symbols=int(state.get("max_unit_symbols", 3)),
        )
        tokenizer.load_state(state)
        return tokenizer

    def load_state(self, state: dict) -> None:
        self.max_merges = int(state.get("max_merges", self.max_merges))
        self.min_pair_count = int(state.get("min_pair_count", self.min_pair_count))
        self.max_unit_symbols = int(state.get("max_unit_symbols", self.max_unit_symbols))
        self.merges = [tuple(pair) for pair in state.get("merges", [])]
        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.learned_units = {left + right for left, right in self.merges}
        self.observed_words = int(state.get("observed_words", 0))
        self.guided_markers = set(state.get("guided_markers", ()))

    def diagnostics(self) -> dict:
        return {
            "merges": len(self.merges),
            "units": len(self.learned_units),
            "observed_words": self.observed_words,
            "guided_markers": len(self.guided_markers),
            "sample_units": sorted(self.learned_units, key=len, reverse=True)[:10],
        }
