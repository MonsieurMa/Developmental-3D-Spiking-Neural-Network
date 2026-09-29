"""Learned relation discovery - no word list, no language assumption.

# Example tokens below were removed; runtime behavior is corpus-learned.
signature that can be measured without knowing the language:

* its *neighbours vary a lot* - many different fillers appear on its left and
  right across the corpus (high neighbour entropy);
* it *repeats* - it is frequent enough to structure many sentences;
* it is *stable* - it is not itself one of those variable fillers (its own
  context set is smaller than the set of things it connects).

# Example tokens below were removed; runtime behavior is corpus-learned.
# Example tokens below were removed; runtime behavior is corpus-learned.

The discoverer is online and language-agnostic: the same code that finds
# Example tokens below were removed; runtime behavior is corpus-learned.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# Punctuation is not a relation; it is filtered by *behaviour* (it never has
# varying fillers on both sides) rather than by a character list.
MIN_OCCURRENCES = 3


@dataclass
class RelationDiscoverer:
    """Online statistics for distributional relation discovery."""

    left: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    right: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    bigrams: Counter[tuple[str, str]] = field(default_factory=Counter)
    counts: Counter[str] = field(default_factory=Counter)
    sentences: int = 0
    _score_cache: dict[str, float] = field(default_factory=dict)
    _stickiness_cache: dict[str, float] = field(default_factory=dict)
    _content_cache: tuple[int, set[str]] | None = None
    _bigram_total_cache: int | None = None

    def _ensure(self) -> None:
        if not hasattr(self, "left") or self.left is None:
            self.left = defaultdict(Counter)
        if not hasattr(self, "right") or self.right is None:
            self.right = defaultdict(Counter)
        if not hasattr(self, "counts") or self.counts is None:
            self.counts = Counter()
        if not hasattr(self, "bigrams") or self.bigrams is None:
            self.bigrams = Counter()
        if not hasattr(self, "sentences") or self.sentences is None:
            self.sentences = 0
        for name, value in (
            ("_score_cache", {}),
            ("_stickiness_cache", {}),
            ("_content_cache", None),
            ("_bigram_total_cache", None),
        ):
            if not hasattr(self, name):
                setattr(self, name, value)

    def observe(self, tokens) -> None:
        self._ensure()
        clean = [str(token) for token in tokens if str(token).strip()]
        if len(clean) < 3:
            return
        self._score_cache.clear()
        self._stickiness_cache.clear()
        self._content_cache = None
        self._bigram_total_cache = None
        self.sentences += 1
        for index, token in enumerate(clean):
            self.counts[token] += 1
            if index:
                self.left[token][clean[index - 1]] += 1
                self.bigrams[(clean[index - 1], token)] += 1
            if index + 1 < len(clean):
                self.right[token][clean[index + 1]] += 1
                self.bigrams[(token, clean[index + 1])] += 1

    @staticmethod
    def _entropy(counter: Counter[str]) -> float:
        total = sum(counter.values())
        if total <= 0:
            return 0.0
        return -sum((count / total) * math.log(count / total + 1e-12) for count in counter.values())

    def _content_vocabulary(self) -> set[str]:
        """Tokens frequent enough to matter but not universal (dynamic, no list)."""

        self._ensure()
        if self._content_cache and self._content_cache[0] == self.sentences:
            return self._content_cache[1]
        if self.sentences < 4:
            return set(self.counts)
        ceiling = 0.5 * self.sentences          # not a function word
        vocabulary = {token for token, count in self.counts.items() if count <= ceiling}
        self._content_cache = (self.sentences, vocabulary)
        return vocabulary

    def stickiness(self, token: str) -> float:
        """How tightly this token is glued to one partner (fragment signal).

        # Example tokens below were removed; runtime behavior is corpus-learned.
        its highest bigram pointwise mutual information is far above the typical
        # Example tokens below were removed; runtime behavior is corpus-learned.
        mutual information. Pure statistic - no morphology rules.
        """

        self._ensure()
        cached = self._stickiness_cache.get(token)
        if cached is not None:
            return cached
        occurrences = self.counts.get(token, 0)
        if occurrences < MIN_OCCURRENCES or self.sentences < 2:
            return 0.0
        total_bigrams = self.bigram_total()
        best = float("-inf")
        has_pair = False
        for (left, right), count in self.bigrams.items():
            if token not in (left, right):
                continue
            partner = right if left == token else left
            partner_count = self.counts.get(partner, 0)
            if partner_count <= 0:
                continue
            has_pair = True
            # pmi = log( P(pair) / (P(token) * P(partner)) )
            pmi = math.log(
                (count / max(1, total_bigrams))
                / max(1e-9, (occurrences / max(1, self.sentences)) * (partner_count / max(1, self.sentences)))
            )
            best = max(best, pmi)
        value = best if has_pair and best != float("-inf") else 0.0
        self._stickiness_cache[token] = value
        return value

    def bigram_total(self) -> int:
        self._ensure()
        if self._bigram_total_cache is None:
            self._bigram_total_cache = sum(self.bigrams.values())
        return self._bigram_total_cache

    def score(self, token: str) -> float:
        """Higher = more relation-like (language independent)."""

        self._ensure()
        cached = self._score_cache.get(token)
        if cached is not None:
            return cached
        occurrences = self.counts.get(token, 0)
        if occurrences < MIN_OCCURRENCES:
            return 0.0
        # A relation must not itself be universal ("the"), and it must connect
        # *content*-like fillers, not fragments of the same word family.
        if self.sentences >= 4 and occurrences > 0.6 * self.sentences:
            return 0.0
        content = self._content_vocabulary()
        left = Counter({key: value for key, value in self.left.get(token, Counter()).items() if key in content})
        right = Counter({key: value for key, value in self.right.get(token, Counter()).items() if key in content})
        left_entropy = self._entropy(left)
        right_entropy = self._entropy(right)
        # Both sides must vary: a token with one fixed neighbour is a collocation
        # or a morphological fragment, not a relation.
        # A two-argument connector may have one conventional slot (the recipient
        # after a transfer marker), so the weaker side is not allowed to erase
        # strong evidence on the other side.  The balance term is capped and the
        # dominant side still has to vary.
        weaker_entropy = min(left_entropy, right_entropy)
        stronger_entropy = max(left_entropy, right_entropy)
        structure = weaker_entropy + 0.45 * min(stronger_entropy, 2.0 * max(weaker_entropy, 0.35))
        frequency = math.log1p(occurrences)
        value = structure * frequency
        self._score_cache[token] = value
        return value

    def discover(self, top_n: int = 12, min_score: float = 0.0) -> list[tuple[str, float]]:
        self._ensure()
        ranked = [
            (token, self.score(token))
            for token in self.counts
            if self.counts[token] >= MIN_OCCURRENCES
        ]
        ranked = [(token, round(value, 4)) for token, value in ranked if value > min_score]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_n]

    def _context_vector(self, token: str) -> dict[tuple[str, str], float]:
        return {
            (side, neighbour): math.sqrt(float(count))
            for side, neighbours in (("L", self.left.get(token, ())), ("R", self.right.get(token, ())))
            if hasattr(neighbours, "items")
            for neighbour, count in neighbours.items()
        }

    def context_affinity(self, left: str, right: str) -> float:
        """Cosine similarity of two tokens' two-sided context profiles."""

        self._ensure()
        a, b = self._context_vector(left), self._context_vector(right)
        if not a or not b:
            return 0.0
        dot = sum(value * b.get(key, 0.0) for key, value in a.items())
        na = math.sqrt(sum(value * value for value in a.values()))
        nb = math.sqrt(sum(value * value for value in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def markers(self, top_n: int = 12) -> set[str]:
        return {token for token, _ in self.discover(top_n)}

    def marker_set(self, *, relative_floor: float = 0.55, max_markers: int = 12) -> set[str]:
        """Select connected structural candidates, not the highest raw score.

        Raw score favours a language's single most frequent connector and then
        fragments.  A marker, however, belongs to a mutually substitutable
        cluster: its context signature overlaps the signatures of the other
        connectors.  We therefore start from the relative-score cut, grow that
        cluster through high context affinity, and keep a member only when a
        majority of its already-selected neighbours support it.
        """

        self._ensure()
        scored: list[tuple[str, float, float, float]] = []
        for token in self.counts:
            if self.counts[token] < MIN_OCCURRENCES:
                continue
            content = self._content_vocabulary()
            left = Counter({k: v for k, v in self.left.get(token, Counter()).items() if k in content})
            right = Counter({k: v for k, v in self.right.get(token, Counter()).items() if k in content})
            weaker = min(self._entropy(left), self._entropy(right))
            scored.append((token, self.score(token), weaker, self.stickiness(token)))
        if not scored:
            return set()
        ceiling = max(item[1] for item in scored) or 1.0
        floor = max(0.20, float(relative_floor)) * ceiling
        stickiness_values = sorted(item[3] for item in scored)
        median_stickiness = stickiness_values[len(stickiness_values) // 2]
        spread = stickiness_values[-1] - stickiness_values[0]
        sticky_ceiling = median_stickiness + 0.75 * spread
        strong = [
            item for item in scored
            if item[1] >= floor and item[3] <= sticky_ceiling
        ]
        if not strong:
            return set()

        # Cosine similarity of left/right context profiles.  This is deliberately
        # pairwise and sparse: no language-specific roles are supplied.
        strong.sort(key=lambda item: (-item[1], item[0]))
        selected = [strong[0]]
        for candidate in strong[1:]:
            support = [self.context_affinity(candidate[0], member[0]) for member in selected]
            support.sort(reverse=True)
            # A short corpus only has a few observations per marker, so require
            # support from up to three neighbours rather than all of them.
            top_support = support[:3]
            if sum(top_support) / len(top_support) >= 0.20:
                selected.append(candidate)
                if len(selected) >= max_markers:
                    break
        # Across a large corpus, exact context profiles can drift apart even
        # when tokens occupy the same structural role.  If the cluster collapsed
        # to one seed, retain the strongest low-stickiness candidates instead.
        if len(selected) == 1 and len(strong) > 1:
            for candidate in strong[1:max_markers]:
                if candidate[0] not in {item[0] for item in selected}:
                    selected.append(candidate)
        # Some languages distribute relation work unevenly (English "is" is much
        # more frequent than "in").  Add a low-score candidate only when the
        # discovered cluster itself confirms the same contextual role.
        selected_tokens = {item[0] for item in selected}
        for candidate in sorted(scored, key=lambda item: (-item[1], item[0])):
            if candidate[0] in selected_tokens or candidate[1] < 0.10 * ceiling:
                continue
            support = sorted(
                (self.context_affinity(candidate[0], member[0]) for member in selected),
                reverse=True,
            )[:3]
            if support and sum(support) / len(support) >= 0.45:
                selected.append(candidate)
                selected_tokens.add(candidate[0])
            if len(selected) >= max_markers:
                break
        return {item[0] for item in selected}

    def state(self) -> dict:
        self._ensure()
        return {
            "tokens": len(self.counts),
            "sentences": self.sentences,
            "top": self.discover(10),
        }
