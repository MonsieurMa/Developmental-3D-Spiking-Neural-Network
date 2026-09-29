"""Learned context features: semantic geometry without global gradients.

The original design gave every token a random assembly, so "cat" and "dog" were
as far apart as "cat" and "table" - semantic similarity was physically
impossible. This module learns shared context features with a *local,
competitive, error-driven* rule (the backprop idea without a global backward
pass):

* every token is explained by the feature cell whose member set best matches the
  token's current context (winner-take-all competition);
* the winner's members are pulled towards the observed context (Hebbian);
* neighbours of the winner are nudged too (a soft neighbourhood, as in
  self-organising maps), which lets features split smoothly;
* a homeostatic count keeps one feature from swallowing the whole vocabulary.

Tokens that appear in similar contexts end up sharing feature cells, so their
assemblies overlap and the existing machinery (hippocampal Jaccard, association
support, coverage) inherits a real similarity metric.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import math
from typing import Iterable


@dataclass
class TokenStatistics:
    """Online document-frequency statistics for language-neutral salience.

    No language's function words are named here. A token becomes low-salience
    because it has appeared in many observed contexts; punctuation is excluded
    by the structural ``isalnum`` property, not by a punctuation table.
    """

    document_frequency: Counter[str] = field(default_factory=Counter)
    final_frequency: Counter[str] = field(default_factory=Counter)
    observations: int = 0
    salience_warmup: int = 4
    final_particle_affinity: float = 0.55
    min_support: int = 2

    def observe(self, tokens: Iterable[str], *, sentence_final: bool = False) -> None:
        values = [str(token) for token in tokens if str(token)]
        if not values:
            return
        self.observations += 1
        self.document_frequency.update(set(values))
        if sentence_final:
            for token in reversed(values):
                if token and all(char.isalnum() for char in token):
                    self.final_frequency[token] += 1
                    break

    @staticmethod
    def is_word_like(token: str) -> bool:
        """Structural test for a non-punctuation unit (Unicode, not a lexicon)."""

        return bool(token) and all(char.isalnum() for char in token)

    def weight(self, token: str) -> float:
        """Statistical outlier coverage, robust to corpus size and script.

        A function word is not defined by a name; it covers far more contexts
        than the token population.  The z-score form keeps the mapping smooth
        and does not need a language-specific threshold table.
        """

        counts = [count for count in self.document_frequency.values() if count]
        if self.observations < self.salience_warmup or not counts:
            return 1.0
        mean = sum(counts) / len(counts)
        variance = sum((count - mean) ** 2 for count in counts) / len(counts)
        deviation = math.sqrt(variance)
        if deviation <= 0.0:
            return 1.0
        excess = (self.document_frequency.get(str(token), 0) - mean) / deviation
        return 1.0 / (1.0 + math.log1p(max(0.0, excess)))

    def final_affinity(self, token: str) -> float:
        """How often an observed token occurs in utterance-final position."""

        occurrences = self.document_frequency.get(str(token), 0)
        if occurrences < self.min_support:
            return 0.0
        return self.final_frequency.get(str(token), 0) / occurrences

    def is_final_particle(self, token: str) -> bool:
        """Did this token learn an utterance-final, low-content role?"""

        return (
            self.weight(token) < 0.40
            and self.final_affinity(token) >= self.final_particle_affinity
        )


def content_tokens(tokens, statistics: TokenStatistics | None = None) -> list[str]:
    """Tokens that carry topic meaning (grammar words removed)."""

    out = []
    for token in tokens:
        text = str(token)
        if not text.strip() or not all(char.isalnum() for char in text):
            continue
        if statistics is not None and statistics.weight(text) < 0.40:
            continue
        out.append(text)
    return out


@dataclass
class ContextFeatureSpace:
    """Online competitive feature learning over co-occurring tokens."""

    n_features: int = 256
    top_k: int = 4
    learning_rate: float = 0.35
    neighbourhood: int = 1
    history: int = 12
    match_threshold: float = 0.35
    # Learning is gated by surprise: predictable input must not reshape the
    # representation (that is what makes the world model stable).
    min_gain: float = 0.25
    members: dict[int, Counter[str]] = field(default_factory=dict)
    token_features: dict[str, Counter[int]] = field(default_factory=dict)
    feature_usage: Counter[int] = field(default_factory=Counter)
    document_frequency: Counter[str] = field(default_factory=Counter)
    # Accommodation: recent observations per feature, used to detect that one
    # feature is explaining two incompatible contexts.
    feature_observations: dict[int, deque] = field(default_factory=dict)
    # Grouping floor for the observation graph. The actual decision is relative
    # (see ``split_separation``): a feature splits only when the groups are much
    # less similar to each other than they are internally, so a fixed threshold
    # can neither leave a merged blob (0.25) nor shatter a coherent cluster.
    split_jaccard: float = 0.25
    split_separation: float = 0.6
    split_min_observations: int = 5
    split_min_component: int = 2
    splits: int = 0
    observations: int = 0
    token_statistics: TokenStatistics = field(default_factory=TokenStatistics)
    update_statistics: bool = True

    # ---------------- learning ----------------
    def _ensure(self) -> None:
        for name, default in (
            ("members", {}), ("token_features", {}),
            ("feature_usage", Counter()), ("observations", 0),
            ("document_frequency", Counter()),
            ("feature_observations", {}),
            ("splits", 0),
        ):
            if not hasattr(self, name) or getattr(self, name) is None:
                setattr(self, name, default)
        if not hasattr(self, "token_statistics") or self.token_statistics is None:
            self.token_statistics = TokenStatistics()
        if not hasattr(self, "update_statistics"):
            self.update_statistics = True

    def _best_feature(self, context: set[str]) -> int:
        """Winner-take-all: the feature whose members best explain the context."""

        best_id = -1
        best_score = 0.0
        context_weight = sum(self._weight(token) for token in context if self._weight(token) >= 0.40)
        if context_weight <= 0:
            return self._next_unused()
        for feature_id in range(self.n_features):
            members = self.members.get(feature_id)
            if not members:
                continue
            # Example tokens below were removed; runtime behavior is corpus-learned.
            # not make every context look alike.
            overlap = sum(
                min(count, 3) * self._weight(token)
                for token, count in members.items() if token in context
            )
            if overlap <= 0:
                continue
            # Example tokens below were removed; runtime behavior is corpus-learned.
            # pull two different domains into one feature.
            score = (overlap / context_weight) / (1.0 + 0.02 * self.feature_usage[feature_id])
            if score > best_score:
                best_score = score
                best_id = feature_id
        if best_id >= 0 and best_score >= self.match_threshold:
            return best_id
        # Nothing matched: allocate a fresh feature so novel contexts carve out
        # their own cell instead of piling onto feature 0.
        return self._next_unused()

    def _weight(self, token: str) -> float:
        self._ensure()
        return self.token_statistics.weight(token)

    def _next_unused(self) -> int:
        for feature_id in range(self.n_features):
            if not self.members.get(feature_id):
                return feature_id
        # All features are in use: no free slot (callers must cope).
        return -1

    def observe(self, tokens, gain: float = 1.0) -> None:
        """Learn from one context window; ``gain`` is the prediction-error drive."""

        self._ensure()
        if gain < self.min_gain:
            return
        weight_scale = min(2.0, 0.5 + float(gain))
        clean = [str(token) for token in tokens if str(token).strip()]
        # Frequency must include everything before filtering; that is what lets
        # the learner discover which symbols are ubiquitous in this corpus.
        if self.update_statistics:
            self.token_statistics.observe(clean)
        clean = content_tokens(clean, self.token_statistics)
        if len(clean) < 2:
            return
        if self.token_statistics.observations < self.token_statistics.salience_warmup:
            # A frequency prior learned from one or two observations would just
            # make every symbol look important; wait for a small sample.
            return
        if len(clean) > self.history:
            clean = clean[-self.history:]
        context = set(clean)
        for token in context:
            self.document_frequency[token] += 1
        winner = self._best_feature(context)
        if winner < 0:
            # Space is full: fall back to the least loaded existing feature.
            winner = min(range(self.n_features), key=lambda feature_id: self.feature_usage[feature_id])
        self.observations += 1
        self.feature_usage[winner] += 1
        history = self.feature_observations.setdefault(winner, deque(maxlen=32))
        history.append(tuple(sorted(context)))
        for token in context:
            weight = self._weight(token)
            if weight < 0.40:
                # Near-universal function word: it belongs to every context, so
                # letting it define features destroys the geometry.
                continue
            feature_counts = self.token_features.setdefault(token, Counter())
            feature_counts[winner] += weight * weight_scale
            self.members.setdefault(winner, Counter())[token] += 1
        # Accommodation: if this feature is now trying to explain two contexts
        # that share almost nothing, split it instead of averaging them.
        if len(history) >= self.split_min_observations:
            self._maybe_split(winner)

    def _maybe_split(self, feature_id: int) -> int:
        """Split a feature whose observations form two weakly linked clusters."""

        observations = [set(item) for item in self.feature_observations.get(feature_id, ())]
        if len(observations) < self.split_min_observations:
            return 0
        parent = list(range(len(observations)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                a, b = observations[i], observations[j]
                union_size = len(a | b)
                if union_size and len(a & b) / union_size >= self.split_jaccard:
                    union(i, j)
        groups: dict[int, list[int]] = {}
        for index in range(len(observations)):
            groups.setdefault(find(index), []).append(index)
        clusters = [sorted(members) for members in groups.values() if len(members) >= self.split_min_component]
        if len(clusters) < 2:
            return 0
        clusters.sort(key=len, reverse=True)

        def jaccard(left: set[str], right: set[str]) -> float:
            union = len(left | right)
            return len(left & right) / union if union else 0.0

        within: list[float] = []
        cross: list[float] = []
        for cluster_index, cluster in enumerate(clusters):
            for i, left_index in enumerate(cluster):
                for right_index in cluster[i + 1:]:
                    within.append(jaccard(observations[left_index], observations[right_index]))
            for other in clusters[cluster_index + 1:]:
                for left_index in cluster:
                    for right_index in other:
                        cross.append(jaccard(observations[left_index], observations[right_index]))
        mean_within = sum(within) / len(within) if within else 0.0
        mean_cross = sum(cross) / len(cross) if cross else 0.0
        # Only a *clearly* separated pair of domains justifies splitting;
        # otherwise the feature stays coherent (avoids shattering one concept).
        if mean_within <= 0.0 or mean_cross > self.split_separation * mean_within:
            return 0
        created = 0
        for cluster in clusters[1:]:
            new_id = self._next_unused()
            if new_id < 0:
                break
            self.members[new_id] = Counter()
            for index in cluster:
                for token in observations[index]:
                    self.members[new_id][token] += 1
                    counts = self.token_features.setdefault(token, Counter())
                    counts[new_id] += 1.0
                    if counts.get(feature_id):
                        counts[feature_id] = max(0.0, counts[feature_id] - 1.0)
            self.feature_observations[new_id] = deque(
                (tuple(sorted(observations[index])) for index in cluster), maxlen=32
            )
            self.feature_usage[new_id] += len(cluster)
            created += 1
        self.feature_observations[feature_id] = deque(
            (tuple(sorted(observations[index])) for index in clusters[0]), maxlen=32
        )
        self.splits += created
        return created

    # ---------------- inference ----------------
    def features(self, token: str) -> tuple[int, ...]:
        """The features a token belongs to (most used first)."""

        self._ensure()
        counts = self.token_features.get(str(token))
        if not counts:
            return ()
        ranked = [(feature_id, weight) for feature_id, weight in counts.most_common() if weight > 0]
        return tuple(feature_id for feature_id, _ in ranked[: max(1, self.top_k)])

    def similarity(self, left: str, right: str) -> float:
        """Shared-feature similarity between two tokens (0 when nothing is learned)."""

        a, b = set(self.features(left)), set(self.features(right))
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    def neighbours(self, token: str, top_k: int = 6) -> list[tuple[str, float]]:
        """Tokens that share the most features with ``token``."""

        self._ensure()
        target = set(self.features(token))
        if not target:
            return []
        scores: Counter[str] = Counter()
        for other, counts in self.token_features.items():
            if other == token:
                continue
            overlap = len(target & set(counts))
            if overlap:
                scores[other] = overlap
        return [(other, round(overlap / max(1, len(target)), 3)) for other, overlap in scores.most_common(top_k)]

    def state(self) -> dict:
        self._ensure()
        used = sum(1 for feature_id in range(self.n_features) if self.members.get(feature_id))
        return {
            "features": self.n_features,
            "used_features": used,
            "tokens": len(self.token_features),
            "observations": self.observations,
            "splits": self.splits,
        }
