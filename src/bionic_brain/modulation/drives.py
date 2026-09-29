"""Intrinsic motivation: the network decides what is worth learning.

Three biologically motivated components:

* novelty   - how unfamiliar the current pattern is versus recent experience
* surprise  - how badly the network's own next-token prediction was violated
* progress  - whether surprise is falling over the recent window (learning)

The resulting drive gates local plasticity and orders hippocampal replay, so
learning is not driven only by whoever is teaching. No backpropagation, no
external objective, no labels.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class IntrinsicDrive:
    novelty_weight: float = 0.35
    surprise_weight: float = 0.45
    progress_weight: float = 0.20
    window: int = 40
    history: deque[float] = field(default_factory=lambda: deque(maxlen=40))
    total_signals: int = 0

    def __post_init__(self) -> None:
        self.history = deque(self.history or [], maxlen=self.window)

    def progress(self) -> float:
        """Relative drop in surprise over the recent window (0..1)."""

        if len(self.history) < 4:
            return 0.0
        values = list(self.history)
        half = max(1, len(values) // 2)
        early = sum(values[:half]) / half
        late = sum(values[half:]) / max(1, len(values) - half)
        if early <= 1e-9:
            return 0.0
        return max(0.0, min(1.0, (early - late) / early))

    def signal(self, *, novelty: float, surprise: float) -> float:
        """Return a 0..1 curiosity drive and record the surprise trajectory."""

        surprise = max(0.0, float(surprise))
        self.history.append(min(1.0, surprise / 4.0))
        self.total_signals += 1
        return self.score(novelty=novelty, surprise=surprise)

    def score(self, *, novelty: float, surprise: float) -> float:
        """Same combination without recording, for tagging episodic events."""

        novelty_term = max(0.0, min(1.0, float(novelty)))
        surprise_term = max(0.0, min(1.0, surprise / 4.0))
        drive = (
            self.novelty_weight * novelty_term
            + self.surprise_weight * surprise_term
            + self.progress_weight * self.progress()
        )
        return max(0.0, min(1.0, drive))

    def state(self) -> dict:
        return {
            "signals": self.total_signals,
            "mean_recent_surprise": round(sum(self.history) / len(self.history), 4) if self.history else 0.0,
            "progress": round(self.progress(), 4),
        }
