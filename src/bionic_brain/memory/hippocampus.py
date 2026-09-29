from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class EventSnapshot:
    timestamp: float
    sdr_index: tuple[int, ...]
    content_neurons: tuple[int, ...]
    motor_neurons: tuple[int, ...] = ()
    context_neurons: tuple[int, ...] = ()
    answer_phrase: str = ""
    amygdala_valence: float = 0.0
    da_level: float = 0.0
    goal_relevance: float = 0.0
    region_rates: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    consolidated: bool = False

    @property
    def importance(self) -> float:
        return (
            0.30 * self.scores.get("novelty", 0.0)
            + 0.30 * abs(self.amygdala_valence)
            + 0.20 * abs(self.da_level)
            + 0.20 * self.goal_relevance
            # Curiosity marks what the network itself found worth learning, so
            # replay (and capacity pressure) prefer it over mere recency.
            + 0.08 * self.scores.get("curiosity", 0.0)
        )


class Hippocampus:
    """Selective episodic index, not a full text/vector database."""

    def __init__(self, buffer_size=1_000, capacity=10_000, retain_consolidated=True):
        # The buffer is the fast, recency-weighted index; the store is the
        # longer-term index. Retrieval searches both, so a memory does not
        # vanish from recall just because newer events pushed it out of the
        # fixed-size buffer.
        self.episodic_buffer: deque[EventSnapshot] = deque(maxlen=int(buffer_size))
        self.episodic_store: list[EventSnapshot] = []
        self.capacity = int(capacity)
        self.retain_consolidated = bool(retain_consolidated)
        self.evictions = 0

    def _ensure_fields(self) -> None:
        """Checkpoints written before these fields existed must still load."""

        if not hasattr(self, "evictions"):
            self.evictions = 0
        if not hasattr(self, "retain_consolidated"):
            self.retain_consolidated = True
        if not hasattr(self, "episodic_store") or self.episodic_store is None:
            self.episodic_store = []
        if not hasattr(self, "capacity"):
            self.capacity = 10_000

    @staticmethod
    def similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
        a, b = set(left), set(right)
        union = len(a | b)
        return len(a & b) / union if union else 0.0

    def novelty(self, sdr: tuple[int, ...], recent: int = 100) -> float:
        candidates = list(self.episodic_buffer)[-recent:]
        if not candidates:
            return 1.0
        return 1.0 - max(self.similarity(sdr, item.sdr_index) for item in candidates)

    def should_encode(self, sdr, salience: float, rpe: float, relevance: float, unfinished: bool) -> tuple[bool, dict]:
        scores = {
            "novelty": self.novelty(sdr),
            "salience": max(0.0, salience),
            "rpe": abs(float(rpe)),
            "relevance": max(0.0, relevance),
            "unfinished": 1.0 if unfinished else 0.0,
        }
        threshold_pass = (
            scores["novelty"] > 0.30
            or scores["salience"] > 0.50
            or scores["rpe"] > 0.30
            or scores["relevance"] > 0.40
            or unfinished
        )
        total = 0.4*scores["novelty"] + 0.2*scores["salience"] + 0.2*scores["rpe"] + 0.15*scores["relevance"] + 0.05*scores["unfinished"]
        return bool(threshold_pass and total > 0.35), {**scores, "total": total}

    def question_key(self, event: EventSnapshot) -> str:
        return str(event.scores.get("_question", "")).strip()

    def write(self, event: EventSnapshot) -> bool:
        """Upsert a question-keyed episode.

        Re-teaching an identical cue updates the same index instead of leaving
        several competing traces. This makes later correction/identity teaching
        stable while preserving the most recent event.
        """
        self._ensure_fields()
        question = self.question_key(event)
        if question:
            in_store = False
            in_buffer = False
            for index, old in enumerate(self.episodic_store):
                if self.question_key(old) == question:
                    self.episodic_store[index] = event
                    in_store = True
                    break
            for index, old in enumerate(self.episodic_buffer):
                if self.question_key(old) == question:
                    self.episodic_buffer[index] = event
                    in_buffer = True
                    break
            if not in_store:
                self.episodic_store.append(event)
            if not in_buffer:
                self._append_to_buffer(event)
        else:
            self._append_to_buffer(event)
            self.episodic_store.append(event)
        self._evict_if_needed()
        return True

    def _append_to_buffer(self, event: EventSnapshot) -> None:
        """Insert into the fast index, evicting the least important entry when full."""

        self._ensure_fields()
        limit = self.episodic_buffer.maxlen
        if limit and len(self.episodic_buffer) >= int(limit):
            victim = min(self.episodic_buffer, key=lambda item: (item.importance, item.timestamp))
            self.evictions += 1
            if victim.importance > event.importance:
                # A weak new event must not displace a stronger memory.
                return
            self.episodic_buffer.remove(victim)
        self.episodic_buffer.append(event)

    def _search_pool(self) -> list[EventSnapshot]:
        pool: list[EventSnapshot] = []
        seen: set[int] = set()
        for event in (*self.episodic_buffer, *self.episodic_store):
            key = id(event)
            if key in seen:
                continue
            seen.add(key)
            pool.append(event)
        return pool

    def retrieve(self, cue: tuple[int, ...], top_k: int = 5) -> list[EventSnapshot]:
        scored = [(self.similarity(cue, event.sdr_index), event.timestamp, event) for event in self._search_pool()]
        # Similarity first; newer memories break ties, so later identity teaching
        # can revise an earlier answer without erasing the episodic trace.
        scored.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
        return [event for score, timestamp, event in scored[:top_k] if score > 0.05]

    def retained(self) -> dict:
        """Retention summary used by diagnostics."""

        self._ensure_fields()
        return {
            "buffer": len(self.episodic_buffer),
            "buffer_maxlen": self.episodic_buffer.maxlen,
            "store": len(self.episodic_store),
            "capacity": self.capacity,
            "evictions": self.evictions,
            "question_keyed": sum(1 for event in self._search_pool() if self.question_key(event)),
        }

    def select_replay(self, count: int = 50) -> list[EventSnapshot]:
        unconsolidated = [event for event in self.episodic_store if not event.consolidated]
        unconsolidated.sort(key=lambda event: event.importance, reverse=True)
        return unconsolidated[:max(0, count)]

    def consolidate(self, count: int = 50) -> list[EventSnapshot]:
        self._ensure_fields()
        selected = self.select_replay(count)
        for event in selected:
            event.consolidated = True
        if not getattr(self, "retain_consolidated", True):
            self.episodic_store = [event for event in self.episodic_store if not event.consolidated]
        return selected

    def _evict_if_needed(self) -> None:
        self._ensure_fields()
        while len(self.episodic_store) > self.capacity:
            victim = min(self.episodic_store, key=lambda event: (event.importance, event.timestamp))
            self.episodic_store.remove(victim)
            try:
                self.episodic_buffer.remove(victim)
            except ValueError:
                pass
            self.evictions += 1

    def to_dict(self) -> dict:
        return {"buffer": list(self.episodic_buffer), "store": list(self.episodic_store), "capacity": self.capacity}

    @classmethod
    def from_dict(cls, data: dict) -> "Hippocampus":
        result = cls(capacity=data.get("capacity", 10_000), buffer_size=max(1, len(data.get("buffer", [])) or 1))
        result.episodic_buffer = deque(data.get("buffer", []), maxlen=max(1, len(data.get("buffer", [])) or 1))
        result.episodic_store = list(data.get("store", []))
        return result
