from dataclasses import dataclass, field
import math
import random
import numpy as np


@dataclass
class Synapse:
    """Delay-carrying chemical synapse with eligibility and DA modulation."""

    pre: int
    post: int
    weight: float
    delay: float
    position: np.ndarray
    release_probability: float = 0.82
    eligibility: float = 0.0
    kind: str = "excitatory"
    active: bool = True
    uses: int = 0
    last_pre: float = -1_000.0
    last_post: float = -1_000.0
    metadata: dict = field(default_factory=dict)

    def pre_spike(self, time: float, cfg) -> float | None:
        """Update presynaptic STDP and return delivered charge if release occurs."""
        self.last_pre = time
        if time - self.last_post <= cfg.stdp_window:
            self.weight += 0.035 * self.plastic_gain()
        self.eligibility = min(2.0, self.eligibility + 0.22)
        self.uses += 1
        if not self.active or self.weight <= cfg.prune_weight:
            return None
        release_seed = (self.pre * 100_003 + self.post * 65_537 + int(time * 1_000)) & 0xFFFFFFFF
        if random.Random(release_seed).random() > self.release_probability:
            return None
        return self.weight if self.kind == "excitatory" else -abs(self.weight)

    def post_spike(self, time: float, cfg) -> None:
        self.last_post = time
        if time - self.last_pre <= cfg.stdp_window:
            self.weight -= 0.025 * self.plastic_gain()
        self.eligibility = min(2.0, self.eligibility + 0.08)

    def apply_da(self, dopamine: float, receptor_density: float, cfg) -> None:
        self.weight += dopamine * cfg.da_gain * self.eligibility * receptor_density * 0.01
        self.weight = max(-abs(cfg.learned_weight) * 1.5, min(abs(cfg.learned_weight) * 2.0, self.weight))
        self.eligibility *= 0.70
        self.active = abs(self.weight) >= cfg.prune_weight

    def plastic_gain(self) -> float:
        return max(0.10, self.metadata.get("plasticity", 1.0))



