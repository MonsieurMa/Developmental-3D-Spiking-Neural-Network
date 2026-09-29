from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
from ..atlas.regions import REGION_BY_CODE


RECEPTOR_MAP = {
    "high": 1.0,
    "medium": 0.60,
    "low": 0.25,
}

MODULATORS = ("DA", "5HT", "ACh", "NE", "Histamine", "GABA")
MODULATOR_SOURCES = {
    "DA": ("SN", "Stri"),
    "5HT": ("Pons", "Med"),
    "ACh": ("Thal", "DLPFC"),
    "NE": ("MB", "Pons"),
    "Histamine": ("Hypo",),
    "GABA": ("Stri", "GP"),
}
MODULATOR_BASELINE = {"DA": 0.0, "5HT": 0.50, "ACh": 0.45, "NE": 0.45, "Histamine": 0.45, "GABA": 0.40}


def _region_receptors(region: str) -> dict[str, str]:
    if region in {"Stri", "DLPFC", "VMPFC", "OFC"}:
        da = serotonin = ach = ne = "high"
    elif region in {"Hipp", "EC"}:
        da, serotonin, ach, ne = "medium", "medium", "high", "medium"
    elif region == "Amy":
        da = serotonin = ne = "high"
        ach = "medium"
    elif region in {"V1", "V2", "V4", "MT", "CB"}:
        da, serotonin, ach, ne = "low", "low", "medium", "medium"
    else:
        da, serotonin, ach, ne = "medium", "medium", "medium", "medium"
    return {"DA": da, "5HT": serotonin, "ACh": ach, "NE": ne, "GABA": "medium", "Histamine": "medium"}


@dataclass
class NeuromodulatorField:
    """Small 3-D volume for diffusion and local chemical interaction.

    This is deliberately much coarser than the morphogen field: neuromodulators
    are volume signals, not developmental identity gradients.
    """

    bounds: tuple[float, float, float, float, float, float]
    shape: tuple[int, int, int] = (12, 14, 11)
    diffusion: float = 0.060
    decay: float = 0.018
    time: float = 0.0
    concentration: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.concentration = np.array([MODULATOR_BASELINE[name] for name in MODULATORS], dtype=float)[
            :, None, None, None
        ] * np.ones((len(MODULATORS), *self.shape), dtype=float)

    def _region_index(self, region: str) -> tuple[int, int, int]:
        from ..atlas.regions import REGION_BY_CODE

        lo = self.bounds[0::2]
        hi = self.bounds[1::2]
        center = REGION_BY_CODE[region].center
        fractions = [
            (float(center[axis]) - lo[axis]) / (hi[axis] - lo[axis])
            for axis in range(3)
        ]
        return tuple(
            int(np.clip(round(value * (count - 1)), 0, count - 1))
            for value, count in zip(fractions, self.shape)
        )

    def inject_region(self, region: str, modulator: str, amount: float, radius: int = 1) -> None:
        if modulator not in MODULATORS or region not in REGION_BY_CODE:
            return
        channel = MODULATORS.index(modulator)
        ix, iy, iz = self._region_index(region)
        x0, x1 = max(0, ix - radius), min(self.shape[0], ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(self.shape[1], iy + radius + 1)
        z0, z1 = max(0, iz - radius), min(self.shape[2], iz + radius + 1)
        self.concentration[channel, x0:x1, y0:y1, z0:z1] += float(amount)

    def diffuse(self, dt: float = 10.0) -> None:
        c = self.concentration
        lap = np.zeros_like(c)
        for axis in range(3):
            center = [slice(None), slice(None), slice(None), slice(None)]
            before = list(center); after = list(center)
            center[axis + 1] = slice(1, -1)
            before[axis + 1] = slice(None, -2)
            after[axis + 1] = slice(2, None)
            lap[tuple(center)] += c[tuple(before)] + c[tuple(after)] - 2.0 * c[tuple(center)]
        # Effective rates are normalized; this keeps Euler integration stable.
        rate = min(0.14, self.diffusion * dt)
        loss = min(0.20, self.decay * dt)
        c += rate * lap
        # Recovery, depletion, and inhibitory/stabilizing reactions.
        da, serotonin, ach, ne, histamine, gaba = c
        gaba_effect = np.maximum(0.0, gaba - MODULATOR_BASELINE["GABA"])
        serotonin_effect = np.maximum(0.0, serotonin - MODULATOR_BASELINE["5HT"])
        c[0] -= 0.055 * gaba_effect * dt + 0.035 * serotonin_effect * dt
        c[2] -= 0.045 * gaba_effect * dt
        c[3] -= 0.050 * gaba_effect * dt + 0.030 * serotonin_effect * dt
        c[4] -= 0.035 * gaba_effect * dt
        for index, name in enumerate(MODULATORS):
            target = MODULATOR_BASELINE[name]
            c[index] += (target - c[index]) * min(0.12, 0.006 * dt)
        np.clip(c, 0.0, 2.0, out=c)
        self.time += dt

    def means(self) -> dict[str, float]:
        return {name: float(self.concentration[index].mean()) for index, name in enumerate(MODULATORS)}

    def state(self) -> dict:
        return {
            "shape": self.shape,
            "time": self.time,
            "means": self.means(),
        }


@dataclass
class ModulatorySystem:
    """Global neuromodulators, arousal, affect, fatigue, attention, motivation."""

    concentrations: dict = field(default_factory=lambda: dict(MODULATOR_BASELINE))
    states: dict = field(default_factory=lambda: {
        "arousal": 0.78, "valence": 0.0, "fatigue": 0.0,
        "attention_focus": "language", "motivation": 0.55,
    })
    reward_history: list = field(default_factory=list)

    def receptor_density(self, region: str, modulator: str) -> float:
        levels = _region_receptors(region)
        return RECEPTOR_MAP[levels.get(modulator, "medium")]

    def thalamic_gain(self, region: str = "Thal") -> float:
        attention = max(0.0, self.states["arousal"]) * max(0.0, self.concentrations["ACh"])
        fatigue_penalty = 1.0 / (1.0 + self.states["fatigue"])
        return float(min(2.0, max(0.0, 0.35 + 1.25 * attention * fatigue_penalty)))

    def neuronal_gain(self, region: str) -> float:
        rec = {m: self.receptor_density(region, m) for m in ("DA", "5HT", "ACh", "NE", "GABA")}
        gain = 1.0
        gain += 0.22 * self.concentrations["DA"] * rec["DA"]
        gain += 0.18 * self.concentrations["ACh"] * rec["ACh"]
        gain += 0.18 * self.concentrations["NE"] * rec["NE"] * self.states["arousal"]
        gain -= 0.18 * self.concentrations["GABA"] * rec["GABA"]
        gain -= 0.18 * self.states["fatigue"]
        gain += 0.05 * self.states["valence"]
        return float(max(0.35, min(2.25, gain)))

    def plasticity_factor(self, region: str) -> float:
        density = self.receptor_density(region, "DA")
        fatigue = 1.0 / (1.0 + self.states["fatigue"])
        return float(max(0.05, (0.4 + self.concentrations["DA"] * density) * fatigue))

    def apply_reward(self, reward_prediction_error: float, source: str = "explicit") -> None:
        self.concentrations["DA"] = float(np_clip(self.concentrations["DA"] + 0.35 * reward_prediction_error))
        self.states["valence"] = float(np_clip(self.states["valence"] + 0.18 * reward_prediction_error, -1.0, 1.0))
        self.reward_history.append((round(self.time if hasattr(self, "time") else 0.0, 3), float(rpe_safe(reward_prediction_error)), source))
        if len(self.reward_history) > 100:
            del self.reward_history[:-100]

    def tick(self, seconds: float, field_means: dict[str, float] | None = None) -> None:
        if field_means:
            # Volume chemistry slowly entrains the global state.
            blend = min(0.25, seconds * 0.008)
            for name in MODULATORS:
                self.concentrations[name] = float(
                    np_clip((1.0 - blend) * self.concentrations[name] + blend * field_means[name])
                )
        else:
            decay = math.exp(-seconds / 250.0)
            self.concentrations["DA"] = float(np_clip(self.concentrations["DA"] * decay, -1.0, 1.5))
        self.states["fatigue"] = float(np_clip(self.states["fatigue"] + seconds * 0.000005, 0.0, 1.0))
        self.states["arousal"] = float(np_clip(
            0.20 + 0.65 * self.concentrations["NE"] + 0.25 * self.concentrations["Histamine"]
            - 0.25 * self.states["fatigue"], 0.0, 1.0,
        ))
        self.states["motivation"] = float(np_clip(
            0.35 + 0.30 * self.concentrations["DA"] + 0.15 * self.states["valence"], 0.0, 1.0,
        ))

    def to_dict(self) -> dict:
        return {
            "concentrations": self.concentrations.copy(),
            "states": self.states.copy(),
            "reward_history": self.reward_history[-20:],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModulatorySystem":
        result = cls()
        result.concentrations.update(data.get("concentrations", {}))
        result.states.update(data.get("states", {}))
        result.reward_history = list(data.get("reward_history", []))
        return result


def np_clip(value: float, lower: float = -1.0, upper: float = 1.5) -> float:
    return max(lower, min(upper, value))


def rpe_safe(value: float) -> float:
    return max(-1.0, min(1.5, float(value)))

