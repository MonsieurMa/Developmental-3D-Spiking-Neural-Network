from dataclasses import dataclass, field
from collections import deque
import math
import numpy as np


def _hh_rates(v: float) -> tuple[float, float, float, float, float, float]:
    v_clamped = float(np.clip(v, -120.0, 60.0))
    am = 0.1 * (v_clamped + 40.0) / (1.0 - math.exp(-(v_clamped + 40.0) / 10.0))
    bm = 4.0 * math.exp(-(v_clamped + 65.0) / 18.0)
    ah = 0.07 * math.exp(-(v_clamped + 65.0) / 20.0)
    bh = 1.0 / (1.0 + math.exp(-(v_clamped + 35.0) / 10.0))
    denom = 1.0 - math.exp(-(v_clamped + 55.0) / 10.0)
    an = 0.01 * (v_clamped + 55.0) / denom if abs(denom) > 1e-9 else 0.1
    bn = 0.125 * math.exp(-(v_clamped + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


@dataclass
class Neuron:
    """A spatially embedded HH/LIF neuron or progenitor-derived cell."""

    id: int
    position: np.ndarray
    region: str
    cell_type: str = "excitatory"
    model: str = "lif"
    role: str = "neuron"
    word: str | None = None
    V: float = -65.0
    m: float = 0.0529
    h: float = 0.5961
    n: float = 0.3177
    g_e: float = 0.0
    g_i: float = 0.0
    e_e: float = 0.0
    e_i: float = -80.0
    g_leak: float = 0.30
    threshold: float = 0.0
    threshold_offset: float = 0.0
    noise_state: float = 0.0
    energy: float = 1.0
    age: float = 0.0
    plasticity: float = 1.0
    alive: bool = True
    divisions: int = 0
    last_spike: float = -1_000.0
    refractory_until: float = -1.0
    spike_times: deque = field(default_factory=lambda: deque(maxlen=2_000))

    def advance(
        self,
        dt: float,
        current: float,
        cfg,
        rng: np.random.Generator,
        modulation_gain: float = 1.0,
        now: float | None = None,
    ) -> bool:
        if not self.alive:
            return False
        self.age += dt
        self.energy = max(0.0, self.energy - dt * (0.00004 + 0.0003 * abs(current)))
        if self.role == "glia":
            self.energy = min(1.5, self.energy + dt * 0.0002)
            return False

        # Ornstein-Uhlenbeck-like membrane noise.
        self.noise_state += dt * (-self.noise_state / cfg.noise_tau)
        self.noise_state += math.sqrt(max(1e-12, dt)) * cfg.noise_strength * float(rng.normal(0.0, 1.0))
        drive = float(current) * modulation_gain + self.noise_state

        # Homeostatic threshold plasticity also accepts checkpoints created
        # before this field existed.
        effective_threshold = self.threshold + float(getattr(self, "threshold_offset", 0.0))
        if self.model == "hh":
            am, bm, ah, bh, an, bn = _hh_rates(self.V)
            m_inf = am / max(1e-9, am + bm)
            self.m += dt * (am * (1.0 - self.m) - bm * self.m)
            self.h += dt * (ah * (1.0 - self.h) - bh * self.h)
            self.n += dt * (an * (1.0 - self.n) - bn * self.n)
            self.m = max(m_inf * 0.85 + self.m * 0.15, 0.0)
            g_na = cfg.g_na * self.m ** 3 * self.h
            g_k = cfg.g_k * self.n ** 4
            ionic = g_na * (self.V - cfg.e_na) + g_k * (self.V - cfg.e_k) + cfg.g_leak * (self.V - cfg.e_leak)
            self.V += dt * (drive - ionic) / cfg.c_m
            fired = self.V >= effective_threshold
        else:
            tau = cfg.lif_tau
            self.V += dt * (-(self.V + 65.0) + drive) / tau
            fired = self.V >= effective_threshold

        if fired and t_refractory_ok(self, dt):
            self._fire(dt, cfg, now)
            return True

        if self.model == "hh" and self.V > 55.0:
            self.V = 55.0
        elif self.model != "hh" and self.V > 30.0:
            self.V = 30.0
        if self.V < -95.0:
            self.V = -95.0
        if self.age > 1.0:
            self.energy = min(1.5, self.energy + dt * 0.00001)
        return False

    def _fire(self, dt: float, cfg, now: float | None = None) -> None:
        self.last_spike = self.age
        self.refractory_until = self.age + cfg.refractory_ms
        self.spike_times.append(self.age if now is None else float(now))
        self.energy = max(0.0, self.energy - 0.004)
        if self.model == "hh":
            self.V = -75.0
            self.m = 0.05
            self.h = 0.60
            self.n = 0.32
        else:
            self.V = cfg.lif_reset

    def recent_rate(self, now: float | None = None, window_ms: float = 100.0) -> float:
        reference = self.age if now is None else float(now)
        while self.spike_times and reference - self.spike_times[0] > window_ms:
            self.spike_times.popleft()
        return len(self.spike_times) / (window_ms / 1000.0)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "position": self.position.tolist(), "region": self.region,
            "cell_type": self.cell_type, "model": self.model, "role": self.role,
            "word": self.word, "V": self.V, "m": self.m, "h": self.h, "n": self.n,
            "g_e": self.g_e, "g_i": self.g_i, "e_e": self.e_e, "e_i": self.e_i,
            "g_leak": self.g_leak, "threshold": self.threshold,
            "threshold_offset": float(getattr(self, "threshold_offset", 0.0)),
            "noise_state": self.noise_state, "energy": self.energy, "age": self.age,
            "plasticity": self.plasticity, "alive": self.alive, "divisions": self.divisions,
            "last_spike": self.last_spike, "refractory_until": self.refractory_until,
            "spike_times": list(self.spike_times),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Neuron":
        result = cls(id=int(data["id"]), position=np.asarray(data["position"], dtype=float), region=data["region"])
        for key, value in data.items():
            if key in {"id", "position"}:
                continue
            if key == "spike_times":
                result.spike_times = deque(value, maxlen=2_000)
            else:
                setattr(result, key, value)
        return result


def t_refractory_ok(neuron: Neuron, dt: float) -> bool:
    return neuron.age >= neuron.refractory_until - dt * 0.5





