"""Learned cognitive subsystems for grounding, binding, and prediction.

Every neural component is mapped to the four structures in
:mod:`bionic_brain.nn`:

* symbol embeddings enter through dendritic receptive fields;
* somatic threshold/surrogate dynamics produce sparse decisions;
* axon delay/myelin preserves temporal credit;
* synapses carry local eligibility or a gradient approximation of top-down
  prediction error.

No method in this module contains a word list, grammar table, regex rule, or
language name.  Vocabularies are grown only from observed symbols; roles and
intents are opaque learned identifiers supplied by callers.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .nn import Dendrite, NeuronCircuit


def _normalise(vector: Tensor) -> Tensor:
    norm = float(vector.norm())
    return vector / norm if norm else vector


class CorticalGrounding(nn.Module):
    """Learned character/phoneme fields gated before the semantic hub.

    Characters or extracted phonemes become embeddings; a dendritic convolution
    learns local receptive fields.  ``gate`` models thalamic selective gain.  A
    symbol vocabulary is grown from data, so any script or phoneme set works.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 16,
        projection_dim: int = 16,
        capacity: int = 4096,
        kernel_size: int = 3,
        branches: int = 4,
        seed: int = 1907,
    ) -> None:
        super().__init__()
        if projection_dim % branches:
            projection_dim += branches - projection_dim % branches
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.symbols: list[str] = []
        self.symbol_ids: dict[str, int] = {}
        self.embedding = nn.Embedding(self.capacity, embedding_dim)
        self.dendrite = Dendrite(
            embedding_dim,
            projection_dim,
            branches=branches,
            kernel_size=kernel_size,
            branch_nonlinearity="tanh",
        )
        self.gate_bias = nn.Parameter(torch.zeros(1))
        self.gain = nn.Parameter(torch.ones(1))
        torch.manual_seed(seed)
        with torch.no_grad():
            self.embedding.weight.normal_(0.0, 0.25)

    def observe(self, text: str) -> int:
        """Grow the symbol vocabulary from any observed string."""

        added = 0
        for symbol in str(text):
            if not symbol.strip() or symbol in self.symbol_ids:
                continue
            if len(self.symbols) >= self.capacity:
                break
            self.symbol_ids[symbol] = len(self.symbols)
            self.symbols.append(symbol)
            added += 1
        return added

    def encode(self, text: str, *, gate: float = 1.0) -> Tensor:
        """Return a gated dendritic vector for a written/phonetic form."""

        self.observe(text)
        symbols = [self.symbol_ids[symbol] for symbol in str(text) if symbol in self.symbol_ids]
        if not symbols:
            return torch.zeros(self.dendrite.out_features)
        indices = torch.tensor(symbols, dtype=torch.long)
        embedded = self.embedding(indices).transpose(0, 1).unsqueeze(0)  # 1,C,L
        field = self.dendrite(embedded)[0].mean(dim=1)
        return torch.tanh(self.gain * float(gate) + self.gate_bias) * field

    def learn_form(self, text: str, *, epochs: int = 1, lr: float = 0.01) -> float:
        """Local reconstruction error: repeated forms sharpen their own field."""

        if not text:
            return 0.0
        target = self.encode(text).detach()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_value = 0.0
        for _ in range(max(1, epochs)):
            optimizer.zero_grad()
            reconstructed = self.encode(text)
            loss = ((reconstructed - target) ** 2).mean() - 0.01 * reconstructed.mean()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
        return loss_value

    def rank_cells(self, vector: Tensor, cell_ids: list[int]) -> list[int]:
        """Rank cortical cells by learned-field similarity, not by a code table."""

        query = _normalise(vector.detach().cpu())
        anchors = torch.stack([self.receptor_vector(cell_id) for cell_id in cell_ids])
        scores = torch.nn.functional.cosine_similarity(query.unsqueeze(0), anchors, dim=1)
        order = sorted(zip(scores.tolist(), cell_ids), key=lambda pair: (-pair[0], pair[1]))
        return [cell_id for _, cell_id in order]

    def receptor_vector(self, cell_id: int) -> Tensor:
        """Stable dendritic receptor seed for a cortical cell."""

        generator = torch.Generator().manual_seed(self.seed + int(cell_id) * 7919 + 104729)
        vector = torch.randn(self.dendrite.out_features, generator=generator) * 0.25
        return _normalise(vector)

    def rank_cells(self, vector: Tensor, cell_ids: list[int], anchors: dict[int, Tensor] | None = None) -> list[int]:
        """Rank receptors by learned-field similarity to their receptor seeds."""

        query = _normalise(vector.detach().cpu())
        receptor_anchors = anchors or {}
        anchor_vectors = torch.stack([receptor_anchors.get(cell_id, self.receptor_vector(cell_id)) for cell_id in cell_ids])
        scores = torch.nn.functional.cosine_similarity(query.unsqueeze(0), anchor_vectors, dim=1)
        order = sorted(zip(scores.tolist(), cell_ids), key=lambda pair: (-pair[0], pair[1]))
        return [cell_id for _, cell_id in order]

    def learn_receptor(self, cell_id: int, vector: Tensor, rate: float = 0.08) -> None:
        """Local Hebbian update: an active receptor aligns with its input field."""

        key = int(cell_id)
        if not hasattr(self, "receptor_fields") or self.receptor_fields is None:
            self.receptor_fields = {}
        current = self.receptor_fields.get(key, self.receptor_vector(key))
        updated = _normalise((1.0 - float(rate)) * current + float(rate) * vector.detach().cpu())
        self.receptor_fields[key] = updated


class TensorProductBinding:
    """Holographic role/filler binding using circular convolution.

    Binding corresponds to dendritic coincidence of role and filler fields.
    Unbinding is an approximate inverse correlation implemented with the same
    spectral pathway; a small cleanup memory then performs competitive recall.
    """

    def __init__(self, dimension: int = 64, *, seed: int = 1907) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = int(dimension)
        self.seed = int(seed)
        generator = torch.Generator().manual_seed(seed)
        self.role_vectors: dict[str, Tensor] = {}
        self.filler_vectors: dict[str, Tensor] = {}
        self._generator_state = generator.get_state()

    def _random_vector(self, seed_offset: int) -> Tensor:
        generator = torch.Generator().manual_seed(self.seed + 10_000 + seed_offset)
        return torch.randn(self.dimension, generator=generator)

    def role(self, role: str) -> Tensor:
        key = str(role)
        if key not in self.role_vectors:
            self.role_vectors[key] = _normalise(self._random_vector(len(self.role_vectors)))
        return self.role_vectors[key]

    def filler(self, filler: str) -> Tensor:
        key = str(filler)
        if key not in self.filler_vectors:
            stable = sum((index + 1) * (ord(char) + 1) for index, char in enumerate(key))
            vector = torch.zeros(self.dimension)
            for offset in range(3):
                vector[(stable + offset * 17) % self.dimension] = 1.0
            self.filler_vectors[key] = _normalise(vector)
        return self.filler_vectors[key]

    @staticmethod
    def bind(role: Tensor, filler: Tensor) -> Tensor:
        return torch.fft.irfft(torch.fft.rfft(role) * torch.fft.rfft(filler), len(role))

    @staticmethod
    def unbind(bound: Tensor, role: Tensor) -> Tensor:
        spectrum = torch.fft.rfft(bound) * torch.conj(torch.fft.rfft(role))
        return torch.fft.irfft(spectrum / (float(role.norm()) ** 2 + 1e-8), len(role))

    @staticmethod
    def similarity(left: Tensor, right: Tensor) -> float:
        denominator = float(left.norm() * right.norm())
        return float(torch.dot(left, right) / denominator) if denominator else 0.0

    def bind_pair(self, role: str, filler: str) -> Tensor:
        return self.bind(self.role(role), self.filler(filler))

    def query(self, trace: Tensor, role: str) -> tuple[str, float]:
        estimate = self.unbind(trace, self.role(role))
        ranked = sorted(
            ((key, self.similarity(estimate, vector)) for key, vector in self.filler_vectors.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[0] if ranked else ("", 0.0)

    def compose(self, traces: list[Tensor], weights: list[float] | None = None) -> Tensor:
        if not traces:
            return torch.zeros(self.dimension)
        gains = weights or [1.0] * len(traces)
        return _normalise(sum(float(weight) * trace for weight, trace in zip(gains, traces)))


@dataclass
class DiscourseState:
    """Salience-based role tracking with holographic cleanup memory."""

    binding: TensorProductBinding
    decay: float = 0.88
    roles: dict[str, str] = field(default_factory=dict)
    salience: dict[str, float] = field(default_factory=dict)
    history: deque = field(default_factory=lambda: deque(maxlen=32))
    trace: Tensor | None = None

    def introduce(self, entity: str, salience: float = 1.0) -> None:
        self.roles[str(entity)] = str(entity)
        self.salience[str(entity)] = max(float(self.salience.get(entity, 0.0)), float(salience))
        self.binding.filler(entity)

    def bind_role(self, role: str, entity: str) -> Tensor:
        self.introduce(entity)
        contribution = self.binding.bind_pair(role, entity)
        if self.trace is None:
            self.trace = contribution
        else:
            self.trace = self.decay * self.trace + contribution
        self.history.append((role, entity))
        return self.trace

    def resolve(self, probe: str) -> tuple[str, float]:
        """Return the most salient compatible entity for an opaque probe."""

        probe_vector = self.binding.filler(probe)
        ranked = sorted(
            self.salience,
            key=lambda entity: (
                self.binding.similarity(probe_vector, self.binding.filler(entity))
                + 0.25 * self.salience[entity],
                entity,
            ),
            reverse=True,
        )
        if not ranked:
            return "", 0.0
        return ranked[0], float(self.binding.similarity(probe_vector, self.binding.filler(ranked[0])))

    def reinforce(self, entity: str, amount: float = 0.15) -> None:
        for key in self.salience:
            self.salience[key] *= self.decay
        self.salience[str(entity)] = float(self.salience.get(entity, 0.0)) + float(amount)

    def timeline(self) -> list[tuple[str, str]]:
        return list(self.history)


class IntentLearner(nn.Module):
    """Supervised intent/slot classifier over learned token embeddings.

    Labels may come from a teacher, but the trained model alone is used at
    inference.  A dendritic receptive field forms the hidden state; backprop is
    the local top-down prediction-error approximation for this decision circuit.
    """

    def __init__(self, *, embedding_dim: int = 16, hidden_dim: int = 16, capacity: int = 8192, lr: float = 0.01):
        super().__init__()
        self.capacity = int(capacity)
        self.embedding = nn.Embedding(self.capacity, embedding_dim)
        self.circuit = NeuronCircuit(embedding_dim, hidden_dim, branches=1, depth=1)
        self.output = nn.Linear(hidden_dim, 1)
        self.token_ids: dict[str, int] = {}
        self.intents: list[str] = []
        self.intent_ids: dict[str, int] = {}
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        with torch.no_grad():
            self.embedding.weight.normal_(0.0, 0.2)

    def _token(self, token: str) -> int | None:
        key = str(token)
        if key not in self.token_ids and len(self.token_ids) < self.capacity:
            self.token_ids[key] = len(self.token_ids)
        return self.token_ids.get(key)

    def intent_id(self, intent: str) -> int:
        key = str(intent)
        if key not in self.intent_ids:
            self.intent_ids[key] = len(self.intents)
            self.intents.append(key)
        return self.intent_ids[key]

    def observe(self, tokens: list[str], intent: str) -> float:
        ids = [self._token(token) for token in tokens]
        ids = [value for value in ids if value is not None]
        if not ids:
            return 0.0
        target_id = self.intent_id(intent)
        target = torch.tensor([2.0 * target_id - 1.0], dtype=torch.float32)
        self.optimizer.zero_grad()
        embedded = self.embedding(torch.tensor(ids, dtype=torch.long)).mean(dim=0)
        hidden = self.circuit(embedded.unsqueeze(0).transpose(0, 1).unsqueeze(0))[0].mean(dim=1)
        prediction = self.output(hidden.unsqueeze(0)).squeeze(0)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def predict(self, tokens: list[str]) -> tuple[str, float]:
        ids = [self.token_ids.get(str(token)) for token in tokens]
        ids = [value for value in ids if value is not None]
        if not ids or not self.intents:
            return "", 0.0
        with torch.no_grad():
            embedded = self.embedding(torch.tensor(ids, dtype=torch.long)).mean(dim=0)
            hidden = self.circuit(embedded.unsqueeze(0).transpose(0, 1).unsqueeze(0))[0].mean(dim=1)
            value = float(self.output(hidden.unsqueeze(0)).squeeze(0))
        index = max(0, min(len(self.intents) - 1, round((value + 1.0) / 2.0)))
        return self.intents[index], min(1.0, abs(value))


class DifferentiableSequenceModel(nn.Module):
    """Next-symbol prediction over dendrites, soma, axon, and synapses.

    The causal dendritic convolution and recurrent soma replace a surface
    n-gram table.  Cross-entropy backprop is treated as a top-down local
    prediction-error approximation; ``surprise`` reports the natural-log error.
    """

    END = object()

    def __init__(self, *, embedding_dim: int = 16, hidden_dim: int = 16, capacity: int = 8192, context_window: int = 8, lr: float = 0.01):
        super().__init__()
        self.capacity = int(capacity)
        self.context_window = int(context_window)
        self.embedding = nn.Embedding(self.capacity, embedding_dim)
        self.circuit = NeuronCircuit(embedding_dim, hidden_dim, branches=1, depth=1)
        self.output = nn.Linear(hidden_dim, self.capacity)
        self.symbol_ids: dict[str, int] = {}
        self.symbols: list[str] = []
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.updates = 0
        self.total_error = 0.0
        # This small recurrent circuit is faster single-threaded; native loops in
        # the soma otherwise pay more for thread handoff than for computation.
        torch.set_num_threads(1)
        with torch.no_grad():
            self.embedding.weight.normal_(0.0, 0.2)

    def symbol_id(self, token: str) -> int:
        key = str(token)
        if key not in self.symbol_ids:
            if len(self.symbol_ids) >= self.capacity:
                key = next(iter(self.symbol_ids))
            else:
                self.symbol_ids[key] = len(self.symbols)
                self.symbols.append(key)
        return self.symbol_ids[key]

    def _hidden(self, ids: list[int]) -> Tensor:
        torch.set_num_threads(1)
        embedded = self.embedding(torch.tensor([ids], dtype=torch.long))
        return self.circuit(embedded.transpose(1, 2))[0].mean(dim=1)

    def update(self, tokens: list[str], *, surprise_gain: float = 1.0) -> float:
        """One local prediction-error update over a contiguous token window."""

        torch.set_num_threads(1)
        clean = [str(token) for token in tokens if str(token)]
        if len(clean) < 2:
            return 0.0
        if len(clean) > 256:
            # A long paper chunk is still one experience; this homeostatic cap
            # prevents one document from monopolising plasticity.
            clean = clean[:256]
        ids = [self.symbol_id(token) for token in clean]
        # One causal pass predicts every next symbol simultaneously.  This is
        # the dendritic/somatic sequence field, not a corpus-wide replay loop.
        inputs = torch.tensor([ids[:-1]], dtype=torch.long)
        targets = torch.tensor([ids[1:]], dtype=torch.long)
        hidden = self.circuit(self.embedding(inputs).transpose(1, 2))
        logits = self.output(hidden.transpose(1, 2))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.capacity), targets.reshape(-1)
        ) * max(0.05, float(surprise_gain))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        value = float(loss.detach())
        self.updates += len(ids) - 1
        self.total_error += value
        return value

    def distribution(self, tokens: list[str]) -> dict[str, float]:
        ids = [self.symbol_id(str(token)) for token in tokens if str(token)]
        if not ids:
            return {}
        with torch.no_grad():
            probabilities = torch.softmax(self.output(self._hidden(ids[-self.context_window:])).squeeze(0), dim=0)
        return {
            self.symbols[index]: float(probabilities[index])
            for index in range(len(self.symbols))
        }

    def predict_next(self, tokens: list[str]) -> tuple[str, float]:
        ranked = sorted(self.distribution(tokens).items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[0] if ranked else ("", 0.0)

    def surprise(self, context: list[str], actual: str) -> float:
        probability = self.distribution(context).get(str(actual), 0.0)
        return -math.log(max(probability, 1e-12))

    def uncertainty(self, tokens: list[str]) -> float:
        probabilities = sorted(self.distribution(tokens).values(), reverse=True)
        if not probabilities:
            return 1.0
        total = sum(probabilities)
        if total <= 0:
            return 1.0
        return float(-sum((p / total) * math.log(p / total + 1e-12) for p in probabilities))


@dataclass
class ActionLoopResult:
    question: str
    intent: str
    retrieved: list[str]
    consistency_error: float
    output: list[str]
    corrected: bool


class SocialFeedback:
    """Immediate teacher feedback as an in-turn neuromodulatory context."""

    def __init__(self, history: int = 12) -> None:
        self.events: deque[tuple[str, str, float]] = deque(maxlen=history)

    def observe(self, prompt: str, correction: str, reward: float) -> dict:
        self.events.append((str(prompt), str(correction), float(reward)))
        return {"prompt": prompt, "correction": correction, "reward": reward}

    def context(self, prompt: str) -> tuple[str, ...]:
        return tuple(correction for event_prompt, correction, _ in reversed(self.events) if event_prompt == prompt)


class DevelopmentalClock:
    """Replay strength and plasticity gated by a learned/observed phase."""

    def __init__(self) -> None:
        self.observations = 0
        self.phase = "proliferation"

    def observe_progress(self, *, episodes: int, replay_events: int) -> str:
        self.observations += max(int(episodes), int(replay_events))
        if self.observations >= 4:
            self.phase = "mature"
        elif self.observations >= 2:
            self.phase = "differentiation"
        return self.phase

    def replay_fraction(self) -> float:
        return {"proliferation": 0.50, "differentiation": 0.75, "mature": 1.00}[self.phase]

    def plasticity(self) -> float:
        return {"proliferation": 1.00, "differentiation": 0.65, "mature": 0.30}[self.phase]


class DecoderCalibration:
    """Online readout thresholds learned from this brain's own evidence.

    Support, margin, and coverage are retained as distributions.  A small
    quantile model replaces fixed acceptance constants while retaining the
    configured value until enough local evidence exists.
    """

    def __init__(self, history: int = 512, minimum_samples: int = 32) -> None:
        self.support: deque[float] = deque(maxlen=history)
        self.margin: deque[float] = deque(maxlen=history)
        self.coverage: deque[float] = deque(maxlen=history)
        self.minimum_samples = int(minimum_samples)

    def observe(self, *, support: float, margin: float, coverage: float, accepted: bool) -> None:
        if not accepted:
            return
        self.support.append(max(0.0, float(support)))
        self.margin.append(max(0.0, float(margin)))
        self.coverage.append(max(0.0, float(coverage)))

    @staticmethod
    def _quantile(values: deque[float], fraction: float) -> float:
        if len(values) < 2:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(fraction * (len(ordered) - 1))))
        return float(ordered[index])

    def thresholds(
        self,
        *,
        confidence: float,
        margin: float,
        support_per_cell: float,
        coverage: float,
    ) -> dict[str, float]:
        if len(self.support) < self.minimum_samples:
            return {
                "confidence": float(confidence),
                "margin": float(margin),
                "support_per_cell": float(support_per_cell),
                "coverage": float(coverage),
            }
        return {
            "confidence": confidence,
            "margin": max(0.0, self._quantile(self.margin, 0.35)),
            "support_per_cell": max(0.0, self._quantile(self.support, 0.35)),
            "coverage": max(0.0, min(coverage, self._quantile(self.coverage, 0.25))),
        }


class PrototypeField:
    """Online abstraction field over learned distributed vectors.

    Each prototype is a cortical assembly formed by competitive attraction.  A
    novel input either recruits a new assembly or updates the nearest existing
    one; no symbolic category name is required.
    """

    def __init__(self, *, match_threshold: float = 0.58, learning_rate: float = 0.12, max_prototypes: int = 512) -> None:
        self.match_threshold = float(match_threshold)
        self.learning_rate = float(learning_rate)
        self.max_prototypes = int(max_prototypes)
        self.vectors: dict[int, Tensor] = {}
        self.members: dict[int, int] = {}
        self.usage: Counter[int] = Counter()

    def observe(self, vector: Tensor, owner: str | None = None) -> int:
        vector = _normalise(vector.detach().cpu())
        best_id, best_similarity = -1, -1.0
        for prototype_id, prototype in self.vectors.items():
            similarity = self.similarity(vector, prototype)
            if similarity > best_similarity:
                best_id, best_similarity = prototype_id, similarity
        if best_id < 0 or best_similarity < self.match_threshold:
            if len(self.vectors) >= self.max_prototypes:
                best_id = min(self.usage, key=lambda item: (self.usage[item], item))
                self.vectors[best_id] = vector.clone()
            else:
                best_id = len(self.vectors)
                self.vectors[best_id] = vector.clone()
        else:
            updated = (1.0 - self.learning_rate) * self.vectors[best_id] + self.learning_rate * vector
            self.vectors[best_id] = _normalise(updated)
        self.usage[best_id] += 1
        if owner is not None:
            self.members[str(owner)] = best_id
        return best_id

    @staticmethod
    def similarity(left: Tensor, right: Tensor) -> float:
        denominator = float(left.norm() * right.norm())
        return float(torch.dot(left.flatten(), right.flatten()) / denominator) if denominator else 0.0

    def prototype_for(self, owner: str) -> int | None:
        return self.members.get(str(owner))

    def state(self) -> dict:
        return {
            "prototypes": len(self.vectors),
            "observations": sum(self.usage.values()),
            "largest_usage": max(self.usage.values(), default=0),
        }


class ConsistencyMemory:
    """Learned local confidence from this brain's own accepted/rejected acts.

    Patterns are sparse activity sets, not linguistic keys.  Similar past cues
    provide a soft consistency estimate; with no evidence the estimate is
    neutral and cannot invent an answer.
    """

    def __init__(self, history: int = 512, nearest: int = 5) -> None:
        self.patterns: deque[tuple[frozenset[int], float]] = deque(maxlen=history)
        self.nearest = max(1, int(nearest))

    @staticmethod
    def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
        union = len(left | right)
        return len(left & right) / union if union else 0.0

    def observe(self, pattern, outcome: float) -> None:
        clean = frozenset(int(item) for item in pattern)
        if clean:
            self.patterns.append((clean, max(0.0, min(1.0, float(outcome)))))

    def estimate(self, pattern) -> float:
        query = frozenset(int(item) for item in pattern)
        if not query or not self.patterns:
            return 0.5
        scored = sorted(
            ((self._jaccard(query, stored), outcome) for stored, outcome in self.patterns),
            key=lambda pair: -pair[0],
        )[: self.nearest]
        relevant = [pair for pair in scored if pair[0] > 0.05]
        if not relevant:
            return 0.5
        denominator = sum(weight for weight, _ in relevant)
        return sum(weight * outcome for weight, outcome in relevant) / denominator


@dataclass(frozen=True)
class ReplayLink:
    source_timestamp: float
    target_timestamp: float
    source_neurons: tuple[int, ...]
    target_neurons: tuple[int, ...]
    strength: float


class AssociativeReplayPlanner:
    """Plan cross-episode links during offline replay.

    Similar or temporally adjacent episodes can strengthen one another's
    assemblies.  This is learned recombination: no answer is selected here, only
    candidate associative pathways for cortical plasticity.
    """

    def __init__(self, *, max_links: int = 12, max_neurons: int = 8, time_scale: float = 30.0) -> None:
        self.max_links = max(1, int(max_links))
        self.max_neurons = max(1, int(max_neurons))
        self.time_scale = max(1e-6, float(time_scale))

    @staticmethod
    def _overlap(left, right) -> float:
        a, b = set(left), set(right)
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    def plan(self, events) -> list[ReplayLink]:
        ordered = sorted(events, key=lambda event: (float(event.timestamp), id(event)))
        candidates: list[ReplayLink] = []
        for index, source in enumerate(ordered[:-1]):
            target = ordered[index + 1]
            source_neurons = tuple(dict.fromkeys((*source.content_neurons, *source.context_neurons)))[: self.max_neurons]
            target_neurons = tuple(dict.fromkeys(target.motor_neurons))[: self.max_neurons]
            if not source_neurons or not target_neurons:
                continue
            gap = max(0.0, float(target.timestamp) - float(source.timestamp))
            temporal = math.exp(-gap / self.time_scale)
            shared = self._overlap(
                (*source.content_neurons, *source.context_neurons),
                (*target.content_neurons, *target.context_neurons),
            )
            strength = min(1.0, 0.55 * temporal + 0.45 * shared)
            if strength > 0.05:
                candidates.append(ReplayLink(
                    float(source.timestamp), float(target.timestamp),
                    source_neurons, target_neurons, strength,
                ))
        candidates.sort(key=lambda link: (-link.strength, link.source_timestamp, link.target_timestamp))
        return candidates[: self.max_links]
