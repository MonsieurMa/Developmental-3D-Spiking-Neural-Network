from __future__ import annotations

from dataclasses import dataclass, field, fields
from collections import Counter, defaultdict, deque
from pathlib import Path
import gzip
import hashlib
import math
import os
import pickle
import random
import re
import tempfile
import time as wall_time
import uuid

import numpy as np

from .config.defaults import DEFAULT_CONFIG, BionicConfig
from .atlas.regions import REGION_BY_CODE, REGION_CODES
from .atlas.seeds import build_seed_plan
from .cells.neuron import Neuron
from .synapses.model import Synapse
from .morphogens.field import MorphogenField
from .tracts.channel import TractSystem
from .modulation.modulators import ModulatorySystem, NeuromodulatorField
from .simulation.events import EventQueue
from .language.sdr import SDRSpace
from .language.dialogue import core_dialogue_tokens
from .language.features import ContextFeatureSpace, TokenStatistics
from .language.relations import RelationDiscoverer
from .language.subwords import SubwordTokenizer
from .memory.hippocampus import Hippocampus, EventSnapshot
from .memory.learned_schemas import LearnedSchemaMemory
from .modulation.drives import IntrinsicDrive
from .cognition import (
    AssociativeReplayPlanner,
    CorticalGrounding,
    ConsistencyMemory,
    DevelopmentalClock,
    DecoderCalibration,
    DifferentiableSequenceModel,
    DiscourseState,
    IntentLearner,
    PrototypeField,
    SocialFeedback,
    TensorProductBinding,
)

# Peripheral cortex for each sensory channel. All channels converge on the
# shared semantic hub (Wernicke), which is what makes cross-modal transfer
# possible: hearing a word and reading it activate the same meaning assembly.
MODALITY_REGIONS = {"visual": "V1", "auditory": "A1", "somatosensory": "S1"}


@dataclass
class RegionState:
    code: str
    neurons: list[int] = field(default_factory=list)
    activation: deque = field(default_factory=lambda: deque(maxlen=300))
    recent_spikes: deque = field(default_factory=lambda: deque(maxlen=2_000))

    def mean_rate(self, window_ms: float = 100.0) -> float:
        if not self.recent_spikes:
            return 0.0
        latest = self.recent_spikes[-1][0]
        return sum(1.0 for t, _ in self.recent_spikes if latest - t <= window_ms) / (window_ms / 1000.0)


@dataclass
class WordBinding:
    token: str
    token_id: int
    sensory_sdr: tuple[int, ...]
    motor_sdr: tuple[int, ...]
    sensory: tuple[int, ...]
    motor: tuple[int, ...]
    context_sdr: tuple[int, ...]
    context: tuple[int, ...] = ()
    # Modality-specific peripheral assemblies (V1/A1/S1), grown by experience.
    channels: dict[str, tuple[int, ...]] = field(default_factory=dict)


class BionicBrain:
    """Aggregate façade over anatomy, development, simulation, language, and memory.

    The class intentionally contains no backpropagation and no answer table.
    Training uses local eligibility, teaching currents, STDP, and dopamine.
    """

    STATE_VERSION = 3

    def __init__(
        self,
        config: BionicConfig = DEFAULT_CONFIG,
        *,
        fresh: bool = True,
        seed: int = 1907,
        develop: bool = True,
    ) -> None:
        if not fresh:
            raise ValueError("use BionicBrain.load to restore an existing brain")
        self.config = config
        self.random_seed = int(seed)
        self.rng = random.Random(self.random_seed)
        self.np_rng = np.random.default_rng(self.random_seed)
        self.time = 0.0
        self.lifecycle_phase = "proliferation"
        self.events = EventQueue()
        self.neurons: list[Neuron] = []
        self.synapses: list[Synapse] = []
        self.outgoing: dict[int, list[int]] = defaultdict(list)
        self.incoming: dict[int, list[int]] = defaultdict(list)
        self.regions: dict[str, RegionState] = {code: RegionState(code) for code in REGION_CODES}
        self.morphogens = MorphogenField(
            config.space_bounds, config.morphogen_grid,
            config.morphogen_diffusion, config.morphogen_decay,
            config.morphogen_inhibition,
        )
        self.tracts = TractSystem()
        self.modulators = ModulatorySystem()
        self.modulator_field = NeuromodulatorField(
            config.space_bounds, getattr(config, "modulator_grid", (12, 14, 11)),
            getattr(config, "modulator_diffusion", 0.060),
            getattr(config, "modulator_decay", 0.018),
        )
        self.sdr = SDRSpace(k=config.sdr_k)
        self.bindings: dict[str, WordBinding] = {}
        self.word_display: dict[str, str] = {}
        self.dialogue_state = {
            "recent_words": deque(maxlen=config.working_memory_tokens),
            "speaker": "user",
            "turn_count": 0,
            "topic": tuple(),
        }
        self.working_trace: dict[int, float] = {}
        self.active_context: tuple[int, ...] = ()
        self.hippocampus = Hippocampus(config.hippocampus_buffer, config.hippocampus_capacity)
        self.subword_tokenizer = SubwordTokenizer(
            max_merges=int(getattr(config, "subword_max_merges", 512)),
            min_pair_count=int(getattr(config, "subword_min_pair_count", 2)),
            max_unit_symbols=int(getattr(config, "subword_max_unit_symbols", 3)),
        )
        self.subwords_enabled = bool(getattr(config, "use_subword_tokenizer", False))
        self.grounding: CorticalGrounding | None = None
        self.binding_memory: TensorProductBinding | None = None
        self.discourse: DiscourseState | None = None
        self.sequence_model: DifferentiableSequenceModel | None = None
        self.intent_model: IntentLearner | None = None
        self.social_feedback = SocialFeedback()
        self.development_clock = DevelopmentalClock()
        self.decoder_calibration = DecoderCalibration()
        self.prototype_field = PrototypeField()
        self.consistency_memory = ConsistencyMemory()
        self.replay_planner = AssociativeReplayPlanner()
        self._association_cache: dict[tuple[int, int], int] = {}
        self._association_pairs: set[tuple[int, int]] = set()
        self._association_outgoing: dict[int, set[int]] = defaultdict(set)
        self._association_incoming: dict[int, set[int]] = defaultdict(set)
        self.grounding_cell_anchors: dict[str, dict[int, Tensor]] = {}
        self.grounding_cell_owners: dict[str, tuple[int, ...]] = {}
        if bool(getattr(config, "use_sensory_grounding", True)):
            self.grounding = CorticalGrounding(
                embedding_dim=int(getattr(config, "grounding_embedding_dim", 16)),
                projection_dim=int(getattr(config, "grounding_projection_dim", 16)),
                seed=self.random_seed,
            )
        if bool(getattr(config, "use_binding_memory", True)):
            self.binding_memory = TensorProductBinding(dimension=64, seed=self.random_seed)
            self.discourse = DiscourseState(self.binding_memory)
        if bool(getattr(config, "use_learned_intent", True)):
            self.intent_model = IntentLearner(
                embedding_dim=int(getattr(config, "grounding_embedding_dim", 16)),
                hidden_dim=int(getattr(config, "neural_hidden_dim", 16)),
                lr=0.01,
            )
        if bool(getattr(config, "use_neural_sequence", False)):
            self.sequence_model = DifferentiableSequenceModel(
                embedding_dim=int(getattr(config, "grounding_embedding_dim", 16)),
                hidden_dim=int(getattr(config, "neural_hidden_dim", 16)),
                context_window=int(getattr(config, "neural_context_window", 8)),
                lr=0.01,
            )
        self.stimuli: dict[int, tuple[float, float]] = {}
        self.pending_currents: dict[int, float] = defaultdict(float)
        self.recent_synapses: set[int] = set()
        # Hebbian co-activation bookkeeping: tokens that repeatedly fire together
        # end up sharing a few sensory cells, which is what lets a learned cue
        # generalize to a paraphrase that shares no surface token.
        self.cooccurrence: dict[tuple[str, str], int] = {}
        self.shared_cells: dict[str, int] = {}
        # Reverse index: motor cell -> word bindings it belongs to. Decoding then
        # scales with the number of cells that actually fired instead of with
        # the size of the vocabulary.
        self.motor_bindings: dict[int, tuple[str, ...]] = {}
        self._recent_spike_cells: deque[tuple[float, int]] = deque(maxlen=20_000)
        # Ongoing perceptual context: sentences that were read or heard and are
        # still available for role-filler binding (prefrontal working memory).
        self.dialogue_context: deque[tuple[str, ...]] = deque(maxlen=6)
        self.relation_sentences: deque[tuple[str, ...]] = deque(maxlen=256)
        self.token_statistics = TokenStatistics()
        self.schemas = LearnedSchemaMemory(
            statistics=self.token_statistics,
            observe_statistics=False,
            relative_floor=float(getattr(config, "learned_relation_relative_floor", 0.35)),
        )
        self.context_features = ContextFeatureSpace(
            n_features=int(getattr(config, "context_features", 256)),
            top_k=int(getattr(config, "context_feature_top_k", 4)),
            token_statistics=self.token_statistics,
            update_statistics=False,
        )
        self.relation_discoverer = RelationDiscoverer()
        self.schemas.set_relation_source(
            self.relation_discoverer,
            enabled=True,
            threshold=float(getattr(config, "learned_relation_threshold", 0.35)),
        )
        self.feature_cell_ids: dict[int, int] = {}
        self.drive = IntrinsicDrive()
        self.last_learning_synapses: tuple[int, ...] = ()
        self.active_eligibility: set[int] = set()
        self.rehearsal_synapses: set[int] | None = None
        self.suppress_association = False
        self.suppress_tracts = False
        self.last_response_plastic_synapses: tuple[int, ...] = ()
        self.last_response_info: dict = {}
        self.recalled_motor: tuple[int, ...] = ()
        self.recalled_answer: list[str] = []
        self.last_output: list[str] = []
        self.last_region_activity: dict[str, float] = {}
        self.stats = {"predictions": 0, "correct": 0, "train_examples": 0, "spikes": 0}
        self._last_lifecycle = 0.0
        self._last_chem = 0.0
        self._last_neurochem = 0.0
        self._last_scale = 0.0
        self._tract_fire_count: dict[str, int] = defaultdict(int)
        self._tract_last_fire: dict[str, float] = defaultdict(float)
        self._fired_last_step: list[int] = []
        self._active_window: deque[int] = deque(maxlen=3_000)
        self._develop_seeds()
        if develop:
            self.develop()
            self._develop_local_synapses()

    def _tokenize(self, text: str) -> list[str]:
        """Central token gateway: learned subwords, with characters as atoms.

        The fallback exists only for an untrained brain/checkpoint. Once BPE
        merges have been learned from corpora, every runtime path uses the same
        learned units instead of deciding language boundaries by regular
        expression.
        """

        if self.subwords_enabled and self.subword_tokenizer.merges:
            return self.subword_tokenizer.encode(text)
        # Structural fallback: learned/BPE tokenizer before training still only
        # sees Unicode alphanumeric runs and punctuation boundaries.
        return self.subword_tokenizer.encode(text)

    def _sequence_context(self, tokens: list[str]) -> list[str]:
        window = int(getattr(self.config, "neural_context_window", 8))
        return [str(token) for token in tokens][-max(1, window):]

    def _sequence_surprise(self, context: list[str], actual: str) -> float:
        if self.sequence_model is None:
            raise RuntimeError("neural sequence model is required")
        return self.sequence_model.surprise(self._sequence_context(context), actual)

    def _sequence_update(self, tokens: list[str], *, gain: float = 1.0) -> float:
        if self.sequence_model is None:
            raise RuntimeError("neural sequence model is required")
        return self.sequence_model.update(tokens, surprise_gain=gain)

    def _core_dialogue_tokens(self, tokens: list[str]) -> list[str]:
        """Observe token distribution once, then normalize the dialogue cue."""

        self.token_statistics.observe(tokens, sentence_final=True)
        return core_dialogue_tokens(tokens, self.token_statistics, observe=False)

    @staticmethod
    def _is_sentence_boundary(token: str) -> bool:
        """Any non-word token returned by the tokenizer closes a segment."""

        return bool(token) and not all(char.isalnum() for char in token)

    def train_subword_tokenizer(
        self,
        lines,
        *,
        enabled: bool = True,
        reset: bool = True,
    ) -> dict:
        """Fit the BPE learner on raw language data before network training.

        Tokenization is corpus learning, not runtime inference: no teacher or
        rule supplies boundaries, and every resulting unit still has to be bound
        through sensory/motor populations. Fit it before training a fresh
        network; changing units after associations exist would change the
        identity of already-bound engrams.
        """

        if reset:
            self.subword_tokenizer = SubwordTokenizer(
                max_merges=int(getattr(self.config, "subword_max_merges", 512)),
                min_pair_count=int(getattr(self.config, "subword_min_pair_count", 2)),
                max_unit_symbols=int(getattr(self.config, "subword_max_unit_symbols", 3)),
            )
        self.subword_tokenizer.learn(lines)
        self.subwords_enabled = bool(enabled and self.subword_tokenizer.merges)
        return self.subword_tokenizer.diagnostics()

    # ------------------------------------------------------------------
    # Development
    # ------------------------------------------------------------------
    def _develop_seeds(self) -> None:
        plan = build_seed_plan(self.config.seed_profile, self.rng)
        for local_id, seed in enumerate(plan):
            spec = REGION_BY_CODE[seed.region]
            role = seed.role
            cell_type = "source" if role == "source" else "glia" if role == "glia" else "excitatory"
            model = spec.model
            if role == "undiff":
                cell_type = "inhibitory" if self.rng.random() < self.config.inhibitory_fraction else "excitatory"
            neuron = Neuron(
                id=len(self.neurons),
                position=np.asarray(seed.position, dtype=float),
                region=seed.region,
                cell_type=cell_type,
                model=model,
                role=role,
                g_leak=self.config.g_leak,
            )
            self.neurons.append(neuron)

    def develop(self) -> None:
        """Establish identity fields from regional source secretion."""
        for code in REGION_CODES:
            self.morphogens.secrete_region(code, 1.0, radius_cells=2)
        for _ in range(self.config.chem_steps):
            self.morphogens.diffuse(self.config.chem_dt)
        # Undifferentiated progenitors read local gradients and settle into a fate.
        for neuron in self.neurons:
            if neuron.role != "undiff":
                continue
            emitted, strength = self.morphogens.regional_activity(neuron.position)
            if strength > 0.05:
                neuron.region = self.morphogens.classify(neuron.position)
        self.regions = {code: RegionState(code) for code in REGION_CODES}
        for neuron in self.neurons:
            self.regions[neuron.region].neurons.append(neuron.id)

    def _develop_local_synapses(self) -> None:
        """Create sparse local synaptogenesis; no global connectome is supplied."""
        for region in self.regions.values():
            neurons = [self.neurons[i] for i in region.neurons if self.neurons[i].role != "glia"]
            for neuron in neurons:
                others = sorted(
                    (other for other in neurons if other.id != neuron.id),
                    key=lambda other: float(np.linalg.norm(other.position-neuron.position)),
                )
                for target in others[:3]:
                    if self._has_synapse(neuron.id, target.id):
                        continue
                    inhibitory = neuron.cell_type == "inhibitory"
                    weight = -self.rng.uniform(0.15, 0.40) if inhibitory else self.rng.uniform(0.08, 0.28)
                    self._add_synapse(neuron.id, target.id, weight, "inhibitory" if inhibitory else "excitatory")

    def maybe_divide(self) -> int:
        """Energy/rate-triggered proliferation during the overproduction phase."""
        if self.lifecycle_phase != "proliferation" or len(self.neurons) >= self.config.max_neurons:
            return 0
        created = 0
        candidates = [n for n in self.neurons if n.alive and n.role == "neuron" and n.energy > 1.15 and n.recent_rate(self.time) > 3.0]
        self.rng.shuffle(candidates)
        for parent in candidates[:3]:
            if len(self.neurons) >= self.config.max_neurons:
                break
            offset = self.np_rng.normal(0.0, 0.5, 3)
            child = Neuron(
                id=len(self.neurons),
                position=parent.position + offset,
                region=parent.region,
                cell_type=parent.cell_type,
                model=parent.model,
                role="neuron",
                g_leak=parent.g_leak,
                energy=0.85,
            )
            self.neurons.append(child)
            self.regions[child.region].neurons.append(child.id)
            parent.energy *= 0.65
            parent.divisions += 1
            created += 1
        return created

    # ------------------------------------------------------------------
    # Core event-driven simulation
    # ------------------------------------------------------------------
    def step(self, duration: float | None = None) -> None:
        dt = self.config.dt
        steps = int(round((duration or dt) / dt))
        for _ in range(max(1, steps)):
            target_time = self.time + dt
            currents = self.pending_currents
            self.pending_currents = defaultdict(float)
            for event in self.events.due(target_time):
                if event.kind == "synapse":
                    post = int(event.payload["post"])
                    currents[post] += float(event.payload["current"])
                elif event.kind == "activate":
                    self._activate_population(
                        event.payload["ids"], current=float(event.payload["current"]),
                        duration=float(event.payload["duration"]), start=target_time,
                    )
                elif event.kind == "region_signal":
                    self._receive_region_signal(event.payload)
            self._fired_last_step.clear()
            now = self.time + dt
            active_until = getattr(self, "_active_until", None)
            if active_until is None:
                active_until = {}
                self._active_until = active_until
            active_ids = set(currents) | set(self.stimuli) | set(active_until)
            for neuron_id in active_ids:
                if neuron_id >= len(self.neurons):
                    continue
                neuron = self.neurons[neuron_id]
                if not neuron.alive:
                    continue
                current = currents.get(neuron.id, 0.0)
                stim = self.stimuli.get(neuron.id)
                if stim:
                    until, strength = stim
                    if now <= until:
                        current += strength
                    else:
                        self.stimuli.pop(neuron.id, None)
                if current and neuron.cell_type == "inhibitory":
                    current *= -0.8
                fired = neuron.advance(dt, current, self.config, self.np_rng, self.modulators.neuronal_gain(neuron.region), now)
                if fired:
                    self._on_spike(neuron, now)
            self._apply_lateral_inhibition(self._fired_last_step)
            for nid in self._fired_last_step:
                self._active_window.append(nid)
                active_until[nid] = now + float(getattr(self.config, "active_hold_ms", 120.0))
            if self._fired_last_step and len(active_until) > 256:
                # Bound the simulated set by time, not by a 3000-entry ring: a
                # cell that has been silent for longer than the hold window no
                # longer needs to be advanced every micro-step.
                cutoff = now - float(getattr(self.config, "active_hold_ms", 25.0))
                limit = int(getattr(self.config, "active_set_limit", 96))
                kept = {nid: until for nid, until in active_until.items() if until >= cutoff}
                if len(kept) > limit:
                    # Keep only the most recently fired cells; the rest cannot
                    # contribute within the short recall window anyway.
                    kept = dict(sorted(kept.items(), key=lambda item: -item[1])[:limit])
                self._active_until = kept
            self.time = now
            self._step_counter = getattr(self, "_step_counter", 0) + 1
            interval = max(1, int(getattr(self.config, "eligibility_decay_interval", 10)))
            if self._step_counter % interval == 0:
                factor = self.config.eligibility_decay_per_step ** interval
                for syn_id in tuple(self.active_eligibility):
                    if syn_id >= len(self.synapses):
                        self.active_eligibility.discard(syn_id)
                        continue
                    synapse = self.synapses[syn_id]
                    if not synapse.active:
                        self.active_eligibility.discard(syn_id)
                        continue
                    synapse.eligibility *= factor
                    if abs(synapse.eligibility) < 1e-6:
                        synapse.eligibility = 0.0
                        self.active_eligibility.discard(syn_id)
            self.modulators.tick(dt)
            interval = float(getattr(self.config, "neurochem_interval", 10.0))
            if self.time - self._last_neurochem >= interval:
                self.modulator_field.diffuse(min(interval, max(self.config.dt, interval)))
                self.modulators.tick(0.0, self.modulator_field.means())
                self._last_neurochem = self.time
            if self.time - self._last_chem >= self.config.chem_dt:
                self.morphogens.diffuse(self.config.chem_dt)
                self._last_chem = self.time
            if self.time - self._last_lifecycle >= self.config.lifecycle_interval:
                self.lifecycle()
                self._last_lifecycle = self.time
            if self.time - self._last_scale >= self.config.synapse_scale_interval:
                self._scale_synapses(self.config.synapse_scale_factor)
                self._last_scale = self.time

    def _on_spike(self, neuron: Neuron, time: float) -> None:
        self.stats["spikes"] += 1
        self._recent_spike_cells.append((time, neuron.id))
        region = self.regions[neuron.region]
        region.recent_spikes.append((time, neuron.id))
        # Local event synapses.
        for syn_id in self.outgoing.get(neuron.id, []):
            synapse = self.synapses[syn_id]
            if self.suppress_association and synapse.metadata.get("association"):
                continue
            if self.rehearsal_synapses is not None and synapse.metadata.get("association") and syn_id not in self.rehearsal_synapses:
                continue
            charge = synapse.pre_spike(time, self.config)
            limit = int(getattr(self.config, "eligibility_set_limit", 40_000))
            if syn_id in self.active_eligibility or len(self.active_eligibility) < limit:
                self.active_eligibility.add(syn_id)
            if charge is not None:
                self.events.schedule(
                    time + synapse.delay, "synapse",
                    post=synapse.post, current=charge,
                )
        # Postsynaptic STDP for recently presynaptic connections.
        for syn_id in self.incoming.get(neuron.id, []):
            self.synapses[syn_id].post_spike(time, self.config)
            self.active_eligibility.add(syn_id)
        # Regenerative tract bus: avoid emitting one event per neuron.
        self._tract_fire_count[neuron.region] += 1
        elapsed = time - self._tract_last_fire[neuron.region]
        if self._tract_fire_count[neuron.region] >= 3 and elapsed >= 6.0:
            self._tract_fire_count[neuron.region] = 0
            self._tract_last_fire[neuron.region] = time
            rate = region.mean_rate(20.0)
            if not self.suppress_tracts:
                self.tracts.route(neuron.region, min(1.0, rate / 20.0), self._schedule_tract)
        self._fired_last_step.append(neuron.id)

    def _schedule_tract(self, target: str, delay: float, strength: float, tract_name: str) -> None:
        self.events.schedule(self.time + delay, "region_signal", region=target, value=strength, tract=tract_name)

    def _receive_region_signal(self, payload: dict) -> None:
        region = self.regions[payload["region"]]
        candidates = [self.neurons[i] for i in region.neurons if self.neurons[i].alive and self.neurons[i].role != "glia"]
        if not candidates:
            return
        sample = candidates[:max(3, min(8, len(candidates) // 3))]
        value = float(payload["value"])
        for neuron in sample:
            self.pending_currents[neuron.id] += 6.0 * value + 1.0

    def _activate_population(self, ids, current: float, duration: float, start: float | None = None) -> None:
        start = self.time if start is None else start
        until = start + max(self.config.dt, duration)
        for neuron_id in ids:
            if 0 <= int(neuron_id) < len(self.neurons) and self.neurons[int(neuron_id)].alive:
                self.stimuli[int(neuron_id)] = (until, float(current))

    def run(self, duration: float) -> None:
        self.step(duration)

    # ------------------------------------------------------------------
    # Language, dialogue state, and no-lookup learned output
    # ------------------------------------------------------------------
    def _grow_binding_neuron(self, region: str, cell_type: str = "excitatory") -> Neuron:
        """Develop one new local engram cell when no unbound cell is available."""
        spec = REGION_BY_CODE[region]
        offset = self.np_rng.normal(0.0, spec.radius * 0.22, 3)
        neuron = Neuron(
            id=len(self.neurons),
            position=spec.center + offset,
            region=region,
            cell_type=cell_type,
            model=spec.model,
            role="neuron",
            g_leak=self.config.g_leak,
            energy=1.0,
        )
        self.neurons.append(neuron)
        self.regions[region].neurons.append(neuron.id)
        return neuron

    # ------------------------------------------------------------------
    # Learned context features (semantic geometry)
    # ------------------------------------------------------------------
    def _feature_cell(self, feature_id: int) -> int:
        """One shared neuron per context feature, so similar tokens overlap."""

        cell = self.feature_cell_ids.get(int(feature_id))
        if cell is not None:
            return cell
        neuron = self._grow_binding_neuron("Wern")
        neuron.word = f"<feat:{int(feature_id)}>"
        cell = neuron.id
        self.feature_cell_ids[int(feature_id)] = cell
        return cell

    def learn_context_features(self, tokens, gain: float = 1.0) -> int:
        """Local competitive update, driven by prediction error (surprise)."""

        if not bool(getattr(self.config, "context_feature_learning", True)):
            return 0
        clean = [str(token) for token in tokens if str(token).strip()]
        if len(clean) < 2:
            return 0
        self.context_features.observe(clean, gain=float(gain))
        attached = 0
        for token in dict.fromkeys(clean):
            binding = self.ensure_word(token)
            cells = tuple(self._feature_cell(fid) for fid in self.context_features.features(token))
            if not cells:
                continue
            merged = tuple(dict.fromkeys((*binding.sensory, *cells)))
            if merged != binding.sensory:
                binding.sensory = merged
                attached += 1
        return attached

    def _unique_binding_cells(self, region: str, count: int, token: str) -> tuple[int, ...]:
        """Recruit unused cells, developing new ones for a new sparse binding."""
        selected = [
            i for i in self.regions[region].neurons
            if self.neurons[i].alive and self.neurons[i].role != "glia" and self.neurons[i].word is None
        ]
        needed = int(count) - len(selected)
        for _ in range(max(0, needed)):
            selected.append(self._grow_binding_neuron(region).id)
        # Deterministic recruitment order makes checkpoints reproducible.
        selected = selected[:count]
        return tuple(selected)

    def _grounded_binding_cells(self, region: str, count: int, token: str) -> tuple[int, ...]:
        """Recruit and order cells through the learned dendritic token field."""

        if self.grounding is None:
            return self._unique_binding_cells(region, count, token)
        vector = self.grounding.encode(token)
        # New semantic fields recruit fresh receptors.  Related tokens may later
        # share context-feature cells, but identity receptors stay sparse; this
        # prevents a dense field from collapsing every word into every answer.
        candidates = [
            neuron.id for neuron in self.neurons
            if neuron.region == region and neuron.alive and neuron.role != "glia"
            and self.neurons[neuron.id].word is None
        ]
        anchors = self.grounding_cell_anchors.setdefault(region, {})
        needed = max(0, int(count) - len(candidates))
        for _ in range(needed):
            neuron = self._grow_binding_neuron(region)
            candidates.append(neuron.id)
        for cell_id in candidates:
            if cell_id not in anchors:
                anchors[cell_id] = self.grounding.receptor_vector(cell_id)
        ordered = self.grounding.rank_cells(vector, candidates, anchors)
        selected = ordered[:int(count)]
        self.grounding_cell_owners[token] = tuple(dict.fromkeys(selected))
        for cell_id in selected:
            self.grounding.learn_receptor(cell_id, vector)
        return tuple(selected)


    def ensure_word(self, token: str) -> WordBinding:
        original = str(token)
        token = original
        # ASCII bindings are case-insensitive, while episodic answer phrases
        # preserve the spelling used by the latest teaching example.
        if token.isascii():
            token = token.lower()
        self.word_display[token] = original
        if self.grounding is not None:
            self.grounding.observe(token)
        known_word = token in self.bindings
        if not known_word and self.grounding is not None and self.prototype_field is not None:
            self.prototype_field.observe(self.grounding.encode(token), owner=token)
        if token in self.bindings:
            return self.bindings[token]
        token_id = self.sdr.ensure_word(token)
        sensory_sdr = self.sdr.sensory_indices(token)
        motor_sdr = self.sdr.motor_indices(token)
        context_sdr = self.sdr.context_indices(token)
        sensory = self._grounded_binding_cells("Wern", self.config.sdr_k, token)
        if " " in token:
            # A multi-word phrase gets a compact motor plan in M1/Broca output.
            motor = self._unique_binding_cells("M1", 4, token)
        else:
            motor = self._unique_binding_cells("Broca", self.config.motor_sdr_k, token)
        context = self._unique_binding_cells("DLPFC", 4, token)
        binding = WordBinding(
            token, token_id, sensory_sdr, motor_sdr, sensory, motor,
            context_sdr, context,
        )
        for neuron_id in sensory:
            self.neurons[neuron_id].word = token
        for neuron_id in motor:
            self.neurons[neuron_id].word = token
        for neuron_id in context:
            self.neurons[neuron_id].word = token
        self.bindings[token] = binding
        for neuron_id in (*motor, *sensory):
            self.motor_bindings[neuron_id] = (*self.motor_bindings.get(neuron_id, ()), token)
        return binding

    def input_text(self, text: str, modality: str = "visual", *, learn: bool = False, run: bool = True) -> list[str]:
        """Present a text turn and maintain a conjunctive DLPFC context engram."""
        tokens = self._tokenize(text)
        source = {"visual": "V1", "auditory": "A1", "somatosensory": "S1"}.get(modality, "V1")
        gain = self.modulators.thalamic_gain(source)
        self.modulator_field.inject_region("Thal", "ACh", 0.020)
        self.modulator_field.inject_region(source, "NE", 0.012)
        self.modulator_field.inject_region("DLPFC", "ACh", 0.016)
        cue_tokens = self._core_dialogue_tokens(tokens)
        for token in cue_tokens:
            # Peripheral channel + shared semantic hub: the same meaning is
            # reached whether the word is read or heard.
            peripheral = self._peripheral_cells(token, modality)
            binding = self.ensure_word(token)
            self._activate_population(peripheral, self.config.input_current * gain, self.config.stimulus_duration)
            self._activate_population(binding.sensory, self.config.input_current * gain, self.config.stimulus_duration)
            self._update_working_memory(token)
        if len(cue_tokens) > 1:
            # The whole-question engram is a DLPFC conjunction, not an answer.
            phrase = " ".join(cue_tokens)
            phrase_binding = self.ensure_word(phrase)
            self._reinforce_context(phrase_binding.context, 2.0)
        if tokens:
            self.dialogue_state["turn_count"] += 1
            self.dialogue_state["topic"] = tuple(tokens[-8:])
            if run:
                self.step(max(self.config.decode_window, 40.0))
        return tokens

    def _update_working_memory(self, token: str) -> None:
        binding = self.ensure_word(token)
        self.dialogue_state["recent_words"].append(token)
        self._reinforce_context(binding.context, 1.0)

    # ------------------------------------------------------------------
    # Multimodal perception (reading / listening) and schema-bound answers
    # ------------------------------------------------------------------
    def _peripheral_cells(self, token: str, modality: str) -> tuple[int, ...]:
        """Grow, once, the modality-specific peripheral assembly for a token.

        Peripheral cortex keeps the channel's own pattern (written form vs
        spoken form) while converging on the modality-independent semantic hub
        in Wernicke. That hub is what makes cross-modal transfer possible.
        """

        region = MODALITY_REGIONS.get(modality, "V1")
        binding = self.ensure_word(token)
        existing = binding.channels.get(modality)
        if existing:
            return existing
        count = max(3, self.config.sdr_k // 2)
        cells = self._unique_binding_cells(region, count, token)
        binding.channels[modality] = tuple(cells)
        for neuron_id in cells:
            self.neurons[neuron_id].word = token
        for pre in cells:
            for post in binding.sensory:
                self._get_or_create_association(pre, post, 2.0)
        for pre in binding.sensory[:2]:
            for post in cells:
                self._get_or_create_association(pre, post, 2.0)
        return binding.channels[modality]

    def _present_token(self, token: str, modality: str, *, current: float | None = None) -> tuple[int, ...]:
        """Deliver one token through one sensory channel into the shared hub."""

        binding = self.ensure_word(token)
        region = MODALITY_REGIONS.get(modality, "V1")
        gain = self.modulators.thalamic_gain(region)
        peripheral = self._peripheral_cells(token, modality)
        base = self.config.input_current if current is None else float(current)
        self._activate_population(peripheral, base * gain, self.config.stimulus_duration)
        self._activate_population(binding.sensory, self.config.input_current * 0.8, self.config.stimulus_duration)
        self.modulator_field.inject_region(region, "NE", 0.010)
        self.modulator_field.inject_region("Thal", "ACh", 0.014)
        self.modulator_field.inject_region("DLPFC", "ACh", 0.012)
        return peripheral

    def perceive(
        self,
        text: str,
        *,
        modality: str = "visual",
        run: bool = True,
        remember: bool = True,
    ) -> list[str]:
        """Live through a sentence: read it or hear it, with no answer required.

        This is the multimodal input the QA corpus format never provided: the
        network experiences text/audio, updates its own statistics and schemas,
        and only later is asked questions about it.
        """

        tokens = self._tokenize(text)
        if not tokens:
            return []
        cue = self._core_dialogue_tokens(tokens)
        history: list[str] = []
        drive_total = 0.0
        drive_count = 0
        novelty_total = 0.0
        for token in cue:
            self._present_token(token, modality)
            self._update_working_memory(token)
            # Self-supervised: predict what this token would be, then let the
            # prediction error (surprise) set the plasticity gain for the
            # context -> token association. No labels are involved.
            surprise = self._sequence_surprise(history, token) if history else 0.0
            novelty = self._token_novelty(token)
            novelty_total += novelty
            drive = self.drive.signal(novelty=novelty, surprise=surprise)
            drive_total += drive
            drive_count += 1
            window = int(getattr(self.config, "stream_context_tokens", 3))
            self._reinforce_stream(history[-window:], token, gain=drive)
            history.append(token)
        if len(cue) > 1:
            phrase_binding = self.ensure_word(" ".join(cue))
            self._reinforce_context(phrase_binding.context, 2.0)
        # Surprise-gated representation change: only what the network could not
        # predict is allowed to reshape its features (measured before updating
        # the transition model).
        # Novelty gates *whether* the representation changes (familiar material
        # must not keep rewriting it); surprise scales how much.
        mean_novelty = novelty_total / max(1, drive_count)
        mean_drive = drive_total / max(1, drive_count)
        self.learn_context_features(cue, gain=max(mean_novelty, mean_drive))
        self._sequence_update(cue, gain=max(0.2, mean_drive))
        self._observe_cooccurrence(cue)
        if remember:
            self.dialogue_context.append(tuple(cue))
            self.relation_discoverer.observe(cue)
            frame = self.schemas.observe(cue)
            if self.discourse is not None and frame is not None:
                for role, filler in frame.roles.items():
                    self.discourse.bind_role(role, "".join(filler))
        self.dialogue_state["turn_count"] += 1
        self.dialogue_state["topic"] = tuple(cue[-8:])
        self.modulators.apply_reward(0.05, source="perception")
        if run:
            self.step(max(self.config.stimulus_duration, 20.0))
            self._normalize_inputs()
        return tokens

    def read(self, text: str, **kwargs) -> list[str]:
        """Read text through the visual channel."""

        return self.perceive(text, modality="visual", **kwargs)

    def listen(self, text: str, **kwargs) -> list[str]:
        """Hear text through the auditory channel."""

        return self.perceive(text, modality="auditory", **kwargs)

    def answer(
        self,
        question: str,
        *,
        modality: str = "visual",
        max_len: int = 8,
        use_schema: bool = True,
    ) -> list[str]:
        """Answer a question about the situation currently being perceived.

        The schema layer supplies a *filler binding* (who filled the queried
        role in the sentence just read/heard); the filler is then covertly
        rehearsed so the answer is produced by motor population decoding, not
        by string lookup.
        """

        question_tokens = self._tokenize(question)
        # Example tokens below were removed; runtime behavior is corpus-learned.
        # filler* to expect. The vocabulary is whatever the network read as that
        # slot - nothing is hard-coded.
        intent_fillers = self.schemas.expected_fillers(question_tokens)
        intent = self.schemas.query(question_tokens) if use_schema else None
        resolution = self.schemas.resolve(question_tokens, tuple(self.dialogue_context)) if use_schema else None
        binding_info: dict = {}
        if resolution is not None and resolution.resolved:
            bound = self._binding_readout(resolution.filler, question, modality=modality, max_len=max_len)
            if bound:
                self.last_response_info["schema"] = {
                    "role": resolution.role,
                    "filler": list(resolution.filler),
                    "reason": resolution.reason,
                }
                return bound
        # No statement about this subject: infer the slot from feature-similar
        # subjects (analogy), then read it out through the motor population.
        if use_schema and intent is not None and intent.kind == "statement":
            analogy = self._analogy_filler(intent.subject, intent.ask)
            if analogy:
                bound = self._binding_readout(analogy, question, modality=modality, max_len=max_len)
                if bound:
                    self.last_response_info["schema"] = {
                        "role": intent.ask,
                        "filler": list(analogy),
                        "reason": "analogy",
                    }
                    return bound
        # Association fallback: nothing declarative was stored about this
        # subject, so guess from the words it keeps company with (the network's
        # Example tokens below were removed; runtime behavior is corpus-learned.
        if use_schema:
            guess = self.schemas.guess(question_tokens)
            if guess:
                bound = self._binding_readout(guess, question, modality=modality, max_len=max_len)
                if bound:
                    self.last_response_info["schema"] = {
                        "role": "guess",
                        "filler": list(guess),
                        "reason": "association-guess",
                    }
                    return bound
        extra_prior: dict[str, float] = {}
        rehearse: tuple[str, ...] = ()
        for token in intent_fillers:
            extra_prior[token] = 1.0
        if resolution is not None and resolution.resolved:
            for token in resolution.filler:
                extra_prior[token] = 1.0
            rehearse = tuple(resolution.filler)
        output = self.respond(
            question,
            max_len=max_len,
            modality=modality,
            # The question itself is the cue: associations formed while reading
            # or listening already connect its tokens to the answer. Feeding the
            # whole dialogue back in would let every perceived token vote and
            # wash out the discriminating ones.
            context=(),
            extra_prior=extra_prior,
            rehearse_tokens=rehearse,
        )
        info = self.last_response_info
        if resolution is not None:
            info["schema"] = {
                "role": resolution.role,
                "filler": list(resolution.filler),
                "reason": resolution.reason,
            }
            if output and resolution.resolved:
                info["route"] = "schema-binding"
        return output

    def _binding_readout(
        self,
        filler_tokens: tuple[str, ...],
        prompt: str,
        *,
        modality: str = "visual",
        max_len: int = 8,
    ) -> list[str]:
        """Read out a working-memory role binding through motor spikes.

        Same principle as the hippocampal pointer path: the binding selects a
        motor assembly, that assembly must actually spike, and only then are its
        tokens reported. The filler came from the situation just perceived, not
        from a stored question→answer entry.
        """

        if not filler_tokens:
            return []
        start_time = self.time
        perf_start = wall_time.perf_counter()
        self.events.clear()
        self.stimuli.clear()
        self.suppress_association = False
        self.suppress_tracts = True
        self.input_text(prompt, run=False, modality=modality)
        region = MODALITY_REGIONS.get(modality, "V1")
        recall_window = float(getattr(self.config, "recall_window_ms", 20.0))
        context_cells = [nid for sentence in self.dialogue_context for token in sentence for nid in self.ensure_word(token).sensory]
        if context_cells:
            self._activate_population(context_cells, self.config.input_current * self.modulators.thalamic_gain(region), recall_window)
        motor: list[int] = []
        for token in filler_tokens:
            binding = self.ensure_word(token)
            self._activate_population(binding.motor, self.config.teach_current, recall_window)
            self._activate_population(binding.sensory, self.config.input_current * 0.5, recall_window)
            motor.extend(binding.motor)
        self.step(recall_window)
        spikes = self._spike_count(motor, start_time)
        if spikes <= 0:
            return []
        output = [token for token in filler_tokens if self._spike_count(self.ensure_word(token).motor, start_time) > 0]
        if not output:
            return []
        self.last_output = output[:max_len]
        self.last_response_info = {
            "route": "schema-binding",
            "memory_pointer": False,
            "neural_decode": True,
            "readout": "targeted-motor-assembly",
            "binding": list(filler_tokens),
            "motor_cells": motor,
            "spikes_used": spikes,
            "output": self.last_output,
            "prompt": prompt,
            "modality": modality,
            "timing_ms": {"total": (wall_time.perf_counter() - perf_start) * 1000.0},
        }
        return self.last_output

    def _token_novelty(self, token: str) -> float:
        binding = self.ensure_word(token)
        return float(self.hippocampus.novelty(self._event_sdr(binding.sensory)))

    def _analogy_filler(self, subject_tokens, ask: str) -> tuple[str, ...]:
        """Infer a slot by learned feature similarity, then a learned relation.

        The analogue is found by shared context features.  Among statements for
        that analogue, relation strength learned by ``RelationDiscoverer``
        chooses which local association reaches the readout.
        """

        features = getattr(self, "context_features", None)
        if features is None:
            return ()
        candidates: list[tuple[float, str]] = []
        for token in subject_tokens:
            for neighbour, score in features.neighbours(token, top_k=6):
                candidates.append((float(score), neighbour))
        for _, neighbour in sorted(candidates, reverse=True):
            # Example tokens below were removed; runtime behavior is corpus-learned.
            # Example tokens below were removed; runtime behavior is corpus-learned.
            pool = self.schemas.by_subject.get(neighbour)
            if not pool:
                pool = [st for st in self.schemas.statements if neighbour in st.subject]
            if pool:
                best = max(
                    pool,
                    key=lambda statement: float(self.relation_discoverer.score(statement.relation)),
                )
                return tuple(best.value)
        return ()

    # ------------------------------------------------------------------
    # Action loop: decide -> retrieve -> check consistency -> credit
    # ------------------------------------------------------------------
    def reason(self, question: str, *, max_len: int = 8) -> list[str]:
        """Answer through an explicit action loop rather than a bare recall.

        1. the question is parsed into an *intent* (which slot is asked for);
        2. the network retrieves from its own memory for that slot;
        3. it checks whether what came back matches the expected kind of filler
           (the prediction error of its own retrieval);
        4. the error drives credit assignment: a consistent retrieval is
           reinforced with dopamine, an inconsistent one is handed to the local
           correction path.
        """

        question_tokens = self._tokenize(question)
        intent = self.schemas.query(question_tokens)
        if self.intent_model is not None and intent is not None:
            learned_intent, learned_confidence = self.intent_model.predict(question_tokens)
            if learned_intent:
                # A learned opaque intent, not a surface template, gates which
                # retrieval action is selected.  The schema role remains the
                # neural slot label until the old checkpoint retires.
                intent.ask = learned_intent
        expected = self.schemas.expected_fillers(question_tokens)
        retrieved = self.schemas.lookup(intent) if intent is not None else ()
        matched = bool(set(retrieved) & expected) if expected else bool(retrieved)
        error = 0.0 if matched else 1.0

        if retrieved:
            # Prime the retrieved slot so the motor population can decode it.
            self._activate_population(
                [nid for token in retrieved for nid in self.ensure_word(token).motor], 6.0, 12.0
            )
        output = self.answer(question, max_len=max_len)
        info = self.last_response_info
        info["action"] = {
            "intent": intent.ask if intent is not None else None,
            "retrieved": list(retrieved),
            "expected_type_known": bool(expected),
            "consistency_error": error,
        }
        if self.intent_model is not None and intent is not None:
            self.intent_model.observe(question_tokens, intent.ask)

        if retrieved and matched:
            self.reward(0.4)                      # RPE: the retrieval was consistent
        elif retrieved:
            self.correct(question, " ".join(retrieved), reward=1.0)   # self-correction
            self.reward(0.8)
        return output

    def self_practice(self, *, rounds: int = 8, max_len: int = 6) -> dict:
        """Generate questions from the network's own knowledge and answer them.

        This is the smallest self-driven practice loop: the network takes a
        statement it read, turns it into a question, answers with ``reason``,
        and keeps the result only if it is consistent with what it read.
        """

        statements = list(self.schemas.statements)
        if not statements:
            return {"attempted": 0, "consistent": 0, "asked": []}
        sample = statements[-max(1, rounds * 3):]
        self.rng.shuffle(sample)
        attempted = consistent = 0
        asked: list[dict] = []
        for statement in sample[: max(1, rounds)]:
            subject = " ".join(statement.subject)
            reference = " ".join(statement.value)
            if not subject or not reference or not statement.relation:
                continue
            # The learned relation itself is the probe: subject + relation asks
            # for its bound value without inventing a language-specific question.
            question = f"{subject} {statement.relation}"
            got = " ".join(self.reason(question, max_len=max_len))
            hit = "".join(got.split()) == "".join(reference.split())
            attempted += 1
            consistent += int(hit)
            if not hit:
                self.correct(question, reference, reward=1.0)
            asked.append({"question": question, "expected": reference, "got": got, "consistent": hit})
        return {"attempted": attempted, "consistent": consistent, "asked": asked}

    def _reinforce_stream(self, context_tokens: list[str], token: str, *, gain: float) -> int:
        """Surprise-gated local Hebbian: context -> the token actually observed."""

        if not context_tokens:
            return 0
        target = self.ensure_word(token)
        sources = [nid for item in context_tokens for nid in self.ensure_word(item).sensory]
        if not sources:
            return 0
        strength = max(0.05, min(1.0, float(gain)))
        weight_target = self.config.learned_weight * (0.6 + 0.4 * strength)
        modulated = 0
        for pre in dict.fromkeys(sources):
            for post in target.motor:
                syn_id = self._get_or_create_association(pre, post, 2.0)
                if syn_id < 0:
                    continue
                synapse = self.synapses[syn_id]
                synapse.eligibility = min(2.0, synapse.eligibility + 0.25 * strength)
                synapse.apply_da(self.modulators.concentrations["DA"], 0.8, self.config)
                synapse.weight = max(synapse.weight, weight_target)
                modulated += 1
        if modulated:
            self.modulators.apply_reward(0.10 + 0.30 * strength, source="stream")
            self.modulator_field.inject_region("SN", "DA", 0.05 * strength)
        return modulated

    def observe_stream(
        self,
        text: str,
        *,
        modality: str = "visual",
        learn: bool = True,
        predict: bool = True,
        remember: bool = True,
    ) -> dict:
        """Learn from a text/audio stream with no labels (reading, listening).

        For every token the network first *predicts* what comes next, then the
        real token arrives, and the prediction error (surprise) drives intrinsic
        motivation and the local plasticity gain. Predictable spans barely
        change weights; surprising spans do. This is the self-supervised path
        that QA pairs cannot provide.
        """

        tokens = self._tokenize(text)
        stats = {
            "tokens": 0, "predictions": 0, "hits": 0, "modulated": 0,
            "mean_surprise": 0.0, "mean_curiosity": 0.0, "sentences": 0,
            "prediction_accuracy": 0.0,
        }
        if not tokens:
            return stats

        history: list[str] = []
        pending: list[str] = []
        sentence_units: list[tuple[str, ...]] = []
        sentences: list[tuple[str, ...]] = []
        surprise_total = 0.0
        curiosity_total = 0.0
        for token in tokens:
            context = self._sequence_context(history)
            surprise = self._sequence_surprise(history, token) if history else 0.0
            novelty = self._token_novelty(token)
            drive = self.drive.signal(novelty=novelty, surprise=surprise)
            if predict and context:
                neural_top = self.sequence_model.predict_next(context) if self.sequence_model else ("", 0.0)
                ranked = [(neural_top[0], neural_top[1])] if neural_top[0] else []
                stats["predictions"] += 1
                if ranked and ranked[0][0] == token:
                    stats["hits"] += 1
            if learn:
                window = int(getattr(self.config, "stream_context_tokens", 3))
                stats["modulated"] += self._reinforce_stream(history[-window:], token, gain=drive)
            history.append(token)
            history = history[-64:]
            pending.append(token)
            if self._is_sentence_boundary(token):
                sentence_units.append(tuple(pending))
                pending = []
                stats["sentences"] += 1
            surprise_total += surprise
            curiosity_total += drive
            stats["tokens"] += 1
        if pending:
            sentence_units.append(tuple(pending))
            stats["sentences"] += 1
        # Update the transition model once for the whole document: chunking at
        # Example tokens below were removed; runtime behavior is corpus-learned.
        # which is exactly what continuation needs.
        self.learn_context_features(tokens, gain=curiosity_total / max(1, stats["tokens"]))
        self._sequence_update(tokens, gain=max(0.2, curiosity_total / max(1, stats["tokens"])))
        self._observe_cooccurrence(tokens)
        if remember:
            for sentence in sentence_units:
                cue = self._core_dialogue_tokens(list(sentence))
                if cue:
                    self.dialogue_context.append(tuple(cue))
                    self.relation_discoverer.observe(cue)
                    frame = self.schemas.observe(cue)
                    if self.discourse is not None and frame is not None:
                        for role, filler in frame.roles.items():
                            self.discourse.bind_role(role, "".join(filler))
                    sentences.append(tuple(cue))
                    self.learn_context_features(cue)
                    self.relation_sentences.append(tuple(cue))
        if sentences and self.schemas.use_learned_relations and self.schemas.reparse_needed:
            # Re-read: the discoverer only becomes reliable after a few
            # sentences, so earlier statements are re-parsed with what it
            # learned (the same way a human re-reads a page).
            self.schemas.reparse(list(self.relation_sentences))
        count = max(1, stats["tokens"])
        stats["mean_surprise"] = round(surprise_total / count, 4)
        stats["mean_curiosity"] = round(curiosity_total / count, 4)
        stats["prediction_accuracy"] = round(stats["hits"] / max(1, stats["predictions"]), 3)
        self._normalize_inputs()
        return stats

    def _reinforce_context(self, neuron_ids, strength: float = 1.0) -> None:
        decay = 0.88
        for neuron_id in list(self.working_trace):
            self.working_trace[neuron_id] *= decay
            if self.working_trace[neuron_id] < 0.02:
                del self.working_trace[neuron_id]
        for neuron_id in neuron_ids:
            self.working_trace[neuron_id] = min(4.0, self.working_trace.get(neuron_id, 0.0) + strength)
        self.active_context = tuple(sorted(self.working_trace, key=self.working_trace.get, reverse=True)[:max(self.config.context_k, 8)])
        self._activate_population(self.active_context, 7.0, 20.0)

    def learn_pair(self, question: str, answer: str, reward: float = 1.0, *, simulate: bool = False) -> list[str]:
        """Teach a conversational turn through local association.

        The answer is used only to deliver teaching currents during training.
        Inference reads supported Broca population activity; it does not query
        this corpus or select an answer from a table.
        """
        answer_tokens = self._tokenize(answer)
        if not answer_tokens:
            return []
        phrase = " ".join(answer_tokens)
        phrase_binding = self.ensure_word(phrase)
        question_tokens = self._core_dialogue_tokens(self._tokenize(question))
        self.working_trace.clear()
        self.active_context = ()
        for token in question_tokens:
            self._update_working_memory(token)
        if len(question_tokens) > 1:
            question_binding = self.ensure_word(" ".join(question_tokens))
            self._reinforce_context(question_binding.context, 2.0)
        presented_wern = [nid for token in question_tokens for nid in self.ensure_word(token).sensory]
        source_population = list(dict.fromkeys((*presented_wern, *self.active_context)))
        target_population = list(phrase_binding.motor)
        newly_wired: list[int] = []
        for pre in source_population:
            for post in target_population:
                syn_id = self._get_or_create_association(pre, post, 2.0)
                if syn_id >= 0:
                    newly_wired.append(syn_id)
        if simulate:
            self.input_text(question, learn=True)
            self._activate_population(target_population, self.config.teach_current, 20.0)
            self.step(38.0)
        self.modulators.apply_reward(reward, source="teaching")
        self.modulator_field.inject_region("SN", "DA", 0.12 * reward)
        self.modulator_field.inject_region("Hipp", "ACh", 0.035)
        for syn_id in dict.fromkeys(newly_wired):
            synapse = self.synapses[syn_id]
            synapse.apply_da(self.modulators.concentrations["DA"], 0.8, self.config)
            synapse.weight = max(synapse.weight, self.config.learned_weight)
            self.recent_synapses.add(syn_id)
        self.last_learning_synapses = tuple(dict.fromkeys(newly_wired))
        self.stats["train_examples"] += 1
        # Learn the turn as a token chain (generation) and let repeatedly
        # co-activated tokens share sensory cells (generalization).
        turn_tokens = [*question_tokens, *answer_tokens]
        self._sequence_update(turn_tokens, gain=max(0.2, reward))
        shared = self._observe_cooccurrence(turn_tokens)
        self.stats["shared_cells"] = self.stats.get("shared_cells", 0) + shared
        # Fast association still needs a monotonically newer episodic timestamp.
        self.time += 0.001
        self._write_episodic_event(question, answer_tokens, reward, content_neurons=presented_wern)
        self.last_output = answer_tokens
        self._normalize_inputs()
        return answer_tokens

    def _association_synapses_to(self, posts, sources) -> tuple[int, ...]:
        """Return association synapses from active cue cells to motor plan cells."""
        sources = set(sources)
        result: set[int] = set()
        for post in posts:
            for syn_id in self.incoming.get(post, []):
                if self.synapses[syn_id].pre in sources:
                    result.add(syn_id)
        return tuple(sorted(result))

    def respond(
        self,
        prompt: str,
        max_len: int = 6,
        *,
        modality: str = "visual",
        context: tuple[tuple[str, ...], ...] = (),
        extra_prior: dict[str, float] | None = None,
        rehearse_tokens: tuple[str, ...] = (),
        use_sequence_prior: bool = True,
    ) -> list[str]:
        """Respond through neural motor populations.

        Hippocampus may provide a cue->motor-plan pointer, but it never supplies
        the returned text directly. The selected motor cells must spike and be
        population-decoded.

        ``modality`` picks the sensory channel used to present the prompt,
        ``context`` adds sentences still held in working memory (so a question
        can be about what was just read or heard), ``extra_prior`` carries a
        binding/expectation prior, and ``rehearse_tokens`` covertly rehearses a
        bound filler so the motor population can decode it.
        """
        start_time = self.time
        perf_start = wall_time.perf_counter()
        self.working_trace.clear()
        self.active_context = ()
        # Clear *before* presenting. input_text() installs the cue stimulus, so
        # clearing afterwards silently dropped the whole prompt and left the
        # hippocampal pointer as the only thing that could ever drive a reply.
        self.events.clear()
        self.stimuli.clear()
        # Recall keeps the learned cortex path online: the cue population must be
        # able to drive Broca/M1 through its association synapses. Suppressing
        # associations here made every answer depend on the episodic pointer and
        # left the network with no path that could generalize.
        self.suppress_association = False
        self.suppress_tracts = True
        tokens = self.input_text(prompt, run=False, modality=modality)
        cue_tokens = self._core_dialogue_tokens(tokens)
        cue_neurons = [nid for token in cue_tokens for nid in self.ensure_word(token).sensory]
        if context:
            context_tokens = [item for sentence in context for item in sentence]
            context_cells = [nid for token in context_tokens for nid in self.ensure_word(token).sensory]
            cue_neurons = list(dict.fromkeys([*cue_neurons, *context_cells]))
        recall_window = float(getattr(self.config, "recall_window_ms", 20.0))
        # Only the working-memory window stays active during the recall window.
        # Simulating every token of a long prompt cost ~60 s per generated
        # token; earlier tokens still reach the semantic hub during
        # presentation, they are simply not re-simulated.
        cue_window = int(getattr(self.config, "recall_cue_tokens", 8))
        decode_cue = cue_tokens[-cue_window:] if cue_window > 0 else list(cue_tokens)
        recent_cells = [nid for token in decode_cue for nid in self.ensure_word(token).sensory]
        for neuron_id in set(cue_neurons) - set(recent_cells):
            self.stimuli.pop(neuron_id, None)
        # Hold the cue (and its DLPFC context engram) for the whole recall window.
        self._activate_population(
            recent_cells,
            self.config.input_current * self.modulators.thalamic_gain(MODALITY_REGIONS.get(modality, "V1")),
            max(recall_window, self.config.stimulus_duration),
        )
        if self.active_context:
            self._activate_population(self.active_context, 7.0, max(recall_window, 20.0))
        cue = self._event_sdr((*cue_neurons, *self.active_context))
        retrieve_started = wall_time.perf_counter()
        # CA3-style pattern completion: the best-matching stored cue biases the
        # answer, which is what disambiguates "which lesson does this context
        # belong to" when several taught answers share cue tokens.
        recalled = self.hippocampus.retrieve(cue, top_k=3)
        retrieve_ms = (wall_time.perf_counter() - retrieve_started) * 1000.0
        top = recalled[0] if recalled else None
        similarity = self.hippocampus.similarity(cue, top.sdr_index) if top else 0.0
        hit_threshold = float(getattr(self.config, "memory_hit_threshold", 0.55))
        memory_hit = bool(top and top.motor_neurons and similarity >= hit_threshold)
        self.recalled_motor = top.motor_neurons if memory_hit else ()
        self.recalled_answer = []
        plastic_sources = list(dict.fromkeys((*cue_neurons, *self.active_context)))

        if memory_hit:
            # The episodic pointer biases the recalled plan; it is no longer the
            # only thing that can drive it.
            gain = float(getattr(self.config, "pointer_bias_gain", 0.6))
            self._activate_population(self.recalled_motor, self.config.teach_current * gain, 20.0)
        if rehearse_tokens:
            # Covert rehearsal of a bound filler: prefrontal binding keeps the
            # slot active long enough for the motor population to decode it.
            self._primed_tokens = {str(token) for token in rehearse_tokens}
            for token in rehearse_tokens:
                filler = self.ensure_word(token)
                self._activate_population(filler.motor, self.config.teach_current, 20.0)
                self._activate_population(filler.sensory, self.config.teach_current * 0.5, 15.0)
        else:
            self._primed_tokens = set()
        # A recall window always runs, so motor populations can be decoded from
        # real spikes with or without a hippocampal hit.
        step_started = wall_time.perf_counter()
        self.step(recall_window)
        step_ms = (wall_time.perf_counter() - step_started) * 1000.0
        prior = self._sequence_prior(decode_cue) if use_sequence_prior else {}
        if extra_prior:
            for key, value in extra_prior.items():
                prior[key] = max(prior.get(key, 0.0), float(value))
        if top is not None and top.answer_phrase and similarity >= hit_threshold * 0.6:
            for item in self._tokenize(top.answer_phrase):
                prior[item] = max(prior.get(item, 0.0), similarity)
        candidate_token = self.ensure_word(top.answer_phrase).token if memory_hit else None
        candidate_binding = self.bindings.get(candidate_token) if candidate_token else None
        decode_started = wall_time.perf_counter()
        candidates = self._decode_motor_candidates(
            start_time, None, prior=prior, cue_tokens=decode_cue, cue_cells=tuple(cue_neurons)
        )
        pointer_spikes = self._spike_count(self.recalled_motor, start_time) if memory_hit else 0
        strong_pointer_threshold = float(getattr(self.config, "memory_hit_threshold", 0.55)) + 0.25

        output: list[str] = []
        # A very-high-similarity hippocampal pointer is read out from its own
        # targeted motor assembly. Broad WTA can become state-dependent after
        # long training, so it must not erase an exact episodic memory.
        if memory_hit and pointer_spikes > 0 and similarity >= strong_pointer_threshold and candidate_binding:
            display = self.word_display.get(candidate_binding.token, candidate_binding.token)
            output = self._tokenize(display)[:max(1, max_len)]
            plastic = self._association_synapses_to(candidate_binding.motor, plastic_sources)
            decode_ms = (wall_time.perf_counter() - decode_started) * 1000.0
            self.recalled_motor = ()
            self.recalled_answer = []
            self.consistency_memory.observe(self._event_sdr(cue_neurons), 1.0)
            self.last_response_plastic_synapses = plastic
            self.last_output = output
            self.suppress_association = False
            self.suppress_tracts = False
            self.last_response_info = {
                "route": "hippocampus-motor",
                "memory_pointer": True,
                "neural_decode": True,
                "readout": "targeted-motor-assembly",
                "similarity": float(similarity),
                "confidence": float(similarity),
                "motor_cells": list(candidate_binding.motor),
                "spikes_used": pointer_spikes,
                "output": output,
                "prompt": prompt,
                "timing_ms": {"retrieve": retrieve_ms, "motor_step": step_ms, "decode": decode_ms, "total": (wall_time.perf_counter()-perf_start)*1000},
                "association_synapses": plastic,
            }
            self.stats["predictions"] += 1
            return output
        winner = candidates[0] if candidates else None
        winner_binding = self.bindings.get(winner[2]) if winner else None
        winner_support = max(0.0, winner[3]) if winner else 0.0
        winner_coverage = float(winner[4]) if winner else 0.0
        runner_support = max(0.0, candidates[1][3]) if len(candidates) > 1 else 0.0
        # Competition is pairwise against the runner-up: with rich association
        # structure many plans carry some support, and requiring a majority of
        # *all* support rejected correct winners.
        ratio = winner_support / max(winner_support + runner_support, 1e-9)
        support_per_cell = winner_support / max(1, len(winner_binding.motor)) if winner_binding else 0.0
        confidence = 0.0 if not winner else ratio * (1.0 - math.exp(-support_per_cell / 4.0))
        # Recall uses its own gates: a partial cue produces less support than an
        # exact one, and requiring exact-cue confidence here was rejecting every
        # answer that was not already in episodic memory.
        thresholds = self.decoder_calibration.thresholds(
            confidence=float(getattr(self.config, "recall_confidence_threshold", 0.45)),
            margin=float(getattr(self.config, "recall_margin_threshold", 1.25)),
            support_per_cell=float(getattr(self.config, "recall_support_per_cell", 0.5)),
            coverage=float(getattr(self.config, "recall_coverage_minimum", 0.25)),
        ) if bool(getattr(self.config, "use_decoder_calibration", False)) else {
            "confidence": float(getattr(self.config, "recall_confidence_threshold", 0.45)),
            "margin": float(getattr(self.config, "recall_margin_threshold", 1.25)),
            "support_per_cell": float(getattr(self.config, "recall_support_per_cell", 0.5)),
            "coverage": float(getattr(self.config, "recall_coverage_minimum", 0.25)),
        }
        confidence_threshold = float(thresholds["confidence"])
        margin_threshold = float(thresholds["margin"])
        support_threshold = float(thresholds["support_per_cell"])
        coverage_minimum = float(thresholds["coverage"])
        cue_pattern = self._event_sdr(cue_neurons)
        local_consistency = self.consistency_memory.estimate(cue_pattern)
        # Learned consistency is a soft prior, not a gate by itself.  After the
        # brain has observed enough cue/outcome pairs, locally reliable contexts
        # gain confidence and locally hallucination-prone contexts lose some.
        if len(self.consistency_memory.patterns) >= 8:
            confidence *= 0.75 + 0.50 * local_consistency
        accepted = bool(
            winner and winner_support > 0.0
            and (runner_support <= 0.0 or winner_support / max(runner_support, 1e-9) >= margin_threshold)
            and support_per_cell >= support_threshold
            and winner_coverage >= coverage_minimum
        )
        if accepted:
            self.consistency_memory.observe(cue_pattern, 1.0)
            self.decoder_calibration.observe(
                support=winner_support / max(1, len(winner_binding.motor)),
                margin=winner_support / max(runner_support, 1e-9),
                coverage=winner_coverage,
                accepted=True,
            )
            token = winner[2]
            display = self.word_display.get(token, token)
            if " " in display:
                output.extend(self._tokenize(display)[:max_len])
            else:
                output.append(display)
            posts = winner_binding.motor
            active_source = (
                set(self._recently_active_neurons("Wern", 180.0))
                | set(self._recently_active_neurons("Broca", 180.0))
                | set(self._recently_active_neurons("M1", 180.0))
                | set(self.active_context)
            )
            plastic = self._association_synapses_to(posts, active_source)
        else:
            plastic = ()
            self.consistency_memory.observe(cue_pattern, 0.0)

        decode_ms = (wall_time.perf_counter() - decode_started) * 1000.0
        self.recalled_motor = ()
        self.recalled_answer = []
        self.last_response_plastic_synapses = plastic
        self.last_output = output
        self.suppress_association = False
        self.suppress_tracts = False
        self.last_response_info = {
            "route": "hippocampus-motor" if memory_hit and output else ("population" if output else "rejected"),
            "memory_pointer": memory_hit,
            "neural_decode": bool(output),
            "similarity": float(similarity),
            "confidence": float(confidence),
            "coverage": float(winner_coverage),
            "winner_support": float(winner_support),
            "winner_ratio": float(ratio),
            "local_consistency": round(float(local_consistency), 4),
            "support_per_cell": float(support_per_cell),
            "motor_cells": list(winner_binding.motor) if winner_binding else [],
            "spikes_used": self._spike_count(winner_binding.motor, start_time) if winner_binding else 0,
            "candidates": [
                {"token": token, "support": float(support), "coverage": float(coverage)}
                for _, _, token, support, coverage in candidates[:6]
            ],
            "output": output,
            "prompt": prompt,
            "timing_ms": {"retrieve": retrieve_ms, "motor_step": step_ms, "decode": decode_ms, "total": (wall_time.perf_counter()-perf_start)*1000},
            "association_synapses": plastic,
        }
        self.stats["predictions"] += 1
        return output

    def language_answer_tokens(self, answer: str) -> list[str]:
        """Tokenize an expected answer for evaluation only."""
        return self._tokenize(answer)

    # ------------------------------------------------------------------
    # Sequence generation and Hebbian generalization
    # ------------------------------------------------------------------
    def _sequence_prior(self, context_tokens: list[str]) -> dict[str, float]:
        """Learned next-token expectation, used to bias motor competition.

        This is the network's own transition statistics (the same model that
        ``predict_language`` exposes), normalised to 0..1. It only re-ranks
        plans that already spiked; it cannot invent a plan with no support.
        """

        if not context_tokens or self.sequence_model is None:
            return {}
        distribution = self.sequence_model.distribution(self._sequence_context(context_tokens))
        positive = {token: score for token, score in distribution.items() if score > 0.0}
        best = max(positive.values(), default=0.0)
        return {token: score / best for token, score in positive.items()} if best else {}

    def learn_sequence(self, text: str) -> dict:
        """Observe a token chain so the network can continue it during generation."""

        tokens = self._tokenize(text)
        if not tokens:
            return {"tokens": 0, "updates": self.sequence_model.updates}
        error = self._sequence_update(tokens)
        shared = self._observe_cooccurrence(tokens)
        return {
            "tokens": len(tokens),
            "prediction_error": error,
            "shared_cells": shared,
        }

    def respond_sequence(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        max_len_per_step: int = 8,
        stop_tokens: tuple[str, ...] | None = None,
    ) -> list[str]:
        """Generate a longer utterance by feeding decoded tokens back as context.

        Each step is a normal neural recall pass: the previously emitted tokens
        become part of the cue, motor populations must spike, and the learned
        expectation primes the plan it predicts next.

        The network decides when to stop; ``max_tokens`` is only a runaway
        safety cap. Automatic stopping happens when:

        * the expectation or the readout produces a learned end marker
          (``</s>``/``<eos>``) - the model's own "sentence finished" signal;
        * no motor plan is supported any more (rejected readout);
        * the same plan repeats (degenerate loop);
        * the expectation stays below ``generation_min_probability`` for
          ``generation_low_confidence_steps`` steps (it has nothing to say);
        * fatigue crosses ``generation_fatigue_limit`` (satiation).

        ``self.last_generation_info`` records why it stopped.
        """

        stop = set(stop_tokens) if stop_tokens is not None else set(
            getattr(self.config, "generation_end_tokens", ("</s>", "<eos>"))
        )
        safety = int(max_tokens) if max_tokens else int(getattr(self.config, "generation_safety_cap", 256))
        min_probability = float(getattr(self.config, "generation_min_probability", 0.12))
        low_conf_limit = max(1, int(getattr(self.config, "generation_low_confidence_steps", 3)))
        fatigue_limit = float(getattr(self.config, "generation_fatigue_limit", 0.85))
        produced: list[str] = []
        context = prompt
        last_chunk: list[str] = []
        low_conf_streak = 0
        probabilities: list[float] = []
        stop_reason = "safety-cap"
        steps = 0
        for _ in range(max(1, safety)):
            # Ordered expectation, not bag-of-context: the learned n-gram model
            # covertly primes the plan it expects next, exactly like prefrontal
            # expectation priming a word before you say it. The output still has
            # to come from motor population spikes.
            context_tokens = self._tokenize(context)
            extra_prior: dict[str, float] = {}
            rehearse: tuple[str, ...] = ()
            expectation = None
            if context_tokens:
                expectation = self.sequence_model.predict_next(self._sequence_context(context_tokens))
            if expectation:
                expected_token, expected_probability = expectation
                probabilities.append(float(expected_probability))
                if expected_token in stop:
                    stop_reason = "model-end-marker"
                    break
                extra_prior[expected_token] = 1.0
                rehearse = (expected_token,)
            tokens = self.respond(
                context,
                max_len=max(1, int(max_len_per_step)),
                extra_prior=extra_prior,
                rehearse_tokens=rehearse,
                # Generation is driven by the ordered expectation, not by the
                # bag-of-context transition prior (which prefers frequent short
                # contexts and derailed continuation in testing).
                use_sequence_prior=False,
            )
            if not tokens:
                stop_reason = "no-supported-plan"
                break
            steps += 1
            if tokens == last_chunk:
                # Degenerate loop: the same plan keeps winning, so the utterance
                # is finished rather than repeated forever.
                stop_reason = "repetition"
                break
            last_chunk = list(tokens)
            hit_stop = any(token in stop for token in tokens)
            produced.extend(token for token in tokens if token not in stop)
            if hit_stop:
                stop_reason = "model-end-marker"
                break
            if expectation and float(expectation[1]) < min_probability:
                low_conf_streak += 1
            else:
                low_conf_streak = 0
            if low_conf_streak >= low_conf_limit:
                stop_reason = "expectation-collapsed"
                break
            if float(self.modulators.states.get("fatigue", 0.0)) >= fatigue_limit:
                stop_reason = "fatigue"
                break
            context = " ".join([prompt, *produced])
        self.last_generation_info = {
            "tokens": len(produced),
            "steps": steps,
            "stop_reason": stop_reason,
            "safety_cap": safety,
            "mean_expectation_probability": round(sum(probabilities) / len(probabilities), 4) if probabilities else 0.0,
            "fatigue": round(float(self.modulators.states.get("fatigue", 0.0)), 4),
        }
        if self.sequence_model is not None and context_tokens:
            self.last_generation_info["neural_uncertainty"] = round(
                self.sequence_model.uncertainty(context_tokens), 4
            )
        return produced

    def _observe_cooccurrence(self, tokens, window: int = 12) -> int:
        """Hebbian association: tokens that repeatedly fire together share cells.

        Identical tokens already share an assembly by construction; this extends
        the overlap to tokens that are *used* together, which is what allows a
        paraphrase with no shared surface form to reach a learned answer.
        """

        budget = int(getattr(self.config, "sdr_share_budget", 4))
        minimum = float(getattr(self.config, "cooccurrence_min", 2.0))
        share = int(getattr(self.config, "sdr_share_cells", 1))
        unique = [token for token in dict.fromkeys(str(item) for item in tokens)][:window]
        for index, left in enumerate(unique):
            for right in unique[index + 1:]:
                key = (left, right) if left <= right else (right, left)
                self.cooccurrence[key] = self.cooccurrence.get(key, 0) + 1
        if share <= 0 or budget <= 0:
            return 0
        created = 0
        for (left, right), count in list(self.cooccurrence.items()):
            if count < minimum:
                continue
            if self.shared_cells.get(left, 0) < budget and self._share_sensory_cells(right, left, share):
                created += 1
            if self.shared_cells.get(right, 0) < budget and self._share_sensory_cells(left, right, share):
                created += 1
        return created

    def _share_sensory_cells(self, source: str, target: str, count: int) -> bool:
        """Copy a few sensory cells from one token assembly into another."""

        donor = self.bindings.get(source)
        receiver = self.bindings.get(target)
        if donor is None or receiver is None or source == target:
            return False
        borrowed = [cell for cell in donor.sensory if cell not in receiver.sensory][: max(1, int(count))]
        if not borrowed:
            return False
        keep = max(len(receiver.sensory), len(borrowed))
        merged = tuple(dict.fromkeys((*borrowed, *receiver.sensory)))[:keep]
        if merged == receiver.sensory:
            return False
        receiver.sensory = merged
        self.shared_cells[target] = self.shared_cells.get(target, 0) + len(borrowed)
        return True

    def _decode_motor_candidates(
        self,
        start_time: float,
        candidate_tokens: set[str] | None = None,
        prior: dict[str, float] | None = None,
        cue_tokens: list[str] | None = None,
        cue_cells: tuple[int, ...] = (),
    ) -> list[tuple[float, int, str, float]]:
        """Decode spiked motor populations, optionally only a pointer-selected set.

        ``prior`` is the learned next-token expectation from the sequence model.
        It biases competition between motor plans that actually spiked; it can
        never create support for a plan that no synapse and no spike supports.
        """
        cell_limit = int(getattr(self.config, "active_source_cells", 40))
        # The cue cells are evidence by construction; the rank-limited "recently
        # active" set alone made support depend on which cue cell happened to
        # spike most (the correct continuation's presynaptic cells often missed
        # the top-40 and the answer was rejected).
        active_source = (
            set(cue_cells)
            | set(self._recently_active_neurons("Wern", 180.0, cell_limit))
            | set(self._recently_active_neurons("Broca", 180.0, cell_limit))
            | set(self.active_context)
        )
        prior_weight = float(getattr(self.config, "sequence_prior_weight", 3.0))
        cue_words = {str(token) for token in (cue_tokens or ())}
        # Which cue tokens own each shared context feature cell (used to give a
        # feature-mediated match credit for covering those tokens).
        feature_owner_map: dict[int, set[str]] = {}
        features = getattr(self, "context_features", None)
        if features is not None:
            for word in cue_words:
                for feature_id in features.features(word):
                    cell = self.feature_cell_ids.get(int(feature_id))
                    if cell is not None:
                        feature_owner_map.setdefault(cell, set()).add(word)
        collected: list[dict] = []
        if candidate_tokens is not None:
            tokens_to_score = [token for token in candidate_tokens if token in self.bindings]
        else:
            # Only bindings whose cells fired in the recall window can compete.
            spiked = set()
            for time, neuron_id in reversed(self._recent_spike_cells):
                if time < start_time:
                    break
                spiked.add(neuron_id)
            tokens_to_score = []
            seen_tokens: set[str] = set()
            for neuron_id in spiked:
                for token in self.motor_bindings.get(neuron_id, ()):
                    if token not in seen_tokens:
                        seen_tokens.add(token)
                        tokens_to_score.append(token)
        for token in tokens_to_score:
            binding = self.bindings.get(token)
            if binding is None:
                continue
            count = self._spike_count(binding.motor, start_time)
            if count <= 0:
                continue
            contributions: list[tuple[int, float]] = []
            supported_times = []
            for post in binding.motor:
                for syn_id in self.incoming.get(post, []):
                    synapse = self.synapses[syn_id]
                    if synapse.pre in active_source and synapse.weight >= self.config.learned_weight * 0.55:
                        weight = synapse.weight
                        pre_word = self.neurons[synapse.pre].word
                        if pre_word and pre_word.startswith("<feat:"):
                            weight *= float(getattr(self.config, "feature_support_gain", 3.0))
                        contributions.append((synapse.pre, weight))
                if post in self.recalled_motor:
                    contributions.append((-1, 100.0))
                times = self._spike_times_since(post, start_time)
                if times:
                    supported_times.append(min(times))
            if not contributions:
                continue
            collected.append({
                "token": token, "binding": binding, "count": count,
                "contributions": contributions, "times": supported_times,
            })

        # Discrimination: a cue cell that feeds many competing plans carries
        # Example tokens below were removed; runtime behavior is corpus-learned.
        # Example tokens below were removed; runtime behavior is corpus-learned.
        # candidates that cell supports is the sparse-coding analogue of IDF.
        document_frequency: dict[int, int] = defaultdict(int)
        for item in collected:
            for cell, _ in item["contributions"]:
                if cell >= 0:
                    document_frequency[cell] += 1

        candidates = []
        for item in collected:
            token = item["token"]
            binding = item["binding"]
            display = self.word_display.get(token, token)
            support = 0.0
            if prior:
                # The prior gates the association evidence instead of adding to
                # it: a plan that the recalled lesson / learned transitions
                # expect gets its support amplified, a plan they do not expect
                # keeps its raw support. Zero-support plans still stay at zero.
                emitted = self._tokenize(display)
                prior_score = sum(prior.get(item_token, 0.0) for item_token in emitted) / math.sqrt(max(1, len(emitted)))
                support_gain = 1.0 + prior_weight * prior_score
            else:
                support_gain = 1.0
            covered: set[int] = set()
            primed = getattr(self, "_primed_tokens", set())
            if token in primed:
                # An expectation that has already been primed counts as
                # evidence, exactly like the hippocampal pointer's bonus: the
                # plan's assembly is active now because the network expected it.
                support += float(getattr(self.config, "primed_support_bonus", 30.0))
            for cell, weight in item["contributions"]:
                if cell < 0:
                    support += weight
                    continue
                support += weight / max(1, document_frequency.get(cell, 1))
                covered.add(cell)
            if token in cue_words:
                # Echo suppression: the tokens you were just asked with are
                # primed as cues, not as answers, so a single-token plan that
                # Example tokens below were removed; runtime behavior is corpus-learned.
                support *= float(getattr(self.config, "recall_echo_penalty", 0.45))
            support *= support_gain
            # Prefer the plan that explains more of the cue. A cue token that was
            # taught with several different answers otherwise produces a tie and
            # the margin gate rejects a question a human finds unambiguous.
            if cue_words:
                # Coverage is measured over cue *tokens*, not over the capped
                # Example tokens below were removed; runtime behavior is corpus-learned.
                # Example tokens below were removed; runtime behavior is corpus-learned.
                covered_words: set[str] = set()
                for cell in covered:
                    word = self.neurons[cell].word
                    if not word:
                        continue
                    if word.startswith("<feat:"):
                        # A shared context feature stands for every cue token
                        # that owns it - that is exactly the analogy evidence.
                        covered_words |= feature_owner_map.get(cell, set())
                    else:
                        covered_words.add(word)
                coverage = len(covered_words & cue_words) / max(1, len(cue_words))
            else:
                coverage = len(covered) / max(1, len(active_source))
            if token in primed:
                # A plan that the ordered expectation primed is explained by the
                # whole context, so it is not held back by association coverage.
                coverage = 1.0
            support *= (0.3 + 0.7 * coverage) * support_gain
            penalty_scale = float(getattr(self.config, "answer_fan_in_penalty", 60.0))
            if penalty_scale > 0 and token not in primed:
                tagged_fan_in = sum(
                    1 for post in binding.motor
                    for syn_id in self.incoming.get(post, [])
                    if self.synapses[syn_id].metadata.get("association")
                ) / max(1, len(binding.motor))
                support *= 1.0 / (1.0 + tagged_fan_in / penalty_scale)
            first = min(supported_times) if supported_times else min(
                (time for nid in binding.motor for time in self._spike_times_since(nid, start_time)),
                default=self.time,
            )
            candidates.append((first, -item["count"], token, support, coverage))
        candidates.sort(key=lambda item: (-item[3], item[0], item[1]))
        return candidates

    def correct(self, prompt: str, expected: str, reward: float = 1.0) -> None:
        feedback = self.social_feedback.observe(prompt, expected, reward)
        answer_tokens = self.learn_pair(prompt, expected, reward)
        self.last_response_plastic_synapses = tuple(getattr(self, "last_learning_synapses", ()))
        self.last_response_info = {
            **self.last_response_info,
            "route": "corrected",
            "confidence": 0.90,
            "output": answer_tokens,
            "prompt": prompt,
            "social_feedback": feedback,
        }

    def reward(self, value: float, synapse_ids: tuple[int, ...] | None = None) -> dict:
        """Apply dopamine only to synapses participating in the last response."""
        value = float(max(-1.0, min(1.5, value)))
        self.modulators.apply_reward(value, source="feedback")
        self.modulator_field.inject_region("SN", "DA", 0.15 * value)
        self.modulator_field.inject_region("Amy", "5HT", 0.02 if value < 0 else -0.01)
        ids = synapse_ids if synapse_ids is not None else self.last_response_plastic_synapses
        changed = 0
        for syn_id in ids:
            if not 0 <= int(syn_id) < len(self.synapses):
                continue
            synapse = self.synapses[int(syn_id)]
            region = self.neurons[synapse.post].region
            synapse.apply_da(value, self.modulators.receptor_density(region, "DA"), self.config)
            changed += 1
        self._normalize_inputs()
        return {"modulated": changed, "rpe": value, "route": self.last_response_info.get("route")}

    def predict_language(self, text_or_tokens: str | list[str], top_k: int = 6) -> list[str]:
        tokens = self._tokenize(text_or_tokens) if isinstance(text_or_tokens, str) else list(text_or_tokens)
        if self.sequence_model is None:
            return []
        ranked = sorted(
            self.sequence_model.distribution(self._sequence_context(tokens)).items(),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return [token for token, _ in ranked[:max(1, top_k)]]

    def answer_world_question(self, question: str) -> list[str]:
        """Neural-only grounded answer: cue -> hippocampus/motor -> decode."""
        tokens = self.respond(question, max_len=16)
        route = self.last_response_info.get("route")
        if tokens and route in {"hippocampus-motor", "population"}:
            return tokens
        return []

    def answer_reasoning_question(self, question: str) -> str:
        """Neural-only causal/temporal answer through motor population decode."""
        tokens = self.respond(question, max_len=20)
        route = self.last_response_info.get("route")
        if tokens and route in {"hippocampus-motor", "population"}:
            return "".join(tokens)
        return "<unk>"

    def answer_compositional_question(self, question: str) -> str:
        """Answer only through neural cue->motor readout.

        The composition model remains available for offline diagnostics/schema
        counts, but it is deliberately not used as an answer fallback.
        """
        tokens = self.respond(question, max_len=16)
        info = self.last_response_info
        route = info.get("route")
        if tokens and route in {"hippocampus-motor", "population"}:
            return "".join(tokens)
        return "<unk>"

    # ------------------------------------------------------------------
    # Hippocampus and plasticity support
    # ------------------------------------------------------------------
    def _event_sdr(self, neuron_ids) -> tuple[int, ...]:
        return tuple(sorted({(int(neuron_id) * 2654435761 + self.random_seed) % self.sdr.dimension for neuron_id in neuron_ids}))

    def _write_episodic_event(self, text: str, answer_tokens: list[str], rpe: float, content_neurons: tuple[int, ...] = ()) -> None:
        content = tuple(dict.fromkeys(content_neurons or self._recently_active_neurons("Wern", 150.0)))
        phrase_binding = self.ensure_word(" ".join(answer_tokens))
        motor = phrase_binding.motor
        sdr = self._event_sdr((*content, *self.active_context))
        salience = float(min(1.0, self.regions["Amy"].mean_rate(100.0) / 10.0))
        relevance = 0.8 if self.dialogue_state["recent_words"] else 0.1
        should, scores = self.hippocampus.should_encode(
            sdr, salience=salience, rpe=rpe, relevance=relevance,
            unfinished=bool(self.dialogue_state["topic"]),
        )
        if not should:
            return
        event = EventSnapshot(
            timestamp=self.time, sdr_index=sdr, content_neurons=tuple(content),
            motor_neurons=motor, context_neurons=self.active_context,
            amygdala_valence=self.modulators.states["valence"], da_level=self.modulators.concentrations["DA"],
            goal_relevance=relevance, answer_phrase=" ".join(answer_tokens),
            region_rates=self.get_region_activity(),
            scores={
                **scores,
                # Curiosity lets replay prefer what the network itself found
                # informative rather than only what was rewarded externally.
                "curiosity": self.drive.score(novelty=scores.get("novelty", 0.0), surprise=abs(float(rpe))),
                "_question": " ".join(self._tokenize(text)),
            },
        )
        self.hippocampus.write(event)

    def relax_chemistry(self, simulated_ms: float = 100.0) -> dict:
        """Let volume chemistry diffuse/recover without evoking action."""
        steps = max(1, int(simulated_ms / 10.0))
        for _ in range(steps):
            self.modulator_field.diffuse(10.0)
        self.modulators.tick(simulated_ms, self.modulator_field.means())
        return {"simulated_ms": simulated_ms, "means": self.modulator_field.means()}

    def rehearse(self, prompt: str, answer: str, duration: float = 30.0) -> dict:
        """Run a focused cue->motor rehearsal without broad tract traffic.

        This is the small biological training loop: active Wernicke/DLPFC cells
        drive a freshly associated Broca/M1 plan, allowing local STDP/eligibility
        traces to act while the answer is taught.
        """
        answer_tokens = self._tokenize(answer)
        if not answer_tokens:
            return {"spikes_before": self.stats["spikes"], "spikes_after": self.stats["spikes"], "active": 0}
        phrase_binding = self.ensure_word(" ".join(answer_tokens))
        target = list(phrase_binding.motor)
        tokens = self._tokenize(prompt)
        cue_cells = [nid for token in tokens for nid in self.ensure_word(token).sensory]
        self.working_trace.clear()
        self.active_context = ()
        for token in tokens:
            self._update_working_memory(token)
        if len(tokens) > 1:
            question_binding = self.ensure_word(" ".join(tokens))
            self._reinforce_context(question_binding.context, 2.0)
        sources = list(dict.fromkeys((*cue_cells, *self.active_context)))
        old_association = self.suppress_association
        old_tracts = self.suppress_tracts
        self.events.clear()
        self.stimuli.clear()
        self._active_window.clear()
        self.suppress_association = False
        self.suppress_tracts = True
        self.rehearsal_synapses = set(self._association_synapses_to(target, sources))
        spikes_before = self.stats["spikes"]
        gain = self.modulators.thalamic_gain("Thal")
        self._activate_population(sources, self.config.input_current * gain, duration)
        self._activate_population(target, self.config.teach_current, duration)
        try:
            self.step(duration)
        finally:
            self.suppress_association = old_association
            self.suppress_tracts = old_tracts
            self.rehearsal_synapses = None
        self._normalize_inputs()
        self.modulators.apply_reward(0.12, source="rehearsal")
        self.modulator_field.inject_region("SN", "DA", 0.05)
        return {
            "spikes_before": spikes_before,
            "spikes_after": self.stats["spikes"],
            "active": len(set((*sources, *target))),
        }

    def recall(self, cue: str = "", top_k: int = 3) -> list[EventSnapshot]:
        tokens = self._tokenize(cue)
        cue_neurons = [nid for token in tokens for nid in self.ensure_word(token).sensory] if tokens else self._recently_active_neurons("Wern", 100.0)
        return self.hippocampus.retrieve(self._event_sdr(cue_neurons), top_k)

    def cognitive_state(self) -> dict:
        """Expose learned grounding, binding, and discourse state for tests."""

        return {
            "grounding_symbols": len(self.grounding.symbol_ids) if self.grounding else 0,
            "binding_fillers": len(self.binding_memory.filler_vectors) if self.binding_memory else 0,
            "binding_roles": len(self.binding_memory.role_vectors) if self.binding_memory else 0,
            "discourse_events": len(self.discourse.timeline()) if self.discourse else 0,
            "intents": len(self.intent_model.intents) if self.intent_model else 0,
            "neural_sequence_updates": self.sequence_model.updates if self.sequence_model else 0,
            "social_feedback": len(self.social_feedback.events),
            "development_phase": self.development_clock.phase,
            "semantic_prototypes": self.prototype_field.state() if self.prototype_field else {"prototypes": 0},
            "consistency_observations": len(self.consistency_memory.patterns) if self.consistency_memory else 0,
        }

    def sleep(self, replay_count: int = 50) -> dict:
        """Consolidate episodic indexes into cortical association synapses."""
        replay_count = max(
            1,
            round(int(replay_count) * self.development_clock.replay_fraction()),
        )
        selected = self.hippocampus.consolidate(replay_count)
        strengthened = 0
        associations = 0
        for event in selected:
            sources = tuple(dict.fromkeys((*event.content_neurons, *event.context_neurons)))
            posts = tuple(event.motor_neurons)
            for pre in sources:
                if not 0 <= pre < len(self.neurons) or not self.neurons[pre].alive:
                    continue
                for post in posts:
                    if not 0 <= post < len(self.neurons) or not self.neurons[post].alive or pre == post:
                        continue
                    syn_id = self._get_or_create_association(pre, post, 2.0)
                    if syn_id < 0:
                        continue
                    synapse = self.synapses[syn_id]
                    synapse.weight = max(synapse.weight, self.config.learned_weight * 0.85)
                    synapse.eligibility = min(2.0, synapse.eligibility + 0.08)
                    associations += 1
                    if syn_id % 7 == 0:
                        strengthened += 1
        cross_links = self.replay_planner.plan(selected)
        cross_associations = 0
        for link in cross_links:
            # Cross-episode links are weaker than direct cue->motor replay; they
            # form associative bridges, not a new answer lookup.
            weight_target = self.config.learned_weight * (0.55 + 0.20 * link.strength)
            for pre in link.source_neurons:
                if not 0 <= pre < len(self.neurons) or not self.neurons[pre].alive:
                    continue
                for post in link.target_neurons:
                    if not 0 <= post < len(self.neurons) or not self.neurons[post].alive or pre == post:
                        continue
                    syn_id = self._get_or_create_association(pre, post, 1.0)
                    if syn_id < 0:
                        continue
                    synapse = self.synapses[syn_id]
                    synapse.weight = max(synapse.weight, weight_target)
                    synapse.eligibility = min(1.5, synapse.eligibility + 0.05 * link.strength)
                    cross_associations += 1
        self.modulators.apply_reward(0.08, source="sleep-replay")
        self.modulator_field.inject_region("SN", "DA", 0.04)
        self.modulator_field.inject_region("Hipp", "ACh", 0.03)
        self._scale_synapses(0.98)
        self._normalize_inputs()
        phase = self.development_clock.observe_progress(
            episodes=int(self.stats.get("train_examples", 0)),
            replay_events=len(selected),
        )
        self.modulators.states["fatigue"] = max(0.0, self.modulators.states["fatigue"] - 0.35)
        return {
            "replayed": len(selected),
            "associations": associations,
            "strengthened": strengthened,
            "cross_episode_links": len(cross_links),
            "cross_episode_associations": cross_associations,
            "phase": phase,
            "plasticity": self.development_clock.plasticity(),
        }

    def mature(self) -> None:
        self.lifecycle_phase = "mature"
        for neuron in self.neurons:
            neuron.plasticity = 0.25
        for synapse in self.synapses:
            synapse.metadata["plasticity"] = 0.25

    # ------------------------------------------------------------------
    # Synapse helpers and lifecycle
    # ------------------------------------------------------------------
    def _has_synapse(self, pre: int, post: int) -> bool:
        return any(self.synapses[i].post == post for i in self.outgoing.get(pre, []))

    def _add_synapse(self, pre: int, post: int, weight: float, kind: str, delay: float | None = None, plasticity: float = 1.0) -> int:
        # Per-neuron synapse budget (measured from the current connectome):
        # strongly excitatory cells are allowed a proportional extra allowance.
        # The expensive strength scan only runs once a neuron is at its base
        # budget, so bulk wiring stays cheap.
        base_out = int(getattr(self.config, "synapse_out_cap_per_neuron", 1982))
        base_in = int(getattr(self.config, "synapse_in_cap_per_neuron", 284))
        if base_out > 0 and len(self.outgoing.get(pre, ())) >= base_out:
            if len(self.outgoing.get(pre, ())) >= self._synapse_budget(pre, "out"):
                return -1
        if base_in > 0 and len(self.incoming.get(post, ())) >= base_in:
            if len(self.incoming.get(post, ())) >= self._synapse_budget(post, "in"):
                return -1
        pre_n, post_n = self.neurons[pre], self.neurons[post]
        distance = float(np.linalg.norm(post_n.position-pre_n.position))
        delay = float(np.clip(distance / 3.2, self.config.synapse_delay_min, self.config.synapse_delay_max))
        synapse = Synapse(
            pre=pre, post=post, weight=float(weight), delay=delay,
            position=(pre_n.position + post_n.position) * 0.5,
            release_probability=self.config.release_probability,
            kind=kind, metadata={"plasticity": plasticity},
        )
        syn_id = len(self.synapses)
        self.synapses.append(synapse)
        self.outgoing[pre].append(syn_id)
        self.incoming[post].append(syn_id)
        return syn_id

    def _synapse_budget(self, neuron_id: int, direction: str) -> int:
        """Synapse budget for one cell; strong excitation earns a bonus."""

        base = int(
            getattr(self.config, "synapse_out_cap_per_neuron", 1982)
            if direction == "out"
            else getattr(self.config, "synapse_in_cap_per_neuron", 284)
        )
        ids = self.outgoing.get(neuron_id, ()) if direction == "out" else self.incoming.get(neuron_id, ())
        if not ids:
            return base
        threshold = float(getattr(self.config, "strong_weight_threshold", 6.0))
        strong = sum(
            1 for syn_id in ids
            if self.synapses[syn_id].kind == "excitatory" and abs(self.synapses[syn_id].weight) >= threshold
        )
        if strong / len(ids) >= float(getattr(self.config, "strong_fraction_for_bonus", 0.30)):
            base = int(base * (1.0 + float(getattr(self.config, "strong_synapse_bonus", 0.25))))
        return base

    def _get_or_create_association(self, pre: int, post: int, delay: float) -> int:
        """Wire (or strengthen) one learned cue->plan association.

        Returns ``-1`` when the postsynaptic cell has reached its learned
        association capacity and this pair is new; callers skip the update. The
        cap keeps the connectome (and therefore decoding) bounded.
        """

        cap = int(getattr(self.config, "association_fan_in_cap", 240))
        cache_key = (int(pre), int(post))
        if cache_key in self._association_pairs:
            syn_id = self._association_cache.get(cache_key)
            if syn_id is None or not 0 <= syn_id < len(self.synapses):
                syn_id = next(iter(self._association_outgoing.get(pre, set()) & self._association_incoming.get(post, set())), -1)
                self._association_cache[cache_key] = syn_id
            if 0 <= syn_id < len(self.synapses):
                synapse = self.synapses[syn_id]
                synapse.weight = max(synapse.weight, self.config.learned_weight)
                synapse.delay = min(synapse.delay, delay)
                synapse.metadata["association"] = True
                synapse.release_probability = 0.97
                synapse.active = True
                return syn_id
        cached = self._association_cache.get(cache_key)
        if cached is not None and 0 <= cached < len(self.synapses):
            synapse = self.synapses[cached]
            if synapse.pre == pre and synapse.post == post:
                synapse.weight = max(synapse.weight, self.config.learned_weight)
                synapse.delay = min(synapse.delay, delay)
                synapse.metadata["association"] = True
                synapse.release_probability = 0.97
                synapse.active = True
                return cached
        tagged: list[int] = []
        for syn_id in self.outgoing.get(pre, []):
            synapse = self.synapses[syn_id]
            if synapse.post == post:
                self._association_cache[cache_key] = syn_id
                self._association_pairs.add(cache_key)
                self._association_outgoing[pre].add(syn_id)
                self._association_incoming[post].add(syn_id)
                synapse.weight = max(synapse.weight, self.config.learned_weight)
                synapse.delay = min(synapse.delay, delay)
                # Tag reused developmental synapses too; otherwise they stay in
                # the homeostatic normalization pool and get rescaled below the
                # decoder's support floor.
                synapse.metadata["association"] = True
                synapse.release_probability = 0.97
                synapse.active = True
                return syn_id
        if cap > 0:
            if len(self._association_incoming.get(post, ())) >= cap:
                return -1
        syn_id = self._add_synapse(pre, post, self.config.learned_weight, "excitatory", delay)
        if syn_id < 0:
            return -1
        self._association_pairs.add(cache_key)
        self._association_outgoing[pre].add(syn_id)
        self._association_incoming[post].add(syn_id)
        self._association_cache[cache_key] = syn_id
        self.synapses[syn_id].metadata["association"] = True
        self.synapses[syn_id].release_probability = 0.97
        return syn_id

    def _normalize_inputs(self) -> None:
        # Homeostatic normalization applies to the developmental/sensory drive.
        # Learned cue->plan associations are exempt: clamping them to the same
        # 120-point budget pushed every fresh association to ~3.15, below the
        # 3.3 support floor the decoder requires, so teaching something new
        # silently erased the recall path for everything already learned.
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, synapse in enumerate(self.synapses):
            if synapse.active and synapse.kind == "excitatory" and not synapse.metadata.get("association"):
                grouped[synapse.post].append(index)
        for indices in grouped.values():
            total = sum(self.synapses[i].weight for i in indices)
            if total > self.config.input_weight_capacity:
                scale = self.config.input_weight_capacity / total
                for i in indices:
                    self.synapses[i].weight *= scale

    def _apply_da(self, value: float) -> None:
        for synapse in self.synapses:
            region = self.neurons[synapse.post].region
            synapse.apply_da(value, self.modulators.receptor_density(region, "DA"), self.config)
        self._normalize_inputs()

    def _scale_synapses(self, factor: float) -> None:
        floor = float(getattr(self.config, "association_floor", 0.85)) * abs(self.config.learned_weight)
        for synapse in self.synapses:
            if synapse.active:
                if synapse.metadata.get("association"):
                    # Engrams decay, but never below the point where the
                    # population decoder can still read them back.
                    synapse.weight = max(synapse.weight * factor, floor)
                else:
                    synapse.weight *= factor
                synapse.active = abs(synapse.weight) >= self.config.prune_weight

    def lifecycle(self) -> None:
        if self.lifecycle_phase == "proliferation" and self.time >= self.config.prolif_ms:
            self.lifecycle_phase = "elimination"
        elif self.lifecycle_phase == "elimination" and self.time >= self.config.prune_ms:
            self.lifecycle_phase = "mature"
        if self.lifecycle_phase == "proliferation":
            self.maybe_divide()
        target = float(getattr(self.config, "homeostatic_target_rate", 4.0))
        offset_max = float(getattr(self.config, "homeostatic_threshold_max", 6.0))
        offset_min = float(getattr(self.config, "homeostatic_threshold_min", -3.0))
        for neuron in self.neurons:
            if not neuron.alive:
                continue
            if neuron.role == "source":
                neuron.energy = min(1.5, neuron.energy + 0.01)
                continue
            rate = neuron.recent_rate(self.time, 1000.0)
            # Active homeostasis: overly fast cells raise their effective firing
            # threshold; silent but viable cells regain a little excitability.
            if neuron.role == "neuron":
                offset = float(getattr(neuron, "threshold_offset", 0.0))
                if rate > target * 2.0:
                    offset += min(0.35, (rate - target * 2.0) * 0.005)
                elif rate < target * 0.15 and neuron.energy > 0.35:
                    offset -= 0.05
                neuron.threshold_offset = float(max(offset_min, min(offset_max, offset)))
            if neuron.role != "glia" and neuron.energy < self.config.apoptosis_energy and rate < self.config.apoptosis_rate:
                protect = bool(getattr(self.config, "protect_binding_cells", True)) and neuron.word is not None
                if protect:
                    # An engram cell is silent by design between uses; starving
                    # it would delete learned content, so it is refuelled instead.
                    neuron.energy = max(neuron.energy, self.config.apoptosis_energy * 1.5)
                else:
                    neuron.alive = False
        # Region-level adaptive inhibition recruits GABA when a field overheats.
        for code, region_state in self.regions.items():
            rate = region_state.mean_rate(1000.0)
            if rate > target * 6.0:
                self.modulator_field.inject_region(code, "GABA", min(0.12, rate / 4000.0))
        for synapse in self.synapses:
            if abs(synapse.weight) < self.config.prune_weight:
                synapse.active = False

    def _apply_lateral_inhibition(self, fired_ids: list[int]) -> int:
        """Suppress just-fired neighbours in a region (fast local WTA)."""
        if len(fired_ids) < 2:
            return 0
        groups: dict[str, list[int]] = defaultdict(list)
        for nid in fired_ids:
            if 0 <= nid < len(self.neurons) and self.neurons[nid].alive:
                groups[self.neurons[nid].region].append(nid)
        inhibited = 0
        window = float(getattr(self.config, "lateral_inhibition_window", 5.0))
        gain = float(getattr(self.config, "lateral_inhibition_gain", 9.0))
        for region, ids in groups.items():
            if len(ids) < 2:
                continue
            ranked = sorted(ids, key=lambda nid: (-self.neurons[nid].recent_rate(self.time, window), -self.neurons[nid].V, nid))
            for loser in ranked[3:]:
                neuron = self.neurons[loser]
                threshold = neuron.threshold + float(getattr(neuron, "threshold_offset", 0.0))
                neuron.V = min(neuron.V, threshold - 6.0)
                self.stimuli.pop(loser, None)
                self.pending_currents[loser] -= gain
                inhibited += 1
        if inhibited:
            self.modulator_field.inject_region("Stri", "GABA", min(0.025, inhibited / 400.0))
        return inhibited

    # ------------------------------------------------------------------
    # Introspection, visualization, persistence
    # ------------------------------------------------------------------
    def _recently_active_neurons(self, region: str, window_ms: float = 100.0, limit: int | None = None) -> tuple[int, ...]:
        since = self.time - window_ms
        region_state = self.regions[region]
        counts: dict[int, int] = defaultdict(int)
        for t, nid in region_state.recent_spikes:
            if t >= since:
                counts[nid] += 1
        k = int(limit or self.config.sdr_k)
        return tuple(nid for nid, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:k])

    def _spike_count(self, neuron_ids, since: float) -> int:
        # spike_times is time-ordered, so a reverse scan can stop at the first
        # older spike instead of walking the whole 2000-entry buffer per cell.
        return sum(len(self._spike_times_since(nid, since)) for nid in neuron_ids)

    def _spike_times_since(self, neuron_id, since: float) -> list[float]:
        """Spike times after ``since`` (newest first)."""

        out: list[float] = []
        for time in reversed(self.neurons[int(neuron_id)].spike_times):
            if time < since:
                break
            out.append(time)
        return out

    def get_region_activity(self) -> dict[str, float]:
        return {code: round(state.mean_rate(100.0), 4) for code, state in self.regions.items()}

    def get_neuron_state(self, neuron_id: int) -> dict:
        return self.neurons[int(neuron_id)].to_dict()

    def get_synapse_state(self, index: int) -> dict:
        syn = self.synapses[int(index)]
        return {**syn.__dict__, "position": syn.position.tolist()}

    def show(self) -> dict:
        return {
            "time_ms": round(self.time, 3),
            "phase": self.lifecycle_phase,
            "neurons": sum(n.alive for n in self.neurons),
            "seed_neurons": len(self.neurons),
            "active_synapses": sum(s.active for s in self.synapses),
            "words": len(self.bindings),
            "events": len(self.events),
            "episodic_events": len(self.hippocampus.episodic_buffer),
            "last_output": self.last_output,
            "last_response": {key: value for key, value in self.last_response_info.items() if key != "association_synapses"},
            "modulators": self.modulators.to_dict(),
            "neuromodulator_field": self.modulator_field.state(),
        }

    @staticmethod
    def _finalize_loaded(brain: "BionicBrain") -> "BionicBrain":
        config_values = {field.name: getattr(brain.config, field.name, field.default) for field in fields(BionicConfig)}
        brain.config = BionicConfig(**config_values)
        defaults = {
            "word_display": {},
            "last_response_plastic_synapses": (),
            "last_learning_synapses": (),
            "last_response_info": {},
            "recalled_answer": [],
            "suppress_association": False,
            "suppress_tracts": False,
            "rehearsal_synapses": None,
            "_last_neurochem": getattr(brain, "time", 0.0),
            "cooccurrence": {},
            "shared_cells": {},
            "_active_until": {},
            "dialogue_context": deque(maxlen=6),
            "relation_sentences": deque(maxlen=256),
            "_recent_spike_cells": deque(maxlen=20_000),
        }
        for name, value in defaults.items():
            if not hasattr(brain, name):
                setattr(brain, name, value)
        if not isinstance(getattr(brain, "schemas", None), LearnedSchemaMemory):
            brain.schemas = LearnedSchemaMemory(
                statistics=brain.token_statistics,
                observe_statistics=False,
            )
        if not hasattr(brain, "context_features"):
            brain.context_features = ContextFeatureSpace(
                n_features=int(getattr(brain.config, "context_features", 256)),
                top_k=int(getattr(brain.config, "context_feature_top_k", 4)),
            )
        if not hasattr(brain, "token_statistics"):
            brain.token_statistics = brain.context_features.token_statistics
        # Loaded models share one online distribution, exactly as a fresh brain
        # does.  The owners that receive already-observed runtime tokens must
        # not count the same sentence again.
        brain.context_features.token_statistics = brain.token_statistics
        brain.context_features.update_statistics = False
        brain.schemas.token_statistics = brain.token_statistics
        brain.schemas.observe_statistics = False
        if not hasattr(brain, "feature_cell_ids"):
            brain.feature_cell_ids = {}
        if not hasattr(brain, "drive"):
            brain.drive = IntrinsicDrive()
        if not hasattr(brain, "motor_bindings"):
            brain.motor_bindings = {}
            for token, binding in brain.bindings.items():
                for neuron_id in (*binding.motor, *binding.sensory):
                    brain.motor_bindings[neuron_id] = (*brain.motor_bindings.get(neuron_id, ()), token)
        for token, binding in brain.bindings.items():
            if not hasattr(binding, "channels"):
                binding.channels = {}
        for name in ("DA", "5HT", "ACh", "NE", "Histamine", "GABA"):
            brain.modulators.concentrations.setdefault(name, {"DA": 0.0, "5HT": 0.50, "ACh": 0.45, "NE": 0.45, "Histamine": 0.45, "GABA": 0.40}[name])
        if not hasattr(brain.modulators, "reward_history"):
            brain.modulators.reward_history = []
        if not hasattr(brain, "active_eligibility"):
            brain.active_eligibility = {
                index for index, synapse in enumerate(brain.synapses)
                if synapse.active and abs(synapse.eligibility) >= 1e-6
            }
        if not hasattr(brain, "modulator_field"):
            brain.modulator_field = NeuromodulatorField(
                brain.config.space_bounds,
                getattr(brain.config, "modulator_grid", (12, 14, 11)),
                getattr(brain.config, "modulator_diffusion", 0.060),
                getattr(brain.config, "modulator_decay", 0.018),
            )
        if not hasattr(brain, "subword_tokenizer") or brain.subword_tokenizer is None:
            brain.subword_tokenizer = SubwordTokenizer(
                max_merges=int(getattr(brain.config, "subword_max_merges", 512)),
                min_pair_count=int(getattr(brain.config, "subword_min_pair_count", 2)),
                max_unit_symbols=int(getattr(brain.config, "subword_max_unit_symbols", 3)),
            )
        if not hasattr(brain, "subwords_enabled"):
            brain.subwords_enabled = False
        if not hasattr(brain, "social_feedback"):
            brain.social_feedback = SocialFeedback()
        if not hasattr(brain, "development_clock"):
            brain.development_clock = DevelopmentalClock()
        if not hasattr(brain, "decoder_calibration"):
            brain.decoder_calibration = DecoderCalibration()
        if not hasattr(brain, "prototype_field"):
            brain.prototype_field = PrototypeField()
        if not hasattr(brain, "consistency_memory"):
            brain.consistency_memory = ConsistencyMemory()
        if not hasattr(brain, "replay_planner"):
            brain.replay_planner = AssociativeReplayPlanner()
        if not hasattr(brain, "_association_cache"):
            brain._association_cache = {}
        if not hasattr(brain, "_association_pairs") or not hasattr(brain, "_association_outgoing") or not hasattr(brain, "_association_incoming"):
            brain._association_pairs = set()
            brain._association_outgoing = defaultdict(set)
            brain._association_incoming = defaultdict(set)
            for syn_id, synapse in enumerate(brain.synapses):
                if not synapse.metadata.get("association"):
                    continue
                key = (synapse.pre, synapse.post)
                brain._association_pairs.add(key)
                brain._association_outgoing[synapse.pre].add(syn_id)
                brain._association_incoming[synapse.post].add(syn_id)
        if not hasattr(brain, "grounding") and bool(getattr(brain.config, "use_sensory_grounding", True)):
            brain.grounding = CorticalGrounding(
                embedding_dim=int(getattr(brain.config, "grounding_embedding_dim", 16)),
                projection_dim=int(getattr(brain.config, "grounding_projection_dim", 16)),
                seed=brain.random_seed,
            )
        if not hasattr(brain, "binding_memory") and bool(getattr(brain.config, "use_binding_memory", True)):
            brain.binding_memory = TensorProductBinding(dimension=64, seed=brain.random_seed)
            brain.discourse = DiscourseState(brain.binding_memory)
        if not hasattr(brain, "intent_model") and bool(getattr(brain.config, "use_learned_intent", True)):
            brain.intent_model = IntentLearner(
                embedding_dim=int(getattr(brain.config, "grounding_embedding_dim", 16)),
                hidden_dim=int(getattr(brain.config, "neural_hidden_dim", 16)),
                lr=0.01,
            )
        if not hasattr(brain, "sequence_model"):
            brain.sequence_model = DifferentiableSequenceModel(
                embedding_dim=int(getattr(brain.config, "grounding_embedding_dim", 16)),
                hidden_dim=int(getattr(brain.config, "neural_hidden_dim", 16)),
                context_window=int(getattr(brain.config, "neural_context_window", 8)),
                lr=0.01,
            )
        # Old checkpoints may predate per-neuron homeostatic thresholds.
        for neuron in brain.neurons:
            if not hasattr(neuron, "threshold_offset"):
                neuron.threshold_offset = 0.0
        return brain

    def save(self, path: str | Path, keep: int = 10) -> Path:
        """Atomically save a checksummed gzip checkpoint.

        The model is pickled first, checksummed, then written to a temporary
        file and moved into place. A process crash therefore cannot leave a
        half-written final checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        brain_blob = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        payload = {
            "format": "bionicbrain",
            "version": self.STATE_VERSION,
            "checksum": hashlib.sha256(brain_blob).hexdigest(),
            "brain_blob": brain_blob,
        }
        tmp_name = f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        tmp_path = path.parent / tmp_name
        try:
            with gzip.open(tmp_path, "wb", compresslevel=6) as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        snapshots = sorted(path.parent.glob(f"{path.stem}*.pkl"))
        if keep > 0 and len(snapshots) > keep:
            for old in snapshots[:-keep]:
                old.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BionicBrain":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
        version = payload.get("version")
        if version == 3:
            blob = payload.get("brain_blob")
            if not isinstance(blob, bytes):
                raise ValueError("checkpoint is missing its model blob")
            checksum = hashlib.sha256(blob).hexdigest()
            if checksum != payload.get("checksum"):
                raise ValueError("checkpoint checksum mismatch; file is damaged")
            brain = pickle.loads(blob)
        elif version == 2:
            # Read the old pre-checksum format, then callers may re-save as v3.
            brain = payload.get("brain")
        else:
            raise ValueError(f"unsupported checkpoint version: {version}")
        if not isinstance(brain, cls):
            raise TypeError("checkpoint contains an incompatible model object")
        return cls._finalize_loaded(brain)

    def snapshot(self, directory: str | Path = "data/checkpoints") -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = wall_time.strftime("%Y%m%d_%H%M%S")
        return self.save(directory / f"brain_{stamp}.pkl", keep=10)


























