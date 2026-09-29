from dataclasses import dataclass


@dataclass(frozen=True)
class BionicConfig:
    """Simulation, development, plasticity, language, and persistence parameters."""

    # Time and space. The coordinate origin is the brainstem.
    dt: float = 0.02
    chem_dt: float = 10.0
    chem_steps: int = 48
    space_bounds: tuple[float, float, float, float, float, float] = (
        -70.0, 70.0, -80.0, 90.0, -40.0, 90.0,
    )

    # Anatomy/progenitors. Region centers are defined in atlas.regions.
    seed_profile: str = "standard"
    max_neurons: int = 10_000
    inhibitory_fraction: float = 0.20

    # Hodgkin-Huxley and LIF dynamics.
    c_m: float = 1.0
    g_na: float = 120.0
    g_k: float = 36.0
    e_na: float = 50.0
    e_k: float = -77.0
    e_leak: float = -54.387
    g_leak: float = 0.30
    lif_tau: float = 18.0
    lif_reset: float = -70.0
    lif_threshold: float = -52.0
    noise_tau: float = 8.0
    noise_strength: float = 2.0
    refractory_ms: float = 2.0

    # Morphogens.
    morphogen_grid: tuple[int, int, int] = (18, 21, 16)
    morphogen_diffusion: float = 0.045
    morphogen_decay: float = 0.010
    morphogen_inhibition: float = 0.035
    modulator_grid: tuple[int, int, int] = (12, 14, 11)
    modulator_diffusion: float = 0.060
    modulator_decay: float = 0.018
    neurochem_interval: float = 10.0

    # Synapses and local plasticity.
    synapse_delay_min: float = 0.5
    synapse_delay_max: float = 4.0
    release_probability: float = 0.82
    stdp_window: float = 20.0
    eligibility_decay_per_step: float = 0.985
    da_gain: float = 2.0
    input_weight_capacity: float = 120.0
    prune_weight: float = 0.04
    synapse_scale_interval: float = 5_000.0
    synapse_scale_factor: float = 0.95

    # Development and metabolism.
    prolif_ms: float = 5_000.0
    prune_ms: float = 15_000.0
    lifecycle_interval: float = 100.0
    apoptosis_energy: float = 0.10
    apoptosis_rate: float = 0.20
    homeostatic_target_rate: float = 4.0
    homeostatic_threshold_max: float = 6.0
    homeostatic_threshold_min: float = -3.0
    lateral_inhibition_gain: float = 9.0
    lateral_inhibition_window: float = 5.0

    # Modulation.
    baseline_attention: float = 0.85
    baseline_arousal: float = 0.78

    # Language and memory.
    sdr_k: int = 10
    motor_sdr_k: int = 6
    context_k: int = 5
    decode_window: float = 50.0
    stimulus_duration: float = 25.0
    teach_current: float = 12.0
    input_current: float = 18.0
    learned_weight: float = 6.0
    memory_hit_threshold: float = 0.55
    strong_memory_hit_threshold: float = 0.80
    response_confidence_threshold: float = 0.68
    response_margin_threshold: float = 1.65
    response_support_per_cell: float = 1.5
    working_memory_tokens: int = 20
    hippocampus_buffer: int = 1_000
    hippocampus_capacity: int = 10_000
    novelty_threshold: float = 0.30
    salience_threshold: float = 0.50
    rpe_threshold: float = 0.30
    relevance_threshold: float = 0.40
    encode_threshold: float = 0.50

    # Recall, consolidation and generalization.
    #
    # Every knob below is read through ``getattr(config, name, default)`` so
    # checkpoints written before these fields existed still load.
    recall_window_ms: float = 16.0          # motor simulation window inside respond()
    # How long a fired cell keeps being advanced. This must stay short: cells
    # that linger in the active set are re-integrated every micro-step, and the
    # set otherwise accumulates noise-driven spikes until one answer costs
    # minutes (measured: 8 s -> 108 s across a session at 120 ms).
    active_hold_ms: float = 25.0
    active_set_limit: int = 96
    pointer_bias_gain: float = 0.6          # hippocampal pointer is a bias, not the only driver
    association_floor: float = 0.85         # association weights decay down to this × learned_weight
    protect_binding_cells: bool = True      # engram cells are not killed by the apoptosis sweep
    hippocampus_retain_consolidated: bool = True
    cooccurrence_min: float = 2.0           # co-activations before two tokens share sensory cells
    sdr_share_cells: int = 1                # cells borrowed per repeated co-occurrence
    sdr_share_budget: int = 4               # max borrowed cells per token, keeps SDRs sparse
    sequence_prior_weight: float = 3.0      # additive motor support from learned token transitions
    # Automatic termination. The network decides when it is done; the safety cap
    # only exists so a runaway loop cannot hang a run, and is not a content
    # length. `</s>`/`<eos>` are learned document-end markers (the transition
    # model emits them at the end of every read text).
    generation_end_tokens: tuple[str, ...] = ("</s>", "<eos>")
    generation_safety_cap: int = 256
    generation_min_probability: float = 0.12
    generation_low_confidence_steps: int = 3
    generation_fatigue_limit: float = 0.85
    # Recall gates for the cortex path. A partial cue legitimately produces less
    # support than an exact cue, so recall uses its own (looser) gates; zero
    # support still yields no candidate at all, so unknown input stays rejected.
    recall_confidence_threshold: float = 0.45
    recall_margin_threshold: float = 1.25
    recall_support_per_cell: float = 0.5
    recall_coverage_minimum: float = 0.25
    recall_echo_penalty: float = 0.45
    # Synaptic capacity: a postsynaptic cell can only hold so many learned
    # associations, so new experience competes with old instead of growing the
    # graph without bound (the first mixed-corpus run tripled synapses to 575k
    # and pushed answer latency to ~30 s).
    association_fan_in_cap: int = 240
    stream_context_tokens: int = 8
    # Eligibility traces are decayed in batches: decaying every synapse on every
    # micro-step cost ~140M operations per answer (measured 60-90 s). Applying
    # the same total decay every N steps is mathematically identical.
    eligibility_decay_interval: int = 10
    eligibility_set_limit: int = 40_000
    # Per-neuron synapse budget, measured from the brian_mix_v2 connectome
    # (max out-degree 1982, max in-degree 284). Strongly excitatory cells may
    # exceed the base budget by ``strong_synapse_bonus``.
    synapse_out_cap_per_neuron: int = 1982
    synapse_in_cap_per_neuron: int = 284
    strong_synapse_bonus: float = 0.25
    strong_weight_threshold: float = 6.0
    strong_fraction_for_bonus: float = 0.30
    # Continuation: how much context a token condition on, and how deep the
    # next-token model looks back.
    sequence_order: int = 5
    # Recall conditions on the recent working-memory window, not the whole
    # prompt: a 30-token prompt otherwise spreads evidence so thin that no plan
    # clears the coverage gate (measured: long-text continuation produced 0
    # tokens even right after reading the document).
    recall_cue_tokens: int = 8
    active_source_cells: int = 40
    primed_support_bonus: float = 30.0
    # A plan whose motor cells connect to almost everything carries little
    # information (的/是/了 win every competition otherwise). The penalty is
    # derived from the network's own fan-in, not from an external stop-word
    # list: score *= 1 / (1 + mean_tagged_fan_in / scale).
    answer_fan_in_penalty: float = 60.0
    # Learned context features (competitive, local): tokens used in similar
    # contexts come to share assemblies, which is what makes semantic
    # similarity - and therefore analogy - possible at all.
    context_features: int = 256
    context_feature_top_k: int = 4
    context_feature_learning: bool = True
    # A shared context feature is strong evidence of category membership, so its
    # contribution to an answer counts with a gain (otherwise the geometry is
    # computed but never reaches the readout).
    feature_support_gain: float = 3.0
    # Learned relations (P1-3b): use the distributional discoverer instead of the
    # hand-written relation lexicons. Off by default until it matches the
    # lexicon on the hardcode-dependence ledger.
    use_learned_relations: bool = True
    learned_relation_threshold: float = 0.35
    learned_relation_relative_floor: float = 0.35
    # P1-P4 learned cognitive switches.  Grounding, sequence and intent are
    # available for new candidates, but remain off until checkpoint-level A/B
    # and production regression evidence satisfy the TODO promotion gate.
    use_sensory_grounding: bool = True
    use_binding_memory: bool = True
    use_neural_sequence: bool = True
    use_learned_intent: bool = True
    use_decoder_calibration: bool = False
    # P1-1: learned subword units replace the language-specific regex split.
    # This stays off for old checkpoints until a tokenizer has been trained and
    # A/B-tested; enabling it before learning falls back to the structural
    # character tokenizer rather than inventing an empty vocabulary.
    use_subword_tokenizer: bool = False
    subword_max_merges: int = 512
    subword_min_pair_count: int = 2
    subword_max_unit_symbols: int = 3
    grounding_embedding_dim: int = 16
    grounding_projection_dim: int = 16
    neural_hidden_dim: int = 16
    neural_context_window: int = 8

    # Runtime defaults.
    default_checkpoint: str = ""


DEFAULT_CONFIG = BionicConfig()


def seed_counts(profile: str, region_count: int) -> int:
    counts = {"minimal": 5, "standard": 20, "large": 50}
    if profile not in counts:
        raise ValueError(f"unknown seed profile: {profile}")
    return counts[profile]




