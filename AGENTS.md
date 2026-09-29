# Repository Guidelines

## What This Project Is

`BionicBrain` (persona "Brian") is a developmental whole-brain structured spiking neural network: 33 atlas regions, morphogen fields, HH/LIF cells, event-driven synapses, white-matter tracts, neuromodulators, selective hippocampal memory, and a text-in / motor-decoding-out language loop.

Hard constraints for anything inside the network:

- **Bionic first.** The four neuron structures must stay explicit: dendrite (local input integration), soma (threshold/spike dynamics), axon (conduction delay/myelin), synapse (local plasticity/modulation). Inputs must pass sensory/thalamic routing; outputs must use motor population decoding.
- **No language knowledge in code.** Zero hardcoded linguistic rules: no relation word lists, no question templates, no stop-word lists, no grammar lexicons, no regex tokenisation rules that encode a language. Language knowledge must be learned from data. See `docs/TODO_ZERO_HARDCODED.md` for the current violation list (H1-H12) and the replacement plan.
- **Language portability is an acceptance criterion.** Learning another language must require *only new corpora*, never edits to `.py` files.
- **Tooling is relaxed, mapping is mandatory.** Backpropagation, convolutions, attention/Transformers, PyTorch and other differentiable tools are allowed, but every differentiable component must state which neuron structure it models and what brain-side process the backward pass approximates (e.g. convolution = dendritic receptive field, attention = thalamic selective gain, backprop = top-down prediction error). Tooling that only makes the model a generic deep net is rejected.
- Still forbidden: static answer lookup, a preset complete connectome, and hand-written answer tables.
- Learning must remain interpretable as local plasticity plus prediction error; weight updates must be attributable (ablation must move the score).

An external chat model may only act as a *teacher/reviewer/labeller* from outside the network (see "Local LLM Teacher"). It must never appear in Brian's inference path.

## Project Structure & Module Organization

Implementation lives in `src/bionic_brain/`:

- `brain.py` — `BionicBrain` orchestrator (large; do not grow it with new mechanisms).
- `atlas/` (`regions.py`, `seeds.py`) — 33 named region centers and seed scaling.
- `cells/neuron.py` — HH + LIF cells, noise, refractory, energy, age.
- `synapses/model.py` — event synapses, delay, release probability, STDP, DA.
- `morphogens/field.py`, `modulation/modulators.py` — developmental morphogens and 6 neuromodulators.
- `tracts/channel.py` — white-matter pathways with delay/attenuation/myelin.
- `memory/hippocampus.py` — gating, event SDR, retrieval, replay.
- `language/` (`tokenizer`, `sdr`, `corpus`, `dialogue`, `world_model`) — text I/O.
- `cognition.py` — learned grounding, binding, discourse, intent, sequence, and development loops.
- `learning/` (`sequence`, `composition`, `reasoning`, `narrative`) — local learning curricula.
- `agents/` — trainer-side agents only: `logic_tutor.py` (rule-based teacher) and `llm_teacher.py` (local chat-model teacher).
- `interfaces/` (`cli.py`, `public_api.py`) — CLI and Python API (new CLI code belongs here).
- `simulation/`, `visualization/`, `config/defaults.py`, `persistence/`.

`docs/PROJECT_STRUCTURE.md` defines the target boundaries and still lists packages that are not implemented yet (`senses/`, `thalamus/`, `motor/`, `cerebellum/`, `regions/`, `development/`). Put new mechanisms in the closest existing package; create a planned package only when a mechanism genuinely needs it.

Other paths:

- `docs/` — `PRD.md`, `in-out.md`, `海马体的设计.md`, `PROJECT_STRUCTURE.md`.
- `tests/` — `test_bionic_brain.py`, `test_llm_teacher.py` (flat today; `unit/ integration/ smoke/ acceptance/` are the target layout).
- `tools/` — training and diagnostic entry points (`train_brian_curriculum.py`, `train_all_streams.py`, `train_city_streams.py`, `train_narrative_book.py`, `selftrain_logic.py`, `train_llm_teacher.py`, `probe_llm_teacher.py`, `logic_agent.py`, `eval_checkpoint.py`, `diagnose_memory.py`, `probe_recall.py`). Do not add experiments to the repository root.
- `data/corpora/` — JSONL corpora (`{"question": "...", "answer": "..."}`); `data/corpora/llm_generated/` caches teacher-drafted items.
- `data/checkpoints/`, `runs/{logs,metrics,figures}/`, `configs/`, `archive/{legacy_scripts,experiments}/`.

Generated `.pkl`, caches, IDE files, and plots stay out of Git (`.gitignore` already covers `*.pkl`, `runs/`, `data/checkpoints/`, `__pycache__/`, `.idea/`).

## Build, Test, and Development Commands

- `uv sync` — synchronize dependencies (numpy, matplotlib, and the CPU/GPU torch wheel used by `src/bionic_brain/nn/`).
- `.\.venv\Scripts\python.exe verify_prd.py` — PRD smoke test.
- `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` — mechanism tests.
- `.\.venv\Scripts\python.exe main.py --mode train --epochs 3 --save data/checkpoints/brain.pkl` — train.
- `.\.venv\Scripts\python.exe main.py --mode evaluate --load data/checkpoints/brian_neural_ocean.pkl` — evaluate.
- `.\.venv\Scripts\python.exe main.py --mode chat --load data/checkpoints/brian_neural_ocean.pkl` — interactive mode.
- `.\.venv\Scripts\python.exe tools\train_brian_curriculum.py --base data/checkpoints/brian_neural_ocean.pkl` — staged curriculum training.

Set `$env:PYTHONIOENCODING = 'utf-8'` on Windows before runs that print Chinese; the default console encoding mangles tokens.

## Training Runs & Checkpoints

**Neural-first policy.** Checkpoints are experiment artifacts, not acceptance targets. Default inference must use learned/ neural mechanisms; symbolic language fallbacks live only under `src/bionic_brain/legacy/` and must not be imported by `BionicBrain`. Experimental checkpoints belong in `archive/experiments/`.

Current general model (2026-09-19, round 4): 82k neurons, 1.52M active synapses, 3,865 words, phase `proliferation`, sha256 `60ea10284d8bfc96...`. It is one network trained in sequence: 12 short-phrase language categories → long-form documents → four rounds of a teacher-generated Python-programming corpus (`data/corpora/python_v1/`, 626 records over 14 categories including operators, common methods, paraphrase variants, forward/reverse/purpose triples and canonicalized correction pairs). Its measured position: language 13/13, Python paraphrase 7/10, Python reverse 4/6, never-taught advanced topics 0/4, long-form continuation 46 tokens to its own end marker.

Measured after the second Python round: language probe 13/13 (no catastrophic forgetting), Python family probe paraphrase 5/10 → **7/10**, reverse-direction 0/6 → **2/6**, never-taught advanced topics 0/4 (expected — coverage, not generalization), earlier plain Python probe 0/14 → 7/14, long-form continuation 46 tokens to its own end marker, ~20-50 s per answer at this size.

**Answer-form dilution is a real failure mode.** The third Python round fed back every failed eval item as several new phrasings, which taught the same fact as "方括号", "使用方括号[]", and "使用 方 括 号" - the motor readout then split its support and questions that used to be answered came back **empty**: paraphrase 7/10 → 6/10, reverse 4/6 → 3/6. That round was rejected and archived (`brian_python_v3_rejected_answer_dilution.pkl`). The fix is canonical answers: one shortest form per knowledge point. `tools/canonicalize_corpus.py` collapses a verbose answer onto the short form only when it appears as a contiguous token run and the surrounding tokens are filler (a dry run on the padded corpus proposed 4 sane rewrites; a naive substring rule proposed 80 and produced garbage like `使用 def 关键字…` → `键`). Corpus prompts now demand the shortest canonical answer. With that fix the re-run restored 7/10 paraphrase and 4/6 reverse (`brian_python_v4`), matching the round-2 baseline while covering more knowledge.

**How to build a topic probe** (`tools/build_topic_eval.py`): generate three families and tag them, because one accuracy number conflates them — `paraphrase` (the fact *was* taught, the wording is new: this is the generalization measure), `reverse` (same fact asked from the other direction), `new_topic` (never taught: a coverage/calibration measure, expected to fail and not evidence about generalization). Literal repeats of training questions are dropped automatically.

Superseded checkpoints from 2026-09-18 (repair lineage, teacher runs, pre-fix mixed runs, per-stage snapshots, `brian_mix_v2`, and the older degraded `brian_neural_ocean`) are in `archive/experiments/superseded_20260919/`.

- One `respond` episode costs roughly 15-30 s at this size, so bound runs explicitly (`--max-episodes`) and iterate; do not launch unbounded training without a stop mechanism.
- Long-running trainers follow the `tools/selftrain_logic.py` conventions: a stop file (`data/selftrain_*.stop`), an atomically written status JSON (`data/selftrain_*.status.json`), periodic `--save-every` checkpoints, optional `--snapshot-every`, and a final save on interrupt.
- Before assuming a run is active, verify the PID recorded in the status file is actually alive; stale `"running": true` entries exist in `data/`.
- Persist RNG state and pass explicit `--seed` for reproducible curricula.
- Always record before/after evidence: keep evaluation output in `runs/metrics/` (JSON) and logs in `runs/logs/`. Never claim improved accuracy without a measured baseline from the same checkpoint lineage.
- Save to a new checkpoint name before overwriting `brian_neural_ocean.pkl` when an experiment is exploratory.

## Recall, Retention and Generalization Invariants

These mechanisms are load-bearing; several of them fix measured failure modes. Do not "simplify" them away without re-running `tools/diagnose_memory.py`.

- **Retrieval spans both memory tiers.** `Hippocampus.episodic_buffer` is the recency index; `episodic_store` is the long-term index. `retrieve()` searches both, so a lesson stops disappearing the moment 1000 newer events arrive. Eviction is importance-based, and `consolidate()` (sleep replay) marks events without deleting them (`hippocampus_retain_consolidated`).
- **`respond()` always runs the network.** It clears queues *before* `input_text()`, re-activates the cue for the whole `recall_window_ms`, and steps the simulation even with no hippocampal hit. Clearing after presentation (the old order) silently dropped the prompt and left the episodic pointer as the only possible answer path.
- **The hippocampus is a bias, not the answer.** A strong pointer (similarity ≥ 0.8) still reads out its targeted motor assembly for stability; otherwise the best-matching stored cue and the learned transitions act as a multiplicative prior that gates association evidence (`sequence_prior_weight`). Zero support still means no candidate, so unknown input stays rejected.
- **Association synapses are protected.** `_get_or_create_association` tags reused developmental synapses with `metadata["association"]`; `_normalize_inputs()` skips tagged synapses and `_scale_synapses()` floors them at `association_floor × learned_weight` (0.85 × 6.0 = 5.1). The decoder needs ≥ `learned_weight × 0.55` (3.3) to count support — normalization used to push every fresh association to ~3.15, so teaching new material erased recall of the old.
- **Engram cells survive the apoptosis sweep.** Cells with a `word` label are silent by design between uses; `lifecycle()` refuels them instead of killing them (`protect_binding_cells`).
- **Generalization is Hebbian, not backprop.** `_observe_cooccurrence()` counts co-activated tokens and `_share_sensory_cells()` copies a few sensory cells between them (`cooccurrence_min`, `sdr_share_cells`, `sdr_share_budget`). This is what lets a paraphrase with no shared surface token reach a learned assembly.
- **Long output is generated, not looked up.** `respond_sequence()` recalls a phrase, feeds it back as context, and continues through the next learned lesson; a repetition guard stops degenerate loops. `learn_sequence()` records token transitions.
- **Speed depends on two knobs.** `recall_window_ms` (default 20 ms) sets the simulated recall window and `active_hold_ms` (default 120 ms) bounds which cells keep being advanced. `_spike_times_since()` early-exits the time-ordered spike deques instead of scanning 2000 entries per cell.
- **Two performance invariants found during the first mixed run.** (1) Association growth is capped per postsynaptic cell (`association_fan_in_cap`, default 240): `_get_or_create_association` returns `-1` at capacity and callers skip, otherwise stream learning tripled synapses to 575k and answers slowed to ~30 s. (2) Eligibility traces decay in batches (`eligibility_decay_interval`, default 10) with a bounded set (`eligibility_set_limit`): decaying every synapse on every micro-step cost ~140M operations per answer (profiled at 60-90 s), while `decay**interval` applied every `interval` steps is mathematically identical. `recall_window_ms` is 12 ms and `active_hold_ms` 25 ms with `active_set_limit` 96 for the same reason — a cell that lingers in the active set is re-integrated every micro-step.
- **Per-neuron synapse budgets come from measurement.** `synapse_out_cap_per_neuron = 1982` and `synapse_in_cap_per_neuron = 284` are the maxima measured on `brian_mix_v2` (p99: 344 / 189). `_add_synapse` refuses to create a synapse once a cell is at budget and returns `-1`; a cell whose synapses are ≥30% strong excitatory (|w| ≥ `strong_weight_threshold`) may exceed the base budget by `strong_synapse_bonus` (25%). Re-measure before changing these numbers.
- **Continuation is expectation-driven, not bag-of-context.** Long output uses `SequentialPredictor.best_next()` (strict longest-context backoff) to prime the expected next plan; the primed plan counts as evidence (`primed_support_bonus`) and as full cue coverage, and generation disables the summed transition prior (`use_sequence_prior=False`). `observe_stream` feeds the transition model the whole document at once — chunking at sentence ends dropped cross-sentence transitions ("后果。" → 他) and broke continuation. Only the `recall_cue_tokens` window is re-simulated during recall; holding a whole long prompt active cost ~60 s per generated token.
- **Output length is decided by the network, not by code.** `respond_sequence()` takes `max_tokens=None`; `generation_safety_cap` (256) is only a runaway fuse. It stops by itself on the learned end marker (`generation_end_tokens`: `</s>`/`<eos>`, which the transition model emits at the end of every read document), on an unsupported readout, on repetition, when the expectation stays below `generation_min_probability` for `generation_low_confidence_steps`, or on `generation_fatigue_limit` satiation. `brain.last_generation_info` reports `stop_reason`. Measured: after reading a six-sentence story, prompting with the first sentence emitted the remaining 46 tokens verbatim and stopped with `stop_reason=model-end-marker` (no length given); the same document with a 40-token fuse stopped at `safety-cap` mid-sentence, which is why a fixed length was truncating output.

Measured on a fresh minimal brain with `tools/diagnose_memory.py` (before → after): recalled probes after 1200 interfering lessons 0/4 → 4/4; cortex-only recall (hippocampus disabled) 0/4 → 4/4; paraphrase recall 2/4 → 3/4; cross-form generalization 1/3 → 2/3; generated continuation 1 token → 7 tokens; association weights 3.15 (120/120 below the 3.3 floor) → 5.6-6.4 (0 below floor).

## Mixed-Input Training Pipeline

Knowledge comes from mixed human-like input, not from a QA set (see `docs/EMERGENCE.md`). Two tools implement the pipeline:

- `tools/build_training_mix.py` — generates the corpus with the local teacher into `data/corpora/mix_v1/<category>.jsonl`. Twelve categories span the acquisition typology: channel (auditory/visual), source (caregiver/peer/teacher/media/self), interactivity (one-way / interactive / grounded), genre (naming, narrative, instruction, exposition, meta-language), and stage (infant → adult). Every record carries `stage/modality/kind/genre/source` metadata. `--write-probe <path>` also emits a held-out QA probe.
- `tools/train_mixed_curriculum.py` — walks the stages in order and presents each category the way a human meets it: `observe_stream` for one-way input (self-supervised), `perceive(run=False)` for channel development and role-filler frames, `learn_pair` + teacher-judged feedback for interactive QA, a `--cross-modal-ratio` share of texts replayed through the other channel, and `sleep` consolidation between categories. Stop file `data/train_mixed.stop`, status `data/train_mixed.status.json`, per-stage checkpoints (`brian_mix_<stage>.pkl`) plus the final `--save` target.
- Long-form categories `longform_story` / `longform_exposition` (150-400 characters per document) drive continuation: `--context-window` sets how many tokens each prediction conditions on, and `--continuation-per-epoch` rehearses generation (generate a continuation, then read the real continuation).

Measured on `brian_repair2.pkl` → `brian_mix_v2.pkl` (3 epochs, 360 records, 4191 tokens, 79 cross-modal presentations, 30 QA pairs, 15 sleeps): judged accuracy on the 13-item probe 3/13 → 13/13, dialogue 0/10 → 10/10, mean answer latency 47 s → 7.9 s. The probe's dialogue items are part of the curriculum, so this measures learning; the held-out generalization numbers come from `tools/emergence_report.py`.

Continuation, after `tools/probe_recall.py --continue` and the mixed run: reading a story and prompting with its first sentence produces the true next tokens (`他 急 忙 跳 下 床 ， 胡 乱 套 上 衬`), judged `partial` (truncated mid-sentence) rather than the 0 tokens measured before the expectation/coverage fixes. `brian_long_v1.pkl` (2 epochs over the 132-record corpus with long-form, 9384 tokens, 66 cross-modal, 4 continuation rehearsals) reproduces the same continuation; it is larger (61k neurons, 1.07M synapses) and costs ~34 s per generated token, so bound generation runs.

## Judging Generalization and Emergence

`tools/generalization_probe.py` is the instrument for "did it generalize, or did it get lucky"; `docs/EMERGENCE.md` defines the protocol. Three rules are load-bearing:

1. **Verify the probe is unseen.** Every question is checked against `data/corpora/**/*.jsonl`; a probe whose *question* string appears in a corpus is reported as `seen_in_corpora` and must not be counted as generalization.
2. **Always run the controls.** `--ablate-schema` disables the binding/association readout (proves a mechanism is causal: one-shot and composition drop 1.00 → 0.00 while association-based paraphrase stays 1.00), and `--untrained` scores a fresh brain built with the same code, which separates innate scaffolding from learned ability (an untrained brain hallucinates on unknown words — 0.00 on calibration — while the trained model refuses all of them).
3. **Separate the two kinds of claim.** Scaffolding abilities (one-shot binding, role composition) work on an untrained brain but collapse under ablation. Learned abilities (calibration, consolidated QA, retention, continuation) are absent in the untrained brain. An emergence claim needs both: the mechanism's ablation gap *and* a training gap on unseen items.

Current numbers (2026-09-19, general model, all items verified unseen, judged by the local teacher): paraphrase 1.00, one-shot facts 1.00 (ablation 0.00, untrained 1.00), composition 1.00 (ablation 0.00), cross-modal 1.00, unknown-word calibration 1.00 (untrained 0.00 — it invents answers), 13-item consolidated QA 13/13 at ~20 s per answer, long-form continuation 46 tokens to its own end marker.

## Answer Evaluation Policy

**Every answer is judged by the local teacher model, never by string equality.**
The network emits token lists, so answers that mean the same thing look different
("地 湿" vs "地湿", "thank you" vs "谢谢", "小 美" vs "小美"). Character
comparison is not evidence of quality and must not be used as a score.

- Judge implementation: `LlmTeacherAgent.judge(question, candidate, reference=None)` in `src/bionic_brain/agents/llm_teacher.py`. It returns `{verdict: correct|partial|wrong, score: 0.0|0.5|1.0, reason, judge, elapsed_ms}`, where `judge` records `model@base_url`.
- Scoring tools use it directly: `tools/eval_checkpoint.py` (headline `judged_correct` / `mean_score`), `tools/diagnose_memory.py`, `tools/emergence_report.py`, `tools/train_llm_teacher.py`, `tools/selftrain_logic.py`.
- Token-level numbers (`exact_match`, `token_f1`) may still be reported, but only under names ending in `_aux` and never as the headline metric.
- Reports must record the judge identity and the number of offline fallbacks, so a report produced without the teacher is visibly degraded rather than silently different.
- If the endpoint is unreachable and the caller allows it, `judge()` falls back to token comparison and labels the result `judge="offline-fallback"`. Treat those numbers as invalid for claims.
- Frozen headline policy: report `judged_correct` / `mean_score`; support emergence claims with the unseen-item check, mechanism ablation, and untrained control; support language portability with a corpus-only cross-language probe.
- Unit tests are exempt: `tests/` asserts code contracts (does `respond` return the taught phrase, does the decoder reject unknown input). They must run offline with stub clients, and they must not require `127.0.0.1:1919`.
- Do not add new scoring heuristics to tools. If a judgement is needed, call the teacher.

## Local LLM Teacher

A local OpenAI-compatible chat server serves the teacher model:

- Base URL: `http://127.0.0.1:1919/v1` (chat completions at `/chat/completions`, models at `/models`).
- Model id: `Qwen3.6-35B-A3B-NVFP4` (context 262144). No API key required.
- Env overrides: `BIONIC_TEACHER_BASE_URL`, `BIONIC_TEACHER_MODEL`, `BIONIC_TEACHER_API_KEY`.

Use `src/bionic_brain/agents/llm_teacher.py` (`ChatTeacherClient`, `LlmTeacherAgent`) instead of hand-rolled HTTP:

```powershell
# the teacher drafts / grades; Brian still learns through its own local rules
.\.venv\Scripts\python.exe tools\train_llm_teacher.py --dry-run --topic "因果推理：如果…那么…" --draft-size 4
.\.venv\Scripts\python.exe tools\train_llm_teacher.py --max-episodes 20 --save-every 5
.\.venv\Scripts\python.exe tools\probe_llm_teacher.py --max-tokens 2048
```

Server-specific behaviour worth remembering:

- It is a reasoning model: the answer arrives in `message.content`, the thinking block in `message.reasoning_content`. When the token budget is too small, `content` is empty and `finish_reason` is `length`; `ChatTeacherClient` falls back to the thinking block and raises `TeacherUnavailable` with a hint when both are empty.
- Keep `max_tokens >= 4096` for drafting prompts and keep the 600 s timeout; a 4-item draft takes roughly 1-4 minutes on this machine.
- The endpoint is local and optional: trainers must fall back to `data/corpora/*.jsonl` when it is offline (`tools/train_llm_teacher.py --fallback-corpus`).
- Teacher output is training data, never runtime behaviour: no teacher call may appear inside `BionicBrain.respond`, `step`, or any mechanism package.

## Coding Style & Naming Conventions

Python 3.13, four-space indentation, `snake_case` functions/files, `PascalCase` classes, `UPPER_SNAKE_CASE` constants, type hints on public APIs. Keep modules mechanism-specific; `brain.py` must not become a monolith again. Standard library first — do not add dependencies without a clear reason.

## Testing Guidelines

Use unittest or pytest-compatible tests named `test_*.py` under `tests/` (target: `tests/unit/`, `tests/integration/`, `tests/smoke/`, `tests/acceptance/`). Cover anatomy, morphogen boundaries, HH/LIF dynamics, event synapses, STDP/DA, tracts, hippocampal gating, language I/O, checkpoint compatibility, teacher-agent JSON handling, and the recall invariants in `tests/test_emergence.py` (cortex-only recall, retention past the buffer, association floors, co-activation sharing, sequence generation). Teacher tests must run offline with stub servers — never require `127.0.0.1:1919` in the default suite. Run the smoke test and the full suite before review.

## Commit & Pull Request Guidelines

Concise imperative commit subjects, for example `Add hippocampal retrieval gate`. Pull requests describe the PRD/FR change, the commands run, smoke/test output, persistence compatibility notes, and simulation metrics. Do not claim an acceptance criterion without reproducible evidence.

## Dedicated Training Agent

This repository is maintained by a dedicated trainer agent whose only job is to improve the network on measured tasks.

Working agreement:

1. Establish a baseline first (`main.py --mode evaluate`, or a fixed held-out question set) and store it in `runs/metrics/` before changing any checkpoint.
2. Train in bounded increments with the existing tools; prefer `tools/train_llm_teacher.py` for teacher-driven curricula and `tools/train_brian_curriculum.py` for staged corpora.
3. Re-measure with the same evaluation set after each increment and keep only changes that improve the metric; otherwise restore the previous checkpoint.
4. Report `episodes`, `correct`, `corrected`, `accuracy`, `neurons`, `active_synapses`, `words`, elapsed time, and the checkpoint path.
5. Stop cleanly: check the stop file before each episode, save before exit, and leave a status JSON that reflects reality.
6. Never bypass the architecture constraints to raise a metric (no lookup tables, no backprop, no teacher calls inside the network).
