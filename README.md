# BionicBrain / Brian

A developmental whole-brain spiking neural network with explicit dendrite / soma / axon / synapse structure, neuromodulation, hippocampal memory, and text-in → motor-decoding-out language behavior.

Brian is designed as a neural-first system: language is learned from corpora and interaction, not encoded as grammar tables, question templates, relation word lists, or static answer lookup tables.

> Latest code direction: symbolic language fallbacks have been removed from the active inference path. Experimental checkpoints are local artifacts, not acceptance targets.

## Design goals

- **Bionic structure first.** Neurons explicitly model dendrite, soma, axon, and synapse.
- **Local plasticity.** Learning uses eligibility traces, STDP-like updates, reward modulation, prediction error, hippocampal replay, and sleep consolidation.
- **Memory across timescales.** Fast hippocampal episodic memory and slower cortical association memory.
- **Learned language.** Tokenization, relation discovery, identity, and surface-form generalization are learned from corpora.
- **No hard-coded language knowledge in active inference.** Linguistic knowledge must come from data, not Python tables.
- **Differentiable tools are allowed when biologically mapped.** PyTorch components model dendritic integration, somatic thresholding, axonal delay, and synaptic plasticity—not a generic opaque MLP.

## Core architecture

```text
src/bionic_brain/
├── brain.py                    BionicBrain orchestrator
├── cognition.py                grounding, binding, intent, sequence, replay
├── atlas/                      33-region atlas and developmental seeds
├── cells/                      HH + LIF neurons
├── synapses/                   event synapses, STDP, DA
├── morphogens/                 developmental morphogen fields
├── tracts/                     white-matter pathways
├── modulation/                 neuromodulators and drives
├── memory/
│   ├── hippocampus.py          episodic buffer/store/replay
│   └── learned_schemas.py      learned role/frame memory
├── language/
│   ├── subwords.py             learned BPE / structural tokenization
│   ├── relations.py            distributional relation discovery
│   ├── features.py             learned context features
│   ├── dialogue.py             learned boundary handling
│   └── corpus.py               corpus loader
├── nn/                         PyTorch dendrite/soma/axon/synapse modules
└── visualization/              3D inspection tools
```

Hard-coded symbolic language modules are quarantined outside active inference and are not imported by `BionicBrain`.

## Setup

Use Python 3.13 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
```

The default environment includes NumPy, Matplotlib, PyTorch, `pypdf`, and `cryptography`.

For NVIDIA CUDA PyTorch, install the CUDA wheel matching your driver, for example:

```powershell
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu128
```

## Chat with a trained checkpoint

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe main.py --mode chat --load data\checkpoints\neural_identity_surface_v18.pkl
```

Useful chat commands:

```text
!help
!show
!recall <text>
!reward 0.5
!sleep
!save <path>
!exit
```

If the model answers `<unk>`, correct it by entering the desired answer at the feedback prompt, then repeat the question.

## Training

### Mixed corpus / interaction training

Prepare JSONL records with either:

```json
{"text": "..."}
```

or:

```json
{"question": "...", "answer": "..."}
```

Then run:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe tools\train_mixed_curriculum.py `
  --load data\checkpoints\neural_identity_surface_v18.pkl `
  --save data\checkpoints\next_experiment.pkl `
  --mix-dir data\corpora\your_corpus `
  --epochs 1 `
  --sleep-replay 20 `
  --interactive-per-epoch 3
```

The trainer uses:

- `observe_stream()` for self-supervised text;
- `perceive()` for multimodal grounding;
- `learn_pair()` plus teacher feedback for interactive QA;
- cross-modal replay;
- sleep consolidation.

### Paper corpus ingestion

Extract text from research papers:

```powershell
.\.venv\Scripts\python.exe tools\ingest_papers.py `
  --source E:\path\to\papers `
  --output-dir data\corpora\papers_v1 `
  --max-pages 8 `
  --chunk-size 720 `
  --chunk-overlap 80 `
  --max-chunks-per-file 3
```

Supported inputs: `.pdf`, `.md`, `.txt`, and `.docx`.

### Teacher-driven training

A local OpenAI-compatible model may draft and judge training data, but it is never used inside Brian's inference path.

```powershell
.\.venv\Scripts\python.exe tools\train_llm_teacher.py `
  --load data\checkpoints\neural_identity_surface_v18.pkl `
  --save data\checkpoints\teacher_trained.pkl `
  --topic "scientific hypothesis generation" `
  --max-episodes 4
```

## Tests

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_prd.py
```

Current status:

```text
87 tests, OK
BionicBrain smoke: PASS
```

## Evaluation policy

Do not score Brian with raw string equality.

For meaningful evaluation:

1. Use unseen questions;
2. Include paraphrases and reverse questions;
3. Run ablation controls;
4. Run an untrained control;
5. Use a semantic judge when available.

Token overlap or exact match may be reported only as auxiliary metrics.

## Experimental checkpoints

Checkpoints are local experiment artifacts and are not committed.

Examples from recent neural-first runs:

```text
neural_identity_surface_v18.pkl    identity and surface-form generalization
neural_papers_v16.pkl              paper-text training plus LogiQA transfer
neural_science_v4.pkl              science mission corpus
neural_python_v3.pkl               Python knowledge corpus
```

Checkpoints can be large. Keep them under `data/checkpoints/` locally or archive them outside Git.

## Documentation

- `docs/PRD.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/EMERGENCE.md`
- `docs/TODO_ZERO_HARDCODED.md`
- `docs/REALTIME_3D_VISUALIZATION.md`
