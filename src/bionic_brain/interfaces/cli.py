from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..brain import BionicBrain
from ..config.defaults import BionicConfig, DEFAULT_CONFIG
from ..language.corpus import load_corpus
from ..language.subwords import SubwordTokenizer
from ..visualization import setup_chinese_font, visualize_brain


HELP = """Commands:
  text                 train and respond
  !learn text          learn raw text as a self association
  !show                show compact brain state
  !region CODE         show one region rate
  !reward NUMBER       apply dopamine feedback to last response
  !confidence          show last response confidence and route
  !recall text          query hippocampus
  !sleep               replay/consolidate episodic memory
  !plot PATH           save a 3D figure
  !mature              enter mature phase
  !save [PATH]         save checkpoint
  !exit                leave chat
"""


def _read_corpus(path: str | None) -> list[dict]:
    if not path:
        raise ValueError("a corpus path is required; source code contains no answer table")
    return load_corpus(path)


def _attach_subwords(brain: BionicBrain, records: list[dict], *, enabled: bool, state_path: str | None) -> None:
    """Attach a corpus-learned tokenizer; fresh brains may fit it here."""

    if state_path:
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        brain.subword_tokenizer = SubwordTokenizer.from_state(state)
        brain.subwords_enabled = enabled
        return
    if enabled and not brain.bindings:
        brain.train_subword_tokenizer(
            (value for record in records for value in record.values() if isinstance(value, str)),
            enabled=True,
        )


def train(args) -> int:
    setup_chinese_font()
    checkpoint = Path(args.save or DEFAULT_CONFIG.default_checkpoint)
    # A larger integration step is used for the 100-item corpus training run;
    # the default simulation remains dt=0.02. The step is verified by the smoke test.
    config = BionicConfig(dt=args.dt, use_subword_tokenizer=bool(args.use_subwords or args.subword_state))
    brain = BionicBrain(config, fresh=True, seed=args.seed, develop=True)
    records = _read_corpus(args.corpus)
    _attach_subwords(
        brain,
        records,
        enabled=bool(args.use_subwords or args.subword_state),
        state_path=args.subword_state,
    )
    for epoch in range(1, args.epochs + 1):
        for record in records:
            brain.learn_pair(record["question"], record["answer"], reward=1.0)
        print(
            f"epoch={epoch:03d} examples={len(records)} target_extra={len(target)} "
            f"state={brain.show()}",
            flush=True,
        )
        brain.snapshot(checkpoint.parent)
    brain.save(checkpoint)
    print(f"saved: {checkpoint}", flush=True)
    return 0

def evaluate(args) -> int:
    brain = BionicBrain.load(args.load or DEFAULT_CONFIG.default_checkpoint)
    records = _read_corpus(args.corpus)
    token_hits = token_total = full_hits = 0
    for record in records:
        expected = brain.language_answer_tokens(record["answer"])
        predicted = brain.respond(record["question"], max_len=max(8, len(expected)))
        overlap = len(set(predicted) & set(expected))
        token_hits += overlap
        token_total += len(expected)
        full_hits += predicted == expected
        print(f"Q: {record['question']}  expected: {expected}  predicted: {predicted}")
    print(f"full={full_hits}/{len(records)} ({full_hits/len(records):.1%}) token={token_hits}/{token_total} ({token_hits/max(1,token_total):.1%})")
    return 0


def chat(args) -> int:
    setup_chinese_font()
    path = Path(args.load or DEFAULT_CONFIG.default_checkpoint)
    brain = BionicBrain.load(path) if path.exists() else BionicBrain(DEFAULT_CONFIG, fresh=True)
    print("BionicBrain interactive mode. Type !help for commands.")
    while True:
        try:
            line = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "!exit":
            break
        if line == "!help":
            print(HELP)
            continue
        if line.startswith("!learn "):
            tokens = brain.input_text(line[7:].strip(), learn=True)
            print("learned:", tokens)
            continue
        if line == "!show":
            print(json.dumps(brain.show(), ensure_ascii=False, indent=2))
            continue
        if line == "!confidence":
            print(json.dumps(brain.last_response_info, ensure_ascii=False, indent=2))
            continue
        if line.startswith("!region "):
            code = line.split(maxsplit=1)[1]
            print(code, brain.get_region_activity().get(code))
            continue
        if line.startswith("!reward "):
            brain.reward(float(line.split()[1]))
            continue
        if line.startswith("!recall "):
            events = brain.recall(line[8:].strip())
            print([(round(event.timestamp, 1), round(event.importance, 3)) for event in events])
            continue
        if line == "!sleep":
            print(brain.sleep())
            continue
        if line.startswith("!plot"):
            parts = line.split(maxsplit=1)
            visualize_brain(brain, parts[1] if len(parts) > 1 else "runs/figures/brain.png")
            continue
        if line == "!mature":
            brain.mature()
            print("mature")
            continue
        if line.startswith("!save"):
            parts = line.split(maxsplit=1)
            target = Path(parts[1]) if len(parts) > 1 else path
            print("saved:", brain.save(target))
            continue
        print("DEBUGLINE", repr(line), file=sys.stderr, flush=True)
        output = brain.respond(line)
        info = brain.last_response_info
        confidence = float(info.get("confidence", 0.0))
        print("脑:", " ".join(output) if output else "<unk>")
        print(f"[route={info.get('route', 'unknown')} confidence={confidence:.2f}]")
        feedback = input("反馈(Enter=好/确认, c=纠正, x=忽略, 或直接输入正确回答): ").strip()
        raw = feedback.lower()
        if feedback == "c":
            expected = input("期望: ").strip()
            if expected:
                brain.correct(line, expected, 1.0)
                result = brain.reward(0.8)
                saved = brain.save(path)
                print("纠正并强化", result["modulated"], "条突触；已保存", saved)
        elif feedback == "x":
            print("忽略")
        elif raw in {"good", "对", "正确", "right", "correct", "y", "yes", "好"}:
            result = brain.reward(0.8)
            print(f"奖励 {result['modulated']} 条突触")
        elif raw in {"bad", "wrong", "错", "不 对", "n", "no"}:
            result = brain.reward(-0.8)
            print(f"惩罚 {result['modulated']} 条突触；建议按 c 给出正确回答")
        elif feedback == "":
            if info.get("route") in {"hippocampus", "hippocampus-motor"} and confidence >= 0.70 and output:
                result = brain.reward(0.6)
                print(f"确认高置信回答，奖励 {result['modulated']} 条突触")
            else:
                print("低置信回答保持中性；输入正确回答可纠正")
        else:
            brain.correct(line, feedback, 1.0)
            result = brain.reward(0.8)
            saved = brain.save(path)
            print("已将输入作为正确回答进行纠正", result["modulated"], "条突触；已保存", saved)
    brain.save(path)
    print(f"saved: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BionicBrain developmental whole-brain spiking network")
    parser.add_argument("--mode", choices=["train", "inference", "chat", "evaluate"], help="run mode")
    parser.add_argument("--corpus", help="JSONL corpus with question/answer fields")
    parser.add_argument("--load", help="checkpoint to load")
    parser.add_argument("--save", help="checkpoint to save")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1907)
    parser.add_argument("--dt", type=float, default=0.02, help="integration step; training large corpora may use 0.1")
    parser.add_argument("--use-subwords", action="store_true", help="fit/use learned BPE units from --corpus")
    parser.add_argument("--subword-state", help="load a learned subword tokenizer state JSON")
    # Transitional compatibility with the old monolithic command line.
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "train" or args.fresh or args.train_only:
        return train(args)
    if args.mode == "evaluate" or args.evaluate:
        return evaluate(args)
    if args.mode == "inference":
        brain = BionicBrain.load(args.load or DEFAULT_CONFIG.default_checkpoint)
        if sys.stdin.isatty():
            prompt = input("prompt: ")
        else:
            raise ValueError("non-interactive inference requires an explicit prompt channel")
        print(" ".join(brain.respond(prompt)))
        return 0
    return chat(args)


if __name__ == "__main__":
    raise SystemExit(main())

