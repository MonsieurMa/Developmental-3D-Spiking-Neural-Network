r"""Train Brian on a human-like mixed input stream.

Instead of a QA corpus, the loop walks the acquisition stages and presents each
category the way a human would meet it:

* one-way input  -> ``observe_stream`` (self-supervised prediction learning)
* grounded scene -> ``perceive`` (channel development + role-filler frames)
* interactive QA -> ``learn_pair`` plus teacher-judged feedback
* cross-modal    -> a share of visual texts is also heard, and vice versa
* consolidation  -> ``sleep`` between categories

Conventions match the other trainers: stop file, atomically written status JSON,
periodic checkpoints, and metrics under ``runs/metrics/``.

Usage::

    .\.venv\Scripts\python.exe tools\train_mixed_curriculum.py `
        --load data/checkpoints/brian_repair2.pkl `
        --save data/checkpoints/brian_mix_v1.pkl --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain import BionicBrain
from bionic_brain.agents.llm_teacher import LlmTeacherAgent
from bionic_brain.config.defaults import BionicConfig
from bionic_brain.language.subwords import SubwordTokenizer
from bionic_brain.language.subwords import SubwordTokenizer

STAGE_ORDER = ["infant", "toddler", "child", "school", "adult"]
STAGE_CHECKPOINT = {
    "infant": "brian_mix_infant.pkl",
    "toddler": "brian_mix_toddler.pkl",
    "child": "brian_mix_child.pkl",
    "school": "brian_mix_school.pkl",
    "adult": "brian_mix_adult.pkl",
}


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_mix(directory: Path) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get("prompt") and not item.get("question"):
                    item["question"] = item["prompt"]
                if item.get("response") and not item.get("answer"):
                    item["answer"] = item["response"]
                items.append(item)
        if items:
            records[path.stem] = items
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Brian on mixed human-like input")
    parser.add_argument("--load", default="data/checkpoints/brian_repair2.pkl")
    parser.add_argument("--fresh", action="store_true", help="grow a new candidate instead of loading --load")
    parser.add_argument("--subword-state", help="learned BPE state JSON for a fresh candidate")
    parser.add_argument("--learned-relations", action="store_true", help="grow a fresh candidate with learned relation discovery")
    parser.add_argument("--save", default="data/checkpoints/brian_mix_v1.pkl")
    parser.add_argument("--mix-dir", default="data/corpora/mix_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sleep-replay", type=int, default=40)
    parser.add_argument("--interactive-per-epoch", type=int, default=4, help="teacher-judged QA probes per epoch")
    parser.add_argument("--cross-modal-ratio", type=float, default=0.25)
    parser.add_argument("--context-window", type=int, default=8, help="tokens of context each prediction conditions on")
    parser.add_argument("--continuation-per-epoch", type=int, default=2, help="long-text continuation rehearsals per epoch")
    parser.add_argument("--continuation-tokens", type=int, default=64, help="safety cap; generation stops by itself")
    parser.add_argument("--max-minutes", type=float, default=0.0, help="0 means run every configured epoch")
    parser.add_argument("--stop-file", default="data/train_mixed.stop")
    parser.add_argument("--status-file", default="data/train_mixed.status.json")
    parser.add_argument("--metrics-file", default="")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--no-stage-checkpoints", dest="stage_checkpoints", action="store_false", default=True,
                        help="skip per-stage snapshots (keeps exactly one model in data/checkpoints)")
    parser.add_argument("--no-judge", action="store_true", help="skip teacher feedback for interactive probes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)

    stop_file = ROOT / args.stop_file
    status_file = ROOT / args.status_file
    save_path = ROOT / args.save
    if stop_file.exists():
        print(f"stop file exists: {stop_file}", flush=True)
        return 0

    if args.fresh:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            raise FileExistsError(f"refusing to overwrite fresh candidate: {save_path}")
        brain = BionicBrain(seed=args.seed, develop=True)
        if args.subword_state:
            state_path = ROOT / args.subword_state
            state = json.loads(state_path.read_text(encoding="utf-8"))
            brain.subword_tokenizer = SubwordTokenizer.from_state(state)
            brain.subwords_enabled = True
        brain.config = replace(
            brain.config,
            use_subword_tokenizer=brain.subwords_enabled,
            use_learned_relations=bool(args.learned_relations),
            stream_context_tokens=max(1, int(args.context_window)),
        )
    else:
        if args.subword_state:
            raise ValueError("a trained checkpoint must keep the tokenizer it was born with")
        brain = BionicBrain.load(ROOT / args.load)
        # BionicConfig is frozen: rebuild it instead of mutating.
        brain.config = replace(brain.config, stream_context_tokens=max(1, int(args.context_window)))
    # BionicConfig is frozen: rebuild it instead of mutating.
    brain.config = replace(brain.config, stream_context_tokens=max(1, int(args.context_window)))
    teacher = None if args.no_judge else LlmTeacherAgent(cache_dir=ROOT / "data/corpora/llm_generated")
    if teacher is not None:
        print(f"judge={json.dumps(teacher.describe(), ensure_ascii=False)}", flush=True)

    mix = load_mix(ROOT / args.mix_dir)
    if not mix:
        print(f"no corpus found in {args.mix_dir}; run tools/build_training_mix.py first", flush=True)
        return 2
    grouped: dict[str, list[tuple[str, dict]]] = {}
    for category, items in mix.items():
        for item in items:
            grouped.setdefault(item.get("stage", "child"), []).append((category, item))

    started = time.time()
    deadline = started + args.max_minutes * 60.0 if args.max_minutes > 0 else None
    stats = {"epochs": 0, "records": 0, "tokens": 0, "streams": 0, "perceptions": 0, "qa_learned": 0,
             "interactive": 0, "judged_correct": 0, "judged_partial": 0, "corrected": 0,
             "sleeps": 0, "checkpoints": 0, "cross_modal": 0, "continuations": 0}
    continuation_log: list[dict] = []
    per_category: dict[str, dict] = {}

    def write_status(event: str, extra: dict | None = None) -> None:
        payload = {
            "running": True,
            "pid": os.getpid(),
            "unix_time": time.time(),
            "elapsed_sec": round(time.time() - started, 1),
            "load": "" if args.fresh else args.load,
            "fresh": args.fresh,
            "subword_state": args.subword_state,
            "learned_relations": args.learned_relations,
            "save": args.save,
            "stats": stats,
            "per_category": per_category,
            "last_event": {"event": event, **(extra or {})},
            "brain": {k: v for k, v in brain.show().items() if k != "last_response"},
        }
        atomic_json(status_file, payload)

    if args.fresh:
        print(
            f"fresh=true seed={args.seed} subwords={brain.subwords_enabled} "
            f"merges={len(brain.subword_tokenizer.merges)}",
            flush=True,
        )
    else:
        print(f"loaded={args.load}", flush=True)
    print(f"categories={sorted(mix)} records={sum(len(v) for v in mix.values())}", flush=True)

    def present_text(text: str, meta: dict, category: str) -> dict:
        modality = meta.get("modality", "visual")
        result = brain.observe_stream(text, modality=modality)
        # Development (peripheral channels, schema frames, dialogue context) is
        # cheap when we skip the 25 ms simulation inside perceive().
        brain.perceive(text, modality=modality, run=False)
        stats["streams"] += 1
        stats["tokens"] += result["tokens"]
        stats["perceptions"] += 1
        bucket = per_category.setdefault(category, {"records": 0, "tokens": 0, "surprise": 0.0, "prediction_hits": 0, "predictions": 0})
        bucket["records"] += 1
        bucket["tokens"] += result["tokens"]
        bucket["surprise"] += result["mean_surprise"]
        bucket["prediction_hits"] += round(result["prediction_accuracy"] * result["predictions"])
        bucket["predictions"] += result["predictions"]
        if rng.random() < args.cross_modal_ratio:
            other = "auditory" if modality == "visual" else "visual"
            brain.observe_stream(text, modality=other)
            stats["cross_modal"] += 1
        return result

    try:
        for epoch in range(1, max(1, args.epochs) + 1):
            stats["epochs"] = epoch
            for stage in STAGE_ORDER:
                entries = grouped.get(stage)
                if not entries:
                    continue
                for category, item in entries:
                    if stop_file.exists():
                        print(f"stop file detected: {stop_file}", flush=True)
                        raise KeyboardInterrupt
                    if deadline and time.time() > deadline:
                        print("time budget reached", flush=True)
                        raise KeyboardInterrupt
                    if "text" in item:
                        present_text(item["text"], item, category)
                    question, answer = item.get("question"), item.get("answer")
                    if question and answer:
                        brain.learn_pair(question, answer, reward=1.0)
                        brain.observe_stream(f"{question}。{answer}。", modality=item.get("modality", "auditory"))
                        stats["qa_learned"] += 1
                    stats["records"] += 1
                    write_status("record", {"epoch": epoch, "stage": stage, "category": category})
                replay = brain.sleep(replay_count=args.sleep_replay)
                stats["sleeps"] += 1
                print(
                    f"epoch={epoch} stage={stage:<8} records={stats['records']:<5} tokens={stats['tokens']:<7} "
                    f"words={len(brain.bindings):<5} neurons={len(brain.neurons):<6} replayed={replay['replayed']}",
                    flush=True,
                )
                if args.stage_checkpoints:
                    stage_name = STAGE_CHECKPOINT.get(stage, f"brian_mix_{stage}.pkl")
                    brain.save(save_path.parent / stage_name)
                    stats["checkpoints"] += 1
            # Long-text continuation rehearsal: the network generates a
            # continuation of a long document, then the real continuation is
            # presented, so the next pass can predict further.
            longform = [
                (category, item)
                for category, items in mix.items()
                for item in items
                if category.startswith("longform") and item.get("text")
            ]
            rng.shuffle(longform)
            for category, item in longform[: max(0, args.continuation_per_epoch)]:
                head, sep, tail = item["text"].partition("。")
                if not sep:
                    continue
                produced = brain.respond_sequence(head, max_tokens=args.continuation_tokens, max_len_per_step=8)
                brain.observe_stream(tail, modality=item.get("modality", "visual"))
                stats["continuations"] += 1
                continuation_log.append({
                    "epoch": epoch, "category": category, "prompt": head[:60],
                    "produced": produced, "length": len(produced),
                })
                print(f"  continuation len={len(produced)} prompt={head[:24]!r} produced={' '.join(produced)!r}", flush=True)

            # Interactive feedback: ask, let the teacher judge, then reward or correct.
            qa_pool = [
                item for category, items in mix.items() for item in items
                if item.get("question") and item.get("answer")
            ]
            rng.shuffle(qa_pool)
            for item in qa_pool[: max(0, args.interactive_per_epoch)]:
                question, expected = item["question"], item["answer"]
                got = " ".join(brain.respond(question, max_len=8))
                verdict = "correct"
                if teacher is not None:
                    judgement = teacher.judge(question, got, expected)
                    verdict = judgement["verdict"]
                stats["interactive"] += 1
                if verdict == "correct":
                    brain.reward(0.5)
                    stats["judged_correct"] += 1
                else:
                    if verdict == "partial":
                        stats["judged_partial"] += 1
                    brain.correct(question, expected, reward=1.0)
                    brain.reward(0.8)
                    stats["corrected"] += 1
                print(f"  interactive Q={question!r} expected={expected!r} got={got!r} verdict={verdict}", flush=True)
            brain.save(save_path)
            stats["checkpoints"] += 1
            print(f"epoch={epoch} checkpoint={save_path}", flush=True)
    except KeyboardInterrupt:
        print("interrupted; saving", flush=True)

    brain.save(save_path)
    metrics_path = Path(args.metrics_file) if args.metrics_file else (
        ROOT / "runs" / "metrics" / f"mix_training_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%SZ')}.json"
    )
    report = {
        "argv": sys.argv[1:],
        "load": "" if args.fresh else args.load,
        "fresh": args.fresh,
        "subword_state": args.subword_state,
        "learned_relations": args.learned_relations,
        "tokenizer": brain.subword_tokenizer.diagnostics(),
        "save": args.save,
        "elapsed_sec": round(time.time() - started, 1),
        "stats": stats,
        "per_category": per_category,
        "continuations": continuation_log,
        "brain": {k: v for k, v in brain.show().items() if k != "last_response"},
        "judge": teacher.describe() if teacher is not None else None,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_status("finished")
    payload = json.loads(status_file.read_text(encoding="utf-8"))
    payload["running"] = False
    atomic_json(status_file, payload)
    if stop_file.exists():
        stop_file.unlink()
    print(f"saved={save_path} metrics={metrics_path}", flush=True)
    print(f"stats={json.dumps(stats, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

