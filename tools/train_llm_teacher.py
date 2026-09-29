"""Autonomous trainer that teaches a BionicBrain checkpoint with a local chat model.

The local model (default ``http://127.0.0.1:1919/v1``) only drafts curricula and
grades answers. Every weight change still comes from Brian's own reward,
correction, relearning and hippocampal rehearsal paths.

Conventions match ``tools/selftrain_logic.py``: a stop file, an atomically
written status JSON and periodic checkpoint saves, so a supervising agent can
watch or halt the run without attaching to the process.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain import BionicBrain
from bionic_brain.agents.llm_teacher import LlmTeacherAgent, TeacherUnavailable
from bionic_brain.language.corpus import load_corpus


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teach Brian with a local OpenAI-compatible chat model")
    parser.add_argument("--load", default="")
    parser.add_argument("--save", required=True)
    parser.add_argument("--base-url", default=None, help="defaults to BIONIC_TEACHER_BASE_URL or 127.0.0.1:1919/v1")
    parser.add_argument("--model", default=None, help="defaults to BIONIC_TEACHER_MODEL")
    parser.add_argument("--timeout", type=float, default=None, help="per-request timeout in seconds")
    parser.add_argument("--enable-thinking", action="store_true", help="keep the teacher's thinking block (slower, more tokens)")
    parser.add_argument("--topics", default="", help="comma separated topics; default curriculum topics")
    parser.add_argument("--topic", action="append", default=[], help="single topic, repeatable")
    parser.add_argument("--draft-size", type=int, default=6, help="question/answer pairs requested per draft")
    parser.add_argument("--max-episodes", type=int, default=20, help="0 means run until stop file or Ctrl+C")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--snapshot-every", type=int, default=0, help="0 disables snapshotting")
    parser.add_argument("--rehearse-every", type=int, default=3, help="hippocampal rehearsal every N episodes")
    parser.add_argument("--rehearsal-ms", type=float, default=25.0)
    parser.add_argument("--sleep-replay", type=int, default=0, help="sleep replay count on exit")
    parser.add_argument("--fallback-corpus", default="data/corpora/logic_reasoning.jsonl", help="used when the teacher is offline")
    parser.add_argument("--cache-dir", default="data/corpora/llm_generated")
    parser.add_argument("--stop-file", default="data/selftrain_llm.stop")
    parser.add_argument("--status-file", default="data/selftrain_llm.status.json")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="only probe the teacher and print drafted items")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.base_url:
        os.environ["BIONIC_TEACHER_BASE_URL"] = args.base_url
    if args.model:
        os.environ["BIONIC_TEACHER_MODEL"] = args.model
    # Thinking is off by default on this reasoning model: drafting goes from
    # ~400 completion tokens to ~20 per request.
    os.environ["BIONIC_TEACHER_ENABLE_THINKING"] = "1" if args.enable_thinking else "0"

    topics = [topic.strip() for topic in args.topics.split(",") if topic.strip()] + list(args.topic)
    if not topics:
        raise ValueError("at least one --topic or --topic-file is required")

    teacher = LlmTeacherAgent(cache_dir=ROOT / args.cache_dir)
    if args.timeout:
        teacher.config.timeout = args.timeout
    print(f"teacher={json.dumps(teacher.describe(), ensure_ascii=False)}", flush=True)

    if args.dry_run:
        try:
            items = teacher.draft_curriculum(topics[0], args.draft_size)
        except TeacherUnavailable as error:
            print(f"dry-run failed: {error}", flush=True)
            return 2
        for item in items:
            print(json.dumps(item, ensure_ascii=False), flush=True)
        return 0

    stop_file = ROOT / args.stop_file
    status_file = ROOT / args.status_file
    load_path = ROOT / args.load
    save_path = ROOT / args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if stop_file.exists():
        print(f"stop file exists: {stop_file}", flush=True)
        return 0

    rng = random.Random(args.seed)
    brain = BionicBrain.load(load_path)
    teacher.brain = brain
    started = time.time()
    stats = {
        "started_at": started,
        "episodes": 0,
        "correct": 0,
        "corrected": 0,
        "drafted": 0,
        "teacher_graded": 0,
        "fallback_items": 0,
        "rehearsals": 0,
        "saves": 0,
        "snapshots": 0,
    }
    recent: list[int] = []

    def write_status(last_event: dict) -> None:
        window = recent[-20:]
        payload = {
            "running": True,
            "unix_time": time.time(),
            "elapsed_sec": round(time.time() - started, 3),
            "load": str(load_path),
            "save": str(save_path),
            "teacher": {"base_url": teacher.config.base_url, "model": teacher.config.model},
            "stats": stats,
            "accuracy_recent": round(sum(window) / len(window), 3) if window else 0.0,
            "last_event": last_event,
            "brain": {k: v for k, v in brain.show().items() if k != "last_response"},
        }
        atomic_json(status_file, payload)

    print(f"loaded={load_path}", flush=True)
    summary = {k: v for k, v in brain.show().items() if k != "last_response"}
    print(f"state={json.dumps(summary, ensure_ascii=False)}", flush=True)

    def next_draft(topic: str) -> list[dict[str, str]]:
        try:
            items = teacher.draft_curriculum(topic, args.draft_size)
            stats["drafted"] += len(items)
            return items
        except TeacherUnavailable as error:
            print(f"teacher unavailable ({error}); falling back to {args.fallback_corpus}", flush=True)
            records = load_corpus(ROOT / args.fallback_corpus)
            stats["fallback_items"] += len(records)
            return [{"question": record["question"], "answer": record["answer"]} for record in records]

    episode = 0
    queue: list[dict[str, str]] = []
    topic_index = 0
    try:
        while True:
            if stop_file.exists():
                print(f"stop file detected: {stop_file}", flush=True)
                break
            if args.max_episodes > 0 and episode >= args.max_episodes:
                print("max episodes reached", flush=True)
                break
            if not queue:
                topic = topics[topic_index % len(topics)]
                topic_index += 1
                queue = next_draft(topic)
                rng.shuffle(queue)

            item = queue.pop(0)
            prompt, expected = item["question"], item["answer"]
            t0 = time.perf_counter()
            answer = brain.respond(prompt, max_len=12)
            got = " ".join(answer)
            route = brain.last_response_info.get("route", "unknown")

            # Every answer is judged by the teacher model on meaning, never by
            # string comparison: the network emits token lists, so "地 湿" and
            # "地湿" are the same answer.
            review = teacher.judge(prompt, got, expected)
            verdict = review["verdict"]
            if verdict == "correct":
                stats["teacher_graded"] += 1

            if verdict == "correct":
                result = brain.reward(0.5)
                mode = "reward"
                stats["correct"] += 1
            else:
                brain.correct(prompt, expected, reward=1.0)
                result = brain.reward(0.8)
                mode = "correct"
                stats["corrected"] += 1

            rehearsal: dict = {}
            episode += 1
            recent.append(1 if verdict == "correct" else 0)
            if args.rehearse_every > 0 and (mode == "correct" or episode % args.rehearse_every == 0):
                rehearsal = brain.rehearse(prompt, expected, duration=args.rehearsal_ms)
                stats["rehearsals"] += 1

            stats["episodes"] = episode
            if args.save_every > 0 and episode % args.save_every == 0:
                brain.save(save_path)
                stats["saves"] += 1
            if args.snapshot_every > 0 and episode % args.snapshot_every == 0:
                brain.snapshot(save_path.parent)
                stats["snapshots"] += 1

            window = recent[-20:]
            if episode % 5 == 0 or mode == "correct":
                print(
                    f"episode={episode:05d} mode={mode} verdict={verdict} expected={expected} "
                    f"got={got or '<unk>'} route={route} new_syn={result.get('modulated')} "
                    f"acc20={round(sum(window) / len(window), 3)} "
                    f"elapsed={round(time.time() - started, 1)}s neurons={sum(n.alive for n in brain.neurons)} "
                    f"synapses={sum(s.active for s in brain.synapses)} words={len(brain.bindings)}",
                    flush=True,
                )

            write_status(
                {
                    "event": mode,
                    "episode": episode,
                    "prompt": prompt,
                    "expected": expected,
                    "got": got,
                    "verdict": verdict,
                    "route": route,
                    "rewarded_synapses": result.get("modulated"),
                    "rehearsal": rehearsal,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
                }
            )

            if args.delay > 0:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("keyboard interrupt; saving", flush=True)
    finally:
        if args.sleep_replay > 0:
            print(f"sleep_replay={brain.sleep(replay_count=args.sleep_replay)}", flush=True)
        brain.save(save_path)
        payload = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {}
        payload["running"] = False
        payload["stats"] = stats
        payload["unix_time"] = time.time()
        atomic_json(status_file, payload)
        print(f"saved={save_path} stats={json.dumps(stats, ensure_ascii=False)}", flush=True)

    if stop_file.exists():
        stop_file.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

