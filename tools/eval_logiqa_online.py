from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from bionic_brain import BionicBrain
from bionic_brain.agents.llm_teacher import LlmTeacherAgent


def fetch_logiqa(count: int) -> list[dict]:
    url = "https://media.githubusercontent.com/media/openai/evals/main/evals/registry/data/logiqa/logiqa.jsonl"
    with urllib.request.urlopen(url, timeout=60) as response:
        rows = [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]
    return rows[:count]


def user_prompt(row: dict) -> str:
    return next(message["content"] for message in row["input"] if message.get("role") == "user")


def ideal_answer(row: dict) -> str:
    return str(row.get("ideal", "")).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Brian on 10 online LogiQA items")
    parser.add_argument("--load", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-len", type=int, default=8)
    args = parser.parse_args()

    rows = fetch_logiqa(args.count)
    brain = BionicBrain.load(args.load)
    teacher = LlmTeacherAgent(cache_dir="data/corpora/llm_generated")
    rows_out = []
    for index, row in enumerate(rows, 1):
        prompt = user_prompt(row)
        ideal = ideal_answer(row)
        started = time.perf_counter()
        output = brain.respond(prompt, max_len=args.max_len)
        elapsed = time.perf_counter() - started
        output_text = " ".join(output)
        judgement = teacher.judge(prompt, output_text, ideal, offline_fallback=False)
        rows_out.append({
            "index": index,
            "question": prompt,
            "expected": ideal,
            "got": output,
            "got_text": output_text,
            "verdict": judgement["verdict"],
            "score": float(judgement["score"]),
            "reason": judgement.get("reason", ""),
            "judge": judgement.get("judge", ""),
            "elapsed_sec": round(elapsed, 3),
        })
        print(f"[{index:02d}/{len(rows)}] {judgement['verdict']} score={judgement['score']:.2f} got={output_text!r}", flush=True)
    summary = {
        "dataset": "OpenAI Evals LogiQA (online media.githubusercontent.com)",
        "checkpoint": str(args.load),
        "items": len(rows_out),
        "judged_correct": sum(row["verdict"] == "correct" for row in rows_out),
        "judged_partial": sum(row["verdict"] == "partial" for row in rows_out),
        "judged_wrong": sum(row["verdict"] == "wrong" for row in rows_out),
        "judged_unjudged": sum(row["verdict"] == "unjudged" for row in rows_out),
        "mean_score": sum(row["score"] for row in rows_out) / max(1, len(rows_out)),
        "mean_latency_sec": round(sum(row["elapsed_sec"] for row in rows_out) / max(1, len(rows_out)), 3),
        "rows": rows_out,
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
