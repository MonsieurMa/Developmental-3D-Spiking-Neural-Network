r"""Turn evaluation failures into new training material - with NEW wordings.

Reading an eval report, this tool takes every item the network got wrong and
asks the teacher for several *different* training phrasings of the same
knowledge point. The result is written as an extra corpus category, so the
network is corrected on the knowledge point without memorising the eval
question itself (which would make the next evaluation meaningless).

Usage::

    .\.venv\Scripts\python.exe tools\build_correction_corpus.py `
        --report runs/metrics/eval_..._python_families_after_....json `
        --out data/corpora/python_v1/zz_correction.jsonl --per-item 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain.agents.llm_teacher import LlmTeacherAgent, extract_json

SYSTEM = "你是编程课备课老师。只输出 JSON 数组，不要解释、不要 Markdown 代码块。"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a correction corpus from evaluation failures")
    parser.add_argument("--report", required=True, help="eval JSON report containing items with verdict/score")
    parser.add_argument("--out", default="data/corpora/python_v1/zz_correction.jsonl")
    parser.add_argument("--per-item", type=int, default=3, help="new training phrasings per failed knowledge point")
    parser.add_argument("--meta", default="adult/auditory/interactive/纠错回炉/teacher")
    parser.add_argument("--max-items", type=int, default=20)
    args = parser.parse_args(argv)

    report = json.loads((ROOT / args.report).read_text(encoding="utf-8"))
    failures = [
        item for item in report.get("items", [])
        if str(item.get("verdict")) not in {"correct"} and item.get("expected")
    ][: args.max_items]
    if not failures:
        print("no failures to correct")
        return 0

    stage, modality, kind, genre, source = args.meta.split("/")
    teacher = LlmTeacherAgent(cache_dir=ROOT / "data/corpora/llm_generated")
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(str(json.loads(line).get("question", "")))

    written = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for failure in failures:
            question, expected = failure["question"], failure["expected"]
            prompt = (
                f"学生在这道题上答错了：\n问题：{question}\n正确答案：{expected}\n\n"
                f"请围绕同一个知识点写出 {args.per_item} 条**问法不同**的训练问答，"
                "答案都必须与正确答案等价（可更完整），问法要与上面的问题字面不同，"
                "其中至少一条要换成正向问法、一条换成反向问法或用途问法。\n"
                '只输出 [{"question": "...", "answer": "..."}]。'
            )
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            try:
                payload = extract_json(teacher.client.chat(messages, temperature=0.8, max_tokens=2048))
            except Exception as error:  # noqa: BLE001 - keep going with other items
                print(f"  generation failed for {question!r}: {error}", flush=True)
                continue
            if isinstance(payload, dict):
                payload = payload.get("items", [])
            for item in payload if isinstance(payload, list) else []:
                new_question = str(item.get("question", "")).strip()
                new_answer = str(item.get("answer", "")).strip()
                if not new_question or not new_answer or new_question in seen:
                    continue
                seen.add(new_question)
                handle.write(json.dumps({
                    "question": new_question, "answer": new_answer,
                    "stage": stage, "modality": modality, "kind": kind,
                    "genre": genre, "source": source, "corrected_from": question,
                }, ensure_ascii=False) + "\n")
                written += 1
            print(f"  {question!r} -> +{sum(1 for q in seen)} accumulated", flush=True)

    print(f"correction corpus: +{written} records -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
