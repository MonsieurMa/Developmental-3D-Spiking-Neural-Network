r"""Build a family-tagged evaluation probe for a topic corpus.

The point is to separate three different questions that a single "accuracy"
number conflates:

* ``paraphrase``  - the fact WAS taught, the wording is new (tests whether the
  mapping generalizes beyond the trained surface form);
* ``reverse``     - the same fact asked from the other direction
  ("列表 用 方括号" -> "方括号 是 什么 的 符号");
* ``new_topic``   - the fact was NEVER taught (tests coverage / calibration,
  not generalization).

Every generated question is checked against the corpus: a literal repeat is
dropped, and for paraphrase/reverse the *answer* is expected to be known while
the *question* must be new.

Usage::

    .\.venv\Scripts\python.exe tools\build_topic_eval.py `
        --corpus-dir data/corpora/python_v1 --out data/corpora/python_v1_eval_families.jsonl
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

SYSTEM = "你是教学评测出题人。只输出 JSON 数组，不要解释、不要 Markdown 代码块。"


def load_corpus(directory: Path) -> tuple[list[tuple[str, str]], set[str]]:
    pairs: list[tuple[str, str]] = []
    known: set[str] = set()
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("question") and record.get("answer"):
                pairs.append((str(record["question"]), str(record["answer"])))
            for key in ("question", "answer", "text"):
                if record.get(key):
                    known.add(str(record[key]))
    return pairs, known


def ask(teacher: LlmTeacherAgent, prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    try:
        payload = extract_json(teacher.client.chat(messages, temperature=0.8, max_tokens=3072))
    except Exception as error:  # noqa: BLE001 - report and continue with other families
        print(f"  generation failed: {error}", flush=True)
        return []
    if isinstance(payload, dict):
        payload = payload.get("items", [])
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a family-tagged topic evaluation probe")
    parser.add_argument("--corpus-dir", default="data/corpora/python_v1")
    parser.add_argument("--topic", default="Python 编程")
    parser.add_argument("--out", default="data/corpora/python_v1_eval_families.jsonl")
    parser.add_argument("--paraphrase", type=int, default=10)
    parser.add_argument("--reverse", type=int, default=6)
    parser.add_argument("--new-topic", type=int, default=6)
    parser.add_argument("--advanced-hint", default="装饰器、生成器、虚拟环境、async、装饰器参数、类型注解泛型、上下文管理器协议、元类")
    args = parser.parse_args(argv)

    corpus_dir = ROOT / args.corpus_dir
    pairs, known = load_corpus(corpus_dir)
    teacher = LlmTeacherAgent(cache_dir=ROOT / "data/corpora/llm_generated")
    print(f"corpus={corpus_dir.name} qa_pairs={len(pairs)} teacher={teacher.config.model}")

    rows: list[dict] = []

    # 1) paraphrase: same fact, new wording
    # Keep the sample answerable by the network: short answers only (the motor
    # readout emits a handful of tokens per pass).
    short = [(question, answer) for question, answer in pairs if len(answer) <= 8]
    stride = max(1, len(short) // 16)
    sample = [{"question": question, "answer": answer} for question, answer in short[::stride][:16]]
    prompt = (
        f"下面是 {args.topic} 已教过的问答。请为其中 {args.paraphrase} 条各写出一个**换一种问法**的新问题，"
        "答案必须与原来的完全一致（保持很短的答案，1-8 字），问题必须与原问题字面不同。\n"
        f"已教过的问答：{json.dumps(sample, ensure_ascii=False)}\n"
        '只输出 [{"question": "新的问法", "answer": "原答案", "source_question": "原问题"}]。'
    )
    for item in ask(teacher, prompt):
        question, answer = str(item.get("question", "")).strip(), str(item.get("answer", "")).strip()
        if question and answer and question not in known:
            rows.append({"question": question, "answer": answer, "group": "paraphrase",
                         "source_question": item.get("source_question", "")})

    # 2) reverse direction
    prompt = (
        f"下面是 {args.topic} 已教过的问答。请为其中 {args.reverse} 条各写出一个**反方向**的问题，"
        "即把原来的答案当成问句的主语、把原来的问题当成答案，答案控制在 1-10 字。\n"
        f"已教过的问答：{json.dumps(sample, ensure_ascii=False)}\n"
        '只输出 [{"question": "反方向问题", "answer": "反方向答案"}]。'
    )
    for item in ask(teacher, prompt):
        question, answer = str(item.get("question", "")).strip(), str(item.get("answer", "")).strip()
        if question and answer and question not in known:
            rows.append({"question": question, "answer": answer, "group": "reverse"})

    # 3) new topics that the corpus never covered
    prompt = (
        f"出 {args.new_topic} 道 {args.topic} 的**进阶**题目，只围绕这些主题：{args.advanced_hint}。"
        '每题问题 6-16 字、答案 1-8 字。只输出 [{"question": "...", "answer": "..."}]。'
    )
    for item in ask(teacher, prompt):
        question, answer = str(item.get("question", "")).strip(), str(item.get("answer", "")).strip()
        if question and answer and question not in known:
            rows.append({"question": question, "answer": answer, "group": "new_topic"})

    out_path = ROOT / args.out
    out_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    print(f"probe={out_path} items={len(rows)} families={counts}")
    for row in rows:
        print(f"  [{row['group']:<10}] {row['question']} -> {row['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
