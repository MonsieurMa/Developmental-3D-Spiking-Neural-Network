r"""Collapse verbose answer forms onto one canonical (shortest) form.

Teaching the same fact as "方括号", "使用方括号[]" and "使用 方 括 号" splits the
motor readout's support across three assemblies, so a question that used to be
answered gets *rejected after more training* (measured: paraphrase 7/10 -> 6/10,
reverse 4/6 -> 3/6 in the third Python round).

This tool rewrites an answer to the shortest taught form that it literally
contains, leaving unrelated answers alone. Run with ``--dry-run`` first.

Usage::

    .\.venv\Scripts\python.exe tools\canonicalize_corpus.py --dir data/corpora/python_v1 --dry-run
    .\.venv\Scripts\python.exe tools\canonicalize_corpus.py --dir data/corpora/python_v1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain.language.subwords import SubwordTokenizer
_STRUCTURAL_TOKENIZER = SubwordTokenizer()

# Tokens that may be dropped without changing what the answer means.
FILLER_TOKENS = {
    "的", "了", "用", "是", "会", "要", "在", "和", "与", "就", "都", "也",
    "使用", "可以", "应该", "需要", "表示", "用来", "叫做", "叫", "称为",
    "，", "。", "、", "（", "）", "(", ")", "[", "]", "{", "}", " ",
}

def normalize(text: str) -> str:
    return re.sub(r"[\s。，、；：！？（）()\[\]{}'\"`]+", "", str(text)).lower()


def token_window(long_tokens: list[str], short_tokens: list[str]) -> tuple[bool, float]:
    """Is ``short`` a contiguous token run inside ``long``, and how noisy is the rest?"""

    size = len(short_tokens)
    if size == 0 or size >= len(long_tokens):
        return False, 0.0
    for start in range(len(long_tokens) - size + 1):
        if long_tokens[start:start + size] != short_tokens:
            continue
        extra = long_tokens[:start] + long_tokens[start + size:]
        if not extra:
            return True, 1.0
        filler = sum(1 for token in extra if token in FILLER_TOKENS)
        return True, filler / len(extra)
    return False, 0.0


def is_canonical_candidate(answer: str) -> bool:
    """A short, self-contained answer - not a fragment and not a function word."""

    tokens = _STRUCTURAL_TOKENIZER.encode(answer)
    if not tokens:
        return False
    if len(tokens) >= 2:
        return True
    single = tokens[0]
    return single.isascii() and len(single) >= 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canonicalize answer forms in a corpus")
    parser.add_argument("--dir", default="data/corpora/python_v1")
    parser.add_argument("--max-canonical-chars", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    directory = ROOT / args.dir
    files = sorted(directory.glob("*.jsonl"))
    records: list[tuple[Path, dict]] = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append((path, json.loads(line)))

    answers = {
        str(record["answer"]).strip()
        for _, record in records
        if record.get("answer")
    }
    canonical_pool = [
        answer for answer in answers
        if len(normalize(answer)) <= args.max_canonical_chars and is_canonical_candidate(answer)
    ]

    rewrites: list[tuple[str, str]] = []
    for _, record in records:
        answer = str(record.get("answer", "")).strip()
        if not answer:
            continue
        normalized = normalize(answer)
        long_tokens = _STRUCTURAL_TOKENIZER.encode(answer)
        best: str | None = None
        for candidate in canonical_pool:
            if candidate == answer:
                continue
            matched, filler_ratio = token_window(long_tokens, _STRUCTURAL_TOKENIZER.encode(candidate))
            # Collapse only a verbose restatement: the short form must appear as
            # a contiguous word run and the surrounding words must be filler.
            if not matched or filler_ratio < 0.6:
                continue
            if best is None or len(_STRUCTURAL_TOKENIZER.encode(candidate)) < len(_STRUCTURAL_TOKENIZER.encode(best)):
                best = candidate
        if best is None:
            continue
        record.setdefault("canonical_from", answer)
        record["answer"] = best
        rewrites.append((answer, best))

    per_file: dict[Path, list[dict]] = {}
    for path, record in records:
        per_file.setdefault(path, []).append(record)

    print(f"files={len(files)} qa_records={sum(1 for _, r in records if r.get('answer'))} "
          f"canonical_pool={len(canonical_pool)} rewrites={len(rewrites)}")
    for old, new in rewrites[:12]:
        print(f"  {old!r} -> {new!r}")
    if len(rewrites) > 12:
        print(f"  ... and {len(rewrites) - 12} more")

    if args.dry_run:
        print("dry run: nothing written")
        return 0
    for path, rows in per_file.items():
        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        path.write_text(payload, encoding="utf-8")
    print(f"rewritten {len(rewrites)} answers across {len(per_file)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


