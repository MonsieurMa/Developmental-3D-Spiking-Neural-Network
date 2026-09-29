"""Corpus loading. No default answer table lives in source."""
from __future__ import annotations

import json
from pathlib import Path


def load_corpus(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            question = item.get("question", item.get("prompt", item.get("input")))
            answer = item.get("answer", item.get("response", item.get("output")))
            if question is None or answer is None:
                raise ValueError(f"line {line_number} must contain question/answer fields")
            records.append({"question": str(question), "answer": str(answer)})
    if not records:
        raise ValueError(f"corpus is empty: {path}")
    return records
