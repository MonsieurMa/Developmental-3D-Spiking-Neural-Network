from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain.language.subwords import SubwordTokenizer


def _texts(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {error}") from error
            if isinstance(record, dict):
                yield from (value for value in record.values() if isinstance(value, str))
            elif isinstance(record, str):
                yield record


def main() -> int:
    parser = argparse.ArgumentParser(description="Learn language-neutral subword units from JSONL corpora")
    parser.add_argument(
        "corpora",
        nargs="*",
        help="JSONL files; empty means data/corpora/mix_v1 and data/corpora/python_v1",
    )
    parser.add_argument("--output", default="data/tokenizers/subwords_v1.json")
    parser.add_argument("--max-merges", type=int, default=512)
    parser.add_argument("--min-pair-count", type=int, default=2)
    parser.add_argument("--max-unit-symbols", type=int, default=3)
    parser.add_argument("--sample", action="store_true", help="print encoded examples after fitting")
    args = parser.parse_args()

    paths = [Path(item) for item in args.corpora]
    if not paths:
        paths = sorted((ROOT / "data/corpora").glob("mix_v1/*.jsonl"))
        paths.extend(sorted((ROOT / "data/corpora").glob("python_v1/*.jsonl")))

    tokenizer = SubwordTokenizer(
        max_merges=args.max_merges,
        min_pair_count=args.min_pair_count,
        max_unit_symbols=args.max_unit_symbols,
    )
    examples: list[str] = []
    for path in paths:
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(path)
        for text in _texts(path):
            examples.append(text)
    tokenizer.learn(examples)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tokenizer.state(), ensure_ascii=False, indent=2), encoding="utf-8")

    stats = tokenizer.diagnostics()
    print(json.dumps({"corpora": len(paths), "lines": len(examples), "output": str(output), **stats},
                     ensure_ascii=False, indent=2))
    if args.sample:
        for text in examples[:10]:
            print(json.dumps({"text": text, "units": tokenizer.encode(text)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
