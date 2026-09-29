r"""Reading pass: populate learned context features and then self-practise.

The context-feature space starts empty on checkpoints trained before it existed.
This tool reads the corpora through ``observe_stream`` (surprise/novelty-gated
feature learning), then lets the network quiz itself with ``self_practice``
(generate a question from its own statements, answer with the action loop
``reason``, self-correct the inconsistent ones).

Usage::

    .\.venv\Scripts\python.exe tools\learn_features.py `
        --load  `
        --save data/checkpoints/brian_features_v1.pkl `
        --corpus-dir data/corpora/mix_v1 --corpus-dir data/corpora/python_v1 `
        --practice 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain import BionicBrain


def records(directory: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reading pass for learned context features")
    parser.add_argument("--load", default="")
    parser.add_argument("--save", required=True)
    parser.add_argument("--corpus-dir", action="append", default=[])
    parser.add_argument("--practice", type=int, default=0, help="self-generated question rounds")
    parser.add_argument("--practice-max-len", type=int, default=6)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--metrics", default="")
    args = parser.parse_args(argv)

    directories = [ROOT / item for item in (args.corpus_dir or ["data/corpora/mix_v1", "data/corpora/python_v1"])]
    brain = BionicBrain.load(ROOT / args.load)
    started = time.time()
    stats = {"records": 0, "tokens": 0, "features_used": 0, "feature_tokens": 0, "splits": 0,
             "practice_attempted": 0, "practice_consistent": 0}

    for directory in directories:
        items = records(directory)
        if args.max_records:
            items = items[: args.max_records]
        for item in items:
            text = item.get("text")
            if not text and item.get("question"):
                text = f"{item['question']}。{item.get('answer', '')}。"
            if not text:
                continue
            result = brain.observe_stream(text, modality=item.get("modality", "visual"))
            stats["records"] += 1
            stats["tokens"] += result["tokens"]
        state = brain.context_features.state()
        print(f"read {directory.name}: records={stats['records']} tokens={stats['tokens']} "
              f"features={state['used_features']} splits={state['splits']}", flush=True)

    if args.practice > 0:
        report = brain.self_practice(rounds=args.practice, max_len=args.practice_max_len)
        stats["practice_attempted"] = report["attempted"]
        stats["practice_consistent"] = report["consistent"]
        for item in report["asked"][:6]:
            print(f"  self-practice {item['question']!r} -> {item['got']!r} "
                  f"(read: {item['expected']!r}) {'ok' if item['consistent'] else 'corrected'}", flush=True)

    state = brain.context_features.state()
    stats["features_used"] = state["used_features"]
    stats["feature_tokens"] = state["tokens"]
    stats["splits"] = state["splits"]
    brain.save(ROOT / args.save)
    payload = {
        "argv": sys.argv[1:],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_sec": round(time.time() - started, 1),
        "stats": stats,
        "feature_state": state,
        "brain": {k: v for k, v in brain.show().items() if k != "last_response"},
    }
    metrics_path = Path(args.metrics) if args.metrics else (
        ROOT / "runs" / "metrics" / f"learn_features_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%SZ')}.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={args.save} stats={json.dumps(stats, ensure_ascii=False)}")
    print(f"metrics={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

