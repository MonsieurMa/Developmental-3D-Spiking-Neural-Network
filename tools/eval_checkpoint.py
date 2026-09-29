r"""Fixed-probe evaluation for a BionicBrain checkpoint.

This is the measuring stick required before any checkpoint change: it loads a
checkpoint, answers a fixed probe set through the normal ``respond`` path, and
writes a JSON report under ``runs/metrics/`` plus a concise console summary.
It never mutates, trains, or saves the checkpoint.

Probe items carry a ``group`` tag:

* ``anchor``     - the exact question string exists in a corpus (recall check)
* ``novel``      - the question string exists in no corpus (generalization check)
* ``ambiguous``  - corpora map this question to several answers; scored strictly,
                   with the competing answers recorded for interpretation

Usage::

    .\.venv\Scripts\python.exe tools\eval_checkpoint.py `
        --load 
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
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
_STRUCTURAL_TOKENIZER = SubwordTokenizer()


DEFAULT_PROBE = ROOT / "data" / "corpora" / "eval_probe.jsonl"
CORPUS_DIR = ROOT / "data" / "corpora"
PROBE_NAME = "eval_probe.jsonl"
SCHEMA = "bionicbrain.eval_probe.v1"


def load_probe(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            question = raw.get("question", raw.get("prompt"))
            answer = raw.get("answer", raw.get("response"))
            if question is None or answer is None:
                raise ValueError(f"line {number} must contain question/answer fields")
            items.append(
                {
                    "question": str(question),
                    "answer": str(answer),
                    "group": str(raw.get("group", "ungrouped")),
                }
            )
    if not items:
        raise ValueError(f"probe set is empty: {path}")
    return items


def corpus_answers(question: str) -> dict[str, list[str]]:
    """Distinct answers recorded for this exact question, keyed by answer."""
    found: dict[str, list[str]] = {}
    for path in sorted(CORPUS_DIR.rglob("*.jsonl")):
        if path.name == PROBE_NAME:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_question = record.get("question", record.get("prompt"))
            raw_answer = record.get("answer", record.get("response"))
            if raw_question is None or raw_answer is None:
                continue
            if str(raw_question) == question:
                found.setdefault(str(raw_answer), []).append(path.name)
    return found


def token_scores(predicted: list[str], expected: list[str]) -> tuple[float, float, float, int, int, int]:
    predicted_counts = Counter(predicted)
    expected_counts = Counter(expected)
    overlap = sum((predicted_counts & expected_counts).values())
    predicted_total = sum(predicted_counts.values())
    expected_total = sum(expected_counts.values())
    precision = overlap / predicted_total if predicted_total else 0.0
    recall = overlap / expected_total if expected_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1, overlap, predicted_total, expected_total


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def brain_snapshot(brain: BionicBrain) -> dict:
    state = brain.show()
    return {
        "time_ms": state["time_ms"],
        "phase": state["phase"],
        "neurons": state["neurons"],
        "seed_neurons": state["seed_neurons"],
        "active_synapses": state["active_synapses"],
        "words": state["words"],
        "events": state["events"],
        "episodic_events": state["episodic_events"],
    }


def group_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"items": 0}
    correct = sum(1 for row in rows if row["verdict"] == "correct")
    partial = sum(1 for row in rows if row["verdict"] == "partial")
    return {
        "items": len(rows),
        "judged_correct": correct,
        "judged_correct_rate": correct / len(rows),
        "judged_partial": partial,
        "mean_score": statistics.fmean(row["score"] for row in rows),
        "mean_confidence": statistics.fmean(row["confidence"] for row in rows),
        # Auxiliary string-level diagnostics only; never the headline score.
        "exact_rate_aux": sum(1 for row in rows if row["exact"]) / len(rows),
        "mean_token_f1_aux": statistics.fmean(row["token_f1"] for row in rows),
    }


def evaluate(args) -> int:
    if not args.load:
        raise ValueError("--load is required")
    load_path = Path(args.load)
    probe_path = Path(args.probe)
    items = load_probe(probe_path)
    if args.limit:
        items = items[: args.limit]

    brain = BionicBrain.load(load_path)
    before = brain_snapshot(brain)

    judge = None if args.no_judge else LlmTeacherAgent(cache_dir=ROOT / "data/corpora/llm_generated")
    if judge is not None:
        if args.judge_base_url:
            judge.config.base_url = args.judge_base_url
        if args.judge_model:
            judge.config.model = args.judge_model
        judge.config.timeout = args.judge_timeout
        print(f"judge={judge.config.model}@{judge.config.base_url} online={judge.online}", flush=True)
    else:
        print("judge=disabled (token statistics only, not a score)", flush=True)

    rows: list[dict] = []
    started = time.perf_counter()
    for index, item in enumerate(items, 1):
        expected = _STRUCTURAL_TOKENIZER.encode(item["answer"])
        question = item["question"]
        item_started = time.perf_counter()
        predicted = brain.respond(question, max_len=max(8, len(expected)))
        elapsed = time.perf_counter() - item_started
        info = getattr(brain, "last_response_info", {}) or {}
        precision, recall, f1, overlap, predicted_total, expected_total = token_scores(predicted, expected)
        variants = corpus_answers(question)
        if judge is not None:
            verdict = judge.judge(question, " ".join(predicted), item["answer"])
        else:
            verdict = {"verdict": "unjudged", "score": 0.0, "reason": "judge disabled", "judge": "none"}
        row = {
            "index": index,
            "group": item["group"],
            "question": question,
            "expected": expected,
            "predicted": predicted,
            "verdict": verdict["verdict"],
            "score": float(verdict["score"]),
            "judge_reason": verdict.get("reason", ""),
            "judge": verdict.get("judge", ""),
            "exact": predicted == expected,
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1,
            "token_overlap": overlap,
            "token_predicted": predicted_total,
            "token_expected": expected_total,
            "route": info.get("route"),
            "confidence": float(info.get("confidence", 0.0)),
            "similarity": float(info.get("similarity", 0.0)),
            "memory_pointer": bool(info.get("memory_pointer", False)),
            "spikes_used": int(info.get("spikes_used", 0)),
            "readout": info.get("readout"),
            "seen_in_corpora": sorted(variants),
            "corpus_answer_variants": {answer: sorted(set(files)) for answer, files in sorted(variants.items())},
            "elapsed_sec": elapsed,
        }
        rows.append(row)
        mark = {"correct": "ok  ", "partial": "part", "wrong": "miss", "unjudged": "??  "}[row["verdict"]]
        print(
            f"[{index:02d}/{len(items)}] {mark} {row['group']:<9} "
            f"Q: {question} | expected: {' '.join(expected)} | got: {' '.join(predicted) or '<empty>'} "
            f"| verdict={row['verdict']} score={row['score']:.2f} route={row['route']} "
            f"({elapsed:.1f}s) :: {row['judge_reason']}",
            flush=True,
        )
    elapsed_total = time.perf_counter() - started
    after = brain_snapshot(brain)

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)

    exact_matches = sum(1 for row in rows if row["exact"])
    judged_correct = sum(1 for row in rows if row["verdict"] == "correct")
    judged_partial = sum(1 for row in rows if row["verdict"] == "partial")
    judged_wrong = sum(1 for row in rows if row["verdict"] == "wrong")
    judge_fallbacks = sum(1 for row in rows if row["judge"] == "offline-fallback")
    micro_overlap = sum(row["token_overlap"] for row in rows)
    micro_predicted = sum(row["token_predicted"] for row in rows)
    micro_expected = sum(row["token_expected"] for row in rows)
    micro_precision = micro_overlap / micro_predicted if micro_predicted else 0.0
    micro_recall = micro_overlap / micro_expected if micro_expected else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    metrics = {
        "items": len(rows),
        # Headline: semantic judgement by the local teacher model.
        "judged_correct": judged_correct,
        "judged_correct_rate": judged_correct / len(rows) if rows else 0.0,
        "judged_partial": judged_partial,
        "judged_wrong": judged_wrong,
        "judge_fallbacks": judge_fallbacks,
        "mean_score": statistics.fmean(row["score"] for row in rows) if rows else 0.0,
        "verdicts": dict(sorted(Counter(row["verdict"] for row in rows).items())),
        # Auxiliary string-level diagnostics.
        "exact_matches_aux": exact_matches,
        "exact_rate_aux": exact_matches / len(rows) if rows else 0.0,
        "token_precision_micro_aux": micro_precision,
        "token_recall_micro_aux": micro_recall,
        "token_f1_micro_aux": micro_f1,
        "token_f1_macro_aux": statistics.fmean(row["token_f1"] for row in rows) if rows else 0.0,
        "mean_confidence": statistics.fmean(row["confidence"] for row in rows) if rows else 0.0,
        "routes": dict(sorted(Counter(row["route"] or "none" for row in rows).items())),
        "mean_latency_sec": statistics.fmean(row["elapsed_sec"] for row in rows) if rows else 0.0,
        "elapsed_sec": elapsed_total,
        "by_group": {name: group_metrics(group) for name, group in sorted(groups.items())},
    }

    config = brain.config if isinstance(brain.config, BionicConfig) else BionicConfig()
    report = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "checkpoint": {
            "path": str(load_path.as_posix()),
            "sha256": sha256_file(load_path),
            "size_bytes": load_path.stat().st_size,
            "modified_utc": datetime.fromtimestamp(load_path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        },
        "config": {
            "seed_profile": config.seed_profile,
            "dt": config.dt,
            "memory_hit_threshold": config.memory_hit_threshold,
            "strong_memory_hit_threshold": config.strong_memory_hit_threshold,
            "response_confidence_threshold": config.response_confidence_threshold,
            "random_seed": getattr(brain, "random_seed", None),
        },
        "probe": {"path": str(probe_path.as_posix()), "items": len(rows)},
        "judge": {
            "enabled": judge is not None,
            "model": judge.config.model if judge is not None else None,
            "base_url": judge.config.base_url if judge is not None else None,
            "fallbacks": judge_fallbacks,
        },
        "brain_before": before,
        "brain_after": after,
        "metrics": metrics,
        "items": rows,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_path = Path(args.out) if args.out else out_dir / f"eval_{load_path.stem}{tag}_{stamp}Z.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("", flush=True)
    print(
        f"judged_correct={metrics['judged_correct']}/{metrics['items']} "
        f"({metrics['judged_correct_rate']:.1%})  partial={metrics['judged_partial']}  "
        f"mean_score={metrics['mean_score']:.3f}  fallbacks={metrics['judge_fallbacks']}",
        flush=True,
    )
    print(
        f"aux(exact={metrics['exact_matches_aux']}/{metrics['items']} "
        f"token_f1_micro={metrics['token_f1_micro_aux']:.3f})",
        flush=True,
    )
    for name, group in metrics["by_group"].items():
        if group.get("items"):
            print(
                f"  {name:<9} judged={group['judged_correct']}/{group['items']} "
                f"({group['judged_correct_rate']:.1%}) mean_score={group['mean_score']:.3f}",
                flush=True,
            )
    print(f"routes={metrics['routes']}", flush=True)
    print(f"neurons={after['neurons']} active_synapses={after['active_synapses']} words={after['words']} phase={after['phase']}", flush=True)
    print(f"latency_mean={metrics['mean_latency_sec']:.1f}s elapsed={metrics['elapsed_sec']:.1f}s", flush=True)
    print(f"checkpoint={load_path.as_posix()} sha256={report['checkpoint']['sha256'][:16]}...", flush=True)
    print(f"metrics: {out_path.as_posix()}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-probe evaluation for a BionicBrain checkpoint")
    parser.add_argument("--load", default=BionicConfig().default_checkpoint, help="checkpoint to evaluate")
    parser.add_argument("--probe", default=str(DEFAULT_PROBE), help="JSONL probe set with question/answer/group fields")
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "metrics"), help="directory for the JSON report")
    parser.add_argument("--out", default=None, help="explicit report path (overrides --out-dir naming)")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N probe items")
    parser.add_argument("--tag", default="", help="suffix for the report filename, e.g. 'baseline'")
    parser.add_argument("--no-judge", action="store_true", help="skip semantic judging (string statistics only)")
    parser.add_argument("--judge-base-url", default=None, help="override the judge endpoint")
    parser.add_argument("--judge-model", default=None, help="override the judge model id")
    parser.add_argument("--judge-timeout", type=float, default=600.0, help="judge request timeout in seconds")
    return parser


if __name__ == "__main__":
    raise SystemExit(evaluate(build_parser().parse_args()))



