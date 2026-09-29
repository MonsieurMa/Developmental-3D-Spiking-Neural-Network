"""Probe the local OpenAI-compatible teacher endpoint and report its reply shape.

Reasoning-style servers may leave ``message.content`` empty and place the whole
answer in ``message.reasoning_content``; this tool shows which field carries the
text so training runs can be configured correctly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain.agents.llm_teacher import ChatTeacherClient, TeacherConfig, TeacherUnavailable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe the local chat teacher endpoint")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--message", default='只输出 JSON 数组：[{"question": "1 + 1 = ?", "answer": "2"}]')
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--show-chars", type=int, default=400)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="extra request field, e.g. --param chat_template_kwargs={\"enable_thinking\": false}",
    )
    parser.add_argument("--models-only", action="store_true", help="only list served models")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TeacherConfig()
    if args.base_url:
        config.base_url = args.base_url
    if args.model:
        config.model = args.model
    if args.timeout:
        config.timeout = args.timeout

    client = ChatTeacherClient(config)
    try:
        models = client.list_models()
    except TeacherUnavailable as error:
        print(f"endpoint unreachable: {error}")
        return 2
    print(f"base_url={config.base_url}")
    print(f"served_models={models}")
    if config.model not in models:
        print(f"warning: configured model {config.model!r} is not in the served list")
    if args.models_only:
        return 0

    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": args.message}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    for item in args.param:
        key, _, raw = item.partition("=")
        payload[key.strip()] = json.loads(raw) if raw.strip() else True
    if args.param:
        print(f"extra_params={json.dumps({k: payload[k] for k in payload if k not in {'model', 'messages', 'temperature', 'max_tokens', 'stream'}}, ensure_ascii=False)}")
    started = time.perf_counter()
    try:
        reply = client._request("/chat/completions", payload)
    except TeacherUnavailable as error:
        print(f"chat failed: {error}")
        return 2
    elapsed = time.perf_counter() - started

    choice = reply["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    print(f"finish_reason={choice.get('finish_reason')}")
    print(f"latency_sec={round(elapsed, 2)}")
    print(f"usage={json.dumps(reply.get('usage', {}), ensure_ascii=False)}")
    print(f"message_keys={sorted(message.keys())}")
    print(f"content_chars={len(content)} reasoning_chars={len(reasoning)}")
    print(f"content={content[: args.show_chars]!r}")
    print(f"reasoning={reasoning[: args.show_chars]!r}")
    if not content and reasoning:
        print("note: server returns an empty content field; the teacher agent reads reasoning as a fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
