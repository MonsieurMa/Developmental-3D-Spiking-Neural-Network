"""Unit tests for the local chat teacher agent (stdlib only, offline by default)."""
from __future__ import annotations

import json
import os
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

from bionic_brain.agents.llm_teacher import (
    ChatTeacherClient,
    LlmTeacherAgent,
    TeacherConfig,
    TeacherUnavailable,
    extract_json,
    is_usable_item,
)

class _StubHandler(BaseHTTPRequestHandler):
    reply_content = "[]"
    reply_message: dict | None = None
    requests: list[dict] = []

    def log_message(self, *args):
        return

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path.endswith("/models"):
            self._send({"object": "list", "data": [{"id": "stub-model"}]})
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).requests.append(payload)
        message = self.reply_message or {"role": "assistant", "content": self.reply_content}
        self._send({"choices": [{"message": message, "finish_reason": "stop"}]})


class _StubTeacher:
    """Stand-in client that returns canned JSON without touching the network."""

    def __init__(self, payload, *, offline: bool = False) -> None:
        self.payload = payload
        self.offline = offline
        self.calls: list[list[dict[str, str]]] = []
        self.config = TeacherConfig(model="stub-model", base_url="http://stub.local/v1")

    def available(self) -> bool:
        return not self.offline

    def chat_json(self, messages, **kwargs):
        self.calls.append(list(messages))
        if self.offline:
            raise TeacherUnavailable("offline")
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class _ScriptedTeacher(_StubTeacher):
    """Returns one payload per call, repeating the final payload afterwards."""

    def __init__(self, payloads) -> None:
        super().__init__(payloads[-1])
        self.scripted = list(payloads)

    def chat_json(self, messages, **kwargs):
        if len(self.scripted) > 1:
            self.calls.append(list(messages))
            return self.scripted.pop(0)
        return super().chat_json(messages, **kwargs)


class ExtractJsonTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract_json('[{"a": 1}]'), [{"a": 1}])

    def test_markdown_fence(self):
        text = '好的，如下：\n```json\n[{"question": "1", "answer": "2"}]\n```\n希望有帮助'
        self.assertEqual(extract_json(text), [{"question": "1", "answer": "2"}])

    def test_json_with_prose_and_braces_in_strings(self):
        text = 'prefix {"verdict": "wrong", "feedback": "缺少 } 符号"} suffix'
        self.assertEqual(extract_json(text)["verdict"], "wrong")

    def test_missing_json_raises(self):
        with self.assertRaises(ValueError):
            extract_json("no structured content here")


class ChatTeacherClientTests(unittest.TestCase):
    def setUp(self):
        _StubHandler.requests = []
        _StubHandler.reply_message = None
        _StubHandler.reply_content = '[{"question": "如果 A 那么 B；A；所以？", "answer": "B"}]'
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.config = TeacherConfig(
            base_url=f"http://127.0.0.1:{self.server.server_address[1]}/v1",
            model="stub-model",
        )
        self.client = ChatTeacherClient(self.config)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_models_and_availability(self):
        self.assertEqual(self.client.list_models(), ["stub-model"])
        self.assertTrue(self.client.available())

    def test_chat_sends_openai_payload(self):
        content = self.client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("question", content)
        sent = _StubHandler.requests[-1]
        self.assertEqual(sent["model"], "stub-model")
        self.assertFalse(sent["stream"])
        self.assertEqual(sent["messages"][0]["content"], "hi")

    def test_thinking_is_disabled_by_default(self):
        self.client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(_StubHandler.requests[-1]["chat_template_kwargs"], {"enable_thinking": False})

    def test_thinking_can_be_reenabled(self):
        client = ChatTeacherClient(TeacherConfig(base_url=self.config.base_url, disable_thinking=False))
        client.chat([{"role": "user", "content": "hi"}])
        self.assertNotIn("chat_template_kwargs", _StubHandler.requests[-1])

    def test_extra_payload_is_merged(self):
        config = TeacherConfig(base_url=self.config.base_url, extra_payload={"reasoning_effort": "none"})
        ChatTeacherClient(config).chat([{"role": "user", "content": "hi"}])
        self.assertEqual(_StubHandler.requests[-1]["reasoning_effort"], "none")

    def test_unreachable_endpoint_is_offline(self):
        client = ChatTeacherClient(TeacherConfig(base_url="http://127.0.0.1:1/v1", timeout=1.0))
        self.assertFalse(client.available())

    def test_reasoning_content_is_used_when_content_is_empty(self):
        _StubHandler.reply_message = {
            "role": "assistant",
            "content": "",
            "reasoning_content": '[{"question": "1 + 1 = ?", "answer": "2"}]',
        }
        self.assertEqual(self.client.chat([{"role": "user", "content": "hi"}]), '[{"question": "1 + 1 = ?", "answer": "2"}]')
        self.assertEqual(
            self.client.chat_json([{"role": "user", "content": "hi"}]),
            [{"question": "1 + 1 = ?", "answer": "2"}],
        )

    def test_empty_reply_raises_with_hint(self):
        _StubHandler.reply_message = {"role": "assistant", "content": ""}
        with self.assertRaises(TeacherUnavailable) as context:
            self.client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("empty reply", str(context.exception))


class LlmTeacherAgentTests(unittest.TestCase):
    def test_draft_retries_once_when_the_first_reply_is_malformed(self):
        bad = [{"question": "", "answer": "opaque"}]
        good = [{"question": "opaque source", "answer": "opaque result"}]
        client = _ScriptedTeacher([bad, good])
        with TemporaryDirectory() as tmp:
            teacher = LlmTeacherAgent(client=client, cache_dir=tmp)
            items = teacher.draft_curriculum("因果推理", 2)
        self.assertEqual(items, good)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("上一次输出不合格", client.calls[1][-1]["content"])

    def test_draft_raises_when_both_attempts_are_malformed(self):
        bad = [{"question": "", "answer": "opaque"}]
        teacher = LlmTeacherAgent(client=_ScriptedTeacher([bad, bad]))
        with self.assertRaises(TeacherUnavailable):
            teacher.draft_curriculum("因果推理", 2)

    def test_draft_curriculum_filters_and_caches(self):
        payload = [
            {"question": "如果 下雨 那么 地湿；下雨；所以？", "answer": "地湿"},
            {"question": "", "answer": "坏数据"},
            {"question": "重复", "answer": "答案"},
            {"question": "重复", "answer": "答案"},
        ]
        with TemporaryDirectory() as tmp:
            teacher = LlmTeacherAgent(client=_StubTeacher(payload), cache_dir=tmp)
            items = teacher.draft_curriculum("因果推理", 4)
            self.assertEqual(len(items), 2)
            cached = Path(tmp) / "因果推理.jsonl"
            self.assertTrue(cached.exists())
            self.assertEqual(len(cached.read_text(encoding="utf-8").strip().splitlines()), 2)
            teacher.draft_curriculum("因果推理", 4)
            self.assertEqual(len(cached.read_text(encoding="utf-8").strip().splitlines()), 2)

    def test_offline_teacher_raises_and_reports(self):
        teacher = LlmTeacherAgent(client=_StubTeacher({}, offline=True))
        self.assertFalse(teacher.online)
        with self.assertRaises(TeacherUnavailable):
            teacher.draft_curriculum("任意", 2)

    def test_judge_scores_by_meaning_not_characters(self):
        payload = {"verdict": "correct", "score": 1.0, "reason": "空格差异，含义等价"}
        teacher = LlmTeacherAgent(client=_StubTeacher(payload))
        verdict = teacher.judge("你 是 谁", "我 是 Brian", "我是Brian")
        self.assertEqual(verdict["verdict"], "correct")
        self.assertEqual(verdict["score"], 1.0)
        self.assertIn("stub-model", verdict["judge"])

    def test_judge_labels_offline_items_unjudged(self):
        teacher = LlmTeacherAgent(client=_StubTeacher(TeacherUnavailable("offline")))
        verdict = teacher.judge("你 是 谁", "我 是 人", "我 是 人")
        self.assertEqual(verdict["verdict"], "unjudged")
        self.assertEqual(verdict["judge"], "offline-fallback")
        self.assertEqual(verdict["score"], 0.0)

    def test_judge_repairs_invalid_verdict_and_score(self):
        teacher = LlmTeacherAgent(client=_StubTeacher({"verdict": "partial"}))
        verdict = teacher.judge("苹果 的 英文", "apple", "苹果")
        self.assertEqual(verdict["verdict"], "partial")
        self.assertEqual(verdict["score"], 0.5)

    def test_judge_many_reports_every_item(self):
        payload = {"verdict": "correct", "score": 1.0, "reason": "等价"}
        teacher = LlmTeacherAgent(client=_StubTeacher(payload))
        results = teacher.judge_many([("q1", "a", "a"), ("q2", "b", "b")])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(item["verdict"] == "correct" for item in results))


if __name__ == "__main__":
    unittest.main()
