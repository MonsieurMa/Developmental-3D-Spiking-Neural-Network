"""Trainer-side agent that teaches Brian through a local OpenAI-compatible chat model.

The language model is strictly a *teacher*: it drafts curricula, grades Brian's
motor-language answers and writes feedback. It never sits in Brian's inference
path, so the network keeps its local, biologically constrained learning rules.

The client uses only the standard library, so the repository keeps its
``numpy`` + ``matplotlib`` dependency set.
"""
from __future__ import annotations

import json
import os
import re
import time as time_module
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..brain import BionicBrain
from ..language.subwords import SubwordTokenizer

DEFAULT_BASE_URL = "http://127.0.0.1:1919/v1"
DEFAULT_MODEL = "Qwen3.6-35B-A3B-NVFP4"
DEFAULT_CACHE_DIR = "data/corpora/llm_generated"


def normalize_answer(value: str | list[str]) -> str:
    """Structural comparison key: casefold and remove whitespace only."""

    if isinstance(value, list):
        value = "".join(str(item) for item in value)
    return re.sub(r"\s+", "", str(value).casefold())


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class TeacherUnavailable(RuntimeError):
    """Raised when the local chat endpoint cannot be reached or answered."""


@dataclass
class TeacherConfig:
    """Connection settings for the local OpenAI-compatible chat endpoint."""

    base_url: str = field(
        default_factory=lambda: os.environ.get("BIONIC_TEACHER_BASE_URL", DEFAULT_BASE_URL)
    )
    model: str = field(default_factory=lambda: os.environ.get("BIONIC_TEACHER_MODEL", DEFAULT_MODEL))
    api_key: str = field(default_factory=lambda: os.environ.get("BIONIC_TEACHER_API_KEY", "local"))
    # Local reasoning models can think for minutes on a drafting prompt.
    timeout: float = 600.0
    temperature: float = 0.3
    # Reasoning-style servers spend most of the budget on the thinking block
    # before they emit ``content``; a small cap yields empty answers.
    max_tokens: int = 4096
    # Qwen3-style chat templates: disable the thinking block. On the local
    # Qwen3.6-35B-A3B-NVFP4 server this saves roughly 20x completion tokens.
    disable_thinking: bool = field(
        default_factory=lambda: not _env_flag("BIONIC_TEACHER_ENABLE_THINKING", False)
    )
    extra_payload: dict = field(
        default_factory=lambda: json.loads(os.environ.get("BIONIC_TEACHER_EXTRA_JSON", "{}"))
    )
    cache_dir: str = DEFAULT_CACHE_DIR

    def url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path


CURRICULUM_SYSTEM_PROMPT = """你是 BionicBrain 脉冲神经网络的训练教师。
BionicBrain 不是语言模型：它靠局部突触可塑性和海马重放学习，只消费语料中的 token 序列。

任务：围绕指定主题出题，并给出确定的参考答案。

输出规则（必须严格遵守）：
1. 只输出一个 JSON 数组，不要解释、不要 Markdown 代码块。
2. 每项形如 {"question": "...", "answer": "..."}。
3. question 与 answer 必须使用语料原有的书写约定，并保持同一套 token 边界。
4. answer 必须简短、确定，不要出现多个可选答案。
5. question 必须自包含，只包含题目本身需要的信息，不要出现主题名、提示词或解释。"""


CURRICULUM_RETRY_PROMPT = (
    "上一次输出不合格：question/answer 缺少可学习的稳定映射，或 token 过多。"
    "请严格按示例格式重新输出 JSON 数组。"
)


def is_usable_item(item: dict[str, str]) -> bool:
    """Reject drafts the network cannot learn a stable mapping from."""

    question, answer = item.get("question", ""), item.get("answer", "")
    if not question or not answer:
        return False
    question_tokens = SubwordTokenizer().encode(question)
    if not question_tokens or len(question_tokens) > 24:
        return False
    if len(SubwordTokenizer().encode(answer)) > 8:
        return False
    return True


GRADING_SYSTEM_PROMPT = """你是 BionicBrain 脉冲神经网络的评分教师。
给定问题、参考答案和学员回答，判断学员回答是否表达了同一含义。

输出规则：
1. 只输出一个 JSON 对象，不要解释、不要 Markdown 代码块。
2. 形如 {"verdict": "correct|partial|wrong", "expected": "...", "feedback": "..."}。
3. verdict 只允许这三个取值；expected 保留语义，不要因书写变体而误判。
4. feedback 用一句中文说明差异，便于训练代理强化或纠正。"""


JUDGE_SYSTEM_PROMPT = """你是 BionicBrain 脉冲神经网络的评分教师。
学员是一个脉冲神经网络，它的输出是 token 列表，常见情况是：同义不同字、语序不同、缺字、多字、字符串粘连或重复。
你的任务是判断学员回答与参考答案是否表达了同一含义，而不是比较字符串。

判定标准：
- correct：含义等价。例："thank you" ≈ "谢谢"；"我 是 Brian" ≈ "我是Brian"；"地 湿" ≈ "地湿"；"小 美" ≈ "小美"。
- partial：关键信息正确但不完整、有多余内容或语序导致轻微歧义。
- wrong：错误、无关、答非所问或空回答。

输出规则：只输出一个 JSON 对象，不要解释、不要 Markdown 代码块：
{"verdict": "correct|partial|wrong", "score": 0.0, "reason": "一句中文说明"}
score 必须与 verdict 一致：correct=1.0，partial=0.5，wrong=0.0。
如果完全没有参考答案，就根据问题本身判断学员回答是否是合理回答（合理=correct）。"""


VERDICT_SCORES = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def extract_json(text: str) -> Any:
    """Return the first JSON value embedded in ``text``.

    Chat models often wrap JSON in prose or Markdown fences, so the raw
    ``json.loads`` call is only the fast path.
    """

    raw = str(text).strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("[", "]"), ("{", "}")):
        start = raw.find(opener)
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(raw)):
                char = raw[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(raw[start : index + 1])
                        except json.JSONDecodeError:
                            break
            start = raw.find(opener, start + 1)
    raise ValueError(f"no JSON value found in teacher reply: {raw[:200]!r}")


class ChatTeacherClient:
    """Minimal standard-library client for ``POST /chat/completions``."""

    def __init__(self, config: TeacherConfig | None = None) -> None:
        self.config = config or TeacherConfig()
        self._models: list[str] | None = None

    def _request(self, path: str, payload: dict | None = None) -> Any:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.config.url(path),
            data=data,
            method="GET" if payload is None else "POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            raise TeacherUnavailable(f"{self.config.url(path)} unavailable: {error}") from error

    def list_models(self) -> list[str]:
        """Return served model ids; cached after the first successful probe."""

        if self._models is None:
            payload = self._request("/models")
            entries = payload.get("data", []) if isinstance(payload, dict) else []
            self._models = [str(entry.get("id")) for entry in entries if isinstance(entry, dict)]
        return list(self._models)

    def available(self) -> bool:
        try:
            return bool(self.list_models())
        except TeacherUnavailable:
            return False

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if self.config.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload.update(self.config.extra_payload)
        reply = self._request("/chat/completions", payload)
        try:
            choice = reply["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise TeacherUnavailable(f"unexpected chat payload: {str(reply)[:200]}") from error

        content = str(message.get("content") or "").strip()
        if not content:
            # Reasoning models may return the whole answer in the thinking block.
            content = str(message.get("reasoning_content") or message.get("reasoning") or "").strip()
        if not content:
            finish = choice.get("finish_reason")
            hint = "raise max_tokens; the thinking block consumed the whole budget" if finish == "length" else "check the served model"
            raise TeacherUnavailable(f"teacher returned an empty reply (finish_reason={finish}): {hint}")
        return content

    def chat_json(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> Any:
        return extract_json(self.chat(messages, **kwargs))


class LlmTeacherAgent:
    """Trainer-side teacher agent for the local spiking network."""

    def __init__(
        self,
        brain: BionicBrain | None = None,
        *,
        client: ChatTeacherClient | None = None,
        config: TeacherConfig | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.config = config or TeacherConfig()
        self.client = client or ChatTeacherClient(self.config)
        self.brain = brain
        self.cache_dir = Path(cache_dir or self.config.cache_dir)
        self.graded = 0
        self.drafted = 0

    # ---------------- availability ----------------
    @property
    def online(self) -> bool:
        return self.client.available()

    def describe(self) -> dict:
        return {
            "base_url": self.config.base_url,
            "model": self.config.model,
            "online": self.online,
            "thinking": not self.config.disable_thinking,
            "cache_dir": str(self.cache_dir),
        }

    # ---------------- curriculum drafting ----------------
    def draft_curriculum(self, topic: str, count: int = 8) -> list[dict[str, str]]:
        """Ask the teacher for ``count`` question/answer pairs about ``topic``."""

        user_prompt = f"主题：{topic}\n请生成 {count} 条训练问答。"
        last_error = ""
        for attempt in range(2):
            messages = [
                {"role": "system", "content": CURRICULUM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            if attempt:
                messages.append({"role": "user", "content": CURRICULUM_RETRY_PROMPT})
            try:
                payload = self.client.chat_json(messages, temperature=0.4 if attempt == 0 else 0.2)
                items = [item for item in self._normalize_items(payload) if is_usable_item(item)]
            except ValueError as error:
                last_error = f"teacher reply was not JSON: {error}"
                items = []
            if items:
                self.drafted += len(items)
                self.save_curriculum(topic, items)
                return items
            last_error = last_error or f"teacher returned no usable items for topic {topic!r}"
        raise TeacherUnavailable(last_error)

    def _normalize_items(self, payload: Any) -> list[dict[str, str]]:
        if isinstance(payload, dict):
            for key in ("items", "data", "curriculum", "examples"):
                if key in payload:
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            return []

        items: list[dict[str, str]] = []
        seen: set[str] = set()
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            question = str(entry.get("question", entry.get("prompt", ""))).strip()
            answer = str(entry.get("answer", entry.get("response", ""))).strip()
            structural_tokenizer = SubwordTokenizer()
            if (
                not question
                or not answer
                or not structural_tokenizer.encode(question)
                or not structural_tokenizer.encode(answer)
            ):
                continue
            key = normalize_answer(question)
            if key in seen:
                continue
            seen.add(key)
            items.append({"question": question, "answer": answer})
        return items

    def save_curriculum(self, topic: str, items: list[dict[str, str]]) -> Path | None:
        """Append drafted items to a topic-scoped JSONL cache, skipping duplicates."""

        if not items:
            return None
        slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", topic).strip("_") or "topic"
        path = self.cache_dir / f"{slug}.jsonl"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        known: set[str] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    known.add(normalize_answer(json.loads(line)["question"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        fresh = [item for item in items if normalize_answer(item["question"]) not in known]
        if not fresh:
            return path
        with path.open("a", encoding="utf-8") as handle:
            for item in fresh:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return path

    # ---------------- semantic judging ----------------
    def judge(
        self,
        question: str,
        candidate: str,
        reference: str | None = None,
        *,
        context: str = "",
        offline_fallback: bool = True,
    ) -> dict:
        """Judge one answer semantically with the local chat model.

        Semantic judgement is the only score.  If the teacher is unavailable,
        the item is explicitly ``unjudged``; there is no string-comparison
        fallback.
        """

        parts = [f"问题：{question}", f"学员回答：{candidate or '<空>'}" ]
        if reference:
            parts.insert(1, f"参考答案：{reference}")
        else:
            parts.insert(1, "参考答案：无（请判断学员回答是否合理）")
        if context:
            parts.append(f"情境（学员刚读到/听到的内容）：{context}")
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(parts)},
        ]
        started = time_module.perf_counter()
        try:
            payload = self.client.chat_json(messages, temperature=0.0, max_tokens=512)
        except (TeacherUnavailable, ValueError) as error:
            if not offline_fallback:
                raise
            return {
                "verdict": "unjudged",
                "score": 0.0,
                "reason": f"teacher unavailable: {error}",
                "judge": "offline-fallback",
                "elapsed_ms": round((time_module.perf_counter() - started) * 1000.0, 1),
            }

        if isinstance(payload, list) and payload:
            payload = payload[0]
        if not isinstance(payload, dict):
            verdict = ""
            reason = "评分教师返回格式异常"
            score = None
            judge_model = self.config.model
            judge_url = self.config.base_url
        else:
            verdict = str(payload.get("verdict", "")).strip().lower()
            reason = str(payload.get("reason", payload.get("feedback", ""))).strip()
            score = payload.get("score")
            client_config = getattr(self.client, "config", None)
            judge_model = getattr(client_config, "model", self.config.model)
            judge_url = getattr(client_config, "base_url", self.config.base_url)
        if verdict not in VERDICT_SCORES:
            return {
                "verdict": "unjudged",
                "score": 0.0,
                "reason": reason or "teacher returned no valid verdict",
                "judge": f"{judge_model}@{judge_url}",
                "elapsed_ms": round((time_module.perf_counter() - started) * 1000.0, 1),
            }
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = VERDICT_SCORES[verdict]
        score = max(0.0, min(1.0, score))
        self.graded += 1
        return {
            "verdict": verdict,
            "score": score,
            "reason": reason,
            "judge": f"{judge_model}@{judge_url}",
            "elapsed_ms": round((time_module.perf_counter() - started) * 1000.0, 1),
        }

    def judge_many(self, items, *, progress: bool = False) -> list[dict]:
        """Judge a list of ``(question, candidate, reference)`` triples."""

        results: list[dict] = []
        for index, item in enumerate(items, 1):
            question, candidate, reference = (list(item) + [None, None])[:3]
            verdict = self.judge(question, candidate, reference)
            verdict.update({"question": question, "candidate": candidate, "reference": reference})
            results.append(verdict)
            if progress:
                print(
                    f"  judge {index}/{len(items)} {verdict['verdict']:<7} "
                    f"Q={question!r} got={candidate!r} ref={reference!r}",
                    flush=True,
                )
        return results

    # ---------------- teaching ----------------
    def teach(
        self,
        prompt: str,
        *,
        expected: str | None = None,
        max_len: int = 12,
        rehearse: bool = False,
        rehearsal_ms: float = 25.0,
    ) -> dict:
        """One supervised episode: answer, grade, then reward or correct locally."""

        if self.brain is None:
            raise ValueError("LlmTeacherAgent.teach requires a brain")

        answer = self.brain.respond(prompt, max_len=max_len)
        got = " ".join(answer)
        if expected is None:
            messages = [
                {"role": "system", "content": CURRICULUM_SYSTEM_PROMPT},
                {"role": "user", "content": f"主题：{prompt}\n请给出这一条问题的参考答案（只输出 JSON 数组，1 条）。"},
            ]
            try:
                items = self._normalize_items(self.client.chat_json(messages, temperature=0.2))
            except (TeacherUnavailable, ValueError):
                items = []
            expected = items[0]["answer"] if items else ""
        if not expected:
            return {"prompt": prompt, "got": answer, "expected": "", "verdict": "unknown", "mode": "skipped"}

        review = self.judge(prompt, got, expected, offline_fallback=False)
        if review["verdict"] == "unjudged":
            return {"prompt": prompt, "got": answer, "expected": expected, "verdict": "unjudged", "mode": "skipped", "judge": review["judge"]}
        if review["verdict"] == "correct":
            result = self.brain.reward(0.5)
            mode = "reward"
        else:
            self.brain.correct(prompt, expected, reward=1.0)
            result = self.brain.reward(0.8)
            mode = "correct"

        rehearsal: dict = {}
        if rehearse:
            rehearsal = self.brain.rehearse(prompt, expected, duration=rehearsal_ms)

        return {
            "prompt": prompt,
            "got": answer,
            "expected": expected,
            "verdict": review["verdict"],
            "feedback": review["feedback"],
            "mode": mode,
            "modulated_synapses": result.get("modulated"),
            "rehearsal": rehearsal,
        }




def build_teacher(brain: BionicBrain | None = None, cache_dir: str | Path | None = None) -> LlmTeacherAgent:
    """Convenience factory used by the training tools."""

    return LlmTeacherAgent(brain, cache_dir=cache_dir)
