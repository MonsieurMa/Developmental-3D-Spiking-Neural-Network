r"""Generate a human-like mixed-input corpus with the local teacher model.

The corpus follows the acquisition typology instead of being a QA set:

* channel:   auditory (spoken) / visual (written)
* source:    caregiver, peer, teacher, media, self
* input:     one-way stream, interactive exchange, grounded situation
* genre:     naming, narrative, instruction, exposition, meta-language
* stage:     infant -> toddler -> school -> adult

Written to ``data/corpora/mix_v1/<category>.jsonl``; every record carries its
metadata so the trainer can present it through the right channel and mode.

Usage::

    .\.venv\Scripts\python.exe tools\build_training_mix.py --per-category 12
    .\.venv\Scripts\python.exe tools\build_training_mix.py --only caregiver_speech
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

from bionic_brain.agents.llm_teacher import LlmTeacherAgent, TeacherUnavailable, extract_json
from bionic_brain.language.subwords import SubwordTokenizer
_STRUCTURAL_TOKENIZER = SubwordTokenizer()

OUTPUT_DIR = ROOT / "data" / "corpora" / "mix_v1"

# Each category: name -> (metadata, prompt). Streams are plain text lines;
# interactive categories return question/answer pairs.
CATEGORIES: dict[str, dict] = {
    "caregiver_speech": {
        "meta": {"stage": "infant", "modality": "auditory", "kind": "interactive", "genre": "儿向语", "source": "caregiver"},
        "prompt": "婴儿期儿向语和指物命名。句子要短（3-8 个字）、重复、带呼唤和指认，例如「宝宝 看 这 是 猫 猫」。",
        "fields": ["text"],
    },
    "picture_book": {
        "meta": {"stage": "toddler", "modality": "visual", "kind": "one_way", "genre": "绘本叙事", "source": "caregiver"},
        "prompt": "幼儿绘本共读文本。每句 5-12 字，简单叙事，讲谁在哪里做什么，有重复句式。",
        "fields": ["text"],
    },
    "peer_play": {
        "meta": {"stage": "child", "modality": "auditory", "kind": "interactive", "genre": "同伴游戏", "source": "peer"},
        "prompt": "儿童同伴游戏对话。轮流说话，有请求、协商、分享，例如「该 我 了」「一起 玩 吧」。",
        "fields": ["text"],
    },
    "classroom": {
        "meta": {"stage": "school", "modality": "auditory", "kind": "interactive", "genre": "课堂讲解", "source": "teacher"},
        "prompt": "小学课堂讲解。老师提问、讲解、纠正，学生回应。每句 6-15 字，包含因果和定义。",
        "fields": ["text"],
    },
    "story": {
        "meta": {"stage": "school", "modality": "visual", "kind": "one_way", "genre": "叙事", "source": "book"},
        "prompt": "分级读物叙事。每句 6-16 字，有角色、动作、地点、结果，前后连贯。",
        "fields": ["text"],
    },
    "instruction": {
        "meta": {"stage": "school", "modality": "visual", "kind": "one_way", "genre": "指令说明", "source": "book"},
        "prompt": "生活与学习指令说明。每句 5-14 字，先做什么再做什么，有条件和顺序。",
        "fields": ["text"],
    },
    "exposition": {
        "meta": {"stage": "adult", "modality": "visual", "kind": "one_way", "genre": "说明议论", "source": "book"},
        "prompt": "说明性短文句子。每句 8-20 字，解释原因、机制或对比，例如「因为 水 受热 膨胀 所以 体积 变大」。",
        "fields": ["text"],
    },
    "media_script": {
        "meta": {"stage": "adult", "modality": "auditory", "kind": "one_way", "genre": "广播短视频", "source": "media"},
        "prompt": "广播或短视频口播稿。口语化、有停顿感，每句 6-16 字，含提示和总结。",
        "fields": ["text"],
    },
    "self_talk": {
        "meta": {"stage": "adult", "modality": "visual", "kind": "one_way", "genre": "自我对话日记", "source": "self"},
        "prompt": "自我对话或日记。第一人称，反思、计划、提醒自己，每句 6-16 字。",
        "fields": ["text"],
    },
    "meta_language": {
        "meta": {"stage": "adult", "modality": "visual", "kind": "one_way", "genre": "元语言", "source": "teacher"},
        "prompt": "语言知识讲解：字词释义、用法、纠错。每句 6-18 字，例如「谢谢 是 礼貌 用语 用于 接受 帮助」。",
        "fields": ["text"],
    },
    "grounded_scene": {
        "meta": {"stage": "child", "modality": "visual", "kind": "grounded", "genre": "情境场景", "source": "caregiver"},
        "prompt": "情境化场景句：某人把某物给某人 / 某人在某处做某事。每句 6-14 字，角色和物体用常见中文词。",
        "fields": ["text"],
    },
    "dialogue_qa": {
        "meta": {"stage": "child", "modality": "auditory", "kind": "interactive", "genre": "问答互动", "source": "caregiver"},
        "prompt": "幼儿问答互动。问题 4-10 字，答案 1-6 字，从问题可推出答案，例如问「谁 在 房间」答「小明」。",
        "fields": ["question", "answer"],
    },
    "longform_story": {
        "meta": {"stage": "school", "modality": "visual", "kind": "one_way", "genre": "长文叙事", "source": "book"},
        "prompt": (
            "连贯长文，每条约 6-10 句、150-400 字，写成一个小故事：有角色、地点、起因、经过、结果。"
            "句与句之间要有指代和连接词（他/她/它/然后/于是/因为/所以/后来），整条放进一个 text 字段。"
            "不要分点、不要标题、不要解释，直接写正文。"
        ),
        "fields": ["text"],
        "count": 3,
    },
    "longform_exposition": {
        "meta": {"stage": "adult", "modality": "visual", "kind": "one_way", "genre": "长文说明", "source": "book"},
        "prompt": (
            "连贯说明文，每条约 6-10 句、150-400 字，解释一个日常现象的原因和过程，"
            "要有因果链（因为…所以…，如果…就…）和前后指代，整条放进一个 text 字段。"
            "不要分点、不要标题、不要解释，直接写正文。"
        ),
        "fields": ["text"],
        "count": 3,
    },
}

SYSTEM = """你是语言习得语料生成器。你要为 BionicBrain 脉冲神经网络生成人类式的语言输入。
规则：
1. 只输出 JSON 数组，不要解释、不要 Markdown 代码块。
2. 中文按自然写法书写（不要逐字加空格），标点用中文标点。
3. 内容具体、可理解、贴近真实使用场景，不要抽象空话。
4. 句子之间要有变化，不要重复同一句。"""

# Python-programming profile: same pipeline, different subject matter. Code is
# written naturally so the tokenizer sees English words, digits and punctuation
# as separate tokens (print, (, ", hello, ", )).
PYTHON_CATEGORIES: dict[str, dict] = {
    "py_concept": {
        "meta": {"stage": "adult", "modality": "visual", "kind": "one_way", "genre": "概念讲解", "source": "teacher"},
        "prompt": "Python 核心概念讲解，每句 8-20 字，一次只讲一个概念（变量、列表、字典、循环、函数、条件、异常、类）。",
        "fields": ["text"],
    },
    "py_syntax": {
        "meta": {"stage": "adult", "modality": "visual", "kind": "one_way", "genre": "语法规则", "source": "teacher"},
        "prompt": "Python 语法规则说明，每句 8-20 字，包含关键字和书写要求，例如「for 循环 用 冒号 和 缩进」。",
        "fields": ["text"],
    },
    "py_example": {
        "meta": {"stage": "adult", "modality": "visual", "kind": "one_way", "genre": "代码示例", "source": "teacher"},
        "prompt": "Python 代码示例，一段 1-3 行代码后面跟一句中文说明，代码用自然写法例如「print(\"hello\") 会 输出 hello」。",
        "fields": ["text"],
    },
    "py_terms": {
        "meta": {"stage": "adult", "modality": "visual", "kind": "one_way", "genre": "术语对照", "source": "teacher"},
        "prompt": "Python 术语中英对照，每句形如「列表 的 英文 是 list」，覆盖 list/dict/loop/function/class/string/int/float。",
        "fields": ["text"],
    },
    "py_fact": {
        "meta": {"stage": "adult", "modality": "visual", "kind": "one_way", "genre": "事实陈述", "source": "teacher"},
        "prompt": "Python 事实陈述，每句形如「列表 用 方括号」或「len 函数 返回 长度」，主语明确、可被提问。",
        "fields": ["text"],
    },
    "py_qa": {
        "meta": {"stage": "adult", "modality": "auditory", "kind": "interactive", "genre": "问答互动", "source": "teacher"},
        "prompt": (
            "Python 编程问答，问题 4-12 字，答案 1-6 字且必须是**最短的标准形态**"
            "（例如答「方括号」而不是「使用方括号 []」），从问题可推出唯一答案。"
        ),
        "fields": ["question", "answer"],
    },
    "py_debug": {
        "meta": {"stage": "adult", "modality": "auditory", "kind": "interactive", "genre": "纠错讲解", "source": "teacher"},
        "prompt": "Python 常见错误与修法，一句说现象、一句说原因、一句说改法，每句 6-16 字。",
        "fields": ["text"],
    },
    "py_operator": {
        "meta": {"stage": "adult", "modality": "visual", "kind": "one_way", "genre": "运算符", "source": "teacher"},
        "prompt": (
            "Python 运算符的中文名与用途，每句形如「双斜杠 是 整除 运算符」或「百分号 是 取余 运算符」，"
            "覆盖 // % ** += and or not == != <= >=。"
        ),
        "fields": ["text"],
    },
    "py_method": {
        "meta": {"stage": "adult", "modality": "visual", "kind": "one_way", "genre": "常用方法", "source": "teacher"},
        "prompt": (
            "Python 常用函数与方法，每句形如「append 方法 向 列表 末尾 添加 元素」或「len 函数 返回 长度」，"
            "覆盖 append pop len keys values items split join strip。"
        ),
        "fields": ["text"],
    },
    "py_paraphrase": {
        "meta": {"stage": "adult", "modality": "auditory", "kind": "interactive", "genre": "同义问法", "source": "teacher"},
        "prompt": (
            "同一个 Python 知识点的多种问法，每条答案相同但问法不同。例如「列表 用 什么 括号 / 列表 的 符号 是 什么 / "
            "怎么 创建 列表」都答「方括号」。**同一知识点必须始终使用同一个最短答案写法**（1-6 字），不要出现两种写法。"
            "请覆盖 print、def、缩进、井号、方括号、花括号、import、try 等知识点。"
        ),
        "fields": ["question", "answer"],
    },
    "py_debug_pairs": {
        "meta": {"stage": "adult", "modality": "auditory", "kind": "interactive", "genre": "纠错问答", "source": "teacher"},
        "prompt": (
            "Python 纠错问答：问题是学生写错的做法，答案是正确做法。例如问「变量 用 什么 声明」答「直接 赋值」，"
            "问「输出 用 printf 对吗」答「应该 用 print」。"
        ),
        "fields": ["question", "answer"],
    },
    "py_triple": {
        "meta": {"stage": "adult", "modality": "auditory", "kind": "interactive", "genre": "正反用途三问", "source": "teacher"},
        "prompt": (
            "同一个 Python 知识点要给出**三种问法**，答案各不相同但都围绕同一事实："
            "①正向（例如「列表 用 什么 括号」→「方括号」）；"
            "②反向（例如「方括号 用来 定义 什么」→「列表」）；"
            "③用途（例如「列表 用来 做什么」→「存放 一组 数据」）。"
            "每条一个问答对，覆盖 print/def/缩进/井号/方括号/花括号/import/try/for/while/加号/双等号。"
        ),
        "fields": ["question", "answer"],
    },
    "py_dialogue": {
        "meta": {"stage": "adult", "modality": "auditory", "kind": "one_way", "genre": "课堂对话", "source": "teacher"},
        "prompt": "Python 课堂对话，学生问、老师答，每句 5-15 字，共 3-4 句，话题围绕循环、列表或函数。",
        "fields": ["text"],
    },
}


PROFILES = {"language": CATEGORIES, "python": PYTHON_CATEGORIES}


def prompt_for(spec: dict, count: int) -> str:
    fields = spec["fields"]
    if fields == ["text"]:
        shape = '{"text": "..."}'
    else:
        shape = '{"question": "...", "answer": "..."}'
    return f"""任务：{spec['prompt']}

请生成 {count} 条，每条形如 {shape}。
再次强调：只输出 JSON 数组。"""


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def write_probe(profile_name: str, out_dir: Path, probe_path: Path, teacher: LlmTeacherAgent) -> None:
    """Write a held-out probe; questions already in the corpus are dropped."""

    known: set[str] = set()
    for path in out_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for key in ("question", "answer", "text"):
                if record.get(key):
                    known.add(str(record[key]))
    if profile_name == "language" and (out_dir / "dialogue_qa.jsonl").exists():
        rows = [
            {"question": json.loads(line)["question"], "answer": json.loads(line)["answer"], "group": "dialogue"}
            for line in (out_dir / "dialogue_qa.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        topic = "Python 编程基础" if profile_name == "python" else "日常常识"
        messages = [
            {"role": "system", "content": "你是教学评测出题人。只输出 JSON 数组，不要解释、不要 Markdown。"},
            {"role": "user", "content": (
                f"出 14 道{topic}题，每题问题 4-14 字、答案 1-6 字，问题各不相同，"
                '形如 [{"question": "...", "answer": "..."}]。'
            )},
        ]
        payload = extract_json(teacher.client.chat(messages, temperature=0.8, max_tokens=2048))
        if isinstance(payload, dict):
            payload = payload.get("items", [])
        rows = [
            {"question": str(item.get("question", "")).strip(), "answer": str(item.get("answer", "")).strip(), "group": profile_name}
            for item in (payload if isinstance(payload, list) else [])
            if item.get("question") and item.get("answer")
        ]
    held_out = [row for row in rows if row["question"] not in known]
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in held_out) + "\n", encoding="utf-8")
    print(f"probe={probe_path} items={len(held_out)} (dropped {len(rows) - len(held_out)} seen in corpus)", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a mixed-input training corpus with the local teacher")
    parser.add_argument("--per-category", type=int, default=0, help="0 uses each category's own default count")
    parser.add_argument("--rounds", type=int, default=1, help="generation rounds per category")
    parser.add_argument("--profile", default="language", choices=sorted(PROFILES), help="corpus subject profile")
    parser.add_argument("--only", action="append", default=[], help="limit to these categories")
    parser.add_argument("--out-dir", default="", help="defaults to data/corpora/<profile>_v1")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed-file", action="store_true", help="ignore existing files and regenerate")
    parser.add_argument("--write-probe", default="", help="also write a held-out eval probe to this path")
    args = parser.parse_args(argv)

    teacher = LlmTeacherAgent(cache_dir=ROOT / "data/corpora/llm_generated")
    if args.base_url:
        teacher.config.base_url = args.base_url
    if args.model:
        teacher.config.model = args.model
    print(f"teacher={json.dumps(teacher.describe(), ensure_ascii=False)}", flush=True)
    if not teacher.online:
        print("teacher endpoint unreachable; aborting", flush=True)
        return 2

    profile = PROFILES[args.profile]
    names = args.only or list(profile)
    default_out = OUTPUT_DIR if args.profile == "language" else (ROOT / "data" / "corpora" / f"{args.profile}_v1")
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in names:
        if name not in profile:
            print(f"unknown category: {name}", flush=True)
            continue
        spec = profile[name]
        path = out_dir / f"{name}.jsonl"
        existing = [] if args.seed_file else load_existing(path)
        seen = {json.dumps({k: record.get(k) for k in spec['fields']}, ensure_ascii=False) for record in existing}
        fresh: list[dict] = []
        per_call = int(args.per_category or spec.get("count", 12))
        for round_index in range(max(1, args.rounds)):
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt_for(spec, per_call)},
            ]
            try:
                payload = extract_json(teacher.client.chat(messages, temperature=0.7))
            except (TeacherUnavailable, ValueError) as error:
                print(f"{name}: generation failed ({error})", flush=True)
                break
            if isinstance(payload, dict):
                payload = payload.get("items", [payload])
            for item in payload if isinstance(payload, list) else []:
                if not isinstance(item, dict):
                    continue
                record = {key: str(item.get(key, "")).strip() for key in spec["fields"]}
                if not record["text" if "text" in record else "question"]:
                    continue
                if any(not _STRUCTURAL_TOKENIZER.encode(value) for value in record.values() if value):
                    continue
                key = json.dumps(record, ensure_ascii=False)
                if key in seen:
                    continue
                seen.add(key)
                fresh.append({**record, **spec["meta"]})
        if fresh:
            with path.open("a", encoding="utf-8") as handle:
                for record in fresh:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        total += len(fresh)
        print(f"{name:<18} +{len(fresh):<3} (file total {len(existing) + len(fresh)})  {spec['meta']['genre']}/{spec['meta']['modality']}", flush=True)
    print(f"generated={total} -> {out_dir}", flush=True)

    if args.write_probe:
        write_probe(args.profile, out_dir, Path(args.write_probe), teacher)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


