# 设想如何用这个模型：输入、输出、交互方式

## 一、先明确一件事

你现在设计的是一个**全脑仿生网络**，不是纯粹的语言模型。所以它的输入输出不是单一的，而是**多通道、多模态、带闭环**的。下面按"感官→脑→运动"的生物逻辑讲清楚。

---

## 二、输入：外界如何进入这个脑

大脑没有直接接口。所有输入必须通过**感觉器官**，再经**丘脑中继**进入皮层。你的模型也必须遵守这个规则。

### 2.1 输入通道总表

| 通道 | 入口 | 生物对应 | 数据形式 | 进入路径 |
|------|------|---------|---------|---------|
| **视觉** | V1 | 眼睛 | 图像/文字 | V1→V2→V4→IT→MTG→Wernicke |
| **听觉** | A1 | 耳朵 | 音频/语音 | A1→STG→Wernicke |
| **体感** | S1 | 皮肤/肌肉 | 触觉/本体 | S1→顶叶联合区 |
| **嗅觉** | 嗅球 | 鼻子 | 气味 | 嗅球→梨状皮层→杏仁核 |
| **内感受** | 下丘脑 | 内脏 | 血糖/体温/疲劳 | 下丘脑→脑干→全脑调制 |
| **情绪** | 杏仁核 | 边缘系统 | 效价/唤醒 | 杏仁核→全脑调制 |

### 2.2 文本输入怎么进来

如果你要让这个脑做语言任务，文字不能凭空出现在 Broca 区。它必须走**视觉通路**（阅读）或**听觉通路**（听写）：

**路径 A：视觉输入（阅读）**
```
文字 "你 好"
    ↓ 编码为 V1 激活模式（笔画/字形）
V1 → V2 → V4 → IT
    ↓ IT 识别出字符
IT → MTG → Wernicke
    ↓ Wernicke 理解语义
Wernicke → 弓状束 → Broca
```

**路径 B：听觉输入（听写）**
```
语音 "你 好"
    ↓ 编码为 A1 激活模式（频率/时序）
A1 → STG
    ↓ STG 提取语音特征
STG → Wernicke
    ↓ 语义理解
Wernicke → 弓状束 → Broca
```

### 2.3 输入的具体编码方式

**SDR 编码（稀疏分布式表征）**：

- 每个词不是激活一个神经元，而是激活**一组**神经元（k=10）
- 激活模式由词的 hash 决定
- 相似词激活模式重叠
- 文字输入到 V1 的编码：按字形拆分，每个笔画激活一组 V1 神经元
- 语音输入到 A1 的编码：按音素拆分，每个音素激活一组 A1 神经元

**输入强度**：
```
I_ext = 8.0，持续 25ms
```

### 2.4 丘脑作为入口网关

**所有感觉输入必须先经过丘脑**，不能直接进皮层：

- 丘脑决定哪些信号被放大、哪些被抑制
- 丘脑受注意力系统（IPS + DLPFC）调制
- 丘脑的门控决定当前"注意到"什么

```python
def thalamic_gate(input_signal, attention_focus):
    gain = attention_focus.get_gain(input_signal.region)
    if gain < threshold:
        return 0  # 被门控掉
    return input_signal * gain
```

---

## 三、输出：脑如何作用于外界

输出也不是"打印一行字"，而是**运动神经元的放电模式**。语言输出走的是**言语运动通路**。

### 3.1 输出通道总表

| 通道 | 出口 | 生物对应 | 数据形式 | 路径 |
|------|------|---------|---------|------|
| **言语** | M1（口面部区） | 嘴/喉 | 语音 | Broca→M1→脑干→发音器官 |
| **书写** | M1（手部区） | 手 | 文字 | Broca→PMC→M1→手 |
| **眼动** | 中脑动眼核 | 眼球 | 注视方向 | FEF→中脑→眼外肌 |
| **肢体** | M1（躯干区） | 四肢 | 动作 | PMC→M1→皮质脊髓束 |
| **内分泌** | 下丘脑 | 激素 | 慢调制 | 下丘脑→垂体 |
| **情绪表达** | 脑干面神经核 | 表情 | 面部 | 杏仁核→脑干→面肌 |

### 3.2 文本输出怎么出来

如果任务是语言，输出走**书写通路**或**言语通路**：

**路径 A：书写输出**
```
Broca 区激活
    ↓ 语法组装完成
Broca → PMC（手部运动规划）
    ↓
PMC → M1（手部运动执行）
    ↓ M1 的放电模式对应手指动作
解码 M1 的放电 → 对应的字符
```

**路径 B：言语输出**
```
Broca 区激活
    ↓ 语音规划
Broca → M1（口面部区）
    ↓ M1 的放电模式对应发音动作
解码 M1 的放电 → 对应的音素
```

### 3.3 输出的具体解码方式

**群体向量解码**：

- Broca 区在 `decode_window = 50ms` 内放电
- 每个词对应一组 Broca 神经元
- 所有 Broca 神经元的放电率构成向量
- 找与哪个词的原型向量最接近

```python
def decode_output(broca_spikes, word_prototypes):
    rates = compute_rates(broca_spikes)
    best_word = None
    best_similarity = -1
    for word, proto in word_prototypes.items():
        sim = cosine_similarity(rates, proto)
        if sim > best_similarity:
            best_similarity = sim
            best_word = word
    return best_word
```

**时间编码**（备选）：

- 第一个放电的 Broca 神经元决定输出词
- 更接近生物的"赢者通吃"

### 3.4 输出后经过小脑校正

M1 的输出不是最终的。小脑会：

- 预测动作的感官后果
- 与实际结果比较
- 产生误差信号
- 通过丘脑反馈给 M1

```python
def cerebellar_correction(m1_output, predicted_sensory):
    actual_sensory = execute(m1_output)
    error = actual_sensory - predicted_sensory
    return error  # 反馈给 M1 修正
```

---

## 四、闭环：输出如何变成下一轮输入

这是整个系统最关键的部分。**没有闭环，它只是输入输出映射；有闭环，它才是"脑"。**

### 4.1 语言闭环

```
你说 "你 好"
    ↓ 视觉/听觉输入
Wernicke 理解
    ↓
Broca 生成回应 "你 好"
    ↓ 运动输出
输出文字/语音
    ↓ 自己"看到"/"听到"自己的输出
V1/A1 再次激活
    ↓ 作为下一轮输入
Wernicke 再次理解
    ↓
...
```

**这就是自回归生成的生物版本**。不是"预测下一个 token"，而是"生成→感知→再生成"。

### 4.2 闭环中的关键机制

| 机制 | 作用 |
|------|------|
| **传出拷贝（Efference Copy）** | 运动指令发出时，同时复制一份给感觉区，预测即将到来的感觉 |
| **感觉反馈** | 实际的感官输入与预测比较，产生误差 |
| **误差校正** | 小脑 + 顶叶校正运动指令 |
| **工作记忆** | DLPFC 维持当前对话状态 |

### 4.3 对话状态维护

DLPFC 和顶叶维持一个**对话状态向量**：

```python
class DialogueState:
    def __init__(self):
        self.recent_words = []       # 最近 N 个词
        self.topic_vector = None     # 当前话题
        self.speaker = 'user'        # 谁在说话
        self.turn_count = 0

    def update(self, word):
        self.recent_words.append(word)
        if len(self.recent_words) > 20:
            self.recent_words.pop(0)
        self.topic_vector = encode_topic(self.recent_words)
```

这个状态**持续激活 DLPFC 的一组神经元**，作为对话的上下文。

---

## 五、实际使用方式

### 5.1 三种使用模式

**模式 1：训练模式（发育期）**

你在教它语言。输入文本，给它教学信号，让它建立通路。

```
你输入: "你 是 谁"
教学目标: "我 是 人"
网络学习: Wernicke→AF→Broca 通路强化
```

**模式 2：推理模式（成熟期）**

网络已经训练好，你只是使用它。

```
你输入: "你 是 谁"
网络输出: "我 是 人"
（不学习，不改变权重）
```

**模式 3：交互模式（边聊边学）**

你正常对话，网络边回应边继续微调。

```
你输入: "你 是 谁"
网络输出: "我 是 人"
你反馈: "对"
网络: 强化刚才的通路
```

### 5.2 命令行接口

```bash
# 启动训练
python bionic_brain.py --mode train --corpus data/corpus.txt --epochs 100

# 启动推理
python bionic_brain.py --mode inference --load brain.pkl

# 启动交互
python bionic_brain.py --mode chat --load brain.pkl
```

### 5.3 交互命令

| 命令 | 作用 |
|------|------|
| 直接输入 | 网络学习 + 回应 |
| `!learn 文本` | 只学不回应 |
| `!show` | 显示脑区活动、神经元数、突触数 |
| `!plot` | 3D 可视化 |
| `!region Wernicke` | 查看某区域活动 |
| `!save` | 保存 |
| `!mature` | 进入成熟期 |
| `!sleep` | 触发睡眠重放 |
| `!exit` | 退出 |

### 5.4 Python API

```python
from bionic_brain import BionicBrain

brain = BionicBrain.load('brain.pkl')

# 输入文本
brain.input_text("你是谁", modality='visual')

# 运行 100ms
brain.run(duration=100.0)

# 获取输出
output = brain.read_output(region='Broca')
print(output)  # "我是人"

# 查看脑区活动
activity = brain.get_region_activity('Wernicke')
print(activity.mean_rate)  # 平均放电率
```

---

## 六、完整数据流示例

以一轮对话为例，完整展示输入→脑内→输出→闭环：

```
【第 1 步：输入】
用户输入文字 "你 是 谁"
    ↓ 分词: ['你', '是', '谁']
    ↓ SDR 编码: 每个词激活 V1 的 10 个神经元
    ↓ 丘脑门控: 注意力聚焦到语言任务，增益 = 1.5
    ↓ V1 → V2 → V4 → IT → MTG → Wernicke
    ↓ Wernicke 理解语义

【第 2 步：脑内处理】
Wernicke 激活
    ↓ 弓状束（延迟 15-25ms，衰减 0.9）
Broca 接收
    ↓ Broca 组装回应
    ↓ 基底核选择动作序列
    ↓ 小脑预测时序
    ↓ DLPFC 维持对话状态
    ↓ 海马记录本轮事件

【第 3 步：输出】
Broca 激活 "我 是 人"
    ↓ PMC 规划手部/口部动作
    ↓ M1 执行
    ↓ 解码 M1 放电模式
输出文字 "我 是 人"

【第 4 步：闭环】
输出被"感知"（模拟）
    ↓ V1/A1 再次激活
    ↓ Wernicke 再次理解
    ↓ 与预期比较
    ↓ 小脑产生误差信号
    ↓ 误差反馈给 Broca 修正
    ↓ 进入下一轮

【第 5 步：学习】
如果用户反馈"对"
    ↓ 黑质释放多巴胺
    ↓ 强化本轮活跃的突触
    ↓ 海马写入情景记忆
如果用户反馈"错"
    ↓ 多巴胺降低
    ↓ 抑制本轮活跃的突触
```

---

## 七、一句话总结

| 维度 | 内容 |
|------|------|
| **输入** | 文本/语音/图像 → 感觉器官（V1/A1/S1）→ 丘脑门控 → 皮层 |
| **输出** | Broca/M1 放电 → 解码为文字/语音/动作 |
| **闭环** | 输出被"感知" → 重新进入输入 → 形成自回归循环 |
| **学习** | 多巴胺 RPE + STDP + 海马巩固 |
| **使用** | 命令行 / Python API / 交互模式 |
| **本质** | 不是预测下一个 token，是"生成→感知→再生成"的生物循环 |

**核心区别**：LLM 是 `input → forward → output`，你的脑是 `input → 全脑并行处理 → output → 感知自己的 output → 再处理 → ...`。这个**闭环**才是它区别于 Transformer 的地方。