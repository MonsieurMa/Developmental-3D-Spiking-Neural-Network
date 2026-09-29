# TODO：零硬编码路线图

更新时间：2026-09-21
定位：这是本项目的**最高优先级待办**，取代"用符号规则换分数"的所有做法。
目标修正（2026-09-21）：** neurally-first inference 是唯一目标**。旧的 `brian_neural_ocean.pkl`
不是验收目标；它和所有依赖符号语言回退的 checkpoint 都只作历史材料。新的 `BionicBrain`
默认使用 `LearnedSchemaMemory`、learned relation discovery、holographic binding、neural sequence
和 learned grounding；旧 `memory/schemas.py` 已移到 `legacy/symbolic_schemas.py`，默认推理不导入。

## 0. 不变的宗旨

- **仿生人脑**：模仿人类的思维路径与学习路径（预测 → 互动 → 巩固 → 发育），不是"做一个聊天接口"。
- **神经元四结构必须显式建模**：
  1. **树突**——局部输入整合（分支非线性、输入分组、增益调制）
  2. **胞体**——阈值/发放动力学（HH/LIF、适应、能量）
  3. **轴突**——传导延迟与髓鞘化（已有 tract/delay 机制）
  4. **突触**——局部可塑性（STDP/资格痕迹/调质门控）
- **一切计算都是网络计算**：不允许词表、正则、if-else 规则承担语言知识。
- **允许的数学工具**（约束放宽）：反向传播、卷积、注意力/Transformer、PyTorch、图网络、可微分仿真。
  但**每个可微分组件都必须映射到四结构之一**，并说明它在脑内对应什么（例如：卷积=树突分支的局部感受野；注意力=丘脑门控的选择性增益；反向传播=自上而下预测误差的局部近似）。

## 1. 现状：硬编码清单（要逐条清零）

| # | 位置 | 现在写死了什么 | 为什么学不了新语言 |
|---|---|---|---|
| H1 | `memory/schemas.py` `ROLE_MARKERS`(11) | 把/给/在/是/有/拿/喜欢… | 词典保留为旧 checkpoint 回退；learned path 已过固定无词表探针，但未改默认 |
| H2 | `…` `RELATION_TOKENS`(12)、`PREPOSITION_VERBS`(10) | 关系词与介词动词 | 词典保留为旧 checkpoint 回退；learned path 已过固定无词表探针，但未改默认 |
| H3 | `…` `TRANSFER_MARKERS`(3) | "给/递/送 = 转移" | 语义类别写死 |
| H4 | `…` `QUERY_WORDS` + `query()` 问句模板 | 哪一类/哪里/谁/什么 的解析 | 词典只作回退；`IntentLearner` 已实现从教师标签学习 opaque intent |
| H5 | ~~`PARTICLE_TOKENS`/`STOP_TOKENS`、`dialogue.py` 语气词表~~ | **已移除**；标点用 Unicode 字母/数字结构判定，句末助词和 schema 边界由在线 df/终位频率学习 | 语言相关规则已清除 |
| H6 | ~~`language/features.py` `GRAMMAR_TOKENS`(70)~~ | **已移除**；由在线 df z-score 突出度学习 | 语言相关规则已清除 |
| H7 | `brain.py` `_analogy_filler` | 类比推断规则 | **已改为 learned context features + learned relation score** |
| H8 | `brain.py` `self_practice` | 自出题模板句 | **已改为 learned statement 的 subject/relation probe** |
| H9 | `learning/sequence.py` | n-gram 计数 + 回退 | 词典只作回退；`DifferentiableSequenceModel` 已实现并接线为开关 |
| H10 | `brain.py` 解码阈值/gating 常数 | margin/coverage/support 门限 | 常数只作默认；`DecoderCalibration` 已实现本脑证据学习开关 |
| H11 | `language/tokenizer.py` | 正则分词（中文逐字、拉丁整词） | 正则只作未训练 fallback；BPE/字符 learned path 已接线 |
| H12 | `agents/llm_teacher.py` 的语料格式约束 | "中文逐字拆开"写进提示词 | **已移除语言专用格式；教师只受语料 token 序列约束** |
| H13 | ~~`learning/sequence.py` `BASE_LEXICON`(59)~~ | **已移除**；`LearnedSegmenter` 使用语料学到的 BPE，未训练时只按字母/数字 run 与标点边界 | 换语言只换语料 |
| H14 | ~~`config/defaults.py` `generation_stop_tokens`~~ | **已移除**；段边界用非字母/数字结构判定，文档结束仍由学到的 `</s>/<eos>` 决定 | 语言相关规则已清除 |

**判定标准**：换一种语言（例如英语、日语）只允许换语料，**不允许改任何 .py 文件**。当前代码不满足这一条——这是 P0 的验收口径。

## 2. 替换原则

1. **语言无关**：所有语言知识来自数据，不来自代码。
2. **可归因**：每个新机制都要能被消融（关掉它分数掉多少）。
3. **可测**：沿用现有三对照协议（未见性检查 / 机制消融 / 未训练基线）+ 新增**跨语言迁移**判据。
4. **四结构落点**：每引入一个可微分组件，必须在文档里写清它对应树突/胞体/轴突/突触中的哪一个，以及"反向传播"在这一层代表什么。

## 3. 分阶段 TODO

### P0 基础设施（先做，未完成前不碰上层）
- [x] **P0-1 引入 PyTorch**：CPU wheel 依赖已接入；`src/bionic_brain/nn/device.py` 提供 CPU/GPU 选择与
  NumPy ↔ Tensor 互操作。脉冲状态仍由 `BionicBrain` 持有，张量只负责"可学习的部分"。
- [x] **P0-2 四结构可微模块**：`src/bionic_brain/nn/modules.py` 已定义 `Dendrite`/`Soma`/`Axon`/`Synapse`
  四个结构、局部学习规则、替代梯度和 `NeuronCircuit`；反向传播只作为局部误差近似。
- [x] **P0-3 硬编码依赖度量**：`tools/hardcode_dependence.py` 已固定探针和消融配置，并输出第 7 节基线。
- [x] **P0-4 跨语言评测集（2026-09-19 完成）**：`data/corpora/english_v1/`
  （12 条英文事实 + 4 条同构英文问题）。当前模型读完英文事实后作答：**mean_score = 0.000**
  （输出中文碎片 `值 使 None`、`cat is the`），而同一批数据上 `RelationDiscoverer` 无需任何词表就发现了
  `is / in`。→ 规则是中文专用的，学习组件是跨语言的，证据齐全。
- [x] **P0-5 评测口径冻结**：`judged_correct` / `mean_score`（大模型判分）+ 三对照 + 跨语言迁移已写入 `AGENTS.md`。

### P1 表征与词法（把"字/词/关系"从代码搬到学习）
- [~] **P1-1 子词学习（2026-09-19 进行中）**：`src/bionic_brain/language/subwords.py`
  —— 频率驱动的 BPE 合并，初始符号只有"字符"，零词表零语言假设；含**结构上限**
  `max_unit_symbols`（防止把关系与论元粘成一个单元）。测试：`tests/test_subword_tokenizer.py`。

  实测（14 句中文 + 10 句英文玩具语料）：
  * 学出的单元包含 `小猫/小狗/小兔/喜欢`（中文）与多字符拉丁单元（英文）→ 学习机制成立；
  * 但也会学出 `在房`、`子是`、`furn/itur` 这类跨边界单元 → **需要互引导**：
    用（同样学到的）关系分数约束合并（不合并含关系标记的 pair），再反过来让更干净的单位改进关系发现。
  * **2026-09-20 接线进展**：`BionicBrain._tokenize()` 已替换全部运行时调用点；开关
    `use_subword_tokenizer`，训练入口 `BionicBrain.train_subword_tokenizer()`，CLI 支持
    `--use-subwords/--subword-state`。`tools/train_subword_tokenizer.py` 已在 mix_v1 + python_v1
    学出 512 个单元。合并器现在按 Unicode 字母/数字属性隔离标点，并用学到的关系标记约束后续合并
    （64 次合并引导后刷新，当前 guided markers 含 `在/是/用/的`）。
  * **2026-09-20 进展**：`NarrativeEventParser` 改用 `LearnedSegmenter`，删除 H13 `BASE_LEXICON`。
  * 剩余：同一探针上的词典/BPE A/B，以及在旧任务不退化前提下让“无词表”配置回到 0.75。

  **P1-1 完成标准**：接线（开关式）→ 在大语料上学习 → 与正则分词做 A/B（同一套探针）→
  在**不降低现有分数**的前提下，让 P1-3c 的"无词表"配置回到 0.75。
- [x] **P1-2 虚词/内容词区分（2026-09-20 完成）**：`ContextFeatureSpace` 用在线 df 的 z-score 突出度替换
  `GRAMMAR_TOKENS`；`core_dialogue_tokens` 和 `SchemaMemory` 也改用同一 `TokenStatistics`。
  标点是结构性边界；句末助词必须同时满足高终位亲和度与低 df 突出度。H5/H6 词表已删除，
  新增学习终位助词测试；全量机制测试通过。
- [~] **P1-3 关系词发现**：分布式方法——某个 token 两侧填充项变化越大，越像关系标记；用可微分目标学习 → 替换 H1、H2、H3。
- [x] **P1-3a 关系发现原型（2026-09-19）**：`src/bionic_brain/language/relations.py`
  —— 纯分布统计、零词表、语言无关。中文语料上发现 `在/是/把/吃`（同时被片段 `子/里` 污染），
  英文语料上发现 `is/in`（同一套代码）；`the` 这类全域高频词被自动排除。
  测试：`tests/test_relation_discovery.py`（中文发现是/在/把、英文发现 is/in、分数是相对的）。
- [x] **P1-3b 接线完成，但未达平替（2026-09-19）**：`SchemaMemory` 现在支持
  `use_learned_relations=True`：关系判定来自发现器、角色按**位置**分配（第一个关系=客体槽、
  第二个=接受者槽）、句内取**最强关系**而非第一个匹配、读完整篇后**二次解析**修正早期句子。
  实测 A/B（18 句语料、4 道探针）：

  | 配置 | 学到的标记集 | 结果 |
  |---|---|---|
  | 词典基线（`False`） | — | **4/4 正确** |
  | 学习关系（`True`） | `['在', '子']` | **0/4**（`小猫 是→狗 动 物`） |

  **诊断（P1-3c 要解决的）**：
  1. 相对筛选把 `是` 剔除了（其"较弱一侧熵 ≥0.35"的门限过严），却保留了片段 `子`；
  2. 缺少"是不是词的片段"判据——`子` 几乎总和 `桌/椅/兔` 组成高频二元组；
  3. 18 句语料对分布统计仍然太薄。

  开关默认 `False`（保持词典路径），所以现有分数没有回归。**P1-3c 验收**：学到的标记集必须覆盖
  是/在/把 且不含片段，并让 `tools/hardcode_dependence.py` 里"无词表"配置回到 0.75。

- [x] **P1-3c 机制验收通过（2026-09-21）**：把"取前 N 个"改成**相对分数选择**（相对最佳候选的比例），
  并加入**粘着度**（二元组 PMI，纯统计）作为片段判据。结果：

  | 配置 | 标记集 | 探针 |
  |---|---|---|
  | 词典基线 | — | **4/4** |
  | 学习关系 | `['在', '子', '是', '里']` | **2/4**（陈述类正确，地点类失败） |

  两个发现：
  1. 相对选择是关键——固定"前 N 个"会砍掉真正的标记 `是`，恢复后陈述类立刻正确；
  2. **粘着度无法区分 `子` 与 `在/是`**（实测同为 −0.616），因为 `子` 只是"桌子"被**逐字符切开**的碎片。

  → 结论（有证据的排序修正）：**必须先做 P1-1 学习的子词分词**，把"桌子/椅子"变成单一单元，
  关系发现才有干净输入；否则"碎片 vs 关系"在字符级下是病态问题，不可能靠统计解决。
  **2026-09-20 进展**：子词器已隔离标点并用关系发现器约束合并，大语料 tokenizer 中不再出现 `行。/代码。`
  这类标点粘接；该约束只解决碎片来源，尚未证明 schema 探针达标。
  在 8 题 judged 探针上，`no-relations` 从 2026-09-19 的 0.125 升到 **0.438**，但 baseline 仍为 0.750，
  因此还不能宣称 P1-3c 验收通过。
  2026-09-21 结果：学习标记集为 `{是, 在, 把}`，不含 `子/里`。`tools/hardcode_dependence.py`
  的 `no-relations` 配置启用 learned path 后得到 **0.750**，等于 0.750 验收线。
  这完成了 P1-3c 的固定探针验收；H1/H2 词典仍保留作旧 checkpoint 回退，P1-3 整体仍需跨语言迁移
  与未写过关系泛化达标后才能改默认。
- [x] **P1-4 感官接地（2026-09-21 实现完成）**：`CorticalGrounding` 把任意字符/音素符号学习为嵌入，
  经 dendritic receptive field 和 thalamic gate 进入语义区；`BionicBrain.ensure_word` 已接线，新细胞由
  learned field 排序。默认关闭，等 checkpoint A/B 后晋级。

### P2 句法与绑定（把"推理"从代码搬到网络）
- [x] **P2-1 分布式绑定（2026-09-21 实现）**：`TensorProductBinding` 用 circular convolution bind/unbind，
  cleanup memory 竞争恢复 filler；并行引擎已接 `BionicBrain.binding_memory`，旧 dict/schema 保留回退。
- [x] **P2-2 提问意图学习（2026-09-21 实现）**：`IntentLearner` 从教师提供的 opaque intent/slot 标签学习，
  `reason()` 先用 learned intent，再用 schema role 读出；问句词表不再是主路径。
- [x] **P2-3 未写过关系上的组合（2026-09-21 实现）**：binding trace 可组合两条 learned relation traces，
  测试在 opaque relation/entity 上验证可分别解绑；不做任何关系词假设。
- [x] **P2-4 指代与角色追踪（2026-09-21 实现）**：`DiscourseState` 用 salience decay、holographic role binding
  和 timeline 维护对话角色；schema frame 成功后同步到 `BionicBrain.discourse`。

### P3 预测与生成
- [x] **P3-1 可微下一词预测（2026-09-21 实现）**：`DifferentiableSequenceModel` 走 embedding→dendrite→
  soma→axon→synapse→readout，预测误差反传只作为局部 top-down error 近似；已接 `respond_sequence` 开关。
- [x] **P3-2 自发停止（2026-09-21 实现）**：神经候选在既有模型 end/no-support/repetition/fatigue 基础上
  输出 learned uncertainty；`generation_min_probability` 和 safety cap 只作门控/保险。
- [x] **P3-3 长文结构（2026-09-21 实现）**：`DiscourseState.timeline()` 维持 role-filler 事件序，全局句缓冲
  支持跨调用重读；`BionicBrain.cognitive_state()` 可观察篇章状态。

### P4 人类式学习路径（宗旨层）
- [x] **P4-1 预测误差驱动（2026-09-21 实现）**：surprise 同时调节脑侧 stream plasticity、context feature 更新
  和 neural sequence 更新；可预测输入不再持续重塑表征。
- [x] **P4-2 行动闭环（2026-09-21 实现）**：`reason` 完成 learned intent→retrieval→consistency error→
  reward/self-correction；`self_practice` 用 learned statement 自生成 probe，不再使用句式模板。
- [x] **P4-3 社会反馈（2026-09-21 实现）**：`SocialFeedback` 在同一 turn 保存 prompt/correction/reward；
  `correct()` 会写入 `last_response_info.social_feedback`。
- [x] **P4-4 睡眠巩固与发育阶段（2026-09-21 实现）**：`DevelopmentalClock` 根据学习/重放进展缩放 replay 和
  plasticity；`sleep()` 返回 phase/plasticity。
- [x] **P4-5 多语言迁移（2026-09-21 机制实现）**：grounding/sequence/subword/relation 全部由 observed symbols
  组成词表；测试用混合 opaque scripts 验证同一代码零改动。生产迁移仍需大语料 checkpoint A/B。

## 4. 每项的验收判据（统一口径）

| 类别 | 判据 |
|---|---|
| 语言无关 | 换语言只换语料，评测脚本与代码零改动，分数 ≥ 原语言的 70% |
| 泛化 | 未见题（字符串查重）+ 改写/反向/类比 + 未写过的关系 |
| 归因 | 机制消融后分数显著下降；未训练脑作为下界 |
| 不退化 | 旧任务回归不下降（语言 13 题、Python 分族题） |
| 判分 | 一律本地大模型语义判分，token 级指标仅作 `_aux` |

## 5. 迁移策略（不破坏现有可运行性）

1. **并行双引擎**：旧的规则实现保留在 `bionic_brain/legacy/`（可开关），新引擎在 `bionic_brain/nn/`。
2. **逐条替换**：每次只替换一个 H* 项，用"删词表基线"对齐分数，达不到就不合入。
3. **门槛**：只有当"跨语言迁移"和"未写过关系"两项同时上升，才提升为规范模型。
4. **可回滚**：检查点与语料都保留版本，评测报告写入 `runs/metrics/`。

## 6. 诚实声明

- 现状是"**我们写的规则 + 网络学的内容**"，规则承担了几乎所有推理——这不是涌现。
- 允许反向传播/Transformer 之后，风险变成另一个极端：**把脑变成普通深度网络**。因此每个可微组件必须回答"它对应脑内哪个结构、学的是什么"，否则不予合入。
- 本路线图不承诺"能涌现"，只承诺：**每一步都能被消融、被跨语言检验、被回归验证**。

## 7. P0-3 实测：硬编码贡献量账本（2026-09-19）

工具：`tools/hardcode_dependence.py`（固定 8 题探针 × 6 种消融，大模型判分）。

| 消融 | 平均分 | Δ | 说明 |
|---|---|---|---|
| baseline | 0.750 | — | 直接陈述 1.00 / 类比 1.00 / 地点 1.00；**transfer 0.00** |
| **no-relations**（清空 ROLE_MARKERS、RELATION_TOKENS、PREPOSITION_VERBS） | **0.125** | **−0.625** | **关系词表承担了 83% 的分数** |
| no-analogy（禁掉类比规则） | 0.625 | −0.125 | 类比规则承担 12.5% |
| no-transfer（清空 TRANSFER_MARKERS） | 0.750 | 0.000 | 该能力本身就没达标（基线上是 0.00），测不出贡献 |
| no-grammar（清空虚词黑名单） | 0.750 | 0.000 | 同上，本探针不敏感 |
| no-sequence（清空 n-gram 表） | 0.750 | 0.000 | 同上 |

**结论**：在这套题上，"推理"的 **83% 来自我手写的关系词表**，12.5% 来自我手写的类比规则——**合计 95% 的推理能力不是网络学出来的**。这与代码审计（H1–H14）吻合，也是 P1/P2 替换工作的基线：新机制必须把 `no-relations` 的 0.125 拉回 0.75 以上，而且**不允许再依赖词表**。

**探针缺口（诚实记录）**：transfer 类在基线上就是 0.00，说明该能力当前不可用或探针设计有误；grammar/sequence 两项在这套题上不敏感。P0-4 的跨语言探针会补上这些盲区。

## 8. P1-2 / P1-3c 复测（2026-09-20）

工具：`tools/hardcode_dependence.py`；judge 为
`Qwen3.6-35B-A3B-NVFP4@http://127.0.0.1:1919/v1`，六组共 48 次判分的
`offline_fallbacks = 0`。报告：
`runs/metrics/hardcode_dependence_20260920-learned-boundaries.json`。

| 消融 | 平均分 | Δ | 说明 |
|---|---:|---:|---|
| baseline | 0.750 | — | location 0.50；transfer 仍为 0.00 |
| no-relations | **0.438** | **−0.312** | 旧词表依赖从 0.125 升至 0.438，但仍低于 0.750 验收线 |
| no-transfer | 0.750 | 0.000 | 能力本身未达标，探针无贡献量 |
| no-learned-boundaries | 0.750 | 0.000 | 本探针对 H5/H6 学习边界不敏感 |
| no-analogy | 0.625 | −0.125 | H7 仍在承担责任 |
| no-sequence | 0.750 | 0.000 | 本探针对 H9 不敏感 |

同时，H5/H6/H13/H14 的字符串词表已被删除，策略 ratchet 通过；但**机制验收未完成**：
learned relations 的 `no-relations` 还需回到 0.750，且学习标记集必须覆盖 是/在/把、不含碎片。

## 9. P1-3c 机制验收（2026-09-21）

工具：`tools/hardcode_dependence.py`；judge 为
`Qwen3.6-35B-A3B-NVFP4@http://127.0.0.1:1919/v1`，六组共 51 次判分，
`offline_fallbacks = 0`。报告：
`runs/metrics/hardcode_dependence_20260921-learned-relations.json`。

| 消融 | 平均分 | Δ | 说明 |
|---|---:|---:|---|
| baseline（词典回退） | 0.750 | — | location 0.50；transfer 0.00 |
| **no-relations（learned path）** | **0.750** | **0.000** | 固定无词表探针达标；learned markers = `{是, 在, 把}`，不含 `子/里` |
| no-transfer | 0.750 | 0.000 | transfer 能力本身仍为 0，探针测不出贡献 |
| no-learned-boundaries | 0.750 | 0.000 | H5/H6 学习边界在该探针不敏感 |
| no-analogy | 0.625 | -0.125 | H7 仍在承担责任 |
| no-sequence | 0.750 | 0.000 | H9 在该探针不敏感 |

工程补充：`observe_stream` 现在维护跨调用句级重读缓冲，否则每句都会清空并只重读最后一句；
learned path 在关系选择、subject/value 保留、serial connector 和目标 frame 竞争上不再借用中文词表。
这仍只是固定探针验收；改成默认引擎前还需跨语言迁移和未写过关系泛化。

## 10. relations_v1 训练记录（2026-09-21）

命令使用 `tools/train_mixed_curriculum.py --fresh --learned-relations`，语料为 `data/corpora/mix_v1/`。
候选已按实验策略归档到
`archive/experiments/relations_v1_20260921/brian_relations_v1.pkl`；生产模型未覆盖。

| 项 | 值 |
|---|---:|
| 记录数 | 132 |
| tokens | 4,692 |
| streams / perceptions | 122 / 122 |
| cross-modal | 28 |
| QA learned | 10 |
| teacher interactive | 4/4 correct，fallback 0 |
| sleeps | 5 |
| neurons | 29,134 |
| words | 1,140 |

## 11. 2026-09-21 工程完成状态

P1-4、P2-1～P2-4、P3-1～P3-3、P4-1～P4-5 的可运行机制已实现，新增
`src/bionic_brain/cognition.py` 和 `tests/test_cognition.py`。当前
`90 tests, OK, skipped=7`。生产模型没有训练，也没有覆盖。

这些新引擎按仓库迁移策略默认保持安全开关：

| 开关 | 作用 |
|---|---|
| `use_sensory_grounding` | dendritic symbol field → semantic-cell ordering |
| `use_binding_memory` | holographic role/filler binding + discourse tracking |
| `use_learned_intent` | teacher-labelled opaque intent model |
| `use_neural_sequence` | four-structure next-token prediction |
| `use_decoder_calibration` | online readout thresholds |

剩下不是“写机制”，而是**晋级验收**：训练新候选 checkpoint 后做旧任务/Python/跨语言/未写过
关系 A/B；只有不退化且新能力达标，才把这些开关提升为默认。按当前指令暂不训练。

## 12. 2026-09-22 硬编码清除

目标修正后，继续把“不再使用但仍在源码树中的符号语言规则”移出 `src/`：

- 删除默认 `SchemaMemory` 的旧符号路径；新模型使用 `memory/learned_schemas.py`。
- `legacy/symbolic_schemas.py`、`language/world_model.py`、`learning/reasoning.py`、
  `learning/composition.py`、`learning/narrative.py`、`agents/logic_tutor.py`、
  `language/tokenizer.py` 均不再属于运行源码；因仓库尚无提交历史，先完整快照到
  `archive/experiments/hardcoded_removed_20260922/`。
- 删除源码内默认 QA 表；CLI 必须显式提供语料。
- 删除默认身份探针、旧 `brian_neural_ocean` 默认路径和 teacher 默认主题列表。
- `BionicBrain` 不再初始化 n-gram predictor、symbolic narrative parser、symbolic reasoner、
  symbolic composition model；序列预测使用 `DifferentiableSequenceModel`。
- semantic judge 不再退化为 token/string comparison；teacher 不可用时明确 `unjudged`。

验证：`84 tests, OK`，PRD smoke `PASS`。active `src/bionic_brain` 内语言词表常量扫描为 0。

## 13. 2026-09-27 能力补强与训练

新增三类纯学习机制：

- `PrototypeField`：从 learned grounding field 在线竞争形成抽象 prototype。
- `ConsistencyMemory`：用本脑自己的 accepted/rejected cue pattern 学习局部一致性置信。
- `AssociativeReplayPlanner`：睡眠重放时规划跨 episode associative bridge，而不是只巩固单条 QA。

同时修正：

- `DifferentiableSequenceModel` 改为一次 causal pass 预测整段下一 token。
- learned schema marker 刷新改为周期性，只有 marker 集合变化时才 re-read 旧 clauses，移除 O(N²) 重读。
- active `src/bionic_brain` 内语言词表常量扫描为 0。

最终增量 checkpoint：`data/checkpoints/neural_innovation_v7.pkl`。
科研创新语料为 teacher 生成的 6 条方法/假设训练样本；本次没有在线论文检索，因此不声称已读取真实论文。
