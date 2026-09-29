# 项目目录与清理规划（v2）

更新时间：2026-09-17  
依据：`docs/PRD.md`、`docs/in-out.md`、`docs/海马体的设计.md`。  
本版本取代旧的目录规划：目标是把“全脑发育机制”和“输入输出闭环”都映射到清晰的模块边界，而不是继续堆在 `main.py`。当前阶段仍不实现新机制，只整理资产、清理无用文件、冻结目录规划。

## 1. 结论

- 当前项目只有一个巨型 `main.py`，同时承担细胞、突触、发育、学习、语言、训练、保存、可视化和交互。
- 当前文档已经从“语言网络”扩展为“全脑仿生网络”：输入必须经过感觉通道与丘脑门控，输出必须走运动通路，输出后还要形成感觉反馈和小脑校正。
- 后续统一包名采用 `bionic_brain`，对外类名采用 `BionicBrain`，与 `docs/in-out.md` 的 Python API 设想一致。
- 当前 `brain.pkl` 保留为可运行检查点；实验检查点移入归档目录；一次性调试脚本移入归档；生成缓存和 IDE 工作区删除。
- 暂时不把 `main.py` 移入新包，避免立刻破坏 README 中的现有运行命令。

## 2. 立即清理策略

### 2.1 删除

| 路径 | 理由 |
|---|---|
| `__pycache__/` | Python 生成缓存，不应进入源码树 |
| `.idea/` | IDE 工作区配置，不属于项目实现 |

### 2.2 归档，不直接删除

| 原路径 | 目标路径 | 理由 |
|---|---|---|
| `audit_wm.py` | `archive/legacy_scripts/` | 一次性工作记忆审计脚本 |
| `debug_wm.py` | `archive/legacy_scripts/` | 一次性上下文调试脚本 |
| `inspect_wm.py` | `archive/legacy_scripts/` | 一次性状态检查脚本 |
| `test_teaching.py` | `archive/legacy_scripts/` | 非正式测试，后续重写 |
| `test_wm.py` | `archive/legacy_scripts/` | 非正式测试，后续重写 |
| `brain_wm1.pkl` | `archive/experiments/` | 工作记忆实验检查点 |
| `brain_teach_adaptive1.pkl` | `archive/experiments/` | 自适应教学实验检查点 |

### 2.3 暂时保留在根目录

| 路径 | 理由 |
|---|---|
| `main.py` | 当前入口，迁移到 `src/bionic_brain/interfaces/cli.py` 前不能删除 |
| `verify_prd.py` | 当前 PRD 烟测入口，后续迁到 `tests/smoke/test_prd.py` |
| `brain.pkl` | 当前默认加载/保存路径仍指向根目录 |
| `pyproject.toml`、`uv.lock`、`README.md`、`.python-version` | 项目基础文件 |

## 3. 目标目录树

```text
.
├── docs/
│   ├── PRD.md
│   ├── in-out.md
│   ├── 海马体的设计.md
│   ├── PROJECT_STRUCTURE.md
│   ├── architecture/
│   │   ├── closed-loop.md
│   │   ├── development.md
│   │   ├── memory.md
│   │   └── regions-and-tracts.md
│   └── reports/
│       └── experiments.md
│
├── configs/
│   ├── minimal.json
│   ├── standard.json
│   └── large.json
│
├── data/
│   ├── corpora/
│   ├── checkpoints/
│   └── exports/
│
├── runs/
│   ├── logs/
│   ├── metrics/
│   └── figures/
│
├── src/
│   └── bionic_brain/
│       ├── __init__.py
│       ├── version.py
│       │
│       ├── core/
│       │   ├── types.py
│       │   ├── events.py
│       │   ├── clock.py
│       │   ├── rng.py
│       │   ├── registry.py
│       │   └── state.py
│       │
│       ├── config/
│       │   ├── schema.py
│       │   ├── defaults.py
│       │   └── loading.py
│       │
│       ├── atlas/
│       │   ├── coordinates.py
│       │   ├── regions.py
│       │   ├── seeds.py
│       │   ├── receptor_density.py
│       │   └── layout.py
│       │
│       ├── cells/
│       │   ├── base.py
│       │   ├── hodgkin_huxley.py
│       │   ├── lif.py
│       │   ├── progenitor.py
│       │   └── metabolism.py
│       │
│       ├── synapses/
│       │   ├── model.py
│       │   ├── transmission.py
│       │   ├── plasticity.py
│       │   ├── dopamine.py
│       │   ├── normalization.py
│       │   └── inhibition.py
│       │
│       ├── morphogens/
│       │   ├── field.py
│       │   ├── regional.py
│       │   ├── axes.py
│       │   ├── interaction.py
│       │   └── slices.py
│       │
│       ├── development/
│       │   ├── growth_cone.py
│       │   ├── guidance.py
│       │   ├── synaptogenesis.py
│       │   ├── cell_division.py
│       │   ├── apoptosis.py
│       │   ├── pruning.py
│       │   ├── myelination.py
│       │   └── lifecycle.py
│       │
│       ├── regions/
│       │   ├── base.py
│       │   ├── frontal.py
│       │   ├── parietal.py
│       │   ├── temporal.py
│       │   ├── occipital.py
│       │   ├── subcortical.py
│       │   ├── cerebellum.py
│       │   ├── brainstem.py
│       │   └── registry.py
│       │
│       ├── tracts/
│       │   ├── channel.py
│       │   ├── pathways.py
│       │   ├── growth.py
│       │   ├── routing.py
│       │   └── plasticity.py
│       │
│       ├── modulation/
│       │   ├── modulators.py
│       │   ├── global_state.py
│       │   ├── reward.py
│       │   └── effects.py
│       │
│       ├── senses/
│       │   ├── visual.py
│       │   ├── auditory.py
│       │   ├── somatosensory.py
│       │   ├── interoception.py
│       │   └── emotional.py
│       │
│       ├── thalamus/
│       │   ├── gate.py
│       │   ├── attention.py
│       │   └── routing.py
│       │
│       ├── motor/
│       │   ├── speech.py
│       │   ├── written.py
│       │   ├── limb.py
│       │   ├── oculomotor.py
│       │   ├── endocrine.py
│       │   ├── expression.py
│       │   └── population_decoder.py
│       │
│       ├── cerebellum/
│       │   ├── prediction.py
│       │   ├── efference_copy.py
│       │   └── correction.py
│       │
│       ├── language/
│       │   ├── tokenization.py
│       │   ├── vocabulary.py
│       │   ├── sdr.py
│       │   ├── text_input.py
│       │   ├── text_output.py
│       │   └── prototypes.py
│       │
│       ├── memory/
│       │   ├── representations.py          # SDR、活动向量、稀疏化
│       │   ├── working_memory.py           # DLPFC 短时上下文
│       │   ├── dialogue_state.py           # 最近词、话题、说话人、回合状态
│       │   ├── event.py                    # 压缩事件快照
│       │   ├── gating.py                   # EC → Hipp 编码门控
│       │   ├── criteria.py                 # 新奇 / 显著 / RPE / 目标 / 未完成
│       │   ├── encoding.py                 # DG 模式分离与索引生成
│       │   ├── episodic_buffer.py          # 最近 N 条快速记忆
│       │   ├── episodic_store.py           # 按时间索引的情景库
│       │   ├── retrieval.py                # CA3 模式补全与检索触发
│       │   ├── binding.py                  # CA1 / 皮层状态绑定与再激活
│       │   ├── importance.py               # 巩固排序与淘汰评分
│       │   ├── replay.py                   # 睡眠重放
│       │   ├── semantic.py                 # 新皮层语义记忆库
│       │   └── capacity.py                 # 容量与淘汰策略
│       │
│       ├── learning/
│       │   ├── teaching.py
│       │   ├── curriculum.py
│       │   ├── correction.py
│       │   ├── evaluation.py
│       │   └── sleep.py
│       │
│       ├── simulation/
│       │   ├── runner.py
│       │   ├── hooks.py
│       │   └── recording.py
│       │
│       ├── brain.py
│       │
│       ├── persistence/
│       │   ├── serialization.py
│       │   ├── checkpoints.py
│       │   ├── migrations.py
│       │   └── exports.py
│       │
│       ├── visualization/
│       │   ├── fonts.py
│       │   ├── brain_3d.py
│       │   ├── slices.py
│       │   ├── raster.py
│       │   ├── voltage.py
│       │   ├── connectivity.py
│       │   ├── morphogens.py
│       │   └── dashboard.py
│       │
│       └── interfaces/
│           ├── public_api.py
│           ├── cli.py
│           ├── chat.py
│           └── commands.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── smoke/
│   └── acceptance/
│
├── tools/
│   ├── inspect_brain.py
│   ├── audit_memory.py
│   ├── benchmark.py
│   └── export_report.py
│
├── archive/
│   ├── legacy_scripts/
│   └── experiments/
│
├── main.py                       # 过渡期兼容入口
├── verify_prd.py                 # 过渡期 PRD 烟测
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore
```

## 4. 模块职责与文档映射

### 4.1 PRD 分层映射

| PRD 层 | 目标包 |
|---|---|
| 规则层 / 基因参数 | `config` |
| 细胞层 | `cells` |
| 形态层 | `development` |
| 化学层 | `morphogens` |
| 学习层 | `synapses` + `learning` |
| 区域层 | `regions` |
| 白质通信层 | `tracts` |
| 生命周期层 | `development` |
| 全局调制层 | `modulation` |
| 海马记忆 | `memory` |

### 4.2 输入输出闭环映射

| `in-out.md` 概念 | 目标包 |
|---|---|
| 视觉、听觉、体感、嗅觉、内感受、情绪输入 | `senses` |
| 丘脑入口网关与注意力门控 | `thalamus` |
| Wernicke / Broca 语言处理 | `regions.temporal`、`regions.frontal`、`language` |
| M1 / PMC 言语、书写、肢体输出 | `motor` |
| 小脑预测与误差校正 | `cerebellum` |
| 传出拷贝、感觉反馈、闭环 | `cerebellum` + `senses` + `simulation` |
| 对话状态 / DLPFC 工作记忆 | `memory.working_memory`、`memory.dialogue_state` |
| EC 编码门控：不是所有输入都进海马 | `memory.gating`、`memory.criteria` |
| 事件快照 / SDR 索引，而不是原始文本 | `memory.event`、`memory.representations` |
| 情景缓冲区 / 情景库 / 新皮层语义库三层结构 | `memory.episodic_buffer`、`memory.episodic_store`、`memory.semantic` |
| DG 模式分离、CA3 模式补全、CA1 再激活 | `memory.encoding`、`memory.retrieval`、`memory.binding` |
| 重要性评分、容量淘汰、睡眠巩固 | `memory.importance`、`memory.capacity`、`memory.replay`、`learning.sleep` |
| 多巴胺反馈学习 | `modulation.reward` + `synapses.dopamine` |
| 命令行、交互、Python API | `interfaces` |

### 4.3 海马记忆的特殊边界

海马不是全文向量库，也不是所有输入的记录仪。  
目录规划遵循 `docs/海马体的设计.md`：

1. **编码门控**  
   只有新奇性、显著性、奖励预测误差、目标相关性或未完成事件达到门槛的事件才进入海马。

2. **存事件，不存文本**  
   不保存“用户说过什么”的原始字符串，而是保存时间、SDR 索引、内容 SDR、运动 SDR、情绪效价、多巴胺水平、目标向量和区域活跃度压缩快照。

3. **三层存储**  
   - `episodic_buffer`：最近事件快速缓冲；
   - `episodic_store`：按时间索引的情景记忆；
   - `semantic`：巩固到新皮层的抽象语义记忆。

4. **检索与再激活**  
   检索由低新奇性、目标查询、情绪标记或上下文缺失触发；命中后重新激活 Wernicke / Broca 等皮层模式，而不是把文本拼回上下文。

5. **巩固后释放海马**  
   睡眠时按重要性重放；巩固成功的事件标记后从海马淘汰，长期内容由新皮层承担。

## 5. 对外形态

`docs/in-out.md` 的目标 API 应收敛为：

```python
from bionic_brain import BionicBrain

brain = BionicBrain.load('data/checkpoints/brain.pkl')

brain.input_text('你是谁', modality='visual')
brain.run(duration=100.0)
output = brain.read_output(region='Broca')

activity = brain.get_region_activity('Wernicke')
```

CLI 目标形态：

```powershell
bionic-brain --mode train --corpus data/corpora/qa_dev_zh.jsonl --epochs 40
bionic-brain --mode inference --load data/checkpoints/brain.pkl
bionic-brain --mode chat --load data/checkpoints/brain.pkl
```

过渡期继续保留：

```powershell
.\.venv\Scripts\python.exe verify_prd.py
.\.venv\Scripts\python.exe main.py --fresh --train-only --epochs 40 --save brain.pkl
```

## 6. 迁移阶段

### 阶段 1：清理与文档冻结

- 删除 `__pycache__` 和 `.idea`；
- 归档一次性脚本与实验检查点；
- 更新 `.gitignore`；
- 本文档作为目录边界依据。

### 阶段 2：建立包壳

- 新增 `src/bionic_brain`；
- 只建立包、版本、配置 schema、异常类型和空模块接口；
- `main.py` 保持兼容入口。

### 阶段 3：拆分无风险模块

- 分词、词表、语料；
- 中文字体；
- 序列化与检查点；
- 参数配置。

### 阶段 4：拆分细胞与突触

- HH 状态；
- 事件突触；
- STDP、资格痕迹；
- DA 调制和归一化。

### 阶段 5：拆分发育机制

- 3D 形态发生素；
- 生长锥、突触形成；
- 分裂、凋亡、修剪、髓鞘化；
- 30 区种子配置。

### 阶段 6：拆分闭环接口

- 多模态感觉入口；
- 丘脑门控；
- 运动输出与群体解码；
- 传出拷贝与小脑校正；
- 对话状态和工作记忆。

### 阶段 7：实现记忆与全局调制

- 海马写入 / 检索；
- 睡眠重放；
- 六类调质；
- 区域受体密度效应。

### 阶段 8：替换入口

- `BionicBrain` 成为主要 API；
- `bionic-brain` CLI 替代 `main.py`；
- 旧 `main.py` 进入兼容弃用期。

## 7. 验收口径

目录整理不能只看目录名，还要满足：

1. 根目录不再积累实验脚本、实验模型、IDE 配置和缓存；
2. `BionicBrain` API 与 `docs/in-out.md` 对齐；
3. 所有输入路径都有明确感觉入口和丘脑门控；
4. 所有输出路径都有明确运动出口和解码器；
5. 闭环中的传出拷贝、感觉反馈、小脑误差有独立模块；
6. 工作记忆与海马记忆分开建模；
7. 30 个脑区配置由 `atlas` 驱动；
8. 旧 `main.py` 在新入口稳定前始终可运行；
9. `.pkl` 检查点默认不提交 Git；
10. PRD 烟测迁移后仍覆盖保存、加载、生长、STDP 和可视化。

## 8. 本轮不做的事

- 不实现新的神经机制；
- 不迁移 `main.py` 内部逻辑；
- 不宣称 AC2 通过；
- 不删除当前可运行的 `brain.pkl`；
- 不引入数据库、分布式训练、Transformer、MLP 或反向传播。



