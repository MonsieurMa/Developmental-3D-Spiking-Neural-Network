# 测试审计（2026-09-19）

背景：66 个测试全绿，但"全绿"不等于"有意义"。按零硬编码路线图
（`docs/TODO_ZERO_HARDCODED.md`）重新审计，结论如下。

## 1. 分类结果

| 类别 | 条数（约） | 处置 |
|---|---|---|
| A 有价值：机制/学习/评测契约 | ~35 | 保留，必要时加强 |
| B 重复：同一契约多处断言 | ~8 | 合并 |
| C 空洞：断言恒真 | 2 | 改断言或删除 |
| D 锁死硬编码规则 | 11 | 退役或标记 legacy |
| E 冒烟与单测重复 | ~10 | verify_prd 只留端到端最小集 |
| F 缺失的关键测试 | 3 类 | 新增 |

## 2. D 类明细（锁死硬编码，阻碍路线图）

| 测试 | 锁死的机制 | 处置 |
|---|---|---|
| `test_causal_temporal_reasoner` | `learning/reasoning.py` 纯符号规则引擎 | 退役（随机制删除） |
| `test_composition_model_hypothetical_binding` | `learning/composition.py` 符号框架 | 退役 |
| `test_narrative_world_state_and_grounded_qa` | `language/world_model.py` 符号世界模型 | 退役 |
| `test_sequential_prediction_improves_seen_context` | n-gram 计数表（H9） | 标记 legacy，P3-1 后删除 |
| `test_usable_item_rejects_missing_clause_separators` | 教师语料的"；"格式硬规则 | 标记 legacy |
| `test_grade_falls_back_to_token_compare` | 旧 `grade()` API（已被 `judge()` 取代） | 随 `grade()` 退役 |
| `test_grade_uses_teacher_verdict_when_online` | 同上 | 同上 |
| `test_morphogens_diffuse_and_classify` | 区域身份字符串 `"M_BS"` | 拆：留扩散，去身份断言 |
| `test_high_similarity_pointer_remains_stable_across_neural_state` | 写死阈值 0.80（H10） | 保留，标记待自校准 |

## 3. C 类明细（断言恒真，等于没测）

- `test_event_simulation_and_plasticity`：`assertGreaterEqual(abs(weight), 0.0)` 恒真。
- `test_sleep_creates_cortical_associations`：`assertGreaterEqual(report["associations"], 0)` 恒真。

## 4. F 类（新增）

1. **零硬编码策略测试**：扫描 `src/bionic_brain/`，若出现语言词表或语言相关正则即失败；
   迁移期用显式白名单（当前违规项）做"棘轮"，只允许减少、不允许新增。
2. **评分路径契约**：评测报告必须含 `judge` 身份与 `offline-fallback` 计数（现在只在工具里，没有测试）。
3. **跨语言端口性基线**：同一套题的英语版；当前预期接近 0，作为"不满足语言无关"的证据，
   并在 P1/P2 完成后转为必须通过的判据。

## 5. 执行原则

- **不删历史**：D 类先用 `@unittest.skipUnless(os.environ.get("BIONIC_LEGACY_TESTS") == "1", ...)`
  隔离在默认套件之外；机制真正删除时再一并删除测试。
- **断言要能失败**：C 类改成"有事件时关联数 > 0"这类可失败断言。
- **冒烟只冒烟**：`verify_prd.py` 保留端到端最小断言，细节交给单测。

## 6. 执行结果（2026-09-19）

- 默认套件：**65 个测试，58 通过 + 7 跳过**（原 66 全跑），耗时从 21s 降到 16s。
- 隔离的 legacy 测试（`BIONIC_LEGACY_TESTS=1` 时全部通过，确认只是隔离而非破坏）：
  `test_causal_temporal_reasoner`、`test_composition_model_hypothetical_binding`、
  `test_narrative_world_state_and_grounded_qa`、`test_sequential_prediction_improves_seen_context`、
  `test_usable_item_rejects_missing_clause_separators`、两条 `test_grade_*`。
- 删除重复：`test_generation_stops_instead_of_repeating`（被自动停止测试完全覆盖）。
- 合并重复：`test_context_features` 的三条几何断言合并为一条（域内 > 跨域 + 邻居语义）。
- 修掉恒真断言：`test_event_simulation_and_plasticity` 现在要求 DA 真的改变权重；
  `test_sleep_creates_cortical_associations` 现在要求 `replayed ≥ 1` 且 `associations ≥ 1`。
- 去掉硬编码契约：`test_morphogens_diffuse_and_classify` 不再断言区域身份字符串 `"M_BS"`。
- 新增策略棘轮 `tests/test_no_hardcoded_language.py`：扫描 `src/`，禁止新增语言词表/正则；
  **它立刻抓到两个漏掉的违规**：`learning/sequence.py:BASE_LEXICON`（59 条中文词表）和
  `config/defaults.py:generation_stop_tokens`（标点表），已补入 H13/H14 与白名单。
- `test_nn_structures` 增加"分支局部性"断言（只改动某一分支输入，其余输出必须不变）。
