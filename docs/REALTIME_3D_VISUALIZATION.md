# Realtime 3D Point-Cloud Visualization Design

更新时间：2026-09-29  
状态：设计方案，尚未实现

## 1. 目标

为 `BionicBrain` 增加独立的大脑 3D 点云可视化器：

- 展示所有存活神经元的空间位置；
- 交互回答时实时看到被激活的神经元；
- 激活神经元使用闪光 / 高亮 / 放大效果；
- 未激活神经元保持原本位置、颜色和大小；
- 可按脑区、激活来源、时间窗口过滤；
- 可保存完整激活记录，之后重复回放。

第一版采用 **record-to-disk + independent viewer** 架构，而不是把可视化逻辑塞进 `BionicBrain`。

## 2. 设计原则

1. **可视化不影响推理逻辑。**  
   `BionicBrain.respond()`、`step()`、learning mechanism 不依赖 viewer。

2. **只记录活动事件，不记录答案规则。**  
   记录的是 spike、active assembly、region rate、motor readout 等神经活动，不保存语言规则或答案查找表。

3. **模拟线程不直接写最终可视化文件。**  
   模拟线程只产生 event batch；独立 recorder thread 负责写入硬盘，避免硬盘延迟阻塞推理。

4. **静态数据和动态数据分离。**  
   神经元位置、脑区、基础颜色是静态点云；激活强度是动态事件流。

5. **viewer 只读 committed data。**  
   viewer 不读取正在写入的半行数据，只读取已完成的 segment 和 atomic manifest。

## 3. 总体架构

```text
BionicBrain simulation
        ↓
Activation event queue
        ↓
Recorder thread / process
        ↓ segment files + atomic manifest
Activation log on disk
        ↓ tail / load committed segments
Viewer IO thread
        ↓ decode batches
Viewer render thread
        ↓ update active point colors / sizes
3D point cloud
```

### 3.1 模式 A：Replay 模式

先完整记录一次交互或训练过程，之后由独立 viewer 打开文件回放。

优点：

- 实现简单；
- 不影响 simulation；
- 可以反复观看、慢放、暂停、拖动时间轴；
- 数据格式稳定后再做 live 版本。

### 3.2 模式 B：Near-live tail 模式

viewer 监控当前 session 目录，持续读取最新 committed segment。

优点：

- 接近实时；
- 可以在交互过程中看到激活扩散。

限制：

- `BionicBrain.respond()` 当前是阻塞式；如果推理耗时很长，viewer 只能显示最近已写入的 batch；
- 需要 recorder 使用 atomic manifest，避免 viewer 读到半个 batch。

## 4. 数据结构

### 4.1 静态点云

神经元位置、脑区、基础颜色等不经常变化，只保存一次。

推荐文件：

```text
brain_points.npz
```

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `neuron_id` | `int32[]` | 神经元 ID |
| `position` | `float32[N,3]` | x/y/z 坐标 |
| `region_code` | `int16[]` | 脑区索引 |
| `alive` | `bool[]` | 是否存活 |
| `base_size` | `float32[]` | 基础点大小 |
| `base_color` | `uint8[N,3]` | 基础 RGB |

可选字段：

- cell type；
- sensory / motor 标记；
- development phase；
- position version。

### 4.2 动态激活流

激活记录只包含当前激活的 neuron，不记录未激活 neuron。

推荐每个 batch 一个 JSON Lines 记录：

```json
{
  "seq": 128,
  "sim_time_ms": 12.4,
  "wall_time": 1790000000.123,
  "event": "spike_batch",
  "active": [
    {"id": 1023, "strength": 1.0},
    {"id": 2044, "strength": 0.7},
    {"id": 5091, "strength": 0.4}
  ],
  "regions": {
    "V1": 12.5,
    "Wern": 31.0,
    "Broca": 8.2
  }
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `seq` | 单调递增 batch 编号 |
| `sim_time_ms` | Brian 模拟时间 |
| `wall_time` | 真实时间戳 |
| `event` | batch 类型，例如 `spike_batch`、`motor_readout`、`hippocampal_pointer` |
| `active` | 当前 batch 内激活的 neuron ID 和强度 |
| `regions` | 脑区级激活率，便于右侧面板显示 |

### 4.3 Session 目录结构

```text
runs/viz/session_20260928_001/
  brain_points.npz
  manifest.json
  activations_000000.jsonl
  activations_000001.jsonl
  activations_000002.jsonl
  latest.json
```

`manifest.json` 示例：

```json
{
  "version": 1,
  "session": "session_20260928_001",
  "static_points": "brain_points.npz",
  "segments": [
    {
      "file": "activations_000000.jsonl",
      "seq_start": 0,
      "seq_end": 127,
      "bytes": 284311,
      "committed": true
    }
  ],
  "updated_wall_time": 1790000123.456
}
```

viewer 只读取 `committed: true` 的 segment。

## 5. 写入流程

### 5.1 Simulation side

`BionicBrain` 不直接格式化 JSON，也不直接写最终文件。

它只暴露一个轻量 event sink 或 recorder：

```text
step / respond / observe_stream
        ↓
activation event batch
        ↓
Recorder queue
```

可记录的事件：

| 事件 | 内容 |
|---|---|
| `spike_batch` | 当前 sim window 内 spiking neuron IDs |
| `token_input` | 当前输入 token 对应 sensory cells |
| `motor_readout` | answer motor neurons |
| `hippocampal_pointer` | retrieved episodic pointer neurons |
| `association_update` | 新建 / 强化 association 的前后神经元 |
| `region_rate` | 脑区级活动率 |
| `marker` | answer 开始、结束、错误、reward 等阶段标记 |

### 5.2 Recorder thread

Recorder thread 从 queue 读取 batch，并批量写入 segment：

- 每 50～200 ms flush 一次；
- 每 256～1024 个 batch 创建新 segment；
- segment 写完后 `fsync`；
- 更新 `manifest.json` 时使用 atomic replace；
- simulation 卡顿时丢弃或合并过旧低强度 batch，避免硬盘堆积。

### 5.3 写入格式

第一版使用 JSONL，方便调试。

后期如果数据量大，可升级为：

- NPZ batch；
- MessagePack；
- Protobuf；
- custom binary；
- Zstandard compressed segment。

## 6. Viewer 架构

viewer 应该是独立进程，不和 `BionicBrain` 共享主线程。

### 6.1 推荐技术

优先：

```text
PyVista / VTK
```

原因：

- 支持 Python；
- 支持 80k～500k point cloud；
- 可旋转、缩放、选择；
- 可只更新 active point 的 color / size / opacity；
- 比 Matplotlib 更适合实时 3D。

备选：

- Vispy：更高性能 point sprite；
- Plotly Dash：适合网页端，但实时大点云性能弱于 VTK；
- Three.js / WebGPU：界面最灵活，但需要额外数据服务。

### 6.2 渲染策略

静态点云只加载一次：

```text
load brain_points.npz
create base point cloud
render inactive neurons
```

动态激活不重建整个点云，只更新 active subset：

```text
for active neuron:
    color = base_color + flash_color * intensity
    size  = base_size * (1 + gain * intensity)
    opacity = base_opacity + activation_boost
```

强度随时间衰减：

```text
intensity *= decay
```

例如：

- 激活瞬间 `intensity = 1.0`；
- 每 frame `intensity *= 0.90`；
- 低于 `0.02` 后恢复原样。

### 6.3 Flash 语义

建议不同来源使用不同 flash 颜色：

| 来源 | 建议颜色 |
|---|---|
| sensory input | cyan |
| Wernicke / semantic activation | blue |
| hippocampal pointer | purple |
| DLPFC working memory | green |
| motor readout | orange / red |
| prediction error | white burst |
| association update | thin edge or small pulse |

未激活 neuron 保持原脑区基础颜色。

## 7. 激活定义

viewer 不应该自己猜测“激活”，而应使用明确的 event 来源。

| 层级 | 定义 | 数据来源 |
|---|---|---|
| Spike activation | neuron 在最近 sim window 内真的 spike | neuron spike records |
| Engram activation | neuron 属于当前 cue / answer assembly | word binding sensory/motor cells |
| Motor readout | neuron 属于最终 answer motor population | `last_response_info["motor_cells"]` |
| Hippocampal pointer | episodic retrieval 命中的 neurons | retrieved event |
| Plasticity | 本次交互新建 / 强化的 association | association update event |
| Region rate | 脑区级 aggregate rate | region activity snapshot |

第一阶段可以先显示：

1. spike batch；
2. motor readout；
3. hippocampal pointer；
4. region rate。

后续再加入 plasticity 和 association bridge。

## 8. Replay 控制

viewer 至少支持：

- play / pause；
- stop；
- speed：0.25x / 0.5x / 1x / 2x / 4x；
- jump to time；
- step batch；
- filter by region；
- filter by event type；
- only-active view；
- show all neurons。

时间轴应同时显示：

```text
simulation time
wall time
event batch index
```

这样即使 Brian 回答耗时很长，也能按 simulation time 回放真实激活扩散过程。

## 9. 性能预算

目标规模：

- 50k～100k neurons；
- 100k～3M synapses；
- 每次回答可能产生数千到数万 spike / activation。

### 9.1 静态点云

80k points 对 VTK / Vispy 不是问题。

### 9.2 动态更新

不要每帧更新所有 neuron，只更新 active subset：

```text
active points: 数百～数万
inactive points: 保持原样
```

### 9.3 Disk IO

推荐：

```text
recorder flush: 50～200 ms
viewer poll: 100～250 ms
segment size: 1～8 MB
```

如果 spike 太密集，可以先在 recorder 内合并：

```text
同一 batch 内同一 neuron 多次 spike → 合成 strength / count
```

## 10. 与现有 Matplotlib 可视化的关系

现有 `visualize_brain()` 继续保留为静态诊断图：

- 训练后快照；
- 神经元位置；
- 脑区放电率；
- 简单保存 PNG。

它不适合作为 realtime viewer。

实时 viewer 应新建独立模块，例如：

```text
src/bionic_brain/visualization/pointcloud_viewer.py
tools/view_brain_session.py
```

不要继续扩大 `brain3d.py` 的职责。

## 11. 实现阶段

### Phase 1：Recorder

- 从 `BionicBrain.step()` / `respond()` 抽取 spike batch；
- 记录 static point cloud；
- 写 JSONL segment 和 atomic manifest；
- 提供 session 目录。

### Phase 2：Offline viewer

- 读取 session；
- 显示 3D point cloud；
- 支持 play / pause / speed / time slider；
- active neuron flash；
- inactive neuron unchanged。

### Phase 3：Near-live tail viewer

- viewer 监控 session 目录；
- 只读取 committed segments；
- 在交互过程中近实时显示激活。

### Phase 4：Interactive inspection

增加：

- 点击 neuron；
- 脑区过滤；
- event type filter；
- association edge 显示；
- motor / sensory路径追踪；
- snapshot export。

## 12. 非目标

- 不用 3D 可视化结果参与推理；
- 不把 viewer 放进 `BionicBrain.respond()` 决策路径；
- 不使用语言规则决定 neuron 颜色；
- 不把 viewer 变成答案解释器；
- 不在第一版实现完整突触级别的 million-edge 实时渲染。

## 13. 验收标准

第一阶段验收：

1. 训练或回答过程中能生成 session 目录；
2. `manifest.json` atomically 更新；
3. viewer 能加载 `brain_points.npz`；
4. viewer 能顺序回放 activation segments；
5. active neuron flash，inactive neuron unchanged；
6. 100k point cloud 下能稳定旋转、缩放、播放。

第二阶段验收：

1. viewer 可以 tail 当前 session；
2. 近实时显示最新 committed activation；
3. recorder 不阻塞 `BionicBrain.respond()`；
4. session 文件可离线重复回放；
5. 数据中没有语言规则，只有 learned activation events。
