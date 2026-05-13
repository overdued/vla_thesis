# openpi 微调对比与选型综合手册

> 版本：2026-05-13
> 整理者：Yuhang Zhao
> 配套文档：
> - `COMPANY_FINETUNING_GUIDE.md` —— 公司全量微调手册（cyborg_v2_* 案例）
> - `LORA_FINETUNING_GUIDE.md` —— 自有 LoRA 微调手册（pi05_libero10_lora 实战）

本手册回答四个问题：

1. 两种方案**核心差异**在哪？
2. **何时选哪个**？（决策树）
3. **共同点**有哪些？（哪些代码可复用）
4. 如何**在两种方案之间迁移**？

---

## 目录

1. [一句话总结](#一一句话总结)
2. [全维度对比表](#二全维度对比表)
3. [TrainConfig 关键差异（最核心）](#三trainconfig-关键差异最核心)
4. [显存与时长实测](#四显存与时长实测)
5. [选型决策树](#五选型决策树)
6. [共同点：哪些代码可复用](#六共同点哪些代码可复用)
7. [互相迁移指南](#七互相迁移指南)
8. [混合工作流推荐](#八混合工作流推荐建议)
9. [FAQ：常见误解](#九faq常见误解)

---

## 一、一句话总结

| 方案 | 来源 | 微调方式 | 一句话 |
| --- | --- | --- | --- |
| 公司方案 | `/home/zhaoyuhang/cyb_vla/agibot-world-master/openpi` | **全量微调** | 解锁所有参数，追求最高性能，需 A100/H100 |
| 自有方案 | `/home/zhaoyuhang/openpi` | **LoRA 微调** | 冻结 99% 参数，只训低秩矩阵，单卡 4090 可跑 |

> **代码结构 99% 相同**，区别仅在 TrainConfig 的 4 个开关 + 学习率。

---

## 二、全维度对比表

| 维度 | 公司全量微调 | **自有 LoRA 微调** |
| --- | --- | --- |
| **可训练参数量** | 100%（约 3B） | **~0.1-1%（10~50M）** |
| **GPU 最小显存/卡** | 70+ GB | **22.5 GB** |
| **典型卡型** | A100/H100 80GB | **RTX 4090 / A800** |
| **典型 batch_size** | 64（4 卡） | **32（2 卡）** |
| **学习率峰值** | 2.5e-5（默认） | **5e-5（可更大）** |
| **学习率 warmup** | 10k 步 | **1k 步** |
| **训练步数** | 28k ~ 40k | **10k** |
| **训练时长** | 长 | **短约 3×** |
| **EMA** | 开（decay=0.999） | **关（None）** |
| **save_interval** | 2k ~ 20k | 2k |
| **预训练 base** | `pi0_base` / `pi05_base` | 同左 |
| **数据格式** | LeRobot v2 | 同左 |
| **环境配置** | uv + py311 服务端 / py310 客户端 | 同左 |
| **Policy 文件结构** | `Inputs/Outputs` 类 | 同左 |
| **DataConfig 结构** | `RepackTransform + DeltaActions` | 同左 |
| **部署架构** | C/S WebSocket + ROS Client | 同左 |
| **配置注册位置** | `src/openpi/training/config.py:_CONFIGS` | 同左 |
| **训练入口** | `scripts/train.py` | 同左 |
| **是否需 norm_stats** | ✓ | ✓ |
| **ckpt 大小（单步）** | ~7 GB | ~7 GB（默认存全量 params）<br>~150 MB（按需提取 lora 增量） |
| **最终性能** | 通常最高 | 接近全量（差 ~1-3%） |
| **多任务复用** | 需重新训练 | **一个 base + N 个 adapter** |
| **过拟合风险** | 中等 | **较低**（参数少） |

---

## 三、TrainConfig 关键差异（最核心）

### 3.1 LoRA 四开关速查

| 字段 | 全量微调 | **LoRA 微调** | 说明 |
| --- | --- | --- | --- |
| `model.paligemma_variant` | `"gemma_2b"` | **`"gemma_2b_lora"`** | 主 LLM 是否注入 LoRA |
| `model.action_expert_variant` | `"gemma_300m"` | **`"gemma_300m_lora"`** | Action Expert 是否注入 LoRA |
| `freeze_filter` | 不设置（默认 `nnx.Nothing`） | **`pi0_config.Pi0Config(...).get_freeze_filter()`** | 决定哪些参数冻结 |
| `ema_decay` | `0.999`（默认） | **`None`** | LoRA 场景下 EMA 浪费显存 |

### 3.2 并排代码对比

#### 公司全量微调（pi05_cyborgv2_pick_up_tray_v2_0313）

```python
TrainConfig(
    name="pi05_cyborgv2_pick_up_tray_v2_0313",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=30,
        # 默认 paligemma_variant="gemma_2b"
        # 默认 action_expert_variant="gemma_300m"
    ),
    data=CyborgV2GripperDataConfig(
        repo_id="pick_up_tray_v2_0313",
        use_delta_joint_actions=True,
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=28_000,
    save_interval=2000,
    num_workers=16,
    # 没有 freeze_filter（默认全训练）
    # 没有 ema_decay（默认 0.999）
    # 没有显式 lr_schedule（默认 cosine warmup=10k, peak=2.5e-5）
),
```

#### 自有 LoRA 微调（pi05_libero10_lora）

```python
TrainConfig(
    name="pi05_libero10_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",          # ★ 不同
        action_expert_variant="gemma_300m_lora",    # ★ 不同
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=32,                                  # 可比公司更大
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,                         # ★ 短得多（1k vs 10k）
        peak_lr=5e-5,                               # ★ 大一倍（5e-5 vs 2.5e-5）
        decay_steps=100_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=None,                                 # ★ 关闭
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    freeze_filter=pi0_config.Pi0Config(             # ★ 必须显式
        pi05=True, action_horizon=10, discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    num_train_steps=10_000,                         # ★ 步数减为 1/3
    wandb_enabled=False,
),
```

### 3.3 差异点总结

| 改动 | 必须 | 影响 |
| --- | --- | --- |
| 加 `_lora` 后缀 | ✓ | 在 Gemma 的 Attention/FFN 注入 LoRA 模块 |
| 设置 `freeze_filter` | ✓ | 让 base 参数不被训练 |
| `ema_decay=None` | ✓ | 节约一份 shadow params 显存 |
| 提高 `peak_lr` | 推荐 | LoRA 可承受更大 lr，收敛更快 |
| 缩短 warmup | 推荐 | LoRA 起点接近 0，无需长 warmup |
| 减少 `num_train_steps` | 推荐 | LoRA 收敛快，10k 通常够 |

---

## 四、显存与时长实测

### 4.1 显存占用（单卡）

| 配置 | 模型 | batch | LoRA | EMA | 显存 |
| --- | --- | --- | --- | --- | --- |
| 公司 `pi05_cyborgv2_*` | pi0.5 全量 | 64 (4 卡) | ✗ | ✓ | ~70 GB |
| 公司 `pi0_libero` | pi0 全量 | 32 | ✗ | ✓ | ~65 GB |
| 自有 `pi05_libero10_full` | pi0.5 全量 | 4 (2 卡) | ✗ | ✓ | ~70 GB |
| **自有 `pi05_libero10_lora`** | **pi0.5 + LoRA** | **32 (2 卡)** | **✓** | **✗** | **~35 GB** |

### 4.2 LIBERO-10 训练效果对比（本仓库实测）

| 配置 | 步数 | 单卡显存 | 训练时长（相对） | 成功率（参考） |
| --- | --- | --- | --- | --- |
| `pi05_libero10_full` | 10k | ~70 GB | 1.0× | 92.4 |
| `pi05_libero10_lora` | 10k | ~35 GB | **~0.3×** | 接近全量（差 1~3%） |

> 性能差距主要在长程任务和泛化场景，常规任务上 LoRA 完全够用。

---

## 五、选型决策树

```
                          ┌─ 数据量 > 5k demos？ ─→ 全量微调（公司方案）
                          │
                          ├─ 单卡显存 ≥ 70 GB？ ──→ 全量微调（公司方案）
                          │
                          ├─ 追求最终交付的 SOTA？ → 全量微调（公司方案）
                          │
你的需求 ────────────────┼─ 数据量 < 1k demos？ ─→ ★ LoRA 微调（自有方案）★
                          │
                          ├─ 只有 RTX 4090？ ────→ ★ LoRA 微调 ★
                          │
                          ├─ 需要快速实验/迭代？ → ★ LoRA 微调 ★
                          │
                          ├─ 一个 base 配多个任务？→ ★ LoRA 微调（多 adapter）★
                          │
                          └─ 不确定？ ──────────→ 先 LoRA 跑通流程，再决定要不要全量
```

### 5.1 推荐工作流

1. **第一轮：LoRA 调通**
   - 用 LoRA 在 1~2k 步内验证：数据格式正确、prompt 含义匹配、Policy 切片对齐
   - 显存压力小，迭代快

2. **第二轮（可选）：LoRA 拉满**
   - 加 rank（如 32）、加步数（20k+）、加数据增强
   - 若性能达标 → 完成

3. **第三轮（可选）：全量微调**
   - 仅当 LoRA 已经撞顶且对最终性能极度敏感时
   - 用公司全量配置作为底，加上你的 DataConfig 即可

---

## 六、共同点：哪些代码可复用

两个项目的代码差异主要在 TrainConfig，**其他几乎一致**：

### 6.1 完全可复用的部分

| 文件 / 模块 | 复用方式 |
| --- | --- |
| `src/openpi/policies/*_policy.py` | 直接照搬，Inputs/Outputs 类公司案例可以套 |
| `src/openpi/training/config.py` 中的 `DataConfig` 类 | 直接照搬 `CyborgV2GripperDataConfig` 改名即可 |
| `scripts/compute_norm_stats.py` | 完全不变 |
| `scripts/train.py` | 完全不变 |
| `scripts/serve_policy.py`（环境注册以外） | 框架代码不变 |
| 环境配置（uv sync、conda、FFmpeg、环境变量） | 完全一致 |
| ROS 客户端代码（infer.py） | 公司代码可直接套（在公司 repo），自有项目按相似模板写 |

### 6.2 唯一需要差异化的部分

| 项 | 自有 LoRA 项目要做 |
| --- | --- |
| `model.paligemma_variant` | 加 `_lora` 后缀 |
| `model.action_expert_variant` | 加 `_lora` 后缀 |
| `freeze_filter` | 加 `.get_freeze_filter()` |
| `ema_decay` | 改 `None` |
| `lr_schedule` / `num_train_steps` | 可适当调（非必须） |

---

## 七、互相迁移指南

### 7.1 公司全量 → 自有 LoRA

**场景**：你拿到公司的 `pi05_cyborgv2_*` 配置，想改成 LoRA 跑。

#### Step 1：复制 TrainConfig，改 4 个开关

```python
# 原（公司全量）
TrainConfig(
    name="pi05_cyborgv2_pick_up_tray_v2_0313",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
    data=CyborgV2GripperDataConfig(repo_id="pick_up_tray_v2_0313", ...),
    weight_loader=...,
    num_train_steps=28_000,
)

# 改为 LoRA
TrainConfig(
    name="pi05_cyborgv2_pick_up_tray_v2_0313_lora",      # ← 加 _lora
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=30,
        paligemma_variant="gemma_2b_lora",                # ← 加
        action_expert_variant="gemma_300m_lora",          # ← 加
    ),
    data=CyborgV2GripperDataConfig(repo_id="pick_up_tray_v2_0313", ...),  # ← 不变
    weight_loader=...,                                                       # ← 不变
    freeze_filter=pi0_config.Pi0Config(                  # ← 加
        pi05=True, action_horizon=30,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,                                       # ← 加
    lr_schedule=_optimizer.CosineDecaySchedule(           # ← 推荐改
        warmup_steps=1_000, peak_lr=5e-5,
        decay_steps=30_000, decay_lr=5e-5,
    ),
    num_train_steps=10_000,                               # ← 减为 1/3
)
```

#### Step 2：Policy / DataConfig 完全不动

`CyborgV2GripperInputs/Outputs` 和 `CyborgV2GripperDataConfig` 无需任何修改。

#### Step 3：重算 norm_stats（如果 config name 变了）

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_cyborgv2_pick_up_tray_v2_0313_lora
```

#### Step 4：训练

```bash
CUDA_VISIBLE_DEVICES=0,1 \
    uv run scripts/train.py pi05_cyborgv2_pick_up_tray_v2_0313_lora \
    --exp-name=lora_run1 --batch_size 32
```

显存预期：从 ~70 GB/卡 → ~35 GB/卡

---

### 7.2 自有 LoRA → 公司全量

**场景**：LoRA 撞顶想冲 SOTA，切换到全量。

#### Step 1：摘掉 4 开关

```python
# 原（自有 LoRA）
model=pi0_config.Pi0Config(
    pi05=True, action_horizon=10, discrete_state_input=False,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
),
freeze_filter=pi0_config.Pi0Config(...).get_freeze_filter(),
ema_decay=None,

# 改为全量
model=pi0_config.Pi0Config(
    pi05=True, action_horizon=10, discrete_state_input=False,
    # 默认 paligemma_variant="gemma_2b"
    # 默认 action_expert_variant="gemma_300m"
),
# 删 freeze_filter
ema_decay=0.999,  # 或者直接删（默认就是 0.999）
```

#### Step 2：调学习率、batch、步数

```python
lr_schedule=_optimizer.CosineDecaySchedule(
    warmup_steps=10_000,
    peak_lr=2.5e-5,
    decay_steps=100_000,
    decay_lr=2.5e-5,
),
batch_size=4,        # 显存允许的话可以加大
num_train_steps=30_000,
```

#### Step 3：硬件确认

确认有 ≥ 70 GB/卡 的显存。如果没有：
- 启用 fsdp（`fsdp_devices=4`）拆分模型
- 或考虑梯度累积

---

## 八、混合工作流推荐（建议）

> 这是我们团队的实际作业模式：

```
阶段 1：LoRA 快速调通（自有方案）
   ↓ 1~2 天
   ✓ 验证 Policy/DataConfig 正确
   ✓ 验证 prompt / 任务对齐
   ✓ 验证训练流程通畅
   ↓
阶段 2：LoRA 拉满（自有方案 + 调参）
   ↓ 3~5 天
   ✓ 加 rank / 加步数 / 调 lr
   ✓ 达到目标性能 → 部署完成
   ↓
阶段 3（可选）：全量微调（公司方案）
   ↓ 1~2 周
   ✓ 仅在 LoRA 性能撞顶时启动
   ✓ 用 LoRA 经验确定的超参作为起点
   ✓ 多卡 A100/H100
```

阶段 1~2 在 RTX 4090 / A800 上就够，阶段 3 才动 A100/H100。

---

## 九、FAQ：常见误解

### Q1：LoRA 是不是性能一定比全量差？

**A**：不一定。在数据量小、任务相似的场景下，LoRA 因为参数少而**更不容易过拟合**，反而泛化更好。
只有在大数据量 + 复杂任务下，全量微调才能拉开差距，但代价是显存和时长。

### Q2：可以"半量"微调吗？比如只训 action_expert，主 LLM 冻结？

**A**：可以，但通常不推荐。设置：

```python
paligemma_variant="gemma_2b_lora",        # 主 LLM LoRA
action_expert_variant="gemma_300m",       # action expert 全训练
```

`get_freeze_filter()` 会冻 paligemma 的 base、保留所有 lora 和 action expert 参数。
显存介于纯 LoRA 和纯全量之间。

### Q3：LoRA checkpoint 能不能只存几十 MB？

**A**：默认不能。当前 openpi 把 LoRA 与 base 合在一起存。
要单独存 LoRA 增量需要在 `_checkpoints.save` 加 `.*lora.*` 抽取。
工作量约半天，但对**多任务部署**有巨大价值（base 加载一次 + N 个轻量 adapter 切换）。

### Q4：公司案例里没有 LoRA 配置，但代码里保留了 `pi0_libero_low_mem_finetune` 是干嘛的？

**A**：那是从上游 openpi 继承的官方 LoRA 模板，公司自己**没用过 LoRA**。
你完全可以参考它的写法（在 `cyb_vla/.../config.py:1035-1055`），把公司的 `CyborgV2DataConfig` 套进去就是公司版 LoRA。

### Q5：用 LoRA 微调出来的 checkpoint，能不能用公司的 `serve_policy.py` 部署？

**A**：可以，**完全可以**。两个项目部署代码一致，只需在 `serve_policy.py` 注册你的 EnvMode + Checkpoint 路径即可。
唯一注意：在 `Checkpoint(config=...)` 中要填 LoRA 配置的 `name`（带 `_lora` 后缀），否则模型架构对不上。

### Q6：DataConfig 中 use_delta_joint_actions 在 LoRA 场景下要改吗？

**A**：不用。这是数据层的设置，与 LoRA 无关。
关节用 delta + 夹爪用 absolute 是机器人控制的通用最佳实践，全量和 LoRA 都该这么用。

### Q7：环境变量 `OPENPI_DATA_HOME` 在两个项目共享吗？

**A**：物理上共享（同一个机器），但**路径建议分开**：

```bash
# 公司项目
export OPENPI_DATA_HOME=/home/sirius/data/model_checkpoints/openpi_base_ckpt

# 自有项目
export OPENPI_DATA_HOME=/home/zhaoyuhang/openpi_base_ckpt
```

避免 GCS 下载/缓存互相覆盖。

### Q8：LoRA 微调的 ema_decay 为什么要关？

**A**：EMA 会维护一份 shadow params（与训练 params 同大小）。
全量微调时这两份都对应可训练参数，是合理代价。
但 LoRA 场景下 99% 的参数都是冻结的，shadow params 完全是浪费 → `ema_decay=None`。

实测：开 EMA 在 LoRA 下单卡多占 ~30 GB 显存。

---

## 附录：三份文档的关系

```
COMPANY_FINETUNING_GUIDE.md   ←  公司案例的完整复刻（来自 Sirius@cyborg）
LORA_FINETUNING_GUIDE.md      ←  我们自己的 LoRA 实战手册
FINETUNING_COMPARISON.md      ←  本文档：对比 + 选型 + 迁移
```

| 用途 | 看哪份 |
| --- | --- |
| 想跑公司的 cyborg 全量微调 | `COMPANY_FINETUNING_GUIDE.md` |
| 想在自己的数据上用 LoRA 训练 | `LORA_FINETUNING_GUIDE.md` |
| 想判断该选哪种、或者切换方案 | **本文档** |
| 第一次接触 openpi 微调 | 三份按顺序读：本文档 → LoRA → Company |

---

## 修订记录

| 日期 | 版本 |
| --- | --- |
| 2026-05-13 | 首版：基于双项目对比 + LIBERO-10 实测数据 |
