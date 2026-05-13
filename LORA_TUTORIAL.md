# openpi LoRA 微调完全指南

## 一、什么是 LoRA？为什么用它？

**LoRA (Low-Rank Adaptation)** 是一种参数高效微调方法，核心思想是：
- **冻结**预训练模型的原始权重
- 只训练少量新增的 "LoRA 适配器" 参数（低秩矩阵 A 和 B）
- 推理时：`output = base_output + scale * (x @ A @ B)`

### 优势
| 对比项 | 全量微调 | LoRA 微调 |
|--------|---------|----------|
| 可训练参数量 | 100% (e.g., 3B) | ~0.1-1% (e.g., 10-50M) |
| 显存占用 | 很高 | 显著降低 |
| 训练速度 | 慢 | 快 2-5x |
| 保存成本 | 大（整个模型） | 小（仅 LoRA 权重） |
| 多任务切换 | 需加载完整模型 | 只换 LoRA 权重 |
| 性能 | 通常最优 | 接近全量微调 |

---

## 二、openpi 中 LoRA 的实现原理

### 2.1 LoRA 参数结构（JAX/Flax 实现）

```python
# openpi/models/lora.py
@struct.dataclass
class LoRAConfig:
    rank: int              # 低秩维度 r，越小参数越少
    alpha: float = 1.0     # 缩放系数，通常设 rank 或 2*rank
    init_fn: ...           # 初始化函数（默认 normal stddev=0.01）
    rslora: bool = False   # 是否使用 rank-stabilized LoRA
    axes: tuple = (-2, -1) # 在哪个维度上注入 LoRA
```

### 2.2 注入位置

openpi 在 Gemma Transformer 的以下位置注入 LoRA：

```
Attention 层:
  - qkv_einsum (q, k, v 投影)
  - q_einsum / kv_einsum (GQA 场景)
  - attn_vec_einsum (输出投影)

FeedForward 层:
  - gating_einsum (门控投影)
  - linear (前馈输出投影)
```

### 2.3 冻结策略

```python
# openpi/models/pi0_config.py -> get_freeze_filter()
# 自动决定哪些参数冻结：
# - 若 paligemma_variant 含 "lora": 冻结 llm 参数，保留 LoRA 参数
# - 若 action_expert_variant 含 "lora": 冻结 llm_1 参数，保留 LoRA 参数
# - 所有含 "lora" 的参数都**不**冻结（会被训练）
```

---

## 三、关键配置参数详解

### 3.1 模型变体选择

```python
# gemma.py 中定义的可用变体
Variant = Literal[
    "gemma_300m",         # 300M 参数 base
    "gemma_300m_lora",    # 300M + LoRA (rank=32, alpha=32)
    "gemma_2b",           # 2B 参数 base
    "gemma_2b_lora",      # 2B + LoRA (rank=16, alpha=16)
]
```

### 3.2 训练配置中的关键参数

```python
TrainConfig(
    name="pi05_libero10_lora",

    # ===== 1. 模型配置：启用 LoRA =====
    model=pi0_config.Pi0Config(
        pi05=True,                              # 使用 pi0.5 架构
        action_horizon=10,                      # 动作预测步长
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",      # 主 LLM 启用 LoRA
        action_expert_variant="gemma_300m_lora", # Action Expert 启用 LoRA
    ),

    # ===== 2. 数据配置 =====
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
    ),

    # ===== 3. 加载预训练权重（必须！）=====
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),

    # ===== 4. 冻结策略（LoRA 关键！）=====
    freeze_filter=pi0_config.Pi0Config(
        pi05=True, action_horizon=10,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    # 作用：只有 LoRA 参数参与梯度更新，原始参数被冻结

    # ===== 5. 关闭 EMA（LoRA 必须）=====
    ema_decay=None,
    # 原因：EMA 会维护一套影子参数，LoRA 场景下不需要且浪费显存

    # ===== 6. 学习率和优化器 =====
    lr_schedule=CosineDecaySchedule(
        warmup_steps=1_000,     # 预热步数
        peak_lr=5e-5,           # 峰值学习率（LoRA 通常用较大 lr）
        decay_steps=100_000,
        decay_lr=5e-5,
    ),
    optimizer=AdamW(clip_gradient_norm=1.0),

    # ===== 7. 训练步数和 batch size =====
    batch_size=32,              # 2x A800 GPU 推荐
    num_train_steps=10_000,     # LoRA 收敛更快，不需要太多步
)
```

### 3.3 关键参数含义

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `paligemma_variant` | 主 LLM 变体 | `"gemma_2b_lora"` |
| `action_expert_variant` | Action Expert 变体 | `"gemma_300m_lora"` |
| `freeze_filter` | 冻结非 LoRA 参数 | 调用 `get_freeze_filter()` |
| `ema_decay` | EMA 衰减率 | LoRA 时必须为 `None` |
| `rank` (内置) | LoRA 低秩维度 | 16 (2B) / 32 (300M) |
| `alpha` (内置) | LoRA 缩放系数 | 16 (2B) / 32 (300M) |

---

## 四、训练命令

```bash
cd /home/zhaoyuhang/openpi

# 1. 检查配置是否被识别
uv run scripts/train.py pi05_libero10_lora --help

# 2. 启动训练（2 GPU）
CUDA_VISIBLE_DEVICES=0,1 uv run scripts/train.py pi05_libero10_lora \
    --exp-name=libero10_lora_run1

# 3. 后台训练（推荐）
nohup CUDA_VISIBLE_DEVICES=0,1 uv run scripts/train.py pi05_libero10_lora \
    --exp-name=libero10_lora_run1 > /tmp/lora_train.log 2>&1 &
```

---

## 五、权重加载原理

```python
# weight_loaders.py
class CheckpointWeightLoader:
    def load(self, params):
        loaded = restore_params(checkpoint_path)
        # 关键：LoRA 参数在 checkpoint 中不存在
        # 用 missing_regex=".*lora.*" 从当前模型 params 中补充
        return _merge_params(loaded, params, missing_regex=".*lora.*")
```

这意味着：
1. 从 base checkpoint 加载原始权重
2. LoRA 参数（lora_a, lora_b）由当前模型随机初始化
3. 训练时只更新 LoRA 参数

---

## 六、自定义 LoRA rank 和 alpha

如果想修改 rank（默认 2B 用 16，300M 用 32），需要修改 `gemma.py`：

```python
# openpi/models/gemma.py
if variant == "gemma_2b_lora_custom":
    return Config(
        width=2048, depth=18, mlp_dim=16384,
        num_heads=8, num_kv_heads=1, head_dim=256,
        lora_configs={
            "attn": lora.LoRAConfig(rank=8, alpha=8.0),   # 更小 rank
            "ffn": lora.LoRAConfig(rank=8, alpha=8.0),
        },
    )
```

然后在 Pi0Config 中使用 `"gemma_2b_lora_custom"`。

---

## 七、LoRA vs 全量微调对比

| 场景 | 推荐方式 |
|------|---------|
| 新机器人、新动作空间 | 全量微调 |
| 相似机器人、小数据集 (<1k demos) | **LoRA** |
| 快速实验/迭代 | **LoRA** |
| 追求 SOTA 性能 | 全量微调 |
| 显存有限 (<40GB) | **LoRA** |
| 多任务适配（一个 base，多个 LoRA） | **LoRA** |
