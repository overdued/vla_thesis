# openpi LoRA 微调手册（自有版）

> 版本：2026-05-13
> 整理者：Yuhang Zhao
> 适用仓库：`/home/zhaoyuhang/openpi`（已落地 `pi05_libero10_lora` 实战）
> 微调方式：**LoRA 参数高效微调（Low-Rank Adaptation）**

---

## 目录

1. [概览](#一概览)
2. [环境配置](#二环境配置)
3. [LoRA 核心原理](#三lora-核心原理)
4. [数据集准备](#四数据集准备lerobot-格式)
5. [新增 Policy 文件](#五新增-policy-文件)
6. [新增 DataConfig](#六新增-dataconfig)
7. [注册 LoRA TrainConfig](#七注册-lora-trainconfig关键)
8. [计算 norm_stats](#八计算-norm_stats)
9. [启动 LoRA 训练](#九启动-lora-训练)
10. [训练监控与调试](#十训练监控与调试)
11. [部署：Server + Client](#十一部署server--client)
12. [LoRA 调参指南](#十二lora-调参指南)
13. [故障排查](#十三故障排查faq)

---

## 一、概览

LoRA 工作流（与全量微调结构相同，**唯一不同**在 TrainConfig 中加四个开关）：

```
LeRobot 数据集 → MyRobotPolicy(Inputs/Outputs) → MyRobotDataConfig → TrainConfig
                                                                          ↓
                                                                ★ LoRA 四开关 ★
                                                          ┌────────────────────┐
                                                          │ paligemma_variant  │ = "gemma_2b_lora"
                                                          │ action_expert_variant │ = "gemma_300m_lora"
                                                          │ freeze_filter      │ = .get_freeze_filter()
                                                          │ ema_decay          │ = None
                                                          └────────────────────┘
                                                                          ↓
                                          scripts/compute_norm_stats.py → scripts/train.py
                                                                          ↓
                                                              checkpoints/<name>/<exp>/
                                                                          ↓
                                                            scripts/serve_policy.py
                                                                          ↓
                                                                   客户端推理脚本
```

本仓库已落地的 LoRA 配置（节选自 `src/openpi/training/config.py`）：

| 配置名 | 模型 | 数据集 | 实战状态 |
| --- | --- | --- | --- |
| `pi05_libero10_lora` | pi0.5 + LoRA | physical-intelligence/libero | ✅ 已训练完成（见 `checkpoints/`） |
| `pi05_libero10_full` | pi0.5 全量 | physical-intelligence/libero | ✅ 对照组 |
| `pi0_libero_low_mem_finetune` | pi0 + LoRA | physical-intelligence/libero | 模板（继承自上游） |
| `pi0_fast_libero_low_mem_finetune` | pi0-FAST + LoRA | physical-intelligence/libero | 模板（继承自上游） |

### 1.1 LoRA 一句话

> **冻结**预训练大模型权重，只插入并训练一组低秩矩阵 A、B，参数量 ~0.1-1% of base，显存与训练时长显著降低。

### 1.2 何时选 LoRA

| 选 LoRA ✓ | 选全量微调 |
| --- | --- |
| 数据量少（< 1k demos） | 数据量大（> 5k demos） |
| 单卡 RTX 4090（22.5 GB+） | 多卡 A100/H100（70 GB+） |
| 快速实验、多次迭代 | 追求 SOTA 最终交付 |
| 一个 base + 多个任务 adapter | 单一任务专门优化 |

---

## 二、环境配置

### 2.1 硬件要求

| 项 | 最小 | 推荐 |
| --- | --- | --- |
| GPU 显存 | 22.5 GB / 卡 | 2× A800 (80GB) |
| 内存 | 64 GB | 128 GB |
| 磁盘 | 500 GB SSD | 1 TB NVMe |

> 实测 `pi05_libero10_lora` 配置 + batch_size=32 + 2× A800：单卡占用约 35 GB。

### 2.2 操作系统

- Ubuntu 22.04 已验证
- NVIDIA Driver ≥ 535，CUDA 12.x

### 2.3 安装 `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或升级
uv self update
```

### 2.4 同步项目依赖

```bash
cd /home/zhaoyuhang/openpi

# （可选）国内镜像
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

GIT_LFS_SKIP_SMUDGE=1 uv sync --index-url https://pypi.org/simple
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 2.5 关键依赖（pyproject.toml 已锁定）

| 包 | 版本 | 用途 |
| --- | --- | --- |
| Python | ≥ 3.11 | |
| jax | 0.5.3 + CUDA 12 | 训练后端 |
| flax | 0.10.2 | nnx 模块 |
| torch | 2.7.1 | 数据加载 |
| transformers | 4.53.2 | Tokenizer |
| lerobot | `/tmp/lerobot` | 数据格式 |
| orbax-checkpoint | 0.11.13 | 权重 IO |

### 2.6 客户端环境（独立 conda env）

```bash
conda create -n openpi_client python=3.10 -y
conda activate openpi_client
cd /home/zhaoyuhang/openpi/packages/openpi-client
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.7 缓存与数据目录

```bash
export OPENPI_DATA_HOME=/home/zhaoyuhang/openpi_base_ckpt   # base ckpt 缓存
export HF_LEROBOT_HOME=/home/zhaoyuhang/data/dataset         # LeRobot 数据集根目录
```

不设置则默认是 `~/.cache/openpi`。

### 2.8 验证安装

```bash
cd /home/zhaoyuhang/openpi
uv run scripts/train.py --help
# 应列出所有可用 config name，包括 pi05_libero10_lora
```

---

## 三、LoRA 核心原理

### 3.1 数学形式

```
原始前向：  y = W · x
LoRA 前向： y = W · x + (α/r) · B · A · x
                        └────── 可训练 ──────┘
```

其中 A ∈ ℝ^(r×d_in)、B ∈ ℝ^(d_out×r)，**r << min(d_in, d_out)**。
- A 初始化为 `normal(stddev=0.01)`
- B 通常初始化为 0 → 训练开始时 LoRA 增量为 0，不破坏预训练知识

### 3.2 openpi 中的 LoRAConfig（`src/openpi/models/lora.py`）

```python
@struct.dataclass
class LoRAConfig:
    rank: int              # 低秩维度 r
    alpha: float = 1.0     # 缩放系数
    init_fn: ...           # 默认 normal stddev=0.01
    rslora: bool = False   # rank-stabilized 开关
    axes: tuple = (-2, -1) # 注入维度
```

### 3.3 LoRA 注入位置

`src/openpi/models/gemma.py` 在 Gemma Transformer 的以下 einsum 注入 LoRA：

```
Attention 层:
  - qkv_einsum (统一 QKV 投影)
  - q_einsum / kv_einsum (GQA 场景)
  - attn_vec_einsum (输出投影)

FeedForward 层:
  - gating_einsum (门控)
  - linear (输出)
```

### 3.4 默认 rank/alpha（`gemma.py:88-108`）

| Variant | Attn rank/alpha | FFN rank/alpha | 适用 |
| --- | --- | --- | --- |
| `gemma_2b_lora` | 16 / 16 | 16 / 16 | 主 LLM 通用 |
| `gemma_300m_lora` | 32 / 32 | 32 / 32 | Action Expert 通用 |

> 2B 用更小 rank：参数体量大，r 太大易过拟合
> 300M 用更大 rank：参数体量小，需要补足表达力

### 3.5 冻结策略源码（`pi0_config.py:88-117`）

```python
def get_freeze_filter(self) -> nnx.filterlib.Filter:
    filters = []
    has_lora = False
    gemma_params_filter        = nnx_utils.PathRegex(".*llm.*")
    action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")

    if "lora" in self.paligemma_variant:
        filters.append(gemma_params_filter)
        if "lora" not in self.action_expert_variant:
            filters.append(nnx.Not(action_expert_params_filter))
        has_lora = True
    elif "lora" in self.action_expert_variant:
        filters.append(action_expert_params_filter)
        has_lora = True

    if has_lora:
        # 排除所有 lora 参数 → lora 参数不冻结（参与训练）
        filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))

    return nnx.All(*filters) if filters else nnx.Nothing
```

**一句话**：名字带 `lora` 的参数 → 不冻结；其他匹配到 `llm` 的 → 冻结。

### 3.6 权重加载机制（`weight_loaders.py`）

```python
class CheckpointWeightLoader:
    def load(self, params):
        loaded = restore_params(checkpoint_path)
        # base ckpt 中不存在 lora_a / lora_b
        # 用 missing_regex=".*lora.*" 让 model 的随机初值作为兜底
        return _merge_params(loaded, params, missing_regex=".*lora.*")
```

含义：base 权重正常加载，LoRA 矩阵保持随机初始化，训练开始时 base 行为 = 预训练行为。

---

## 四、数据集准备（LeRobot 格式）

### 4.1 LeRobot v2 目录结构

```
$HF_LEROBOT_HOME/
└── <repo_id>/
    ├── meta/
    │   ├── episodes.jsonl
    │   ├── info.json
    │   ├── stats.json
    │   └── tasks.jsonl
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── ...
    └── videos/
        └── chunk-000/
            └── observation.images.head/
                ├── episode_000000.mp4
                └── ...
```

### 4.2 关键字段

| 字段 | 形状/类型 | 必填 |
| --- | --- | --- |
| `observation.state` | (T, state_dim) float32 | ✓ |
| `observation.images.<cam>` | (T, H, W, 3) uint8 / mp4 | ≥ 1 个相机 |
| `actions` | (T, action_dim) float32 | ✓ |
| `task_index` | int | ✓ |
| `prompt` 或 task 字段 | str | ✓ |

### 4.3 转换示例

参考 `examples/libero/convert_libero_data_to_lerobot.py`：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset.create(
    repo_id="my_robot/pack_drill",
    fps=30,
    features={
        "observation.state": {"dtype": "float32", "shape": (16,)},
        "observation.images.head":        {"dtype": "video", "shape": (224, 224, 3)},
        "observation.images.left_wrist":  {"dtype": "video", "shape": (224, 224, 3)},
        "observation.images.right_wrist": {"dtype": "video", "shape": (224, 224, 3)},
        "actions": {"dtype": "float32", "shape": (16,)},
    },
    use_videos=True,
)

for episode in raw_episodes:
    for frame in episode:
        dataset.add_frame(frame)
    dataset.save_episode(task=instruction)

dataset.consolidate()
```

### 4.4 验证

```bash
ls $HF_LEROBOT_HOME/my_robot/pack_drill/  # 应能看到 meta/ data/ videos/
```

---

## 五、新增 Policy 文件

> 路径：`src/openpi/policies/my_robot_policy.py`
> 参考：`src/openpi/policies/libero_policy.py`

### 5.1 完整模板

```python
import dataclasses
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """LeRobot 存 float32 (C,H,W)，模型要 uint8 (H,W,C)。"""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MyRobotInputs(transforms.DataTransformFn):
    """训练 + 推理共用。"""

    model_type: _model.ModelType  # 由 ModelTransformFactory 自动注入

    def __call__(self, data: dict) -> dict:
        # ===== 1. state 切片 =====
        state = data["observation.state"][..., 0:16]  # 调整为你的维度

        # ===== 2. 图像解码 =====
        base_image        = _parse_image(data["observation.images.head"])
        left_wrist_image  = _parse_image(data["observation.images.left_wrist"])
        right_wrist_image = _parse_image(data["observation.images.right_wrist"])

        # ===== 3. 按模型类型组装 =====
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, left_wrist_image, right_wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model: {self.model_type}")

        inputs = {
            "state": state,
            "image":      dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # ===== 4. 训练阶段才会有 actions =====
        if "actions" in data:
            inputs["actions"] = np.array(data["actions"])[..., 0:16]

        # ===== 5. prompt =====
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class MyRobotOutputs(transforms.DataTransformFn):
    """仅推理使用：模型 → 机器人。"""

    def __call__(self, data: dict) -> dict:
        result = {"actions": np.asarray(data["actions"][:, :16])}
        for key in ("origin_actions", "state"):
            if key in data:
                result[key] = data[key]
        return result
```

### 5.2 image key 命名约定（PaliGemma 视觉编码器固定）

| 模型 | image keys |
| --- | --- |
| pi0 / pi0.5 | `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` |
| pi0-FAST | `base_0_rgb`, `base_1_rgb`, `wrist_0_rgb` |

实际相机缺失时：传 `np.zeros_like(base_image)` 占位 + `image_mask=False`（pi0/pi0.5）或 `True`（pi0-FAST）。

---

## 六、新增 DataConfig

> 路径：在 `src/openpi/training/config.py` 中追加类。

```python
# 文件顶部
import openpi.policies.my_robot_policy as my_robot_policy


@dataclasses.dataclass(frozen=True)
class MyRobotDataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 1. RepackTransform：LeRobot 字段 → 训练用键名
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "actions":                       "actions",
                        "observation.state":             "observation.state",
                        "observation.images.head":       "observation.images.head",
                        "observation.images.left_wrist": "observation.images.left_wrist",
                        "observation.images.right_wrist":"observation.images.right_wrist",
                        "prompt":                        "prompt",
                        "task_index":                    "task_index",
                    }
                )
            ]
        )

        # 2. data_transforms：调用上一节 Policy 类
        data_transforms = _transforms.Group(
            inputs=[my_robot_policy.MyRobotInputs(model_type=model_config.model_type)],
            outputs=[my_robot_policy.MyRobotOutputs()],
        )

        # 3. model_transforms：tokenizer + 标准化
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        # 4. delta action 处理
        if self.use_delta_joint_actions:
            # make_bool_mask(14, -1, -1) → [1]*14 + [0]*1 + [0]*1
            # 双臂关节用 delta，双夹爪保留绝对
            delta_action_mask = _transforms.make_bool_mask(14, -1, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

`make_bool_mask` 用法：

```python
make_bool_mask(7, -3)        # [1]*7 + [0]*3
make_bool_mask(14, -1, -1)   # [1]*14 + [0]*1 + [0]*1
make_bool_mask(6, -1, 6, -1) # [1]*6 + [0]*1 + [1]*6 + [0]*1   ← ALOHA 双臂
```

正数 = delta，负数 = absolute。**夹爪/手指必须 absolute**，否则累积误差爆炸。

---

## 七、注册 LoRA TrainConfig（★关键★）

> 在 `src/openpi/training/config.py` 的 `_CONFIGS = [...]` 列表中追加。
> 参考已落地实现：`src/openpi/training/config.py:770-803`（`pi05_libero10_lora`）

### 7.1 pi0.5 + LoRA（推荐）

```python
TrainConfig(
    name="pi05_my_robot_lora",

    # ★ 开关 1: 模型变体改成 _lora 后缀
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=30,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),

    data=MyRobotDataConfig(
        repo_id="my_robot/pack_drill",
        use_delta_joint_actions=True,
        base_config=DataConfig(prompt_from_task=True),
    ),

    # 加载 base ckpt（LoRA 必须从 base 启动）
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),

    # ★ 开关 2: freeze_filter（model 参数要一致）
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=30,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),

    # ★ 开关 3: 关闭 EMA
    ema_decay=None,

    # ★ 开关 4: 学习率可用较大值（LoRA 抗噪能力强）
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=5e-5,
        decay_steps=30_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),

    batch_size=32,
    num_train_steps=10_000,        # LoRA 收敛快，10k 通常够
    save_interval=2_000,
    num_workers=16,
    wandb_enabled=False,
),
```

### 7.2 已落地：`pi05_libero10_lora`

```python
# src/openpi/training/config.py:770-803（仓库现有）
TrainConfig(
    name="pi05_libero10_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    batch_size=32,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000, peak_lr=5e-5, decay_steps=100_000, decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=None,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True, action_horizon=10, discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    num_train_steps=10_000,
    wandb_enabled=False,
),
```

### 7.3 pi0 + LoRA（旧版兼容）

```python
TrainConfig(
    name="pi0_my_robot_lora",
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=MyRobotDataConfig(repo_id="my_robot/pack_drill"),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,
    num_train_steps=30_000,
),
```

### 7.4 pi0-FAST + LoRA

```python
TrainConfig(
    name="pi0_fast_my_robot_lora",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=7,
        action_horizon=10,
        max_token_len=180,       # 单臂≈180，双臂≈250
        paligemma_variant="gemma_2b_lora",
    ),
    data=MyRobotDataConfig(repo_id="my_robot/pack_drill"),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    freeze_filter=pi0_fast.Pi0FASTConfig(
        action_dim=7, action_horizon=10, max_token_len=180,
        paligemma_variant="gemma_2b_lora",
    ).get_freeze_filter(),
    ema_decay=None,
),
```

### 7.5 LoRA 四开关速查表（最容易踩坑）

| 开关 | 全量微调 | **LoRA 微调** |
| --- | --- | --- |
| `paligemma_variant` | `"gemma_2b"` | **`"gemma_2b_lora"`** |
| `action_expert_variant` | `"gemma_300m"` | **`"gemma_300m_lora"`** |
| `freeze_filter` | 不设置（默认 `Nothing`） | **`pi0_config.Pi0Config(...).get_freeze_filter()`** |
| `ema_decay` | `0.999`（默认） | **`None`** |

> 这四个必须**一起改**，缺一不可。常见错误：只改前两个忘了 freeze_filter，结果模型 base 也被训了，等于半全量微调，显存爆炸。

---

## 八、计算 norm_stats

```bash
cd /home/zhaoyuhang/openpi

HF_LEROBOT_HOME=/home/zhaoyuhang/data/dataset \
    uv run scripts/compute_norm_stats.py --config-name pi05_my_robot_lora
```

输出位置：`assets/pi05_my_robot_lora/<asset_id>/norm_stats.json`

> `asset_id` 由 `AssetsConfig.asset_id` 或 `repo_id` 决定。

---

## 九、启动 LoRA 训练

### 9.1 单卡（开发/调试）

```bash
cd /home/zhaoyuhang/openpi

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
HF_LEROBOT_HOME=/home/zhaoyuhang/data/dataset \
OPENPI_DATA_HOME=/home/zhaoyuhang/openpi_base_ckpt \
    uv run scripts/train.py pi05_my_robot_lora \
    --exp-name=my_robot_lora_run1 \
    --batch_size 16
```

### 9.2 双卡（生产推荐）

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
HF_LEROBOT_HOME=/home/zhaoyuhang/data/dataset \
OPENPI_DATA_HOME=/home/zhaoyuhang/openpi_base_ckpt \
    uv run scripts/train.py pi05_my_robot_lora \
    --exp-name=my_robot_lora_run1 \
    --batch_size 32
```

### 9.3 后台运行

```bash
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
HF_LEROBOT_HOME=/home/zhaoyuhang/data/dataset \
    uv run scripts/train.py pi05_my_robot_lora \
    --exp-name=my_robot_lora_run1 --batch_size 32" \
    > /tmp/lora_train.log 2>&1 &

tail -f /tmp/lora_train.log
```

### 9.4 参数与环境变量

| 项 | 含义 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | 用哪几张卡 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | JAX 预占显存比例（0.9~0.95） |
| `HF_LEROBOT_HOME` | LeRobot 数据集根目录 |
| `OPENPI_DATA_HOME` | base ckpt 缓存目录 |
| `pi05_my_robot_lora` | TrainConfig.name |
| `--exp-name` | 输出落到 `checkpoints/<name>/<exp_name>/` |
| `--batch_size` | 覆盖 TrainConfig 默认值 |
| `--overwrite` | 同名 ckpt 目录直接覆盖 |
| `--resume` | 从 ckpt 续训（与 overwrite 互斥） |

### 9.5 输出目录

```
checkpoints/pi05_my_robot_lora/my_robot_lora_run1/
├── 2000/
├── 4000/
└── 10000/                ← 终态
    ├── params/           # 模型权重（含 LoRA + 冻结的 base 副本）
    ├── opt_state/        # 优化器状态
    └── ...
```

> 当前 openpi 默认保存完整 params（含 base 冻结副本），单个 ckpt 约 7 GB。
> 若想只存 LoRA 增量（约 50~150 MB），需要在 `_checkpoints.save` 里加 `.*lora.*` 抽取（非默认行为）。

---

## 十、训练监控与调试

### 10.1 显存监控

```bash
watch -n 1 nvidia-smi
```

实测 `pi05_libero10_lora`，2× A800，batch_size=32：单卡 35 GB。

### 10.2 日志解读

```
I 12:34:56.789 step=2000 loss=0.0432 grad_norm=1.23 lr=5.0e-05 sps=8.5
```

| 字段 | 含义 |
| --- | --- |
| `loss` | Flow Matching MSE，pi0.5 通常 0.01~0.05 |
| `grad_norm` | 梯度范数（clip 到 1.0） |
| `lr` | 当前学习率（cosine schedule） |
| `sps` | steps per second |

### 10.3 训练曲线判断

| 现象 | 诊断 |
| --- | --- |
| loss 从 1.0 → 0.05 平滑下降 | 正常 |
| loss 卡在 0.2+ 不动 | 欠拟合：加 rank、加步数、检查数据 |
| 训练 loss 降但推理表现差 | 数据/分布 mismatch，检查 repack key 是否对齐 |
| loss NaN | 学习率太大；降到 1e-5 重试 |

### 10.4 WandB（可选）

`TrainConfig(wandb_enabled=True, project_name="my_robot_lora")` 后 `wandb login` 一次。

每个实验自动写 `wandb_id.txt`，断点续训会复用 run id。

---

## 十一、部署：Server + Client

### 11.1 在 serve_policy.py 中注册

```python
# scripts/serve_policy.py
class EnvMode(enum.Enum):
    # ...
    MY_ROBOT = "my_robot"

DEFAULT_CHECKPOINTS = {
    EnvMode.MY_ROBOT: Checkpoint(
        config="pi05_my_robot_lora",
        dir="/home/zhaoyuhang/openpi/checkpoints/pi05_my_robot_lora/my_robot_lora_run1/10000",
    ),
}
```

### 11.2 启动 Server

```bash
cd /home/zhaoyuhang/openpi
uv run scripts/serve_policy.py --env=MY_ROBOT
# 默认 0.0.0.0:8000
```

显式指定 checkpoint：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_my_robot_lora \
    --policy.dir=./checkpoints/pi05_my_robot_lora/my_robot_lora_run1/10000
```

### 11.3 客户端推理

```python
# conda activate openpi_client
from openpi_client import websocket_client_policy, image_tools
import numpy as np

policy = websocket_client_policy.WebsocketClientPolicy(host="0.0.0.0", port=8000)

# key 名必须对齐 MyRobotDataConfig.RepackTransform
obs = {
    "observation.state": qpos,   # (state_dim,)
    "observation.images.head":        image_tools.resize_with_pad(img_h, 224, 224),
    "observation.images.left_wrist":  image_tools.resize_with_pad(img_l, 224, 224),
    "observation.images.right_wrist": image_tools.resize_with_pad(img_r, 224, 224),
    "prompt": instruction,
}

action_chunk = policy.infer(obs)["actions"]   # (action_horizon, action_dim)

for action in action_chunk:
    publish_joint_command(action)
```

### 11.4 客户端 ↔ DataConfig 对齐表

| 客户端字段 | DataConfig.RepackTransform 中的 key | Inputs 类切片 |
| --- | --- | --- |
| `obs["observation.state"]` | `"observation.state"` | `state[..., a:b]` |
| `obs["observation.images.head"]` | `"observation.images.head"` | base_0_rgb |
| `obs["prompt"]` | `"prompt"` | 透传到模型 |
| `policy.infer(obs)["actions"]` | （Outputs 类输出） | 由 AbsoluteActions 还原 |

任一处不匹配 → 模型输出乱动。

---

## 十二、LoRA 调参指南

### 12.1 自定义 rank/alpha

修改 `src/openpi/models/gemma.py`：

```python
# 新增 variant
if variant == "gemma_2b_lora_r8":
    return Config(
        width=2048, depth=18, mlp_dim=16_384,
        num_heads=8, num_kv_heads=1, head_dim=256,
        lora_configs={
            "attn": lora.LoRAConfig(rank=8,  alpha=8.0),
            "ffn":  lora.LoRAConfig(rank=8,  alpha=8.0),
        },
    )
```

并在 `gemma.py:55` 的 `Variant` Literal 中追加 `"gemma_2b_lora_r8"`。

### 12.2 rank 经验值

| 数据量 | rank（2B 部分） | rank（300M 部分） |
| --- | --- | --- |
| < 500 demos | 8 | 16 |
| 500 ~ 2000 demos | **16（默认）** | **32（默认）** |
| > 2000 demos | 32 或考虑全量微调 | 32~64 |

### 12.3 alpha 经验值

通常 `alpha = rank` 或 `alpha = 2 * rank`。
- alpha 越大 → LoRA 输出权重越大 → 学习信号越强
- 若 loss 早期就抖动，先把 alpha 降到 rank 一半

### 12.4 单边 LoRA

只在主 LLM 上加 LoRA、action expert 全训练：

```python
model=pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m",   # ← 不带 _lora
)
```

适用：你认为视觉-语言已足够好，只调动作专家。

### 12.5 学习率

| 配置 | 建议 lr |
| --- | --- |
| 双 LoRA | 5e-5（默认） |
| 单 LoRA + action expert 全训练 | 2.5e-5 |
| rank > 32 大 LoRA | 3e-5 |

---

## 十三、故障排查（FAQ）

| 现象 | 诊断 / 解决 |
| --- | --- |
| `Missing norm stats` | 先跑 `compute_norm_stats.py --config-name <name>` |
| OOM | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` + `--batch_size` 减半 + 增 `fsdp_devices` |
| 加载 base ckpt 时 shape mismatch | 检查 `paligemma_variant` 是否前后一致（TrainConfig + freeze_filter 都要带 `_lora`） |
| 模型实际显存比预期大 | 检查 `ema_decay` 是否还是 0.999（要改成 `None`） |
| 训练正常但推理输出乱 | 99% 是 obs key / qpos 拼接顺序与 DataConfig 不一致 |
| LoRA 训完性能反而比全量差 | 1) 检查 freeze_filter 是否生效；2) 加 rank；3) 加步数 |
| `Cannot resume and overwrite` | 二选一 |
| 多卡训练 hang | 检查 `fsdp_devices`；NCCL 防火墙 |
| WandB 报 not logged in | `wandb login` 或 `wandb_enabled=False` |
| uv sync 依赖冲突 | `rm -rf .venv && uv self update && uv sync` |

---

## 附录：完整命令清单

```bash
# ===== 一次性环境 =====
cd /home/zhaoyuhang/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --index-url https://pypi.org/simple
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 客户端
conda create -n openpi_client python=3.10 -y
conda activate openpi_client
cd /home/zhaoyuhang/openpi/packages/openpi-client && pip install -e .

# ===== LoRA 训练 =====
export HF_LEROBOT_HOME=/home/zhaoyuhang/data/dataset
export OPENPI_DATA_HOME=/home/zhaoyuhang/openpi_base_ckpt
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

uv run scripts/compute_norm_stats.py --config-name pi05_my_robot_lora

CUDA_VISIBLE_DEVICES=0,1 \
    uv run scripts/train.py pi05_my_robot_lora \
    --exp-name=my_robot_lora_run1 --batch_size 32

# ===== 部署 =====
uv run scripts/serve_policy.py --env=MY_ROBOT

conda activate openpi_client
python my_inference_script.py
```

---

## 修订记录

| 日期 | 版本 |
| --- | --- |
| 2026-05-13 | 首版：基于 `pi05_libero10_lora` 实战 + LoRA 调参经验 |
