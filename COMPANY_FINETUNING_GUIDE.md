# openpi 全量微调手册（公司 cyborg 案例版）

> 版本：2026-05-13
> 整理者：Yuhang Zhao
> 原始来源：`/home/zhaoyuhang/cyb_vla/agibot-world-master/openpi`（Sirius@cyborg，2025/10/13）
> 适用对象：cyborg v1.01 / cyborg v2（gripper / hand）等公司机器人平台
> 微调方式：**全量参数微调（Full Fine-Tuning）**

---

## 目录

1. [概览](#一概览)
2. [环境配置](#二环境配置)
3. [pi0 vs pi0.5 区别](#三pi0-vs-pi05-架构区别)
4. [新增 Policy 文件](#四新增-policy-文件inputsoutputs)
5. [新增 DataConfig](#五新增-dataconfig)
6. [注册 TrainConfig](#六注册-trainconfig)
7. [计算 norm_stats](#七计算-norm_stats)
8. [启动训练](#八启动训练)
9. [部署：Server + ROS Client](#九部署server--ros-client)
10. [硬件需求](#十硬件需求)
11. [故障排查](#十一故障排查)

---

## 一、概览

公司在 openpi 框架上的工作流（Sirius@cyborg 版本 1.0，2025/09/10）：

```
LeRobot 数据集          自定义 Policy            自定义 DataConfig
   ├─→ state         →  CyborgV2HandInputs   →  CyborgV2HandDataConfig  →  pi0/pi0.5 模型
   ├─→ images        →  (切片 + repack)         (RepackTransform +
   └─→ actions       →                          DeltaActions)
                                                          ↓
                                                  TrainConfig (全量微调)
                                                          ↓
                                          scripts/compute_norm_stats.py
                                                          ↓
                                            scripts/train.py（多卡 FSDP）
                                                          ↓
                                                checkpoints/<name>/<exp>/
                                                          ↓
                              scripts/serve_policy.py（WebSocket Server）
                                                          ↓
                                scripts/infer.py（ROS 客户端，运行在仿真/真机）
```

实际已落地的训练 config（节选自 `src/openpi/training/config.py`）：

| 配置名 | 模型 | 任务 |
| --- | --- | --- |
| `pi0_cyborgv101_pack_cola`   | pi0    | 装罐 |
| `pi0_cyborgv2_sim`            | pi0    | 仿真 |
| `pi05_cyborgv2_pack_drill_train_1031` | pi0.5 | 打包电钻 |
| `pi05_cyborgv2_pick_up_tray_v2_0313` | pi0.5 | 端托盘 |

---

## 二、环境配置

### 2.1 操作系统

- Ubuntu 22.04（公司唯一验证过的发行版）

### 2.2 安装 uv 与项目依赖（服务端 py311）

```bash
cd /home/zhaoyuhang/cyb_vla/agibot-world-master/openpi

# （可选）使用国内镜像
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

# 同步 .venv（跳过 LFS 大文件）
GIT_LFS_SKIP_SMUDGE=1 uv sync --index-url https://pypi.org/simple

# 激活
source .venv/bin/activate

# 安装为可编辑包
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 2.3 FFmpeg 7（Ubuntu 22.04 默认是 4.x，需升级）

```bash
sudo add-apt-repository ppa:ubuntuhandbook1/ffmpeg7
sudo apt update
sudo apt install -y ffmpeg
sudo apt install -y pkg-config python3-dev \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
```

### 2.4 客户端环境（推理用，py310）

> 服务端 (py311 + JAX) 与客户端 (py310 + ROS) 必须分离，避免依赖冲突。

```bash
conda create -n openpi_client python=3.10 -y
conda activate openpi_client

cd /home/zhaoyuhang/cyb_vla/agibot-world-master/openpi/packages/openpi-client
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.5 base checkpoint 目录与数据集目录

```bash
# Base checkpoint 存放点
export OPENPI_DATA_HOME=/home/sirius/data/model_checkpoints/openpi_base_ckpt

# LeRobot 数据集根目录
export HF_LEROBOT_HOME=~/data/dataset/cyborg_sim_task
```

> 这两个环境变量在公司 README 中**贯穿所有训练命令**，必须设置。

---

## 三、pi0 vs pi0.5 架构区别

公司经历了 **pi0 → pi0.5** 的整体升级（README 节选）：

| 维度 | pi0 | pi0.5 |
| --- | --- | --- |
| **State 输入处理** | 连续值，在 suffix 阶段处理 | 离散化成 token，与语言指令同处于 prefix 阶段 |
| **Timestep 注入** | 传统 MLP：时间 embedding 与 action token 拼接后过 MLP | **AdaRMSNorm**：时间 embedding 经 MLP 后作为条件信号动态调制归一化 |
| **配置写法** | `Pi0Config(pi05=False, action_horizon=30)` 或 `Pi0FASTConfig(...)` | `Pi0Config(pi05=True, action_horizon=30)` |

公司当前主推 pi0.5（带 `pi05=True` 标志）。

---

## 四、新增 Policy 文件（Inputs/Outputs）

> 这是**第一个**要写的文件，训练 + 推理都会用。
> 路径：`src/openpi/policies/<your_robot>_policy.py`
> 参考完整实现：`src/openpi/policies/cyborg_v2_hand_policy.py`

### 4.1 Inputs 类（数据 → 模型）

```python
# src/openpi/policies/cyborg_v2_hand_policy.py
import dataclasses
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class CyborgV2HandInputs(transforms.DataTransformFn):
    model_type: _model.ModelType  # 由 ModelTransformFactory 注入

    def __call__(self, data: dict) -> dict:
        state_read = data["observation.state"]
        # 拼接：右臂 7 维（idx 7:14） + 右手 3 维（idx 20:23）
        state = np.concatenate([
            state_read[..., 7:14],
            state_read[..., 20:23],
        ], axis=-1)

        base_image        = _parse_image(data["observation.images.head"])
        left_wrist_image  = _parse_image(data["observation.images.left_wrist"])
        right_wrist_image = _parse_image(data["observation.images.right_wrist"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, left_wrist_image, right_wrist_image)
                image_masks = (np.True_, np.False_, np.True_)  # 左腕被 mask
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

        if "actions" in data:
            actions_read = np.array(data["actions"])
            actions = np.concatenate([
                actions_read[..., 7:14],
                actions_read[..., 20:23],
            ], axis=-1)
            inputs["actions"] = actions

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs
```

### 4.2 Outputs 类（模型 → 机器人）

```python
@dataclasses.dataclass(frozen=True)
class CyborgV2HandOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 模型输出维度可能 > 实际需求，只取前 10 维（7 臂关节 + 3 手指）
        result = {"actions": np.asarray(data["actions"][:, :10])}
        for key in ("origin_actions", "state"):
            if key in data:
                result[key] = data[key]
        return result
```

### 4.3 Gripper 版本对比（cyborg_v2_gripper_policy.py）

差异点：

| 项 | hand 版 | gripper 版 |
| --- | --- | --- |
| state 维度 | 7 + 3 = 10 | 16（左臂 7 + 右臂 7 + 左 gripper 1 + 右 gripper 1） |
| actions 切片 | `[7:14]` + `[20:23]` | `[0:16]` |
| image_mask | `(True, False, True)`（左腕屏蔽） | `(True, True, True)` |

### 4.4 写自己的 policy 时需要确认

1. 原始 state 各维度物理含义（手册或 ROS topic 文档）
2. 哪些维度真正参与控制
3. 相机配置：哪些存在、哪些占位
4. action 维度：模型输出 32 维（pi0.5 默认），机器人只取前 N 维

---

## 五、新增 DataConfig

> 路径：在 `src/openpi/training/config.py` 中追加一个类。
> 参考完整实现：`src/openpi/training/config.py:411-469`（`CyborgV2GripperDataConfig`）

### 5.1 完整模板（基于 CyborgV2GripperDataConfig）

```python
# src/openpi/training/config.py 顶部 import
import openpi.policies.cyborg_v2_hand_policy as cyborg_v2_hand_policy


@dataclasses.dataclass(frozen=True)
class CyborgV2HandDataConfig(DataConfigFactory):
    """Hand 版本：左臂 7 + 右臂 7 + 左 hand 3 + 右 hand 3 = 20 dim 中取所需部分"""

    # 若 True：关节动作转 delta，gripper/手指保持绝对
    use_delta_joint_actions: bool = True

    # 数据集中无 prompt 字段时的兜底
    default_prompt: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # ===== 1. RepackTransform：LeRobot 字段名 → 训练用的标准键名 =====
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "actions":                       "actions",
                        "observation.state":             "observation.state",
                        "observation.images.head":       "observation.images.head",
                        "observation.images.hand_left":  "observation.images.hand_left",
                        "observation.images.hand_right": "observation.images.hand_right",
                        "prompt":                        "prompt",
                        "task_index":                    "task_index",
                    }
                )
            ]
        )
        # 备注：未在此 dict 列出的字段会被忽略

        # ===== 2. 调用上一节写的 Policy 类 =====
        data_transforms = _transforms.Group(
            inputs=[cyborg_v2_hand_policy.CyborgV2HandInputs(model_type=model_config.model_type)],
            outputs=[cyborg_v2_hand_policy.CyborgV2HandOutputs()],
        )

        # ===== 3. 模型侧（tokenizer + state/action 标准化） =====
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        # ===== 4. delta action 处理（公司 hand 版的实际掩码） =====
        if self.use_delta_joint_actions:
            # make_bool_mask(7, -3) → [1]*7 + [0]*3
            # 即：7 维臂关节用 delta，3 维手指保持绝对
            delta_action_mask = _transforms.make_bool_mask(7, -3)
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

### 5.2 `make_bool_mask` 速查

```python
make_bool_mask(7, -3)        # [1]*7 + [0]*3        ← hand 案例（双足版臂+手指）
make_bool_mask(14, -1, -1)   # [1]*14 + [0]*1 + [0]*1  ← gripper 案例（双臂+双夹爪）
make_bool_mask(7, 7, -6, -6) # [1]*7 + [1]*7 + [0]*6 + [0]*6  ← README 引用的示例
```

正数 → 转 delta，负数 → 保持 absolute。

---

## 六、注册 TrainConfig

> 在 `src/openpi/training/config.py` 的 `_CONFIGS = [...]` 列表中追加。
> 参考完整 entry：`src/openpi/training/config.py:978-999`（`pi05_cyborgv2_pick_up_tray_v2_0313`）

### 6.1 pi0.5 全量微调模板（公司主推）

```python
TrainConfig(
    name="pi05_cyborgv2_pick_up_tray_v2_0313",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=30),
    data=CyborgV2GripperDataConfig(
        repo_id="pick_up_tray_v2_0313",          # 对应 $HF_LEROBOT_HOME 下子目录
        use_delta_joint_actions=True,
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=28_000,
    save_interval=2000,
    num_workers=16,
    # checkpoint_base_dir="/home/sirius/data/model_checkpoints/openpi_finetune"
    # wandb_enabled=False
),
```

### 6.2 pi0 全量微调模板（旧版仍可用）

```python
TrainConfig(
    name="pi0_cyborgv2_sim",
    model=pi0_config.Pi0Config(action_horizon=50),
    data=CyborgV2DataConfig(
        repo_id="pack_in_the_supermarket",
        use_delta_joint_actions=True,
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    num_train_steps=40_000,
    save_interval=20_000,
    num_workers=16,
),
```

### 6.3 关键字段说明

| 字段 | 公司典型值 | 说明 |
| --- | --- | --- |
| `name` | `pi05_cyborgv2_<task>_<date>` | 训练命名约定 |
| `model.pi05` | `True` | 切换 pi0.5 架构 |
| `model.action_horizon` | 30 或 50 | 动作 chunk 长度 |
| `data.repo_id` | 数据集子目录名 | 对应 `$HF_LEROBOT_HOME` 下 |
| `data.use_delta_joint_actions` | `True` | 关节转 delta，夹爪保留绝对 |
| `data.base_config.prompt_from_task` | `True` | 从 LeRobot 的 task 字段读取 prompt |
| `weight_loader` | `pi05_base/params` | 全量微调用 base 起点 |
| `num_train_steps` | 28k~40k | 全量微调步数 |
| `save_interval` | 2k~20k | 保存频率 |
| `num_workers` | 16~32 | 数据加载并发 |

### 6.4 公司**不在** TrainConfig 中显式设置的字段（用 TrainConfig 默认值）

- `batch_size`（默认 32，命令行覆盖：`--batch_size 64`）
- `ema_decay`（默认 0.999，全量微调需要保留）
- `optimizer` / `lr_schedule`（默认 AdamW + Cosine warm 10k）
- `freeze_filter`（默认 `nnx.Nothing`，全部参数训练）

---

## 七、计算 norm_stats

每次新建/更换数据集都要重算一次：

```bash
HF_LEROBOT_HOME=~/data/dataset/cyborg_sim_task \
    uv run scripts/compute_norm_stats.py \
    --config-name pi05_cyborgv2_pick_up_tray_v2_0313
```

输出落到 `assets/<config_name>/<asset_id>/norm_stats.json`。

---

## 八、启动训练

### 8.1 公司标准 4 卡命令（README 原文）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
OPENPI_DATA_HOME=/home/sirius/data/model_checkpoints/openpi_base_ckpt \
HF_LEROBOT_HOME=~/data/dataset/cyborg_sim_task \
    uv run scripts/train.py pi0_cyborgv2_sim \
    --exp-name=pi0_cyborgv2_pack_in_the_supermarket_1 \
    --batch_size 64 \
    --overwrite
```

### 8.2 命令行参数与环境变量解释

| 项 | 含义 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES=0,1,2,3` | 使用 4 张 GPU（不设则用全部可见 GPU） |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` | JAX 预占 95% 显存 |
| `OPENPI_DATA_HOME` | base checkpoint 缓存目录 |
| `HF_LEROBOT_HOME` | LeRobot 数据集根目录 |
| `pi0_cyborgv2_sim` | TrainConfig 中的 `name` |
| `--exp-name` | 实验名 → 落到 `checkpoints/<name>/<exp>/` |
| `--batch_size 64` | 命令行覆盖 TrainConfig 默认值 |
| `--overwrite` | 同名实验目录直接覆盖（重新训练） |

### 8.3 输出目录

```
checkpoints/pi0_cyborgv2_sim/pi0_cyborgv2_pack_in_the_supermarket_1/
├── 2000/
├── 4000/
├── ...
└── 40000/
    ├── params/      # 完整模型权重（~7 GB+）
    ├── opt_state/   # 优化器状态（恢复训练用）
    └── ...
```

---

## 九、部署：Server + ROS Client

> 公司采用 **C/S 架构**：JAX 服务端推理 + ROS 客户端控制机器人。

### 9.1 在 serve_policy.py 中注册环境

```python
# scripts/serve_policy.py
class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    AGIBOT = "agibot"
    CYBORGV101 = "cyborgv101"
    CYBORGV2 = "cyborgv2"
    CYBORG_V2_GRIPPER = "cyborg_v2_gripper"   # ← 新增环境

# 在 DEFAULT_CHECKPOINTS 中映射
DEFAULT_CHECKPOINTS = {
    EnvMode.CYBORGV2: Checkpoint(
        config="pi0_cyborgv2_sim",
        dir="/root/workspace/model_checkpoints/openpi/openpi-assets/checkpoints/cyborg_pack_cola/"
    ),
    # ...
}
```

### 9.2 启动 Server

```bash
cd /home/zhaoyuhang/cyb_vla/agibot-world-master/openpi
uv run scripts/serve_policy.py --env=CYBORGV2
# 默认 0.0.0.0:8000
```

### 9.3 ROS 客户端（参考 scripts/infer.py）

```python
# 客户端环境
# conda activate openpi_client
from openpi_client import websocket_client_policy, image_tools
import numpy as np

policy = websocket_client_policy.WebsocketClientPolicy(host="0.0.0.0", port=8000)

# qpos 拼接顺序必须与 CyborgV2GripperInputs 对应
joint_state = np.array(act_raw.position)
left_arm_qpos    = joint_state[0:7]
left_gripper_pos = np.array([joint_state[7]])
right_arm_qpos   = joint_state[8:15]
right_gripper_pos= np.array([joint_state[15]])

qpos = np.concatenate([
    left_arm_qpos,
    right_arm_qpos,
    left_gripper_pos,
    right_gripper_pos,
])

# obs 字典 key 必须与 CyborgV2GripperDataConfig.RepackTransform 对应
obs = {
    "observation.state": qpos,
    "observation.images.top_head":    image_tools.resize_with_pad(img_h, 224, 224),
    "observation.images.hand_left":   image_tools.resize_with_pad(img_l, 224, 224),
    "observation.images.hand_right":  image_tools.resize_with_pad(img_r, 224, 224),
    "prompt": instruction,
}

action_chunk_dict = policy.infer(obs)
action_queue = list(action_chunk_dict["actions"])

while action_queue:
    is_end = (exec_cnt == ACTIONS_TO_EXECUTE - 1)
    action = action_queue.pop(0)
    sim_ros_node.publish_joint_command(action, is_end)
    exec_cnt += 1
```

### 9.4 与 autorun.sh 集成

公司 autorun.sh 中：

```bash
elif [ "$MODEL" = "openpi" ]; then
    INFER_CMD="docker exec -it $CONTAINER_NAME bash -ic 'cd $MODEL_PATH && conda activate openpi_client && python scripts/infer.py --task_name $TASK_NAME'"
```

### 9.5 ROS 接口约定（infer.py 关键节点）

```python
# 发布关节命令
def publish_joint_command(self, action, is_end=False): ...

# 订阅关节状态
def callback_joint_state(self, msg): ...

# Topic 列表（节选）
sub_img_head        : /sim/head_img
sub_img_left_wrist  : /sim/left_wrist_img
sub_img_right_wrist : /sim/right_wrist_img
sub_js              : /joint_states
sub_infer_start     : /sim/infer_start
pub_joint_command   : /sim/target_joint_state
```

---

## 十、硬件需求

> 节选自公司 README：

| 模式 | 显存需求 | 推荐 GPU |
| --- | --- | --- |
| Inference | > 8 GB | RTX 4090 |
| Fine-Tuning (LoRA) | > 22.5 GB | RTX 4090 |
| **Fine-Tuning (Full)** | **> 70 GB** | **A100 (80GB) / H100** |

> 当前训练脚本**不支持多机训练**，但支持单机多卡 FSDP（通过 `TrainConfig.fsdp_devices`）。

---

## 十一、故障排查

公司 README 整理的常见问题（已落地）：

| 现象 | 解决方案 |
| --- | --- |
| `uv sync` 依赖冲突 | `rm -rf .venv` 后重新 `uv sync`；确认 `uv self update` |
| Training OOM | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 加 `--batch_size` 减半 |
| Policy server 连不上 | 检查端口监听 + 防火墙；确认网段连通 |
| `Missing norm stats` | 先跑 `scripts/compute_norm_stats.py --config-name <name>` |
| Dataset 下载失败 | `huggingface-cli login`；确认网络 |
| CUDA/GPU 报错 | 检查 NVIDIA 驱动、CUDA toolkit、nvidia-container-toolkit |
| Examples import 报错 | `uv sync` 后再激活 `.venv` |
| Action 维度不匹配 | 检查 Policy 类的 state/action 切片与机器人 DoF |

---

## 附录：公司一行命令汇总

```bash
# 1. 环境
GIT_LFS_SKIP_SMUDGE=1 uv sync --index-url https://pypi.org/simple
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 2. 算 norm stats
HF_LEROBOT_HOME=~/data/dataset/cyborg_sim_task \
    uv run scripts/compute_norm_stats.py --config-name pi05_cyborgv2_pick_up_tray_v2_0313

# 3. 4 卡全量训练
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
OPENPI_DATA_HOME=/home/sirius/data/model_checkpoints/openpi_base_ckpt \
HF_LEROBOT_HOME=~/data/dataset/cyborg_sim_task \
    uv run scripts/train.py pi05_cyborgv2_pick_up_tray_v2_0313 \
    --exp-name=run_0313 --batch_size 64 --overwrite

# 4. Server
uv run scripts/serve_policy.py --env=CYBORGV2

# 5. Client（另一终端 / 另一容器）
conda activate openpi_client
python scripts/infer.py --task_name iros_pack_in_the_supermarket
```

---

## 修订记录

| 日期 | 版本 |
| --- | --- |
| 2025/09/10 | 公司 README v1.0（Sirius@cyborg） |
| 2025/10/13 | 增补 How to finetune / deploy 章节 |
| 2026-05-13 | 本文档整理：拆分自原 README，补充字段对照表与全量参数说明 |
