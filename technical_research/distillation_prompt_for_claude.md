# One-Step π0.5: 一致性蒸馏训练完整技术大纲

## 项目目标
将 π0.5 的 10 步 Flow Matching ODE 积分蒸馏为单步生成模型，推理加速 10x，保持成功率不下降。

## 硬件资源
- NVIDIA A800 80GB × 2
- 目标：每张卡使用不超过 50GB 显存

---

## 一、整体架构设计

```
教师模型 Teacher (π0.5, 冻结)          学生模型 Student (One-Step, 可训练)
┌──────────────────────────┐          ┌──────────────────────────┐
│  embed_prefix(obs)       │          │  复用 teacher 的         │
│  embed_suffix(x_t, t)    │          │  embed_prefix            │
│  PaliGemma.llm(.)        │          │  embed_suffix(noise, t=1)│
│  action_out_proj         │          │  PaliGemma.llm(.)        │
│  循环10步: x ← x + dt·v  │          │  action_out_proj         │
│  输出: a* (目标动作)      │          │  输出: a_pred (单步预测)  │
└──────────┬───────────────┘          └──────────┬───────────────┘
           │                                     │
           └────────── a* ─────────────────────→┘
                       ↑
              损失: ||a_pred - a*||²
```

**关键设计决策**：
1. 教师完全冻结（frozen），只用于生成蒸馏目标
2. 学生复用教师的 embed_prefix 和 action 投影层权重作为初始化
3. 学生只训练一个轻量的"去噪头"，将噪声直接映射到干净动作
4. 两张卡并行：卡0跑教师推理，卡1跑学生训练

---

## 二、文件结构与新增文件

在 openpi 仓库基础上，新增以下文件：

```
openpi/
├── src/
│   └── openpi/
│       ├── training/                    # 新增目录
│       │   ├── distillation_trainer.py  # 主训练器
│       │   └── one_step_model.py        # 学生模型定义
│       └── scripts/
│           └── distill_pi05.py          # 训练入口脚本
├── configs/
│   └── distill_pi05_libero.yaml       # 蒸馏训练配置
└── experiments/                         # 实验结果目录
    └── distill/
        ├── logs/
        ├── checkpoints/
        └── eval_results/
```

---

## 三、学生模型设计 (one_step_model.py)

```python
"""
学生模型: One-Step π0.5
将10步Flow Matching压缩为单步前向传播。
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Optional

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at


class OneStepPi05(nnx.Module):
    """
    单步 π0.5 蒸馏学生模型。
    
    设计理念：
    - 复用教师模型的 embed_prefix 和 action 投影层
    - 只添加一个轻量的"去噪头"
    - 输入: observation + 噪声
    - 输出: 直接预测干净动作（跳过10步ODE积分）
    """
    
    def __init__(
        self,
        teacher_config: pi0_config.Pi0Config,
        teacher_model,  # 冻结的教师模型
        rngs: nnx.Rngs,
        use_lora_for_prefix: bool = False,  # 是否冻结prefix
    ):
        # ========== 从教师模型复用组件 ==========
        
        # 1. 复用视觉编码器 (SigLIP) - 冻结
        self.img = teacher_model.PaliGemma.img
        
        # 2. 复用 LLM - 可选冻结或微调
        self.llm = teacher_model.PaliGemma.llm
        
        # 3. 复用 action 投影层 - 可选微调
        self.action_in_proj = teacher_model.action_in_proj
        self.action_out_proj = teacher_model.action_out_proj
        
        # 4. π0.5 的时间MLP - 冻结（单步不需要）
        if hasattr(teacher_model, 'time_mlp_in'):
            self.time_mlp_in = teacher_model.time_mlp_in
            self.time_mlp_out = teacher_model.time_mlp_out
        
        # 5. 状态投影 (π0 用, π0.5 不用)
        if hasattr(teacher_model, 'state_proj'):
            self.state_proj = teacher_model.state_proj
        
        # ========== 新增: 可训练的去噪头 ==========
        
        action_expert_config = _gemma.get_config(teacher_config.action_expert_variant)
        hidden_dim = action_expert_config.width
        
        # 方案A: 轻量MLP去噪头
        self.denoise_head_mlp = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.LayerNorm(hidden_dim, rngs=rngs),
            lambda x: nnx.swish(x),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.LayerNorm(hidden_dim, rngs=rngs),
            lambda x: nnx.swish(x),
        )
        
        # 方案B: 可选的注意力增强去噪头
        self.denoise_attn = nnx.MultiHeadAttention(
            num_heads=8,
            in_features=hidden_dim,
            rngs=rngs,
        )
        
        # 配置
        self.config = teacher_config
        self.action_dim = teacher_config.action_dim
        self.action_horizon = teacher_config.action_horizon
        self.pi05 = teacher_config.pi05
        
        # 标记训练模式
        self.deterministic = True
        
    def embed_prefix(self, obs: _model.Observation):
        """复用教师的 embed_prefix 方法。"""
        input_mask = []
        ar_mask = []
        tokens = []
        
        # 编码图像
        for name in obs.images:
            image_tokens, _ = self.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                jnp.repeat(obs.image_masks[name][:, None], image_tokens.shape[1], axis=1)
            )
            ar_mask += [False] * image_tokens.shape[1]
        
        # 编码语言
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask
    
    def embed_suffix_one_step(self, obs, noisy_actions, time_emb):
        """
        单步版本的 embed_suffix。
        关键区别：不构建时间步循环，直接编码噪声动作 + 时间embedding。
        """
        input_mask = []
        ar_mask = []
        tokens = []
        
        # π0: 添加state token
        if not self.pi05 and hasattr(self, 'state_proj'):
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]
        
        # 编码动作
        action_tokens = self.action_in_proj(noisy_actions)
        
        # π0.5: 时间MLP (使用预计算的 time_emb)
        if self.pi05:
            time_emb_processed = self.time_mlp_out(
                nnx.swish(self.time_mlp_in(time_emb))
            )
            time_emb_processed = nnx.swish(time_emb_processed)
            # 将时间信息注入动作token
            action_tokens = action_tokens + time_emb_processed[:, None, :]
        
        # ========== 关键: 去噪头 ==========
        # 方案A: MLP增强
        enhanced_tokens = self.denoise_head_mlp(action_tokens)
        # 残差连接
        action_expert_tokens = action_tokens + enhanced_tokens
        
        # 方案B: 自注意力增强 (可选)
        attn_out = self.denoise_attn(action_expert_tokens)
        action_expert_tokens = action_expert_tokens + attn_out
        
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        
        return tokens, input_mask, ar_mask
    
    @at.typecheck
    def __call__(
        self,
        obs: _model.Observation,
        noise: at.Float[at.Array, "b ah ad"],
    ) -> _model.Actions:
        """
        单步前向传播: noise → 干净动作。
        
        Args:
            obs: 观察 (图像+语言+状态)
            noise: 随机噪声 (B, action_horizon, action_dim)
            
        Returns:
            actions: 预测的干净动作 (B, action_horizon, action_dim)
        """
        batch_size = noise.shape[0]
        
        # 1. 编码 prefix (观察)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)
        
        # 2. 构建固定的时间embedding (t=1, 纯噪声端)
        time_vec = jnp.ones((batch_size,))  # t=1
        time_emb = self._get_time_embedding(time_vec)
        
        # 3. 编码 suffix (噪声 + 时间)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix_one_step(
            obs, noise, time_emb
        )
        
        # 4. 构建注意力掩码 (同教师模型)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = self._make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        
        # 5. 联合前向传播
        adarms_cond = self._get_adarms_cond(time_emb) if self.pi05 else None
        (prefix_out, suffix_out), _ = self.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        
        # 6. 输出动作预测
        actions = self.action_out_proj(
            suffix_out[:, -self.action_horizon:]
        )
        
        return actions
    
    def _get_time_embedding(self, t):
        """正弦余弦位置编码。"""
        from openpi.models.pi0 import posemb_sincos
        action_expert_config = _gemma.get_config(self.config.action_expert_variant)
        return posemb_sincos(
            t, action_expert_config.width,
            min_period=4e-3, max_period=4.0
        )
    
    def _get_adarms_cond(self, time_emb):
        """π0.5的adaRMS条件。"""
        if not self.pi05:
            return None
        t = self.time_mlp_in(time_emb)
        t = nnx.swish(t)
        t = self.time_mlp_out(t)
        t = nnx.swish(t)
        return t
    
    def _make_attn_mask(self, input_mask, mask_ar):
        """复用教师的注意力掩码逻辑。"""
        from openpi.models.pi0 import make_attn_mask
        return make_attn_mask(input_mask, mask_ar)
```

---

## 四、蒸馏损失函数设计

```python
"""
distillation_trainer.py - 蒸馏训练器

支持多种蒸馏损失：
1. 基础MSE: ||a_student - a_teacher||²
2. 加权MSE: 对时间步加权
3. 特征蒸馏: 对中间层特征做蒸馏
4. 多步蒸馏: 不仅蒸馏最终步，还蒸馏中间步
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple


class DistillationLoss:
    """蒸馏损失函数集合。"""
    
    @staticmethod
    def mse_loss(student_actions, teacher_actions):
        """基础MSE损失。"""
        return jnp.mean(jnp.square(student_actions - teacher_actions))
    
    @staticmethod
    def weighted_mse_loss(student_actions, teacher_actions, weights=None):
        """
        加权MSE损失。
        可以对动作的不同维度或时间步赋予不同权重。
        """
        diff = jnp.square(student_actions - teacher_actions)
        if weights is not None:
            diff = diff * weights
        return jnp.mean(diff)
    
    @staticmethod
    def consistency_loss(student_output, teacher_output, student_features=None, teacher_features=None):
        """
        一致性蒸馏损失（核心）。
        
        不仅蒸馏最终动作输出，还对齐中间特征表示。
        """
        # 动作输出蒸馏
        action_loss = jnp.mean(jnp.square(student_output - teacher_output))
        
        # 特征蒸馏 (如果提供了中间特征)
        feature_loss = 0.0
        if student_features is not None and teacher_features is not None:
            feature_loss = jnp.mean(jnp.square(student_features - teacher_features))
        
        return action_loss + 0.1 * feature_loss
    
    @staticmethod
    def multi_step_distillation_loss(
        student_actions,
        teacher_intermediate_states,  # 教师10步的中间状态
        time_weights=None,
    ):
        """
        多步蒸馏：不仅蒸馏最终动作，还对齐中间ODE轨迹。
        
        教师生成10个中间状态 [x_1.0, x_0.9, ..., x_0.0]
        学生学习匹配其中几个关键状态。
        """
        # 选择关键步数进行蒸馏（如t=1.0, t=0.5, t=0.0）
        key_steps = [0, 5, 9]  # 对应 t=1.0, 0.5, 0.0
        
        losses = []
        for step_idx in key_steps:
            target = teacher_intermediate_states[step_idx]
            # 学生当前是单步，需要通过不同噪声水平模拟
            loss = jnp.mean(jnp.square(student_actions - target))
            losses.append(loss)
        
        return jnp.mean(jnp.array(losses))


def compute_distillation_loss(
    student_params,
    teacher_params,
    observation,
    noise,
    teacher_actions_target,  # 教师生成的目标动作
    config: dict,
) -> Tuple[float, Dict]:
    """
    完整的蒸馏损失计算。
    
    Args:
        student_params: 学生模型参数
        teacher_params: 教师模型参数（冻结）
        observation: 观察
        noise: 随机噪声
        teacher_actions_target: 教师10步ODE积分结果
        config: 损失配置
        
    Returns:
        total_loss: 总损失
        metrics: 各损失分量的字典
    """
    # 学生单步预测
    student_actions = student_model.apply(student_params, observation, noise)
    
    # 计算各项损失
    metrics = {}
    
    # 1. 基础动作蒸馏损失
    action_loss = DistillationLoss.mse_loss(student_actions, teacher_actions_target)
    metrics['action_loss'] = action_loss
    
    # 2. 可选: 多步蒸馏损失
    total_loss = action_loss
    if config.get('use_multi_step', False):
        multi_step_loss = DistillationLoss.multi_step_distillation_loss(
            student_actions,
            teacher_intermediate_states,  # 需要保存教师的中间状态
        )
        total_loss += config.get('multi_step_weight', 0.5) * multi_step_loss
        metrics['multi_step_loss'] = multi_step_loss
    
    # 3. 可选: 特征对齐损失
    if config.get('use_feature_distill', False):
        # 需要提取中间层特征
        feature_loss = ...
        total_loss += config.get('feature_weight', 0.1) * feature_loss
        metrics['feature_loss'] = feature_loss
    
    metrics['total_loss'] = total_loss
    return total_loss, metrics
```

---

## 五、双卡并行训练循环 (distill_pi05.py)

```python
"""
distill_pi05.py - 蒸馏训练主脚本

双A800并行策略：
- GPU-0: 教师模型 (冻结, FP16) - 负责生成蒸馏目标
- GPU-1: 学生模型 (训练, BF16) - 负责学习和优化

通信方式: 只需要传递observation和teacher_output (小数据量)
"""

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding, NamedSharding
import optax
from flax.training import train_state
import numpy as np
from tqdm import tqdm

from openpi.models import pi0_config
from openpi.training.one_step_model import OneStepPi05
from openpi.training.distillation_trainer import compute_distillation_loss


def create_teacher(model_path: str, device):
    """加载并冻结教师模型 π0.5。"""
    # 从checkpoint加载
    config = pi0_config.Pi0Config(pi05=True)
    teacher = load_pi05_checkpoint(model_path, config)
    
    # 冻结所有参数
    teacher_params = jax.tree_map(lambda x: jax.device_put(x, device), teacher.parameters())
    
    return teacher, teacher_params


def create_student(teacher_config, teacher_model, device, rng_seed=42):
    """初始化学生模型。"""
    rngs = nnx.Rngs(rng_seed)
    student = OneStepPi05(
        teacher_config=teacher_config,
        teacher_model=teacher_model,
        rngs=rngs,
    )
    
    # 将学生模型移到指定设备
    student_params = jax.tree_map(lambda x: jax.device_put(x, device), student.parameters())
    
    return student, student_params


def teacher_generate_batch(teacher_params, teacher_apply_fn, observations, rng):
    """
    教师模型生成蒸馏目标。
    
    在GPU-0上运行，对每批数据执行10步ODE积分。
    """
    batch_size = observations.state.shape[0]
    
    # 采样噪声
    noise = jax.random.normal(rng, (batch_size, 50, 18))  # (B, H, D)
    
    # 教师10步推理: noise → actions
    # 使用 teacher_apply_fn (即 teacher_model.sample_actions)
    teacher_actions = teacher_apply_fn(
        teacher_params,
        observations,
        noise=noise,
        num_steps=10,
    )
    
    return noise, teacher_actions


def student_train_step(student_state, teacher_params, batch, rng, loss_config):
    """
    学生模型单步训练。
    
    在GPU-1上运行。
    """
    observations = batch['observation']
    noise = batch['noise']
    teacher_target = batch['teacher_actions']
    
    def loss_fn(student_params):
        # 学生单步预测
        student_actions = student_state.apply_fn(
            student_params,
            observations,
            noise,
        )
        
        # 蒸馏损失
        loss = jnp.mean(jnp.square(student_actions - teacher_target))
        
        # 可选: 加入辅助损失
        if loss_config.get('use_cosine_similarity', False):
            # 余弦相似度损失，保持方向一致
            cosine_loss = -jnp.mean(
                jnp.sum(student_actions * teacher_target, axis=-1) /
                (jnp.linalg.norm(student_actions, axis=-1) * jnp.linalg.norm(teacher_target, axis=-1) + 1e-8)
            )
            loss += loss_config.get('cosine_weight', 0.1) * cosine_loss
        
        return loss
    
    # 计算损失和梯度
    loss, grads = jax.value_and_grad(loss_fn)(student_state.params)
    
    # 更新参数
    student_state = student_state.apply_gradients(grads=grads)
    
    return student_state, {'loss': loss}


def training_loop(config):
    """
    主训练循环。
    
    数据流:
    1. 从dataloader取一批observation
    2. 送到GPU-0，教师生成 (noise, teacher_actions)
    3. 将 (observation, noise, teacher_actions) 打包
    4. 送到GPU-1，学生训练一步
    5. 重复
    """
    
    # ========== 初始化设备 ==========
    devices = jax.devices()[:2]  # 只用前2张A800
    print(f"使用设备: {devices}")
    
    teacher_device = devices[0]  # GPU-0
    student_device = devices[1]  # GPU-1
    
    # ========== 加载教师模型 ==========
    print("加载教师模型 π0.5...")
    teacher, teacher_params = create_teacher(
        config.teacher_checkpoint_path,
        teacher_device,
    )
    print(f"教师模型已加载到 {teacher_device}")
    
    # ========== 初始化学生模型 ==========
    print("初始化学生模型 One-Step π0.5...")
    student, student_params = create_student(
        teacher.config,
        teacher,
        student_device,
    )
    print(f"学生模型已初始化到 {student_device}")
    
    # ========== 优化器 ==========
    # 使用AdamW + 余弦退火学习率
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,  # 建议: 1e-4
        warmup_steps=config.warmup_steps,  # 建议: 1000
        decay_steps=config.total_steps,
        end_value=config.learning_rate * 0.1,
    )
    
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        b1=0.9,
        b2=0.98,
        weight_decay=config.weight_decay,  # 建议: 0.01
    )
    
    # 创建训练状态
    student_state = train_state.TrainState.create(
        apply_fn=student.__call__,
        params=student_params,
        tx=optimizer,
    )
    
    # ========== 数据加载 ==========
    # 使用LIBERO数据集
    dataloader = create_libero_dataloader(
        data_path=config.data_path,
        batch_size=config.batch_size,  # 建议: 2-4 (单卡80GB)
        shuffle=True,
    )
    
    # ========== 训练循环 ==========
    print(f"开始训练，总步数: {config.total_steps}")
    
    rng = jax.random.PRNGKey(config.seed)
    metrics_history = []
    
    for step, batch in enumerate(tqdm(dataloader, total=config.total_steps)):
        if step >= config.total_steps:
            break
        
        rng, step_rng, teacher_rng = jax.random.split(rng, 3)
        
        # ---- Step 1: 教师生成蒸馏目标 (GPU-0) ----
        observations = batch['observation']
        
        # 将obs移到GPU-0
        obs_on_teacher = jax.tree_map(
            lambda x: jax.device_put(x, teacher_device),
            observations,
        )
        
        # 教师10步推理
        noise, teacher_actions = teacher_generate_batch(
            teacher_params,
            teacher.__call__,
            obs_on_teacher,
            teacher_rng,
        )
        
        # ---- Step 2: 数据打包并移到GPU-1 ----
        # 将结果移到GPU-1
        noise_on_student = jax.device_put(noise, student_device)
        teacher_actions_on_student = jax.device_put(teacher_actions, student_device)
        obs_on_student = jax.tree_map(
            lambda x: jax.device_put(x, student_device),
            observations,
        )
        
        distill_batch = {
            'observation': obs_on_student,
            'noise': noise_on_student,
            'teacher_actions': teacher_actions_on_student,
        }
        
        # ---- Step 3: 学生训练 (GPU-1) ----
        student_state, metrics = student_train_step(
            student_state,
            teacher_params,
            distill_batch,
            step_rng,
            config.loss_config,
        )
        
        metrics_history.append(metrics)
        
        # ---- 日志和保存 ----
        if step % config.log_every == 0:
            avg_loss = np.mean([m['loss'] for m in metrics_history[-100:]])
            print(f"Step {step}: loss={avg_loss:.6f}")
        
        if step % config.save_every == 0 and step > 0:
            save_checkpoint(student_state, step, config.output_dir)
            print(f"Checkpoint saved at step {step}")
    
    # 保存最终模型
    save_checkpoint(student_state, config.total_steps, config.output_dir, is_final=True)
    print("训练完成！")
    
    return student_state


if __name__ == "__main__":
    import yaml
    
    # 加载配置
    with open("configs/distill_pi05_libero.yaml") as f:
        config = yaml.safe_load(f)
    
    # 启动训练
    final_state = training_loop(config)
```

---

## 六、配置文件 (distill_pi05_libero.yaml)

```yaml
# configs/distill_pi05_libero.yaml

# 模型配置
model:
  teacher_checkpoint: "gs://openpi-assets/checkpoints/pi05_base"
  pi05: true
  action_dim: 18
  action_horizon: 50
  max_token_len: 512

# 训练配置
training:
  total_steps: 50000           # 50K步
  batch_size: 2                # 单卡80GB安全值
  learning_rate: 1.0e-4        # 蒸馏用较小LR
  warmup_steps: 1000
  weight_decay: 0.01
  
  # 损失配置
  loss:
    use_multi_step: true       # 多步蒸馏
    multi_step_weight: 0.3
    use_cosine_similarity: true
    cosine_weight: 0.1
    
  # 数据配置
  data:
    dataset: "libero"
    data_path: "./data/libero"
    num_workers: 4
    
  # 优化器
  optimizer:
    type: "adamw"
    beta1: 0.9
    beta2: 0.98
    
  # 保存和日志
  log_every: 100
  save_every: 5000
  eval_every: 2000
  
  # 硬件
  devices: [0, 1]              # 使用GPU-0和GPU-1
  mixed_precision: "bf16"      # BF16混合精度

# 输出
output:
  output_dir: "./experiments/distill/pi05_libero"
  wandb_project: "one-step-pi05"
  wandb_run_name: "distill_baseline"
```

---

## 七、评估脚本设计

```python
"""
eval_one_step.py - 评估单步π0.5的性能

评估指标:
1. 推理速度 (ms/step)
2. 成功率 (task success rate)
3. 动作质量 (与教师模型的MSE)
4. 内存占用
"""

import time
import numpy as np


def evaluate_speed(model, dataloader, num_steps=100):
    """评估推理速度。"""
    times = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break
        
        obs = batch['observation']
        noise = np.random.randn(1, 50, 18).astype(np.float32)
        
        start = time.time()
        actions = model(obs, noise)  # 单步
        elapsed = time.time() - start
        
        times.append(elapsed * 1000)  # ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'p50_ms': np.percentile(times, 50),
        'p99_ms': np.percentile(times, 99),
    }


def evaluate_success_rate(model, env, num_episodes=50):
    """在仿真环境中评估任务成功率。"""
    successes = 0
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        
        while not done:
            # 单步生成动作
            noise = np.random.randn(1, 50, 18).astype(np.float32)
            actions = model(obs, noise)
            
            # 执行动作
            obs, reward, done, info = env.step(actions)
        
        if info.get('success', False):
            successes += 1
    
    return {
        'success_rate': successes / num_episodes,
        'num_episodes': num_episodes,
    }


def evaluate_action_quality(student_model, teacher_model, dataloader):
    """评估学生输出与教师输出的差距。"""
    mses = []
    
    for batch in dataloader:
        obs = batch['observation']
        noise = np.random.randn(*batch['actions'].shape).astype(np.float32)
        
        # 学生单步
        student_actions = student_model(obs, noise)
        
        # 教师10步
        teacher_actions = teacher_model.sample_actions(obs, noise=noise, num_steps=10)
        
        mse = np.mean((student_actions - teacher_actions) ** 2)
        mses.append(mse)
    
    return {
        'mean_mse': np.mean(mses),
        'max_mse': np.max(mses),
    }


def compare_with_baseline(student_model, teacher_model, env):
    """
    完整对比实验:
    - 教师10步 (baseline)
    - 学生1步 (ours)
    - 学生2步 (可选, 作为中间点)
    """
    results = {}
    
    # 教师10步
    print("评估教师 10-step...")
    results['teacher_10step'] = evaluate_speed_and_success(
        teacher_model, env, num_steps=10
    )
    
    # 学生1步
    print("评估学生 1-step...")
    results['student_1step'] = evaluate_speed_and_success(
        student_model, env, num_steps=1
    )
    
    # 打印对比
    print("\n" + "="*60)
    print("对比结果:")
    print(f"教师10步: {results['teacher_10step']['speed']['mean_ms']:.1f}ms, "
          f"成功率: {results['teacher_10step']['success']:.2%}")
    print(f"学生1步:  {results['student_1step']['speed']['mean_ms']:.1f}ms, "
          f"成功率: {results['student_1step']['success']:.2%}")
    
    speedup = results['teacher_10step']['speed']['mean_ms'] / results['student_1step']['speed']['mean_ms']
    print(f"加速比: {speedup:.1f}x")
    
    return results
```

---

## 八、实验设计（8周时间表）

| 周 | 任务 | 产出 |
|----|------|------|
| **Week 1** | ① 跑通π0.5推理baseline，记录10步的latency和成功率<br>② 搭建蒸馏代码框架（学生模型+损失函数） | 基线数据 + 可运行代码 |
| **Week 2** | ① 实现双卡并行训练循环<br>② 在LIBERO小数据集上debug | 训练能跑通 |
| **Week 3-4** | ① 完整训练50K步<br>② 监控loss曲线，调超参 | 训练好的模型 + loss曲线 |
| **Week 5** | ① 消融实验：不同损失函数对比（MSE vs 多步蒸馏 vs 余弦相似度）<br>② 学生结构消融（MLP-only vs MLP+Attention） | 消融实验数据 |
| **Week 6** | ① 完整评估：速度、成功率、动作质量<br>② 对比不同步数（1/2/5/10步） | 完整评估报告 |
| **Week 7** | ① 写论文（Method + Experiments）<br>② 整理代码开源 | 论文初稿 + 开源代码 |
| **Week 8** | ① 润色论文<br>② 补充可视化（推理速度对比图、轨迹对比图） | 投稿版本 |

---

## 九、论文故事线

### 标题建议
**"One Step is Enough: Real-Time π0.5 Policy via Consistency Distillation for Robot Manipulation"**

### Abstract 框架
1. **问题**: π0.5 等 Flow Matching 策略推理需要 10 步 ODE 积分，速度慢，无法实时闭环控制
2. **方法**: 提出一致性蒸馏，将 π0.5 蒸馏为单步模型，保持多模态理解能力
3. **创新**: 多步蒸馏损失 + 特征对齐，保留教师模型的分布特性
4. **结果**: LIBERO 基准上推理加速 10x，成功率保持在教师 95% 以上
5. **意义**: 首次对 π0.5 进行蒸馏加速，使实时 VLA 控制成为可能

### 核心图表
1. 教师10步 vs 学生1步的推理速度对比柱状图
2. 不同任务的成功率对比（保持率）
3. 消融实验：不同损失函数的效果
4. 蒸馏过程中的 loss 曲线
5. 动作轨迹可视化（教师 vs 学生 vs 真实动作）

---

## 十、关键风险与应对

| 风险 | 概率 | 应对策略 |
|------|------|---------|
| 学生模型精度下降太多 | 中 | ① 降低学习率 ② 增加多步蒸馏权重 ③ 部分微调prefix |
| 双卡通信瓶颈 | 低 | 教师输出很小(B,50,18)，通信量可忽略 |
| 显存溢出 | 低 | batch_size=2, 使用BF16, 梯度检查点 |
| LIBERO数据集不够 | 中 | 加入DROID数据, 数据增强 |
| 训练不收敛 | 低 | 学习率warmup, 梯度裁剪, 监控梯度范数 |

---

这份大纲可以直接交给 Claude 执行。需要我补充任何具体部分的代码细节吗？