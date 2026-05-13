# Week 1: One-Step π0.5 蒸馏训练 —— 代码框架搭建报告

**日期**: 2026-05-13
**目标**: 搭建蒸馏代码框架，跑通 π0.5 baseline 推理

---

## 一、已完成工作

### 1. 修改核心模型 (`src/openpi/models/pi0.py`)

在 `Pi0` 类中新增两个方法：

- **`compute_distillation_loss(rng, observation, teacher_actions)`**
  - 学生接收纯噪声 (t=1) 和观察
  - 单步前向传播预测动作
  - 计算与教师 10 步 ODE 结果的 MSE 损失
  - 复用 `embed_prefix`, `embed_suffix`, `PaliGemma.llm`, `action_out_proj`

- **`single_step_predict(rng, observation)`**
  - 单步推理入口：从纯噪声直接预测干净动作
  - 跳过 10 步 ODE 积分
  - 用于推理时加速

### 2. 蒸馏冻结策略 (`src/openpi/models/pi0_config.py`)

新增 `get_distill_freeze_filter()` 方法：
- **冻结**: 所有视觉编码器 (img) 和语言模型 (llm) 参数
- **训练**: 只有 `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out`
- 可训练参数量约 0.5% 的 3B 模型

### 3. 训练配置 (`src/openpi/training/distillation_config.py`)

定义 `DistillConfig` 继承自 `TrainConfig`：
- `teacher_checkpoint`: 教师模型路径
- `loss_type`: MSE / weighted_mse / consistency
- `use_multi_step`: 是否使用多步蒸馏损失
- 默认超参：50K 步, batch=16, lr=1e-4, warmup=1K, AdamW + cosine decay

### 4. 蒸馏训练器 (`src/openpi/training/distillation_trainer.py`)

核心函数：
- **`init_teacher_model()`**: 加载并冻结教师模型
- **`distill_train_step()`**: 单步蒸馏训练
  1. 教师 `sample_actions(num_steps=10)` 生成目标
  2. 学生 `compute_distillation_loss()` 计算损失
  3. `nnx.value_and_grad` 只计算可训练参数梯度
  4. 优化器更新
- **`init_student_state()`**: 初始化学生（从教师权重加载）
- **`run_distillation()`**: 主训练循环（复用现有 checkpoint 和 wandb 机制）

### 5. 入口脚本 (`scripts/distill_pi05.py`)

```bash
uv run scripts/distill_pi05.py pi05_libero_distill --exp-name=distill_run1
```

### 6. 评估脚本 (`scripts/eval_one_step.py`)

支持三种模式：
- `teacher`: 评估 10 步 ODE baseline
- `student`: 评估单步模型
- `compare`: 对比两者速度和成功率

### 7. 配置注册 (`src/openpi/training/config.py`)

在 `_CONFIGS` 列表中注册 `pi05_libero_distill`：
- π0.5 架构 (gemma_2b + gemma_300m)
- LIBERO-10 数据集
- 冻结 filter 使用 `get_distill_freeze_filter()`
- FSDP 分片到 2 个 GPU
- 从 π0.5 base checkpoint 初始化

---

## 二、关键设计决策

### 为什么不创建新的模型类？
复用 `Pi0` 更简单，只需添加方法。冻结策略通过 NNX filter 实现，无需修改模型结构。

### 为什么只训练 action 投影层？
- π0.5 的"智能"在 LLM 和视觉编码器中（理解场景和语言）
- Action 投影层是 Flow Matching 特有的，负责条件生成
- 单步蒸馏本质是学习不同的条件生成映射
- 参数量极小 (~50M)，训练效率高

### 为什么不用双设备分教师/学生？
- JAX 跨设备通信复杂，增加延迟
- 更简洁的方式是在同一 mesh 上让 JAX 自动分片
- 2×A800 (80GB each) 足够容纳两个 bf16 模型

---

## 三、文件变更汇总

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/openpi/models/pi0.py` | 修改 | 添加 `compute_distillation_loss` + `single_step_predict` |
| `src/openpi/models/pi0_config.py` | 修改 | 添加 `get_distill_freeze_filter()` |
| `src/openpi/training/config.py` | 修改 | 注册 `pi05_libero_distill` 配置 |
| `src/openpi/training/distillation_config.py` | 新增 | 蒸馏专用配置类 |
| `src/openpi/training/distillation_trainer.py` | 新增 | 蒸馏训练器核心逻辑 |
| `scripts/distill_pi05.py` | 新增 | 蒸馏训练入口 |
| `scripts/eval_one_step.py` | 新增 | 评估脚本 |

---

## 四、编译测试状态

**测试命令**:
```bash
uv run python scripts/distill_pi05.py pi05_libero_distill \
    --exp-name=compile_test --overwrite --num_train_steps=5
```

**状态**: 运行中（JAX 首次编译 3B 模型，预计需要 10-20 分钟）
- GPU 0 已分配 ~20GB 显存（模型加载中）
- 数据加载器初始化成功
- Norm stats 已复制到 `assets/pi05_libero_distill/`

**预期结果**: 5 步训练成功完成，无报错

---

## 五、Week 2 计划

1. **编译测试**: 运行 10 步蒸馏训练，验证框架无报错
2. **小数据训练**: 在 LIBERO-10 上训练 1K 步，验证损失下降
3. **Baseline 评估**: 运行 π0.5 教师模型的推理速度和成功率测试
4. **超参调优**: 调整学习率、batch size 等
