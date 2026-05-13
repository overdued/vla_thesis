#!/usr/bin/env bash
# =============================================================================
# openpi LoRA 微调启动脚本（自有 pi05_libero10_lora 实战）
#
# -----------------------------------------------------------------------------
# 任务说明（What & Why）
# -----------------------------------------------------------------------------
# • 项目：       自有 openpi（/home/zhaoyuhang/openpi）
# • 任务：       pi05_libero10_lora —— LIBERO-10 长程操作任务套件
# • 数据集：     physical-intelligence/libero（开源，10 个长程操作任务）
#                由 LeRobot v2 格式封装，自动从 HuggingFace 拉取
# • 数据形态：   单臂 + 单 gripper，state 8 维，action 7 维（已 padded 至模型 dim）
# • 模型架构：   pi0.5（pi05=True, action_horizon=10, discrete_state_input=False）
# • 微调方式：   ★ LoRA 参数高效微调 ★
#                - Gemma 2B 主 LLM 上注入 LoRA（rank=16, alpha=16）
#                - Gemma 300M Action Expert 上注入 LoRA（rank=32, alpha=32）
#                - 只训练 ~10-50M 个 LoRA 矩阵参数（base 全部冻结）
# • 期望显存：   ~35 GB / 卡（实测 batch_size=32 + 2× A800）
# • 预期产出：   checkpoints/pi05_libero10_lora/<EXP_NAME>/{2000..10000}/
# • 部署方式：   uv run scripts/serve_policy.py --env=LIBERO
#
# -----------------------------------------------------------------------------
# 与 openpi 默认 TrainConfig + 公司全量微调 的差异（重点）
# -----------------------------------------------------------------------------
#   相对 openpi 默认（src/openpi/training/config.py:466-538）：
#     [LoRA 4 开关] paligemma_variant   "gemma_2b"   → "gemma_2b_lora"    ★必改
#     [LoRA 4 开关] action_expert_variant "gemma_300m" → "gemma_300m_lora" ★必改
#     [LoRA 4 开关] freeze_filter        Nothing      → .get_freeze_filter() ★必改
#     [LoRA 4 开关] ema_decay            0.99         → None                ★必改
#                   peak_lr              2.5e-5       → 5e-5                推荐
#                   warmup_steps         10_000       → 1_000               推荐
#                   num_train_steps      30_000       → 10_000              推荐
#                   batch_size           32 (默认)    → 32 (.sh 可覆盖)
#                   pi05                 False        → True                pi0.5 启用
#                   action_horizon       50 (默认)    → 10                  Libero 任务长度
#
#   相对公司 cyborg 全量（pi05_cyborgv2_pick_up_tray_v2_0313）：
#     公司全量没有上面 4 个 LoRA 开关；其余 lr/warmup/steps 也跟 openpi 默认
#     公司用 28k 步，我们 LoRA 只要 10k 步
#     公司单卡 ~70 GB，我们 LoRA 单卡 ~35 GB
#
# -----------------------------------------------------------------------------
# 参数来源分三层
# -----------------------------------------------------------------------------
#   A) src/openpi/training/config.py 中的 TrainConfig(name="pi05_libero10_lora")
#      —— LoRA 4 开关 + lr_schedule + num_train_steps 等都写死在那
#      （已经写好，无需再改；如需新建配置见底部"如何派生新 LoRA 配置"）
#   B) 本脚本顶部变量（GPUs / batch / 路径等）—— 改这里
#   C) train.py 命令行 flag（--batch_size / --exp-name / --overwrite）
#
# 整理者：Yuhang Zhao  日期：2026-05-13
# 参考文档：LORA_FINETUNING_GUIDE.md, FINETUNING_COMPARISON.md
# =============================================================================

set -euo pipefail

# =============================================================================
# 1. 可调参数（顶部变量配置区） —— 改这里
# =============================================================================

# --- 项目根 ---
PROJECT_ROOT="${PROJECT_ROOT:-/home/zhaoyuhang/openpi}"

# --- 训练配置名（必须在 src/openpi/training/config.py 中已注册） ---
# 仓库已有可选 LoRA 配置：
#   pi05_libero10_lora              ← 本脚本默认（pi0.5 + Libero-10）
#   pi0_libero_low_mem_finetune     —— pi0 + LoRA + Libero（模板）
#   pi0_fast_libero_low_mem_finetune —— pi0-FAST + LoRA + Libero（模板）
# 若要新增自己的 LoRA 配置，请按本脚本底部的"如何派生新 LoRA 配置"小节操作
CONFIG_NAME="${CONFIG_NAME:-pi05_libero10_lora}"

# --- 实验名（决定 checkpoints/<CONFIG>/<EXP>/） ---
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_$(whoami)_$(date +%Y%m%d_%H%M)}"

# --- 计算资源 ---
GPUS="${GPUS:-0,1}"                                # LoRA 2 卡足够
BATCH_SIZE="${BATCH_SIZE:-32}"                     # LoRA 实测 2 卡 batch=32 占 ~35 GB
FSDP_DEVICES="${FSDP_DEVICES:-1}"                  # LoRA 通常用不上 FSDP

# --- 显存 ---
XLA_MEM_FRACTION="${XLA_MEM_FRACTION:-0.95}"

# --- 数据集 ---
# physical-intelligence/libero 会从 HuggingFace 自动下载，无需手动放置
# 但要确保 HF_LEROBOT_HOME 已设置（首次下载缓存于此）
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/data/dataset}"

# --- Base ckpt 缓存 ---
# pi05_base 约 7 GB，首次自动从 gs://openpi-assets/checkpoints/pi05_base 下载
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/openpi_base_ckpt}"

# --- 运行行为 ---
OVERWRITE="${OVERWRITE:-1}"                        # 1=加 --overwrite
RESUME="${RESUME:-0}"                              # 1=加 --resume
SKIP_NORM_STATS="${SKIP_NORM_STATS:-0}"            # 1=跳过 norm_stats 计算
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"        # 1=nohup 后台
WANDB_ENABLED="${WANDB_ENABLED:-0}"                # 1=开启 WandB

# =============================================================================
# 2. 配置参数全清单（pi05_libero10_lora 实际生效的所有超参）
# =============================================================================
# 标 ★ = 显式设置且不同于 openpi 默认；标 ★L = LoRA 4 开关；标 (默认) = 用默认值
# 出处：[config.py] = src/openpi/training/config.py:770-803
#       [cmdline]   = uv run scripts/train.py 命令行
#       [shell env] = 本脚本通过环境变量传入
#
# +-----------------------------+-------------------------+--------------------+
# | 参数                        | 取值                    | 出处               |
# +-----------------------------+-------------------------+--------------------+
# | 【顶层 TrainConfig】         |                         |                    |
# | name                        | pi05_libero10_lora      | [config.py] ★      |
# | project_name                | openpi                  | (默认)              |
# | exp_name                    | <由 .sh 决定>           | [cmdline] --exp-name|
# | seed                        | 42                      | (默认)              |
# | batch_size                  | 32                      | [config.py] / cmdline ★|
# | num_workers                 | 2 (默认)                 | (默认)              |
# | num_train_steps             | 10_000                  | [config.py] ★      |
# | log_interval                | 100                     | (默认)              |
# | save_interval               | 1000                    | (默认)              |
# | keep_period                 | 5000                    | (默认)              |
# | overwrite / resume          | 见 .sh                  | [cmdline]          |
# | wandb_enabled               | false                   | [config.py] ★      |
# | fsdp_devices                | 1                       | [config.py]        |
# | checkpoint_base_dir         | ./checkpoints           | (默认)              |
# | assets_base_dir             | ./assets                | (默认)              |
# |                             |                         |                    |
# | 【ModelConfig: Pi0Config】   |                         |                    |
# | pi05                        | True                    | [config.py] ★      |
# | action_horizon              | 10                      | [config.py] ★      |
# | action_dim                  | 32 (pi0.5 默认)          | (默认)              |
# | discrete_state_input        | False                   | [config.py] ★      |
# | paligemma_variant           | "gemma_2b_lora"         | [config.py] ★L     |
# | action_expert_variant       | "gemma_300m_lora"       | [config.py] ★L     |
# |                             |                         |                    |
# | 【LoRA 内部】(gemma.py:88-108) |                       |                    |
# |   2B  attn  rank/alpha      | 16 / 16.0               | [gemma.py]         |
# |   2B  ffn   rank/alpha      | 16 / 16.0               | [gemma.py]         |
# |   300M attn rank/alpha      | 32 / 32.0               | [gemma.py]         |
# |   300M ffn  rank/alpha      | 32 / 32.0               | [gemma.py]         |
# |   rslora                    | False                   | [gemma.py]         |
# |   init_fn                   | normal(stddev=0.01)     | [gemma.py]         |
# |                             |                         |                    |
# | 【优化器 / 学习率】           |                         |                    |
# | optimizer                   | AdamW(clip=1.0)         | [config.py]        |
# | lr_schedule                 | CosineDecaySchedule     | [config.py]        |
# |   warmup_steps              | 1_000                   | [config.py] ★      |
# |   peak_lr                   | 5e-5                    | [config.py] ★      |
# |   decay_steps               | 100_000                 | [config.py] ★      |
# |   decay_lr                  | 5e-5                    | [config.py] ★      |
# | ema_decay                   | None                    | [config.py] ★L     |
# | freeze_filter               | get_freeze_filter()     | [config.py] ★L     |
# |   实际效果                  | 训 .*lora.*，冻 .*llm.* |                    |
# |                             |                         |                    |
# | 【数据 DataConfig】          |                         |                    |
# | data class                  | LeRobotLiberoDataConfig | [config.py] ★      |
# | repo_id                     | physical-intelligence/libero | [config.py] ★ |
# | extra_delta_transform       | False                   | [config.py] ★      |
# | prompt_from_task            | True                    | [config.py] ★      |
# |                             |                         |                    |
# | 【权重加载】                 |                         |                    |
# | weight_loader               | CheckpointWeightLoader  | [config.py] ★      |
# |   checkpoint_path           | gs://openpi-assets/checkpoints/pi05_base/params | ★|
# |   missing_regex             | .*lora.*                | (LoRA 兜底)        |
# |                             |                         |                    |
# | 【环境变量】                 |                         |                    |
# | OPENPI_DATA_HOME            | ~/openpi_base_ckpt      | [shell env]        |
# | HF_LEROBOT_HOME             | ~/data/dataset          | [shell env]        |
# | XLA_PYTHON_CLIENT_MEM_FRACTION | 0.95                 | [shell env]        |
# | CUDA_VISIBLE_DEVICES        | 0,1                     | [shell env]        |
# +-----------------------------+-------------------------+--------------------+
#
# 与 openpi 默认 TrainConfig 相比，本配置 ★ 改动汇总：
#   [LoRA 必改 4 项 ★L]
#     1) model.paligemma_variant      = "gemma_2b_lora"
#     2) model.action_expert_variant  = "gemma_300m_lora"
#     3) freeze_filter                = ....get_freeze_filter()
#     4) ema_decay                    = None
#   [其余 9 项 ★]
#     5) name / exp_name              —— 命名
#     6) model.pi05                   = True
#     7) model.action_horizon         = 10
#     8) model.discrete_state_input   = False
#     9) data                         = LeRobotLiberoDataConfig(libero)
#    10) weight_loader                = pi05_base/params
#    11) lr_schedule warmup           = 1_000  (默认 10_000)
#    12) lr_schedule peak_lr          = 5e-5   (默认 2.5e-5)
#    13) num_train_steps              = 10_000 (默认 30_000)
#    14) wandb_enabled                = False  (默认 True)
# =============================================================================

# =============================================================================
# 3. 颜色 / 日志辅助
# =============================================================================
if [[ -t 1 ]]; then
    CL_RED='\033[0;31m'; CL_GREEN='\033[0;32m'
    CL_YELLOW='\033[0;33m'; CL_BLUE='\033[0;34m'; CL_RESET='\033[0m'
else
    CL_RED=''; CL_GREEN=''; CL_YELLOW=''; CL_BLUE=''; CL_RESET=''
fi
log_info()  { echo -e "${CL_BLUE}[INFO]${CL_RESET}  $*"; }
log_ok()    { echo -e "${CL_GREEN}[ OK ]${CL_RESET}  $*"; }
log_warn()  { echo -e "${CL_YELLOW}[WARN]${CL_RESET}  $*"; }
log_error() { echo -e "${CL_RED}[ERR ]${CL_RESET}  $*" 1>&2; }
log_step()  { echo -e "\n${CL_BLUE}========== $* ==========${CL_RESET}"; }

# =============================================================================
# 4. 打印本次运行配置
# =============================================================================
log_step "本次运行配置（[.sh] 可调）"
cat <<EOF
  PROJECT_ROOT       : ${PROJECT_ROOT}
  CONFIG_NAME        : ${CONFIG_NAME}
  EXP_NAME           : ${EXP_NAME}
  GPUS               : ${GPUS}
  BATCH_SIZE         : ${BATCH_SIZE}
  FSDP_DEVICES       : ${FSDP_DEVICES}
  XLA_MEM_FRACTION   : ${XLA_MEM_FRACTION}
  OPENPI_DATA_HOME   : ${OPENPI_DATA_HOME}
  HF_LEROBOT_HOME    : ${HF_LEROBOT_HOME}
  OVERWRITE          : ${OVERWRITE}
  RESUME             : ${RESUME}
  SKIP_NORM_STATS    : ${SKIP_NORM_STATS}
  RUN_IN_BACKGROUND  : ${RUN_IN_BACKGROUND}
  WANDB_ENABLED      : ${WANDB_ENABLED}
EOF

if [[ "${OVERWRITE}" == "1" && "${RESUME}" == "1" ]]; then
    log_error "OVERWRITE 与 RESUME 不能同时为 1"; exit 1
fi

# =============================================================================
# 5. 环境检查
# =============================================================================
log_step "环境检查"

[[ -d "${PROJECT_ROOT}" ]] || { log_error "项目目录不存在：${PROJECT_ROOT}"; exit 1; }
cd "${PROJECT_ROOT}"; log_ok "项目目录：${PROJECT_ROOT}"

command -v uv >/dev/null 2>&1 \
    || { log_error "未找到 uv：curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
log_ok "uv：$(uv --version)"

if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
    log_warn ".venv 不存在，准备 uv sync ..."
    GIT_LFS_SKIP_SMUDGE=1 uv sync --index-url https://pypi.org/simple
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.venv/bin/activate"
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
fi
log_ok ".venv 已就绪"

mkdir -p "${HF_LEROBOT_HOME}" "${OPENPI_DATA_HOME}"
log_ok "数据目录：${HF_LEROBOT_HOME}"
log_ok "base ckpt 缓存：${OPENPI_DATA_HOME}"

if command -v nvidia-smi >/dev/null 2>&1; then
    log_info "可见 GPU："
    CUDA_VISIBLE_DEVICES="${GPUS}" nvidia-smi \
        --query-gpu=index,name,memory.free --format=csv,noheader 2>&1 \
        | sed 's/^/         /' || true
else
    log_warn "未找到 nvidia-smi"
fi

if ! grep -q "name=\"${CONFIG_NAME}\"" src/openpi/training/config.py; then
    log_error "未在 config.py 中找到 name=\"${CONFIG_NAME}\""
    log_error "可用配置：grep -n 'name=\"' src/openpi/training/config.py"
    exit 1
fi
log_ok "找到配置 ${CONFIG_NAME}"

# LoRA 健康度检查：确认 4 开关都被设置（启发式 grep）
log_info "校验 LoRA 4 开关..."
{
    grep -A 40 "name=\"${CONFIG_NAME}\"" src/openpi/training/config.py \
        | grep -q "_lora"            && echo "  [OK] paligemma/action_expert _lora variant" \
        || echo "  [!!] 似乎未启用 _lora variant，请确认 ${CONFIG_NAME} 真是 LoRA 配置"

    grep -A 40 "name=\"${CONFIG_NAME}\"" src/openpi/training/config.py \
        | grep -q "get_freeze_filter" && echo "  [OK] freeze_filter 已设置" \
        || echo "  [!!] freeze_filter 似乎未设置"

    grep -A 40 "name=\"${CONFIG_NAME}\"" src/openpi/training/config.py \
        | grep -q "ema_decay=None"    && echo "  [OK] ema_decay=None" \
        || echo "  [!!] ema_decay 似乎不是 None"
} | sed 's/^/         /'

# =============================================================================
# 6. 准备日志
# =============================================================================
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
log_info "日志：${LOG_FILE}"

# =============================================================================
# 7. Step 1/2: 计算 norm_stats
# =============================================================================
if [[ "${SKIP_NORM_STATS}" != "1" ]]; then
    log_step "Step 1/2: 计算 norm_stats"
    HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" \
        uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
    log_ok "norm_stats 完成"
else
    log_warn "跳过 norm_stats（SKIP_NORM_STATS=1）"
fi

# =============================================================================
# 8. Step 2/2: 启动训练（LoRA）
# =============================================================================
log_step "Step 2/2: 启动训练（LoRA）"

# 拼接 train.py flag
TRAIN_FLAGS=("--exp-name=${EXP_NAME}" "--batch_size" "${BATCH_SIZE}")
[[ "${OVERWRITE}"     == "1" ]] && TRAIN_FLAGS+=("--overwrite")
[[ "${RESUME}"        == "1" ]] && TRAIN_FLAGS+=("--resume")
[[ "${WANDB_ENABLED}" == "1" ]] && TRAIN_FLAGS+=("--wandb_enabled") \
                               || TRAIN_FLAGS+=("--no-wandb_enabled")
[[ "${FSDP_DEVICES}"  != "1" ]] && TRAIN_FLAGS+=("--fsdp_devices" "${FSDP_DEVICES}")

TRAIN_CMD=(
    env
    CUDA_VISIBLE_DEVICES="${GPUS}"
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION}"
    OPENPI_DATA_HOME="${OPENPI_DATA_HOME}"
    HF_LEROBOT_HOME="${HF_LEROBOT_HOME}"
    uv run scripts/train.py "${CONFIG_NAME}"
    "${TRAIN_FLAGS[@]}"
)

log_info "执行命令："
printf '  %s\n' "${TRAIN_CMD[@]}"
echo

if [[ "${RUN_IN_BACKGROUND}" == "1" ]]; then
    nohup "${TRAIN_CMD[@]}" >"${LOG_FILE}" 2>&1 &
    TRAIN_PID=$!
    echo "${TRAIN_PID}" >"${LOG_DIR}/${EXP_NAME}.pid"
    log_ok "后台 PID：${TRAIN_PID}（写到 ${LOG_DIR}/${EXP_NAME}.pid）"
    log_info "查看：tail -f ${LOG_FILE}"
    log_info "停止：kill ${TRAIN_PID}"
else
    "${TRAIN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
fi

log_step "完成"
log_ok "Checkpoint 目录：${PROJECT_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
log_info "下一步部署："
log_info "  cd ${PROJECT_ROOT} && uv run scripts/serve_policy.py --env=LIBERO \\"
log_info "      policy:checkpoint --policy.config=${CONFIG_NAME} \\"
log_info "      --policy.dir=${PROJECT_ROOT}/checkpoints/${CONFIG_NAME}/${EXP_NAME}/10000"

# =============================================================================
# 附录：如何派生新 LoRA 配置（基于本模板）
# -----------------------------------------------------------------------------
# 若你想训自己的机器人数据（不是 Libero），步骤：
#   1) 在 src/openpi/policies/ 新建 my_robot_policy.py（参考 libero_policy.py）
#   2) 在 src/openpi/training/config.py 新增 MyRobotDataConfig（参考 LeRobotLiberoDataConfig）
#   3) 在 config.py 的 _CONFIGS 列表中追加：
#      TrainConfig(
#          name="pi05_my_robot_lora",
#          model=pi0_config.Pi0Config(
#              pi05=True, action_horizon=30, discrete_state_input=False,
#              paligemma_variant="gemma_2b_lora",          # ★L
#              action_expert_variant="gemma_300m_lora",    # ★L
#          ),
#          data=MyRobotDataConfig(
#              repo_id="my_robot/<your_task>",
#              base_config=DataConfig(prompt_from_task=True),
#          ),
#          weight_loader=weight_loaders.CheckpointWeightLoader(
#              "gs://openpi-assets/checkpoints/pi05_base/params"
#          ),
#          freeze_filter=pi0_config.Pi0Config(
#              pi05=True, action_horizon=30, discrete_state_input=False,
#              paligemma_variant="gemma_2b_lora",
#              action_expert_variant="gemma_300m_lora",
#          ).get_freeze_filter(),                          # ★L
#          ema_decay=None,                                 # ★L
#          lr_schedule=_optimizer.CosineDecaySchedule(
#              warmup_steps=1_000, peak_lr=5e-5,
#              decay_steps=30_000, decay_lr=5e-5,
#          ),
#          batch_size=32,
#          num_train_steps=10_000,
#          save_interval=2_000,
#          num_workers=16,
#          wandb_enabled=False,
#      ),
#   4) 用环境变量 CONFIG_NAME=pi05_my_robot_lora 调用本脚本即可
# =============================================================================
