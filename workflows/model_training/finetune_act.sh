#!/usr/bin/env bash
# finetune_act.sh — 从已有 checkpoint 继续训练新数据集
#
# 一键执行:   ./finetune_act.sh
# 自定义参数: PRETRAINED_CKPT=<path> NEW_DATASET=<path> ./finetune_act.sh
#
# 默认参数在脚本内配置，修改 DEFAULT_* 变量即可

set -euo pipefail

# ============== 配置 ==============
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# 默认参数 —— 按需修改这里即可
DEFAULT_PRETRAINED_CKPT="/home/zzx23457/lerobot_vlahost/outputs/train/act_26-07-21+22+23+25-merged_v2_20260725_024554/checkpoints/400000/pretrained_model"
DEFAULT_NEW_DATASET="datasets/26-07-23+25-merged_v2"  # 改成你的新数据集路径

# 使用环境变量或默认值
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$DEFAULT_PRETRAINED_CKPT}"
NEW_DATASET="${NEW_DATASET:-$DEFAULT_NEW_DATASET}"

# 转换为绝对路径
if [[ ! "$PRETRAINED_CKPT" = /* ]]; then
    PRETRAINED_CKPT="$PROJECT_ROOT/$PRETRAINED_CKPT"
fi
if [[ ! "$NEW_DATASET" = /* ]]; then
    NEW_DATASET="$PROJECT_ROOT/$NEW_DATASET"
fi

# 验证路径
if [[ ! -d "$PRETRAINED_CKPT" ]]; then
    echo "错误: checkpoint 目录不存在: $PRETRAINED_CKPT"
    exit 1
fi

if [[ ! -d "$NEW_DATASET" ]]; then
    echo "错误: 数据集目录不存在: $NEW_DATASET"
    exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/train}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# 训练参数
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-400000}"
EVAL_FREQ="${EVAL_FREQ:-20000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-50}"

WANDB_PROJECT="${WANDB_PROJECT:-lerobot_act_finetune}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*" >&2; }

# 从数据集路径提取标签
dataset_tag() {
    local tag
    tag="$(basename "$NEW_DATASET" | sed -nE 's/.*_(v[0-9]+)$/\1/p')"
    [ -z "$tag" ] && tag="$(basename "$NEW_DATASET")"
    printf '%s' "$tag"
}

# ============== 信息展示 ==============
log "Fine-tune 配置:"
echo "  Pretrained checkpoint: $PRETRAINED_CKPT"
echo "  New dataset:          $NEW_DATASET"
echo "  Batch size:           $BATCH_SIZE"
echo "  Training steps:       $STEPS"
echo "  WandB project:        $WANDB_PROJECT"
echo
read -p "配置正确? 按回车继续，Ctrl+C 取消: " -r
echo

# ============== 快速验证 ==============
log "验证 checkpoint 完整性..."
if [[ ! -f "$PRETRAINED_CKPT/config.json" ]]; then
    err "checkpoint 缺少 config.json"
    exit 1
fi

if [[ ! -f "$PRETRAINED_CKPT/model.safetensors" ]] && [[ ! -f "$PRETRAINED_CKPT/pytorch_model.bin" ]]; then
    err "checkpoint 缺少模型权重文件 (model.safetensors 或 pytorch_model.bin)"
    exit 1
fi
ok "Checkpoint 文件完整"

log "验证新数据集..."
if [[ ! -f "$NEW_DATASET/meta/info.json" ]]; then
    err "数据集缺少 meta/info.json"
    exit 1
fi

# 检查数据集特征维度是否匹配
python3 - <<PY
import json, sys

# 读取 checkpoint 配置
ckpt_config = json.load(open("$PRETRAINED_CKPT/config.json"))
input_shapes = ckpt_config.get("input_shapes", {})
output_shapes = ckpt_config.get("output_shapes", {})

# 读取新数据集 info
dataset_info = json.load(open("$NEW_DATASET/meta/info.json"))
features = dataset_info["features"]

# 检查 state 维度
if "observation.state" in input_shapes:
    ckpt_state_dim = input_shapes["observation.state"][-1]
    dataset_state_shape = features["observation.state"]["shape"]
    if isinstance(dataset_state_shape, list):
        dataset_state_dim = dataset_state_shape[0]
    else:
        dataset_state_dim = dataset_state_shape

    if ckpt_state_dim != dataset_state_dim:
        print(f"✗ State 维度不匹配: checkpoint={ckpt_state_dim}, dataset={dataset_state_dim}", file=sys.stderr)
        sys.exit(1)
    print(f"  ✓ State 维度匹配: {ckpt_state_dim}")

# 检查 action 维度
if "action" in output_shapes:
    ckpt_action_dim = output_shapes["action"][-1]
    dataset_action_shape = features["action"]["shape"]
    if isinstance(dataset_action_shape, list):
        dataset_action_dim = dataset_action_shape[0]
    else:
        dataset_action_dim = dataset_action_shape

    if ckpt_action_dim != dataset_action_dim:
        print(f"✗ Action 维度不匹配: checkpoint={ckpt_action_dim}, dataset={dataset_action_dim}", file=sys.stderr)
        sys.exit(1)
    print(f"  ✓ Action 维度匹配: {ckpt_action_dim}")

print(f"  ✓ 版本: {dataset_info['codebase_version']}")
print(f"  ✓ Episodes: {dataset_info['total_episodes']}")
print(f"  ✓ Frames: {dataset_info['total_frames']}")
PY

if [[ $? -ne 0 ]]; then
    err "数据集验证失败,请确保新数据集的特征维度与 checkpoint 一致"
    exit 1
fi
ok "新数据集验证通过"

# ============== 开始训练 ==============
log "准备 fine-tune 训练..."

TAG="$(dataset_tag)"
RUN_DIR="$OUTPUT_ROOT/act_finetune_${TAG}_${TIMESTAMP}"
mkdir -p "$OUTPUT_ROOT/logs"
TRAIN_LOG="$OUTPUT_ROOT/logs/act_finetune_${TAG}_${TIMESTAMP}.log"

log "输出目录: $RUN_DIR"
log "日志文件: $TRAIN_LOG"
log "开始时间: $(date)"
echo

# 构建训练命令
TRAIN_ARGS=(
    --policy.type=act
    --policy.pretrained_path="$PRETRAINED_CKPT"
    --dataset.repo_id=local
    --dataset.root="$NEW_DATASET"
    --dataset.video_backend=pyav
    --output_dir="$RUN_DIR"
    --job_name="act_finetune_${TAG}"
    --batch_size="$BATCH_SIZE"
    --steps="$STEPS"
    --env_eval_freq="$EVAL_FREQ"
    --save_freq="$SAVE_FREQ"
    --log_freq="$LOG_FREQ"
    --policy.device=cuda
    --policy.push_to_hub="$PUSH_TO_HUB"
    --wandb.enable="$WANDB_ENABLE"
    --wandb.project="$WANDB_PROJECT"
    --tolerance_s=0.0001
)

# 打印完整命令（方便调试）
log "训练命令:"
echo "lerobot-train \\"
for arg in "${TRAIN_ARGS[@]}"; do
    echo "    $arg \\"
done
echo

# 执行训练
lerobot-train "${TRAIN_ARGS[@]}" > "$TRAIN_LOG" 2>&1

EXIT_CODE=$?
log "训练退出码: $EXIT_CODE"
log "结束时间: $(date)"

if [[ $EXIT_CODE -ne 0 ]]; then
    err "训练失败,查看日志: $TRAIN_LOG"
    err "最后 30 行:"
    tail -30 "$TRAIN_LOG" | sed 's/^/    /'
    exit $EXIT_CODE
fi

ok "Fine-tune 完成!"
echo
echo "=================================================="
echo " 训练完成"
echo "=================================================="
echo
echo " 输出目录: $RUN_DIR"
echo " 日志文件: $TRAIN_LOG"
echo " Checkpoints: $RUN_DIR/checkpoints/"
echo
echo " 查看训练曲线:"
echo "   tail -f $TRAIN_LOG"
if [[ "$WANDB_ENABLE" == "true" ]]; then
    echo "   或访问 WandB: $WANDB_PROJECT"
fi
echo "=================================================="

# 保存路径供后续使用
echo "$RUN_DIR" > "$OUTPUT_ROOT/.last_finetune_run"
