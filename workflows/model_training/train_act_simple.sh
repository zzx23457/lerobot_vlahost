#!/usr/bin/env bash
# train_act_simple.sh — 纯训练,不做事前数据校验
# 用法:   ./train_act_simple.sh
# 依赖:   - GPU (CUDA)
#         - lerobot 包已安装
#
# 可调环境变量:
#   DATASET_ROOT   数据集路径           (必填,无默认值)
#   OUTPUT_ROOT    checkpoint 输出根目录  (默认 $PROJECT_ROOT/outputs/train)
#   BATCH_SIZE                          (默认 8)
#   STEPS                               (默认 200000)
#   EVAL_FREQ                           (默认 10000)
#   SAVE_FREQ                           (默认 10000)
#   LOG_FREQ                            (默认 50)
#   PUSH_TO_HUB                         (默认 false)
#   WANDB_ENABLE                        (默认 true)
#   WANDB_PROJECT                       (默认 act_train)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/datasets/26-07-03-14-32-17_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/train}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"


BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-200000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-50}"

WANDB_PROJECT="${WANDB_PROJECT:-act_train}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
log() { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()  { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }

if [[ -z "$DATASET_ROOT" ]]; then
    echo "错误: DATASET_ROOT 未设置" >&2
    echo "用法: DATASET_ROOT=/path/to/dataset $0" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT/logs"
run_dir="$OUTPUT_ROOT/act_${TIMESTAMP}"
log_file="$OUTPUT_ROOT/logs/act_${TIMESTAMP}.log"

log "数据集:    $DATASET_ROOT"
log "输出目录:  $run_dir"
log "日志:      $log_file"
log "steps=$STEPS  batch=$BATCH_SIZE"

lerobot-train \
    --policy.type=act \
    --dataset.repo_id=local \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --output_dir="$run_dir" \
    --job_name="act_${TIMESTAMP}" \
    --batch_size="$BATCH_SIZE" \
    --steps="$STEPS" \
    --env_eval_freq="$EVAL_FREQ" \
    --save_freq="$SAVE_FREQ" \
    --log_freq="$LOG_FREQ" \
    --policy.device=cuda \
    --policy.push_to_hub="$PUSH_TO_HUB" \
    --wandb.enable="$WANDB_ENABLE" \
    --wandb.project="$WANDB_PROJECT" \
    > "$log_file" 2>&1

ok "训练完成,checkpoint: $run_dir/checkpoints/"
ok "日志: $log_file"