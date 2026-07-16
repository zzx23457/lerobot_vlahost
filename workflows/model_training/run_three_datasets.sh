#!/usr/bin/env bash
# run_three_datasets.sh — 临时脚本: 串行跑 v2/v3/v4 三个数据集
# 用法:   ./workflows/model_training/run_three_datasets.sh        # 前台跑
#         nohup ./workflows/model_training/run_three_datasets.sh &  # 后台跑
# 用完可删

set -euo pipefail

# 切到项目根目录(脚本放 workflows/model_training/, 根目录是 ../../)
cd "$(dirname "$0")/../.."

DATASETS=(
    "/home/zzx23457/lerobot/datasets/26-06-17-11-32-27_v2"
    "/home/zzx23457/lerobot/datasets/26-06-17-11-32-27_v3"
    "/home/zzx23457/lerobot/datasets/26-06-17-11-32-27_v4"
)

LOG_DIR=workflows/run_logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/three_datasets_$(date +%Y%m%d_%H%M%S).log"
echo "总日志: $LOG"

for DS in "${DATASETS[@]}"; do
    if [ ! -d "$DS" ]; then
        echo "!!! 数据集不存在: $DS, 跳过" | tee -a "$LOG"
        continue
    fi
    TAG="$(basename "$DS" | sed -nE 's/.*_(v[0-9]+)$/\1/p')"
    [ -z "$TAG" ] && TAG="local"

    echo "==========================================" | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] 开始: tag=$TAG"          | tee -a "$LOG"
    echo "  dataset: $DS"                                | tee -a "$LOG"
    echo "==========================================" | tee -a "$LOG"

    DATASET_ROOT="$DS" STEPS=200000 BATCH_SIZE=2 \
        ./workflows/model_training/train_act.sh train 2>&1 | tee -a "$LOG"

    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        echo "!!! tag=$TAG 失败 exit=$rc, 停止后续。看 $LOG" | tee -a "$LOG"
        exit $rc
    fi
    echo "[$(date +%H:%M:%S)] tag=$TAG 完成 ✓" | tee -a "$LOG"
done

echo "==========================================" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] 三个数据集全部完成 🎉"   | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
