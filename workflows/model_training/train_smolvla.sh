#!/usr/bin/env bash
# train_smolvla.sh — 一键跑完 SmolVLA 训练全流程
# 用法:   ./train_smolvla.sh                # 跑完整 5 个 phase
#         ./train_smolvla.sh smoke          # 只跑烟测 (Phase 2)
#         ./train_smolvla.sh train          # 只跑正式训练 (Phase 3)
#         ./train_smolvla.sh check          # 只跑数据校验 (Phase 1)
#
# 依赖:   - GPU (CUDA, 建议 ≥24GB,SmolVLA 占显存比 ACT 大)
#         - ffmpeg
#         - lerobot[smolvla] 已装
#           `pip install -e ".[smolvla]"`  或  `uv sync --extra smolvla`
#         - 数据集路径: $DATASET_ROOT
#
# SmolVLA 跟 ACT 的关键差异:
#   - VLA 模型,需要 tasks.parquet 里有 ≥1 条语言指令 (Phase 1 会校验)
#   - 图像 resize+pad 到 512x512
#   - 默认 freeze_vision_encoder=True, train_expert_only=True (显存省大半)
#   - chunk_size/n_action_steps 默认 50 (不是 ACT 的 100)
#   - optimizer_lr 默认 1e-4 (不是 ACT 的 1e-5)
#   - 没有 gradient_checkpointing 字段 (官方靠 freeze 降显存)

set -euo pipefail

# ============== 配置 ==============
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/datasets/26-06-26-20-03-34_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/train}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# SmolVLA 显存大, batch=1 起步,跑通后视显存剩余再 batch=2
BATCH_SIZE="${BATCH_SIZE:-2}"
STEPS="${STEPS:-200000}"
EVAL_FREQ="${EVAL_FREQ:-40000}"
SAVE_FREQ="${SAVE_FREQ:-40000}"
LOG_FREQ="${LOG_FREQ:-50}"

# SmolVLA 的超参 (chunk/action/学习率)
POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-50}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-50}"
POLICY_LR="${POLICY_LR:-1e-4}"

# ★ 预训练起点:二选一 (默认用 smolvla_base,效果最好)
#   A) POLICY_PATH=lerobot/smolvla_base  → 官方预训练完整 SmolVLA,推荐
#   B) POLICY_PATH="" 且 LOAD_VLM_WEIGHTS=true → 只下载 VLM backbone,action expert 从零训
#   C) POLICY_PATH="" 且 LOAD_VLM_WEIGHTS=false → 全部从零训(别选)
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
LOAD_VLM_WEIGHTS="${LOAD_VLM_WEIGHTS:-false}"
# 冻结 VLM 主干,只训 action expert
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-true}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"

WANDB_PROJECT="${WANDB_PROJECT:-lerobot_smolvla_v2_26-06-26-20-03-34_3cam}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

# HuggingFace 国内镜像 (默认 hf-mirror.com,首次下载 smolvla_base ~1GB 不走这个会很慢)
# 关掉:HF_ENDPOINT="" ./train_smolvla.sh train
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# HF 缓存目录(可选,默认 ~/.cache/huggingface)
HF_HOME="${HF_HOME:-}"

# 相机 key 重命名:把数据集的 key 翻译成 policy 期望的 key
#   lerobot/smolvla_base 的 input_features 是 ALOHA 风格的 camera1/2/3
#   空间约定: camera1=top/overhead, camera2=left_wrist, camera3=right_wrist
#   你的数据集是 right_eye(前视) / left_wrist / right_wrist
#   ⚠ 映射错了 action expert 会按错误空间关系出动作
#   注意: 必须是 compact JSON(无空格),否则 shell 会把 JSON 切成多段参数
#   关掉: RENAME_MAP="" ./train_smolvla.sh train
RENAME_MAP="${RENAME_MAP:-{\"observation.images.right_eye\":\"observation.images.camera1\",\"observation.images.left_wrist\":\"observation.images.camera2\",\"observation.images.right_wrist\":\"observation.images.camera3\"}}"


# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*" >&2; }

# 构造 lerobot-train 的 policy flag 片段 (二选一,不能同时传)
#   POLICY_PATH=xxx        →  --policy.path=xxx
#                            (类型从 checkpoint config 推,不能再传 --policy.type)
#   POLICY_PATH=""         →  --policy.type=smolvla
#                            (从零 init,或配合 LOAD_VLM_WEIGHTS 下载 VLM)
policy_init_args() {
    if [[ -n "${POLICY_PATH:-}" ]]; then
        printf -- '--policy.path=%q ' "$POLICY_PATH"
    else
        printf -- '--policy.type=%s ' "${POLICY_TYPE:-smolvla}"
    fi
}

# 构造 --rename_map flag (RENAME_MAP 为空时跳过)
# RENAME_MAP 必须是 compact JSON(无空格),用 %s 输出,避免 %q 把 JSON 里的空格转义后被 shell 错切
rename_map_args() {
    if [[ -n "${RENAME_MAP:-}" ]]; then
        printf -- "--rename_map=%s" "$RENAME_MAP"
        printf ' '
    fi
}

# 从 DATASET_ROOT 路径里提取版本标签 (e.g. "..._v2" -> "v2"), 用来给 run_dir/job_name 加后缀
dataset_tag() {
    local tag
    tag="$(basename "$DATASET_ROOT" | sed -nE 's/.*_(v[0-9]+)$/\1/p')"
    [ -z "$tag" ] && tag="local"
    printf '%s' "$tag"
}

# ============== Phase 0: 环境 ==============
phase0_env() {
    log "Phase 0: 环境检查"

    # 导出 HuggingFace 镜像(对 huggingface_hub 和 transformers 都生效)
    if [[ -n "${HF_ENDPOINT:-}" ]]; then
        export HF_ENDPOINT
        ok "HF 镜像: $HF_ENDPOINT  (关闭: HF_ENDPOINT='' ./train_smolvla.sh ...)"
    else
        warn "HF 镜像未设(HF_ENDPOINT=''),将直连 huggingface.co"
    fi
    if [[ -n "${HF_HOME:-}" ]]; then
        export HF_HOME
        ok "HF 缓存目录: $HF_HOME"
    fi

    # GPU
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        err "nvidia-smi 不可用,无法训练"
        exit 1
    fi
    local gpu_mem
    gpu_mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
    ok "GPU 显存: ${gpu_mem} MiB"
    if [ "${gpu_mem:-0}" -lt 20000 ]; then
        warn "显存 < 20GB,SmolVLA batch=1 可能 OOM,建议先 --batch_size=1 --steps=5 试一下"
    fi

    # ffmpeg
    if ! command -v ffmpeg >/dev/null 2>&1; then
        err "ffmpeg 不可用,无法解码视频。请 conda install ffmpeg"
        exit 1
    fi
    ok "ffmpeg: $(ffmpeg -version | head -1 | awk '{print $3}')"

    # lerobot-train
    if ! command -v lerobot-train >/dev/null 2>&1; then
        warn "lerobot-train 不在 PATH,改用 'uv run lerobot-train'"
        alias lerobot-train='uv run lerobot-train'
    fi
    ok "lerobot-train: $(which lerobot-train)"

    # python 验证
    if ! lerobot-train --help >/dev/null 2>&1; then
        err "lerobot-train 启动失败"
        exit 1
    fi

    # 确定 python 解释器: 优先当前 PATH 上的 python, 否则用 uv
    if command -v python >/dev/null 2>&1 && python -c "import lerobot" 2>/dev/null; then
        PYTHON_BIN="$(command -v python)"
    elif command -v uv >/dev/null 2>&1; then
        PYTHON_BIN="uv run python"
    else
        err "找不到能 import lerobot 的 python,请先激活 conda env 或安装 uv"
        exit 1
    fi
    ok "python: $PYTHON_BIN"

    # SmolVLA 依赖检查 (transformers + accelerate + num2words)
    log "SmolVLA 依赖检查 (transformers / accelerate / num2words)"
    if ! $PYTHON_BIN -c "import transformers, accelerate, num2words" 2>/dev/null; then
        err "缺 SmolVLA 依赖,请跑: pip install -e '.[smolvla]'"
        exit 1
    fi
    local tr_ver aa_ver
    tr_ver="$($PYTHON_BIN -c 'import transformers; print(transformers.__version__)')"
    aa_ver="$($PYTHON_BIN -c 'import accelerate; print(accelerate.__version__)')"
    ok "transformers=$tr_ver  accelerate=$aa_ver"

    # SmolVLAConfig 注册检查
    if ! $PYTHON_BIN -c "from lerobot.policies.smolvla import SmolVLAConfig" 2>/dev/null; then
        err "SmolVLAConfig 不可导入,确认 lerobot[smolvla] 装好"
        exit 1
    fi
    ok "SmolVLA 注册可用"

    # 预训练起点可达性检查 (HuggingFace 网络 / 本地路径)
    if [[ -n "${POLICY_PATH:-}" ]]; then
        log "校验预训练起点可达性: $POLICY_PATH"
        if [[ -d "$POLICY_PATH" ]]; then
            ok "本地路径存在: $POLICY_PATH"
        else
            # 远端 HF repo,试 ping 一下,失败不阻塞(可能离线),只警告
            if $PYTHON_BIN -c "
from huggingface_hub import HfApi
import sys
try:
    HfApi().model_info('$POLICY_PATH', timeout=10)
except Exception as e:
    print(f'WARN: {e}', file=sys.stderr); sys.exit(2)
" 2>/dev/null; then
                ok "HuggingFace repo 可达: $POLICY_PATH"
            else
                warn "无法 ping 通 $POLICY_PATH,可能是网络/权限问题,真要跑时会再下载一次"
            fi
        fi
    fi

    ok "Phase 0 通过"
}

# ============== Phase 1: 数据校验 ==============
phase1_check() {
    log "Phase 1: 数据校验"

    if [[ ! -d "$DATASET_ROOT" ]]; then
        err "数据集路径不存在: $DATASET_ROOT"
        exit 1
    fi
    ok "数据集目录存在: $DATASET_ROOT"

    # 必有的子目录
    for d in data meta videos; do
        if [[ ! -d "$DATASET_ROOT/$d" ]]; then
            err "缺少子目录: $d"
            exit 1
        fi
    done
    ok "data/  meta/  videos/ 都在"

    # info.json 关键字段
    python - <<PY
import json, sys
d = json.load(open("$DATASET_ROOT/meta/info.json"))
assert d["codebase_version"] == "v3.0", f"version: {d['codebase_version']}"
print(f"  version:      {d['codebase_version']}")
print(f"  episodes:     {d['total_episodes']}")
print(f"  frames:       {d['total_frames']}")
print(f"  fps:          {d['fps']}")
print(f"  state shape:  {d['features']['observation.state']['shape']}")
print(f"  action shape: {d['features']['action']['shape']}")
assert d["features"]["observation.state"]["shape"] == [16], "state 维度不是 16"
assert d["features"]["action"]["shape"] == [16], "action 维度不是 16"
PY
    ok "info.json 字段符合预期"

    # ★ SmolVLA 特有:必须有 task 描述 (language instruction)
    #   LeRobot v3 schema: task_index 是列,任务文本是 pandas index
    #   写进 parquet 后 index 会变成 __index_level_0__ 列
    log "校验 SmolVLA 必需的语言指令 (tasks.parquet)"
    $PYTHON_BIN - <<PY || { err "tasks.parquet 校验失败,见上"; exit 1; }
import json, os, sys
import pandas as pd

ROOT = "$DATASET_ROOT"
META = os.path.join(ROOT, "meta")
info = json.load(open(f"{META}/info.json"))
assert info["total_tasks"] >= 1, f"SmolVLA 至少要 1 个 task,实际 {info['total_tasks']}"
print(f"  total_tasks:  {info['total_tasks']}")

tp = f"{META}/tasks.parquet"
assert os.path.exists(tp), f"缺 {tp}"

# 用 pandas 读 (LeRobot 官方 load_tasks() 的方式)
df = pd.read_parquet(tp)
# LeRobot v3: 任务文本在 df.index (index.name 应该叫 'task')
# 兜底:如果 index 没名字,rename 成 'task'
if df.index.name is None:
    df.index.name = "task"
print(f"  index name:   {df.index.name}")
print(f"  columns:      {list(df.columns)}")

# 取出 task 文本列表 (兼容 index 是 'task' 或 fallback 叫 '__index_level_0__')
tasks = df.index.tolist()
assert len(tasks) >= 1, "tasks 列表为空"
# 还要确保不是空字符串
tasks = [t for t in tasks if isinstance(t, str) and t.strip()]
assert len(tasks) >= 1, "任务文本全是空字符串"
print(f"  任务数:        {len(tasks)}")
for i, txt in enumerate(tasks[:3]):
    preview = (txt[:80] + '...') if len(txt) > 80 else txt
    print(f"  task[{i}]:      {preview}")
if len(tasks) == 1:
    print("  ⚠ 警告: 只有 1 个 task,SmolVLA 多任务优势发挥不出来")
PY
    ok "tasks.parquet 校验通过"

    # 交叉一致性校验: info / stats / videos/ / episodes parquet 互相不能矛盾
    log "交叉一致性校验 (info.json ↔ stats.json ↔ videos/ ↔ episodes parquet)"
    $PYTHON_BIN - <<PY || { err "一致性校验失败,见上"; exit 1; }
import json, os, sys
import pyarrow.parquet as pq
ROOT = "$DATASET_ROOT"
META = os.path.join(ROOT, "meta")
EP   = os.path.join(META, "episodes/chunk-000")

info  = json.load(open(f"{META}/info.json"))
stats = json.load(open(f"{META}/stats.json"))
video_dirs = set(os.listdir(f"{ROOT}/videos"))

# episodes parquet 里实际引用的 video stream
ep_streams = set()
for f in os.listdir(EP):
    for c in pq.read_table(os.path.join(EP, f)).column_names:
        if c.startswith("videos/"):
            ep_streams.add(c.split("/")[1])

info_video = {k for k,v in info["features"].items() if v.get("dtype")=="video"}
fatal = []

# 1) info 声明了 video stream, 但 videos/ 实际没有 → 致命 (会触发 Hub 下载)
for s in info_video - video_dirs:
    fatal.append(f"info.json 声明了 '{s}', 但 videos/{s}/ 不存在 → 会触发 Hub 拉数据")

# 2) episode parquet 引用了 videos/ 没有的 stream → 致命
for s in ep_streams - video_dirs:
    fatal.append(f"episodes parquet 引用了 '{s}', 但 videos/{s}/ 不存在")

# 3) info 声明的 tabular feature 在 data parquet 里必须存在
data_cols = set()
for f in os.listdir(os.path.join(ROOT, "data/chunk-000")):
    for c in pq.read_table(f"{ROOT}/data/chunk-000/{f}").column_names:
        data_cols.add(c)
for f in (set(info["features"]) - info_video):
    if f not in data_cols:
        fatal.append(f"info.json 声明的 tabular feature '{f}' 在 data parquet 里找不到")

# 4) float 特征必须有 stats (bool 蒙版除外, bool 缺 stats 是设计的 no-op)
for k, v in info["features"].items():
    if v.get("dtype") == "float32" and k not in stats:
        fatal.append(f"float feature '{k}' 在 stats.json 缺 stats, 训练会拿不到信号")

if fatal:
    print("✗ 致命不一致:")
    for x in fatal:
        print(f"    - {x}")
    sys.exit(1)

# 警告级: 跟训练无关但建议关注
warn = []
for k in stats:
    if k.startswith("observation.images.") and k not in info_video:
        warn.append(f"stats.json 里有 '{k}' 但 info.json 没声明 (无害, 占点空间)")
if warn:
    print("  ⚠ 警告 (不阻塞):")
    for x in warn:
        print(f"    - {x}")
else:
    print("  ✓ info / stats / videos / episodes / data 全部一致")
PY
    ok "交叉一致性通过"

    # 跑 sanity_check.py (重构后位于 data_processing/)
    log "跑 sanity_check.py ..."
    $PYTHON_BIN "$PROJECT_ROOT/workflows/data_processing/sanity_check.py" \
        --dataset-root "$DATASET_ROOT" || {
        err "sanity_check.py 失败"
        exit 1
    }
    ok "sanity_check.py 通过"
    ok "Phase 1 通过"
}

# ============== Phase 2: 烟测 ==============
phase2_smoke() {
    log "Phase 2: 烟测 (50 步,batch=$BATCH_SIZE)"

    local smoke_dir="$OUTPUT_ROOT/smolvla_smoke_${TIMESTAMP}"
    rm -rf "$smoke_dir"
    # 注意:lerobot-train 会校验 output_dir 是否已存在,即便为空也拒绝
    # 所以这里**不**要 mkdir -p,让它自己创建
    # 但 tee 写日志需要目录,先准备日志目录,再让 lerobot-train 创建 output_dir
    # 简单做法:把 smoke.log 写到 $OUTPUT_ROOT/ 而不是 $smoke_dir/
    local smoke_log="$OUTPUT_ROOT/smolvla_smoke_${TIMESTAMP}.log"
    rm -f "$smoke_log"

    # 本地数据集要同时给 repo_id(必填,通常填 "local") + root(可选,本地路径)
    # 强制用 pyav 做视频后端(torchcodec 装但系统 lib 不全)
    lerobot-train \
        $(policy_init_args) \
        $(rename_map_args) \
        --dataset.repo_id=local \
        --dataset.root="$DATASET_ROOT" \
        --dataset.video_backend=pyav \
        --output_dir="$smoke_dir" \
        --job_name=smolvla_smoke \
        --batch_size="$BATCH_SIZE" \
        --steps=50 \
        --eval_freq=1000000 \
        --save_freq=1000000 \
        --log_freq=10 \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --policy.chunk_size="$POLICY_CHUNK_SIZE" \
        --policy.n_action_steps="$POLICY_N_ACTION_STEPS" \
        --policy.optimizer_lr="$POLICY_LR" \
        --policy.load_vlm_weights="$LOAD_VLM_WEIGHTS" \
        --policy.freeze_vision_encoder="$FREEZE_VISION_ENCODER" \
        --policy.train_expert_only="$TRAIN_EXPERT_ONLY" \
        --wandb.enable=true \
        2>&1 | tee "$smoke_log" | grep -E "loss|step|grad_norm|error|Error|OOM" -i | head -50

    # 抓最后一个 loss
    local final_loss
    final_loss="$(grep -oE "loss[: =][0-9.]+|train/loss[ =:][0-9.]+" "$smoke_log" | tail -1 || echo "")"
    if [[ -z "$final_loss" ]]; then
        err "烟测日志里没找到 loss,可能训练失败"
        err "最后 30 行:"
        tail -30 "$smoke_log" | sed 's/^/    /'
        exit 1
    fi
    ok "烟测 loss 末值: $final_loss"
    ok "Phase 2 通过 (checkpoint 在 $smoke_dir,日志 $smoke_dir.log)"
}

# ============== Phase 3: 正式训练 ==============
phase3_train() {
    log "Phase 3: 正式训练 (steps=$STEPS, batch=$BATCH_SIZE)"

    local tag
    tag="$(dataset_tag)"
    local run_dir="$OUTPUT_ROOT/smolvla_${tag}_${TIMESTAMP}"
    rm -rf "$run_dir"
    # 注意:lerobot-train 校验 output_dir 不允许预先存在
    # 所以这里**不**mkdir -p,让 lerobot-train 自己创建
    # 日志写到 run_dir 的兄弟目录,避免污染
    mkdir -p "$OUTPUT_ROOT/logs"
    local train_log="$OUTPUT_ROOT/logs/smolvla_${tag}_${TIMESTAMP}.log"
    rm -f "$train_log"

    log "输出目录: $run_dir (由 lerobot-train 自己创建)"
    log "日志文件: $train_log"
    log "开始时间: $(date)"
    log "SmolVLA 超参: chunk_size=$POLICY_CHUNK_SIZE  n_action_steps=$POLICY_N_ACTION_STEPS  lr=$POLICY_LR"
    log "                load_vlm_weights=$LOAD_VLM_WEIGHTS  freeze_vision_encoder=$FREEZE_VISION_ENCODER  train_expert_only=$TRAIN_EXPERT_ONLY"
    if [[ -n "${POLICY_PATH:-}" ]]; then
        log "★ 预训练起点: $POLICY_PATH  (会从 HuggingFace 下载 ~1GB)"
    else
        log "★ 预训练起点: 仅 VLM backbone (HuggingFaceTB/SmolVLM2-500M-Video-Instruct), action expert 从零训"
    fi

    # 后台跑,日志落盘
    lerobot-train \
        $(policy_init_args) \
        $(rename_map_args) \
        --dataset.repo_id=local \
        --dataset.root="$DATASET_ROOT" \
        --dataset.video_backend=pyav \
        --output_dir="$run_dir" \
        --job_name="smolvla_${tag}" \
        --batch_size="$BATCH_SIZE" \
        --steps="$STEPS" \
        --env_eval_freq="$EVAL_FREQ" \
        --save_freq="$SAVE_FREQ" \
        --log_freq="$LOG_FREQ" \
        --policy.device=cuda \
        --policy.push_to_hub="$PUSH_TO_HUB" \
        --policy.chunk_size="$POLICY_CHUNK_SIZE" \
        --policy.n_action_steps="$POLICY_N_ACTION_STEPS" \
        --policy.optimizer_lr="$POLICY_LR" \
        --policy.load_vlm_weights="$LOAD_VLM_WEIGHTS" \
        --policy.freeze_vision_encoder="$FREEZE_VISION_ENCODER" \
        --policy.train_expert_only="$TRAIN_EXPERT_ONLY" \
        --wandb.enable="$WANDB_ENABLE" \
        --wandb.project="$WANDB_PROJECT" \
        > "$train_log" 2>&1

    local exit_code=$?
    log "训练退出码: $exit_code"
    log "结束时间: $(date)"

    if [[ $exit_code -ne 0 ]]; then
        err "训练异常退出,看 $train_log"
        err "最后 20 行:"
        tail -20 "$train_log" | sed 's/^/    /'
        exit $exit_code
    fi

    ok "训练完成,checkpoint 在 $run_dir/checkpoints/"
    echo "$run_dir" > "$OUTPUT_ROOT/.last_smolvla_run"
}

# ============== Phase 4: 评估提示 ==============
phase4_eval_hint() {
    log "Phase 4: 评估 (这里只打印命令,实际跑按需)"
    local run_dir
    run_dir="$(cat "$OUTPUT_ROOT/.last_smolvla_run" 2>/dev/null || echo "$OUTPUT_ROOT/smolvla_$(dataset_tag)_$TIMESTAMP")"
    local last_ckpt="$run_dir/checkpoints/last"

    echo
    echo "=================================================="
    echo " SmolVLA 训练完成,下一步:"
    echo "=================================================="
    echo
    echo " # 1) 列 checkpoints:"
    echo "   ls $run_dir/checkpoints/"
    echo
    echo " # 2) 仿真 eval (假设有 aloha env):"
    echo "   lerobot-eval --policy.path=$last_ckpt --env.type=aloha --eval.n_episodes=20"
    echo
    echo " # 3) 真实机械臂 replay (部署时记得给语言指令):"
    echo "   lerobot-replay --robot.type=<your_robot> --robot.port=/dev/ttyACM0 \\"
    echo "                  --dataset.root=$DATASET_ROOT --replay.episode=0"
    echo
    echo " # 4) 看训练曲线:"
    echo "   tail -f $run_dir/train.log"
    echo "   或 wandb: project=$WANDB_PROJECT"
    echo
    echo " # 5) ★ SmolVLA 部署关键: 推理时要传 'task' 字符串,"
    echo "    不传会 fallback 到空字符串,效果很差"
    echo "=================================================="
}

# ============== 入口 ==============
case "${1:-all}" in
    env)     phase0_env ;;
    check)   phase0_env; phase1_check ;;
    smoke)   phase0_env; phase1_check; phase2_smoke ;;
    train)   phase0_env; phase1_check; phase3_train ;;
    eval)    phase4_eval_hint ;;
    all|"")  phase0_env; phase1_check; phase2_smoke; phase3_train; phase4_eval_hint ;;
    *)
        err "未知参数: $1"
        echo "用法: $0 [env|check|smoke|train|eval|all]"
        exit 1
        ;;
esac
