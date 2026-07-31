#!/usr/bin/env bash
# train_act.sh — 一键跑完 ACT 训练全流程
# 用法:   ./train_act.sh                # 跑完整 5 个 phase
#         ./train_act.sh smoke          # 只跑烟测 (Phase 2)
#         ./train_act.sh train          # 只跑正式训练 (Phase 3)
#         ./train_act.sh check          # 只跑数据校验 (Phase 1)
#
# 依赖:   - GPU (CUDA)
#         - ffmpeg
#         - lerobot 包已安装 (`uv sync --locked --extra aloha` 等)
#         - 数据集路径: $DATASET_ROOT

set -euo pipefail

# ============== 配置 ==============
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/datasets/26-07-21+22+23+25+25-merged_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/train}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-400000}"
EVAL_FREQ="${EVAL_FREQ:-20000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-50}"

WANDB_PROJECT="${WANDB_PROJECT:-202607172057}"
# 默认关 wandb —— 它是可选依赖,不装也能训
# 想用 wandb 时: pip install wandb && WANDB_ENABLE=true ./train_act.sh train
WANDB_ENABLE="${WANDB_ENABLE:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*" >&2; }

# 从 DATASET_ROOT 路径里提取数据集标识 (basename), 用来给 run_dir/job_name 加后缀
# 例如 DATASET_ROOT=.../datasets/20260710_143022_v2 -> tag=20260710_143022_v2
# 这样保存的模型路径里就能直接看出是用哪个数据集训的, 避免和别的数据集训练的模型混淆
dataset_tag() {
    local tag
    tag="$(basename "$DATASET_ROOT")"
    [ -z "$tag" ] && tag="local"
    printf '%s' "$tag"
}

# ============== Phase 0: 环境 ==============
phase0_env() {
    log "Phase 0: 环境检查"

    # GPU
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        err "nvidia-smi 不可用,无法训练"
        exit 1
    fi
    local gpu_mem
    gpu_mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
    ok "GPU 显存: ${gpu_mem} MiB"

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

    # 时间戳对齐检查 (放在 sanity_check 之前, 避免无谓触发 lerobot 内部错误)
    # sanity_check.py 会真的加载样本, 一旦数据有时间戳漂移就会立刻 FrameTimestampError;
    # 提前跑 check_timestamp_alignment.py 可以用更便宜的方式发现并提示用户。
    log "检查数据集时间戳对齐 ..."
    local alignment_check
    local check_exit=0
    alignment_check=$($PYTHON_BIN "$PROJECT_ROOT/workflows/data_processing/check_timestamp_alignment.py" \
        --dataset-root "$DATASET_ROOT" --format json 2>&1) || check_exit=$?

    # 根据 exit code 分流处理:
    #   0 = CLEAN (无脏数据, alignment_check 含 JSON)
    #   1 = DIRTY (有脏数据, alignment_check 含 JSON)
    #   2 = SCRIPT_ERROR (脚本异常, alignment_check 含 traceback)
    if [[ $check_exit -eq 2 ]]; then
        # 脚本异常: 数据集路径不存在 / 视频流找不到 / 其他错误
        err "时间戳对齐检查脚本失败 (exit 2)"
        echo "$alignment_check" | sed 's/^/    /'
        echo
        err "请先修复数据集结构问题, 然后重试"
        exit 1
    fi

    # 处理时间戳检查结果
    if [[ $check_exit -ne 0 ]]; then
        # 有脏数据 (exit 1)
        warn "数据集存在时间戳漂移 (data parquet timestamp 与视频 pts 不同步)"
        echo "$alignment_check" | grep -E "DIRTY|gap_ms" || true
        echo
        warn "这会导致训练时 FrameTimestampError"
        echo
        echo "选项:"
        echo "  1) 自动过滤 DIRTY episodes, 仅用 CLEAN 数据训练 (推荐)"
        echo "  2) 使用全部数据 + 调高 tolerance_s (会有视觉-本体错位)"
        echo "  3) 取消训练"
        echo
        read -p "请选择 [1/2/3]: " -n 1 -r
        echo

        case $REPLY in
            1)
                log "将自动过滤 DIRTY episodes"
                CLEAN_EPISODES=$(echo "$alignment_check" | grep '"clean"' -A 1 | tail -1 | sed 's/[][]//g' | tr -d ' ')
                if [[ -z "$CLEAN_EPISODES" ]]; then
                    err "无法提取 CLEAN episode 列表 (alignment_check 缺 'clean' 字段)"
                    exit 1
                fi
                ok "将使用 CLEAN episodes"
                export USE_CLEAN_EPISODES="[$CLEAN_EPISODES]"
                ;;
            2)
                warn "使用全部数据 + tolerance_s=1"
                warn "注意: DIRTY episodes 会有视觉-本体错位 (最大 2+ 秒)"
                export USE_CLEAN_EPISODES=""
                export OVERRIDE_TOLERANCE_S="1"
                ;;
            3)
                err "用户取消"
                exit 1
                ;;
            *)
                err "无效选择"
                exit 1
                ;;
        esac
    else
        ok "时间戳对齐检查通过, 所有 episodes 都是 CLEAN"
        export USE_CLEAN_EPISODES=""
    fi

    # 跑 sanity_check.py (重构后位于 data_processing/)
    log "跑 sanity_check.py ..."
    $PYTHON_BIN "$PROJECT_ROOT/workflows/data_processing/sanity_check.py" \
        --dataset-root "$DATASET_ROOT" || {
        err "sanity_check.py 失败"
        exit 1
    }
    ok "sanity_check.py 通过"

    # 脏数据检测 (必须检查，但清洗是可选的)
    log "检测脏数据 episode (observation.state 前 7 位异常全零) ..."
    local dirty_report
    dirty_report=$($PYTHON_BIN "$PROJECT_ROOT/workflows/data_processing/clean_dirty_episodes.py" \
        --dataset-path "$DATASET_ROOT" \
        --report-only 2>&1)

    if echo "$dirty_report" | grep -q "No dirty episodes found"; then
        ok "未发现脏数据 episode"
    else
        warn "发现脏数据 episode!"
        echo "$dirty_report" | grep -A 20 "SUMMARY:"
        echo
        warn "脏数据 episode 的特征: observation.state 前 7 位全为 0"
        warn "建议清洗这些 episode 以提高训练质量"
        echo
        echo "选项:"
        echo "  1) 清洗数据集 (自动创建 ${DATASET_ROOT}_cleaned 并使用)"
        echo "  2) 跳过清洗,继续使用原数据集 (不推荐,可能影响训练质量)"
        echo "  3) 取消训练,手动处理"
        echo
        read -p "请选择 [1/2/3]: " -n 1 -r
        echo

        case $REPLY in
            1)
                log "开始清洗数据集..."
                local cleaned_path="${DATASET_ROOT}_cleaned"

                # 如果清洗后的数据集已存在，询问是否覆盖
                if [[ -d "$cleaned_path" ]]; then
                    warn "清洗后的数据集已存在: $cleaned_path"
                    read -p "是否删除并重新清洗? (y/N) " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        rm -rf "$cleaned_path"
                    else
                        log "使用现有清洗后的数据集"
                        DATASET_ROOT="$cleaned_path"
                        ok "已切换到清洗后的数据集: $DATASET_ROOT"
                        ok "Phase 1 通过"
                        return
                    fi
                fi

                $PYTHON_BIN "$PROJECT_ROOT/workflows/data_processing/clean_dirty_episodes.py" \
                    --dataset-path "$DATASET_ROOT" \
                    --output-path "$cleaned_path" || {
                    err "数据清洗失败"
                    exit 1
                }

                # 切换到清洗后的数据集
                DATASET_ROOT="$cleaned_path"
                ok "数据清洗完成,已切换到: $DATASET_ROOT"
                export DATASET_ROOT  # 导出给后续 phase 使用
                ;;
            2)
                warn "跳过清洗,使用原数据集 (可能影响训练质量)"
                ;;
            3)
                log "用户取消训练"
                echo
                echo "手动清洗命令:"
                echo "  python workflows/data_processing/clean_dirty_episodes.py \\"
                echo "    --dataset-path $DATASET_ROOT \\"
                echo "    --output-path ${DATASET_ROOT}_cleaned"
                exit 0
                ;;
            *)
                err "无效选择: $REPLY"
                exit 1
                ;;
        esac
    fi

    # 时间戳对齐检查已合并到 sanity_check 之前

    ok "Phase 1 通过"
}

# ============== Phase 2: 烟测 ==============
phase2_smoke() {
    log "Phase 2: 烟测 (50 步,batch=2)"

    local smoke_dir="$OUTPUT_ROOT/act_smoke_${TIMESTAMP}"
    rm -rf "$smoke_dir"
    # 注意:lerobot-train 会校验 output_dir 是否已存在,即便为空也拒绝
    # 所以这里**不**要 mkdir -p,让它自己创建
    # 但 tee 写日志需要目录,先准备日志目录,再让 lerobot-train 创建 output_dir
    # 简单做法:把 smoke.log 写到 $OUTPUT_ROOT/ 而不是 $smoke_dir/
    local smoke_log="$OUTPUT_ROOT/act_smoke_${TIMESTAMP}.log"
    rm -f "$smoke_log"

    # 本地数据集要同时给 repo_id(必填,通常填 "local") + root(可选,本地路径)
    # 强制用 pyav 做视频后端(torchcodec 装但系统 lib 不全)
    lerobot-train \
        --policy.type=act \
        --dataset.repo_id=local \
        --dataset.root="$DATASET_ROOT" \
        --dataset.video_backend=pyav \
        --output_dir="$smoke_dir" \
        --job_name=act_smoke \
        --batch_size=2 \
        --steps=50 \
        --eval_freq=1000000 \
        --save_freq=1000000 \
        --log_freq=10 \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --wandb.enable=true \
        2>&1 | tee "$smoke_log" | grep -E "loss|step|grad_norm|error|Error" -i | head -50

    # 抓最后一个 loss
    local final_loss
    final_loss="$(grep -oE "loss:[0-9.]+|loss=[0-9.]+|train/loss[ =:][0-9.]+" "$smoke_log" | tail -1 || echo "")"
    if [[ -z "$final_loss" ]]; then
        err "烟测日志里没找到 loss,可能训练失败"
        exit 1
    fi
    ok "烟测 loss 末值: $final_loss"
    ok "Phase 2 通过 (checkpoint 在 $smoke_dir,日志 $smoke_log)"
}

# ============== Phase 3: 正式训练 ==============
phase3_train() {
    log "Phase 3: 正式训练 (steps=$STEPS, batch=$BATCH_SIZE)"

    local tag
    tag="$(dataset_tag)"
    local run_dir="$OUTPUT_ROOT/act_${tag}_${TIMESTAMP}"
    rm -rf "$run_dir"
    # 注意:lerobot-train 校验 output_dir 不允许预先存在
    # 所以这里**不**mkdir -p,让 lerobot-train 自己创建
    # 日志写到 run_dir 的兄弟目录,避免污染
    mkdir -p "$OUTPUT_ROOT/logs"
    local train_log="$OUTPUT_ROOT/logs/act_${tag}_${TIMESTAMP}.log"
    rm -f "$train_log"

    log "输出目录: $run_dir (由 lerobot-train 自己创建)"
    log "日志文件: $train_log"
    log "开始时间: $(date)"

    # 构建训练命令参数
    local train_args=(
        --policy.type=act
        --dataset.image_transforms.enable=true
        --dataset.repo_id=local
        --dataset.root="$DATASET_ROOT"
        --dataset.video_backend=pyav
        --output_dir="$run_dir"
        --job_name="act_${tag}"
        --batch_size="$BATCH_SIZE"
        --steps="$STEPS"
        --env_eval_freq="$EVAL_FREQ"
        --save_freq="$SAVE_FREQ"
        --log_freq="$LOG_FREQ"
        --policy.device=cuda
        --policy.push_to_hub="$PUSH_TO_HUB"
        --wandb.enable="$WANDB_ENABLE"
        --wandb.project="$WANDB_PROJECT"
    )

    # 如果 Phase 1 检测到脏数据并选择了过滤
    if [[ -n "$USE_CLEAN_EPISODES" ]]; then
        log "使用 CLEAN episodes: $USE_CLEAN_EPISODES"
        train_args+=(--dataset.episodes="$USE_CLEAN_EPISODES")
    fi

    # tolerance_s: 优先用 Phase 1 设置的覆盖值, 否则用默认 0.0001
    local tolerance_val="${OVERRIDE_TOLERANCE_S:-0.0001}"
    train_args+=(--tolerance_s="$tolerance_val")
    log "tolerance_s=$tolerance_val"

    # 后台跑,日志落盘
    lerobot-train "${train_args[@]}" > "$train_log" 2>&1

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
    echo "$run_dir" > "$OUTPUT_ROOT/.last_act_run"
}

# ============== Phase 4: 评估提示 ==============
phase4_eval_hint() {
    log "Phase 4: 评估 (这里只打印命令,实际跑按需)"
    local run_dir
    run_dir="$(cat "$OUTPUT_ROOT/.last_act_run" 2>/dev/null || echo "$OUTPUT_ROOT/act_$(dataset_tag)_$TIMESTAMP")"
    local last_ckpt="$run_dir/checkpoints/last"

    echo
    echo "=================================================="
    echo " 训练完成,下一步:"
    echo "=================================================="
    echo
    echo " # 1) 列 checkpoints:"
    echo "   ls $run_dir/checkpoints/"
    echo
    echo " # 2) 仿真 eval (假设有 aloha env):"
    echo "   lerobot-eval --policy.path=$last_ckpt --env.type=aloha --eval.n_episodes=20"
    echo
    echo " # 3) 真实机械臂 replay:"
    echo "   lerobot-replay --robot.type=<your_robot> --robot.port=/dev/ttyACM0 \\"
    echo "                  --dataset.root=$DATASET_ROOT --replay.episode=0"
    echo
    echo " # 4) 看训练曲线:"
    echo "   tail -f $run_dir/train.log"
    echo "   或 wandb: project=$WANDB_PROJECT"
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
