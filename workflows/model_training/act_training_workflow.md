# ACT 训练工作流——`26-06-17-11-32-27_v2`

> **目标**:在 RTX 4090 上,用 v3.0 LeRobot 数据集 (100 episodes / 21,092 帧) 训练一个 ACT 策略。
>
> **预计总耗时**: 数据校验 5 min → 烟测 15 min → 正式训练 6–10 小时 → 评估 30 min。
>
> **核心原则**: 烟测过了再开正式训练,正式训练必须留 checkpoint + wandb。

---

## Phase 0 — 环境与前置(必做,5 min)

```bash
# 0.1 确认 GPU
nvidia-smi | head -15
# 预期:RTX 4090, 显存 ≥ 40 GB 可用,Driver ≥ 535

# 0.2 确认 ffmpeg(视频解码依赖)
ffmpeg -version | head -1
# 预期:ffmpeg 4+ (8.0.1 也 OK)

# 0.3 确认 lerobot-train 在 PATH 上
which lerobot-train
# 预期:任意 python 环境的 bin/lerobot-train(如 uv/conda/venv)

# 0.4 确认 lerobot python 包能 import
uv run python -c "import lerobot; print(lerobot.__version__)"
# 或直接:
python -c "import lerobot; print(lerobot.__version__)"

# 0.5 (可选) wandb 登录,推到云端
wandb login
```

**不通过的处理**:
- 显存 < 24 GB: 把 `batch_size` 降到 2,见 Phase 3。
- ffmpeg 缺失: `conda install -c conda-forge ffmpeg` 或 `apt install ffmpeg`。
- lerobot-train 不在 PATH: 用 `uv run lerobot-train ...` 调用。

---

## Phase 1 — 数据集校验(必做,5 min)

不校验就开训 = 浪费 GPU 小时。

### 1.1 检查 schema 与规模

```bash
DATASET_ROOT=/home/zzx23457/lerobot/datasets/26-06-17-11-32-27_v2

# 必须存在的目录
ls $DATASET_ROOT
# 预期: data/  meta/  videos/  README.md

# info.json 关键字段
python -c "
import json
d = json.load(open('$DATASET_ROOT/meta/info.json'))
print('version:', d['codebase_version'])           # 预期 v3.0
print('total_episodes:', d['total_episodes'])      # 预期 100
print('total_frames:', d['total_frames'])          # 预期 21092
print('fps:', d['fps'])                            # 预期 25
print('state shape:', d['features']['observation.state']['shape'])  # 预期 [16]
print('action shape:', d['features']['action']['shape'])            # 预期 [16]
print('images:', [k for k in d['features'] if 'images' in k])
"
```

### 1.2 跑项目自带的 sanity 脚本

```bash
python workflows/data_processing/sanity_check.py \
  --dataset-root $DATASET_ROOT
```

`sanity_check.py` 会做:
1. 加载 `LeRobotDataset` 并遍历一个 batch
2. 检查视频能正常解码(从 mp4 读出一帧)
3. 检查 `stats.json` 数值合理(无 NaN/Inf,均值在合理范围)
4. 检查 action 维度、image 维度与 info.json 一致
5. 报告每路相机的第 0 帧是否非黑/非全同
6. 构造 ACT 策略 + 校验 input_features / output_features shape 全对得上

**实测通过(2026-06-17)**: 100 episodes · 21,092 帧 · 3 路相机 (right_eye / left_wrist / right_wrist) · state=[16] 度 · action=[16] · ACT 策略构造成功

**不通过的处理**:
- 视频解码失败: 大概率 ffmpeg 路径问题或视频损坏,先 `ffmpeg -i xxx.mp4` 单帧测一下。
- stats.json 异常: 重新跑 [lerobot_compute_stats.py](src/lerobot/scripts/) 或用 [v2_convert.py](../data_processing/v2_convert.py) 重做归一化。
- ACT 构造失败: 检查 `cfg.input_features / output_features` 是否用了 `PolicyFeature` 对象(不是裸 dict)。

### 1.3 (强烈建议) 可视化 2–3 个 episode

```bash
lerobot-dataset-viz \
  --repo-id $DATASET_ROOT \
  --episode-index 0 1 5
```

看图确认:
- 3 路相机画面都对(不是黑屏、不是错位、不是时间错位)
- 关节角度曲线平滑(没有跳变)
- 抓取动作清晰

---

## Phase 2 — 烟测(必做,15 min)

**目的**: 验证整条管线通——dataset → model → loss → backward → optimizer step,只跑 50 步。

```bash
DATASET_ROOT=/home/zzx23457/lerobot/datasets/26-06-17-11-32-27_v2
SMOKE_DIR=outputs/train/act_smoke

lerobot-train \
  --policy.type=act \
  --dataset.repo_id=local \
  --dataset.root=$DATASET_ROOT \
  --output_dir=$SMOKE_DIR \
  --job_name=act_smoke \
  --batch_size=2 \
  --steps=50 \
  --eval_freq=1000000 \
  --save_freq=1000000 \
  --log_freq=10 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false
```

> ⚠️ 关键: lerobot-train 默认 `push_to_hub=true`,**不**指定 `policy.repo_id` 会报错。本地训练要加 `--policy.push_to_hub=false`。想推到 Hub 时再设 `--policy.push_to_hub=true --policy.repo_id=<HF_user>/<name>`。

**判读烟测日志**:
- 启动到第 1 step 用了多久(预期 < 3 分钟,主要是初始化 + 视频预取)
- 50 步后 `train/loss` 数字是否合理(预期 0.1–1.0)
- 有无 `RuntimeError`(尤其是 `CUDA OOM` / `shape mismatch` / `KeyError`)
- grad_norm 是否 < 100(没爆炸)

**不通过的处理**:
- **OOM**: 把 `batch_size=1` 重跑,如果还 OOM,加 `--policy.gradient_checkpointing=true`(ACT 不一定有,可能要在 config 里加)或减少相机。
- **shape mismatch**: 数据集与 ACT 默认的 `input_features` 不兼容,先看日志里的 key 名。
- **loss=NaN**: 学习率过高,加 `--policy.optimizer_lr=1e-5`(默认更高)。ACT 默认 lr 通常是 1e-5,这里出问题通常是 bf16/fp16 数值问题,加 `--policy.dtype=float32`。

---

## Phase 3 — 正式训练(过夜,6–10 小时)

### 3.1 选 batch_size

| GPU 显存 | batch_size | 备注 |
|---|---|---|
| 24 GB | 2 | 保守,稳 |
| 40 GB | 4 | **推荐 (RTX 4090 48 GB 可用)** |
| 80 GB | 8 | 快 |

数据集 21k 帧,batch=4 → 5,273 step/epoch。

### 3.2 选训练步数

数据集不大,3 个 epoch 起步:
- **50k 步** (≈ 4 epoch): 第一次跑看 baseline
- **100k 步** (≈ 8 epoch): 组长期望的"够训"档
- **150k+ 步** (≈ 12+ epoch): 只在 loss 还在降时继续

### 3.3 启动命令(直接用)

```bash
DATASET_ROOT=/home/zzx23457/lerobot/datasets/26-06-17-11-32-27_v2
RUN_DIR=outputs/train/act_v2_$(date +%Y%m%d_%H%M%S)

lerobot-train \
  --policy.type=act \
  --dataset.repo_id=local \
  --dataset.root=$DATASET_ROOT \
  --output_dir=$RUN_DIR \
  --job_name=act_v2 \
  --batch_size=4 \
  --steps=100000 \
  --eval_freq=10000 \
  --save_freq=10000 \
  --log_freq=50 \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=lerobot_act_v2
```

**关键参数说明**:
| 参数 | 值 | 含义 |
|---|---|---|
| `--policy.type=act` | 必填 | 选 ACT |
| `--dataset.repo_id=local` | 必填 | 本地数据集标识,任意字符串(常用 "local") |
| `--dataset.root` | 必填 | 数据集根目录(配合 `repo_id=local` 读本地) |
| `--output_dir` | 必填 | checkpoint + 日志目录 |
| `--batch_size` | 4 | 起步安全值 |
| `--steps` | 200000 | 训练总步 |
| `--eval_freq` | 20000 | 每 20k 步跑一次 eval(暂可设很大跳过) |
| `--save_freq` | 20000 | 每 20k 步存一个 checkpoint,11 个 |
| `--log_freq` | 50 | wandb / 终端每 50 步打一次 |
| `--policy.device=cuda` | 必填 | 不写会让数据 loader 跑在 CPU 上、模型默认 cuda |
| `--wandb.enable=true` | 推荐 | 把曲线推到云端,断网也不影响本地 log |

### 3.4 (可选)后台跑 + 用 tail 监控

```bash
# 启动
nohup lerobot-train ... > logs/act_$(date +%Y%m%d).log 2>&1 &

# 监控
tail -f logs/act_20260611.log | grep -E "loss|step|grad_norm"
# 或:在 wandb 网页看曲线
```

### 3.5 监控要点

训练中盯 3 个数:

| 指标 | 范围 | 异常 → 处理 |
|---|---|---|
| `train/loss` | 头 10k 步下降到 0.05–0.2,后续 0.02–0.1 | 不降 → 数据问题;不收敛 → lr 调小 10× |
| `train/grad_norm` | < 5 正常,5–10 偏高,> 10 要干预 | 加 `--grad_clip_norm=1.0`(默认就有) 或降 lr |
| GPU 显存 | 训练中 < 80% 满 | 80%+ 持续 → 降 batch_size |

---

## Phase 4 — 训练后评估

### 4.1 选 checkpoint

`output_dir` 下会有 `checkpoints/step_20000/`, `step_40000`, ..., `last/`。

- `last/` 是最终 step 的
- 中间 step 通常不是"最好的"(noise 大),看 loss 曲线找**最低点对应 step**

```bash
# 列 checkpoints
ls outputs/train/act_v2_*/checkpoints/

# 挑一个最佳(假设 step_140000 loss 最低)
CKPT=outputs/train/act_v2_*/checkpoints/step_140000
```

### 4.2 在仿真 env 上 eval(如果有)

```bash
lerobot-eval \
  --policy.path=$CKPT \
  --env.type=aloha \
  --eval.n_episodes=20 \
  --eval.batch_size=1
```

### 4.3 真实机械臂上回放(replay 验证数据流)

```bash
# 用训练集前几个 episode 在真机上回放,看动作是否能复现
lerobot-replay \
  --robot.type=<你的 robot type> \
  --robot.port=/dev/ttyACM0 \
  --dataset.repo_id=local \
  --dataset.root=$DATASET_ROOT \
  --dataset.episode=0
```

### 4.4 (如果只有 ACT 模型,没有 env)用 lerobot-rollout 做 rollout

`lerobot-rollout.py` 在 [src/lerobot/scripts/lerobot_rollout.py](src/lerobot/scripts/lerobot_rollout.py)。

---

## Phase 5 — 故障排查(决策树)

### 5.1 训练中

| 症状 | 原因 | 处理 |
|---|---|---|
| `CUDA out of memory` | batch/视频太大 | 降 `batch_size`;检查相机分辨率(540×960 + 3 路很重) |
| `loss=NaN` 第一步 | lr 过高 / dtype 不稳 | 降 lr 10×, `--policy.dtype=float32` |
| 1000 步 loss 还不动 | 数据 stat 错 / 标签错 | 跑 Phase 1 校验;打印一个 batch 看看 |
| 训练中 loss 反弹 | lr 太高或过拟合 | 看 `grad_norm`,降 lr;或提早停 |
| wandb 报 `Connection refused` | 没登录 / 离线 | 加 `WANDB_MODE=offline`,训完再 `wandb sync` |
| 视频解码卡死 | 视频太大 / ffmpeg 线程冲突 | 加 `--dataset.video_backend=pyav` 或 `decord` |

### 5.2 训练后

| 症状 | 原因 | 处理 |
|---|---|---|
| 真机动作抖动 | 数据有噪声 / 模型没收敛 | 加 `--batch_size=8` 训更多步;或加 EMA 权重 |
| 任务成功率 0% | task 描述对不上 / 单位错 | 复检 `tasks.parquet`;确认度/弧度 |
| 训得快但泛化差 | 过拟合 | 加图像增广(--policy.use_aug=true,看 ACT config 是否支持);多训几轮 |

### 5.3 工程类

| 症状 | 处理 |
|---|---|
| `output_dir` 满了 | 加 `--output_dir=新路径`,旧 checkpoint 归档 |
| 断电 / 训练中断 | 加 `--resume=true` 从 `last/` 继续 |
| 想换 lr 重跑 | 在新 `output_dir` 跑,旧数据保留 |

---

## 附录:本工作流假设的目录

```
/home/zzx23457/lerobot/
├── datasets/                                   # ← 数据集根目录
│   └── 26-06-17-11-32-27_v2/  # 训练用数据集(只读)
├── workflows/                                  # ← 本工作流
│   ├── model_training/                         # 模型训练
│   │   ├── README.md
│   │   ├── act_training_workflow.md            # ← 当前文档
│   │   └── train_act.sh                        # 一键脚本
│   └── data_processing/                        # 数据处理
│       ├── v2_convert.py + v2_convert_config.json
│       └── sanity_check.py                     # 校验脚本
└── outputs/                                    # 训练输出(自动创建)
    └── train/
        ├── act_smoke/                          # 烟测
        └── act_v2_20260617_xxxxxx/             # 正式训练
            ├── checkpoints/
            ├── logs/
            └── wandb/
```

## 附录:本工作流中所有的 lerobot CLI 命令

- `lerobot-train` — 训练
- `lerobot-dataset-viz` — 可视化数据
- `lerobot-eval` — 仿真评估
- `lerobot-replay` — 真实机械臂回放
- `lerobot-rollout` — 真实环境 rollout

具体见 [src/lerobot/scripts/](src/lerobot/scripts/)。
