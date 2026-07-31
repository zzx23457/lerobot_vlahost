# `workflows/` 工作流使用手册

> **目标读者**：刚加入项目的新同事。
> **核心结论**：**3 个脚本**就能跑通"采集→训练→部署"全部流程，其他都是被这三个脚本调用的子模块。

---

## 0. 5 分钟上手（只看这一章就够了）

### 0.1 三个核心脚本

| 步骤 | 脚本 | 一句话 |
|---|---|---|
| 数据转换 | `workflows/data_processing/v2_convert_next_joint.py` | 把外部采集器输出的 v1 格式数据集 → 内部 v2 格式（**度**、16 维 action/state） |
| 训练 | `workflows/model_training/train_act.sh` | 一键训练 ACT，自动跑环境检查 → 数据校验 → 50 步烟测 → 正式训练 |
| 部署 | `workflows/robot_interaction/deploy.py` | 把训好的策略推上真机；顺便能做回放 / 相机预览 |



### 0.2 完整流水线（3 步）

```bash
# ==== 步骤 1：把 v1 数据集转成 v2 ====
# 改 workflows/data_processing/v2_convert_config.json 的 datasets 列表后:
python workflows/data_processing/v2_convert_next_joint.py
# 产出：datasets/<原名>_v2/

# ==== 步骤 2：训练（一键 5 phase） ====
DATASET_ROOT=datasets/<原名>_v2 \
BATCH_SIZE=4 STEPS=100000 WANDB_ENABLE=false \
    ./workflows/model_training/train_act.sh train
# 自动做的事：
#   ✓ 检查 GPU / ffmpeg / lerobot-train
#   ✓ 检查数据集目录结构 + info.json + 交叉一致性
#   ✓ 跑 sanity_check.py（视频解码、stats、ACT forward）
#   ✓ 检查时间戳对齐（脏的话让你选：过滤/容忍/取消）
#   ✓ 检测脏数据 episode（前 7 位全零；脏的话让你选：自动清洗/跳过/取消）
#   ✓ 跑 50 步烟测（确认整条管线通）
#   ✓ 正式训练 → checkpoint 写到 outputs/train/act_<tag>_<时间戳>/checkpoints/

# ==== 步骤 3：部署到真机 ====
CKPT=$(cat outputs/train/.last_act_run)/checkpoints/last/pretrained_model
python workflows/robot_interaction/deploy.py --policy-path "$CKPT"
# 自动做的事：
#   ✓ 加载 deploy_config_chunk.yaml 默认配置（chunk 模式，30Hz）
#   ✓ 用 wrapper 子进程跑 lerobot-rollout（自动处理 SIGTERM/僵尸清理）
#   ✓ Ctrl+C 退出时自动把机械臂送回 home 位置并断连
```

**至此，端到端流水线就跑完了。**

### 0.3 想看更细的？

- **训练做了什么**：继续看 [§3.2 train_act.sh](#32-train_actsh--act-一键训练-推荐)
- **部署做了什么**：继续看 [§3.3 deploy.py](#33-deploypy--策略部署主入口)
- **想用浏览器控制台（部署/回放/相机/数据处理/训练全包）**：看 [§3.4 UI 控制台（Gradio）](#34-ui-控制台gradio可选)
- **v1 → v2 转换做了什么**：继续看 [§3.1 v2_convert_next_joint.py](#31-v2_convert_next_jointpy--v1--v2-schema-转换)
- **要换数据集 / 换模型 / 改频率**：看 [§6 参数替换速查表](#6-参数替换速查表)
- **遇到报错**：看 [§7 常见故障排查](#7-常见故障排查)
- **其它子模块的细节**：看 [§5 其他子模块参考](#5-其他子模块参考被集成的子模块)

---

## 1. 整体架构与目录全景（背景知识）

### 1.1 全局架构图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Marvain M6 HTTP Server                          │
│                    http://192.168.10.123:8010                             │
│  GET /state  →  joint_states.positions + gripper_* + quad_image          │
│  POST /action ←  {jointcmd_left, jointcmd_right, gripper_left, _right}   │
│  POST /action_chunk ← 整 chunk 批量下发                                   │
└──────────────────────────────────────────────────────────────────────────┘
                 ▲                  ▲                  ▲
                 │ observations     │ actions          │ actions
                 │                  │                  │
   ┌─────────────┴─────┐   ┌────────┴────────┐  ┌───────┴──────────┐
   │ MarvainM6Http    │   │ deploy.py      │  │ replay.py        │
   │ Robot (driver)   │   │ (policy 推理)   │  │ (重放示教动作)   │
   └──────────────────┘   └─────────────────┘  └──────────────────┘
            ▲                       ▲                  ▲
            │                       │                  │
            │  deploy.py / replay.py / show_cameras.py 通过 --config 读 yaml
            │  ┌──────────────────────────────────────────────────┐
            │  │ workflows/robot_interaction/ui/ (Gradio 控制台) │
            │  └──────────────────────────────────────────────────┘
            └─────────────────────────────────────────────────────

训练侧（独立 GPU 机器）:
   datasets/v1  ──►  v2_convert_next_joint.py  ──►  datasets/v2
                                                        │
                                                        ▼
                                              train_act.sh (含 sanity_check/
                                                  clean_dirty_episodes/
                                                  check_timestamp_alignment)
                                                        │
                                                        ▼
                                              outputs/train/<run>/checkpoints/
                                                        │
                                                        ▼
                                              deploy.py --policy-path=...
```

### 1.2 目录结构（按"被谁调用"分组）

```
workflows/
├── _config_loader.py             ← 共享配置加载器（deploy.py 等 wrapper 引用）
├── _robot_home.py                ← deploy/replay 退出钩子（deploy.py 自动调用）
├── _robot_home_config.py         ← home 姿态中心定义（_robot_home.py 引用）
├── get_robot_state.py            ← 【可选工具】查机器人当前状态
├── arm_control_http.py           ← 【可选工具】HTTP 交互式控制（手动调位置）
├── quickstart.sh                 ← 【可选工具】5 步环境自检
│
├── robot_interaction/            ← 【核心：真机交互】
│   ├── deploy.py                 ← ★ 部署策略主入口
│   ├── replay.py                 ← ★ 数据集回放（deploy.py 也会用到 send_action_chunk）
│   ├── replay_chunk.py           ← ★ chunk 模式回放
│   ├── show_cameras.py           ← ★ 相机预览（deploy.py --show-cameras 时自动启）
│   ├── capture_snapshot.py       ← 【可选】状态 + 图像截取
│   ├── mock_echo_server.py       ← 【可选】假机器人（无真机调试）
│   ├── deploy_config_chunk.yaml  ← ★ deploy.py 默认配置模板
│   ├── deploy_config.yaml        ← ★ sync 模式配置模板
│   ├── deploy_config_hybrid.yaml ← ★ Hybrid 配置模板
│   ├── replay_config.yaml        ← ★ replay.py 默认配置模板
│   ├── replay_chunk_config.yaml  ← ★ replay_chunk.py 默认配置模板
│   └── ui/                       ← 【可选】Gradio Web 控制台（包装 deploy/replay/show_cameras）
│
├── data_processing/              ← 【核心：数据转换】+ 【被 train_act.sh 调用】
│   ├── v2_convert_next_joint.py  ← ★ v1 → v2 schema 转换（使用 NEXT timestep joint_pos 作为 arm_cmd）
│   ├── v2_convert.py             ← 【旧版】同上，但 arm_cmd 来自 v1_actions[:, 14:28]（保留参考）
│   ├── v2_convert_config.json    ← ★ v2_convert_* 的输入配置
│   ├── v2_convert_next_joint.py  ← ★ 同 v2_convert_next_joint.py（见下说明）
│   ├── sanity_check.py           ← ★ train_act.sh Phase 1 调用
│   ├── check_timestamp_alignment.py ← ★ train_act.sh Phase 1 调用
│   ├── clean_dirty_episodes.py   ← ★ train_act.sh Phase 1 调用
│   ├── example_clean_dataset.py  ← 【示例】clean_dirty_episodes.py 的 4 步使用
│   ├── merge_two_datasets.py     ← 【可选】合并两个 LeRobot v3 数据集
│   ├── README.md / QUICKREF.md / 数据清洗分析总结.md
│
└── model_training/               ← 【核心：训练入口】
    ├── train_act.sh              ← ★ ACT 一键训练（5 phase）
    ├── train_act_simple.sh       ← 【可选】ACT 纯训练（无任何校验）
    ├── train_smolvla.sh          ← ★ SmolVLA 一键训练（结构与 train_act.sh 类似）
    ├── finetune_act.sh           ← 【可选】从已有 ckpt 继续 fine-tune
    ├── run_three_datasets.sh     ← 【可选】临时脚本：串行跑多个数据集
    ├── act_training_workflow.md  ← train_act.sh 5 phase 详解
    └── DIRTY_DATA_INTEGRATION.md ← train_act.sh 与 clean_dirty_episodes.py 集成说明
```

> ★ = 核心脚本 / 你会直接用到的  
> 【可选】= 特殊场景才用得到（无真机调试 / 手动调位置 / Gradio UI / fine-tune / 批量跑）  
> 其余 = 被集成的子模块，train_act.sh / deploy.py 帮你自动调用

---

## 2. 环境准备

### 2.1 必备依赖

| 依赖 | 用途 | 安装 |
|---|---|---|
| Python 3.10+ | 解释器 | 系统包 / conda |
| `lerobot` | 主框架 | `uv sync --locked` |
| `requests` | HTTP 客户端 | `pip install requests` |
| `numpy`, `opencv-python` | 图像处理 | `uv sync --locked` |
| `pyyaml` | yaml 解析 | `pip install pyyaml` |
| `ffmpeg` | 视频解码（训练侧必须） | `apt install ffmpeg` 或 `conda install ffmpeg` |
| `pyarrow`, `pandas` | parquet 处理 | `uv sync --locked` |
| `gradio>=4.0,<6.0` | UI（可选） | `uv sync --extra ui` |
| `fastapi`, `uvicorn` | mock echo server（可选） | `pip install fastapi uvicorn` |
| `transformers`, `accelerate`, `num2words` | SmolVLA 训练（可选） | `uv sync --extra smolvla` |
| `wandb` | 训练可视化（可选） | `pip install wandb` |

### 2.2 一键自检

```bash
cd /home/zzx23457/lerobot_vlahost
bash workflows/quickstart.sh
```

`quickstart.sh` 依次检查：Python 版本、curl、网络连接、目录完整性、依赖 import。全部 ✓ 才会提示"环境已就绪"。

### 2.3 真机连通性测试（部署前必做）

```bash
# 1) HTTP 服务端必须能返回 joint_states.positions
curl http://192.168.10.123:8010/state | python3 -m json.tool | head -30
# 期望看到：
# {
#   "joint_states": {"positions": [14 floats, radians], ...},
#   "gripper_left":  ...,
#   "gripper_right": ...,
#   "quad_image":    {"format":"jpeg","data":"..."} 或 {"stream_url":"..."}
# }

# 2) 驱动器模块能正常 import
uv run python -c "from lerobot.robots.marvain_m6_http import MarvainM6HttpRobotConfig, MarvainM6HttpRobot; print('import ok')"
```

如果还没真机，先用 mock 跑通：

```bash
# 终端 1：起假机器人
python workflows/robot_interaction/mock_echo_server.py --port 8010

# 终端 2：deploy 指到 mock
python workflows/robot_interaction/deploy.py --http-base-url http://127.0.0.1:8010 --fps 2
```

---

## 3. 三件套详解

### 3.1 `v2_convert_next_joint.py` — v1 → v2 schema 转换

**功能**：把外部采集器（KM converter）输出的 v1 格式 LeRobot v3 数据集，转成内部统一的 v2 schema。

**与 `v2_convert.py` 的区别**：

| 版本 | arm_cmd 来源 |
|---|---|
| `v2_convert.py`（旧） | `v1_actions[:, 14:28]`（直接用 action 字段的中间 14 维） |
| **`v2_convert_next_joint.py`（推荐）** | **下一帧的 `joint_pos`（v1_actions[:, 42:56]，按 episode 内前向 shift 一次）** |

为什么推荐 `_next_joint` 版本：因为 v1 里 `action[:, 14:28]` 是上一帧的关节角（前馈 / 预测），用作训练 action 会有 1 步相位差；用**下一帧的关节角**作为 action 更接近"控制指令到达后机器人的目标位姿"。

**使用流程**：

```bash
# 步骤 1：编辑配置文件，告诉脚本要转哪些数据集
cat workflows/data_processing/v2_convert_config.json
# {
#   "v2_suffix": "_v2",          # 输出目录后缀
#   "cameras": [0, 1, 1, 1],     # [left_eye, right_eye, left_wrist, right_wrist] 哪个 drop
#   "datasets": ["26-07-01-14-37-04"]  # ← 这里改：要转的 v1 数据集名
# }
# 输出路径：datasets/<v1 名><v2_suffix>/，例如 datasets/26-07-01-14-37-04_v2/

# 步骤 2：跑转换
python workflows/data_processing/v2_convert_next_joint.py
```

**关键转换规则**：

- `action` (56,) → (16,)：arm_command_NEXT (14) + left_grip_next (1) + right_grip_next (1)
- `observation.state` (26,) → (16,)：joint_pos (14) + left_grip_angle (1) + right_grip_angle (1)
- 弧度 → 角度（× 180/π）
- 新增 `action_is_pad`（bool，全 False）
- **视频软链接**（不复制实体，省空间）—— 这意味着**v1 数据集不能删**，否则 v2 视频会断链
- 通过 config `cameras` 数组（按 [left_eye, right_eye, left_wrist, right_wrist] 顺序）选择要丢弃的相机

**安全保证**：

- v1 数据集永远不会被改动；
- rollback 一个数据集就是 `rm -rf <v2_dir>`；
- 输出目录如果已存在会跳过，避免误覆盖。

**何时需要重跑**：改了 v2_convert_config.json 的 datasets / cameras / v2_suffix 时。

---

### 3.2 `train_act.sh` — ACT 一键训练（推荐）

**功能**：分 5 个 phase 的端到端 ACT 训练流程——环境检查 → 数据校验 → 50 步烟测 → 正式训练 → 评估提示。**Phase 1 自动串联了 3 个子模块**（sanity_check / check_timestamp_alignment / clean_dirty_episodes）。

**使用流程**：

```bash
cd /home/zzx23457/lerobot_vlahost

# 完整 5 phase（一键跑完）
DATASET_ROOT=datasets/<你的数据集> \
    ./workflows/model_training/train_act.sh all

# 只跑某个 phase
./workflows/model_training/train_act.sh smoke    # 只跑烟测
./workflows/model_training/train_act.sh train    # 只跑正式训练（跳过校验）
./workflows/model_training/train_act.sh check    # 只跑数据校验
./workflows/model_training/train_act.sh eval     # 只打印评估命令
```

**可调环境变量**：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DATASET_ROOT` | `$PROJECT_ROOT/datasets/26-07-01-14-37-04_0111_v2` | 数据集路径 |
| `OUTPUT_ROOT` | `$PROJECT_ROOT/outputs/train` | checkpoint 输出根 |
| `BATCH_SIZE` | `2` | 起步安全值；24GB 显存可调到 4 |
| `STEPS` | `100000` | 训练总步数 |
| `EVAL_FREQ` | `10000` | 每 N 步跑一次 eval |
| `SAVE_FREQ` | `10000` | 每 N 步存一个 checkpoint |
| `LOG_FREQ` | `50` | wandb / 终端日志频率 |
| `WANDB_PROJECT` | `202607172057` | wandb 项目名 |
| `WANDB_ENABLE` | `false` | 是否启用 wandb |
| `PUSH_TO_HUB` | `false` | 是否推到 HuggingFace Hub |

**典型自定义**：

```bash
# 改数据集 + batch + 步数
DATASET_ROOT=/path/to/my_dataset_v2 BATCH_SIZE=4 STEPS=150000 \
    ./workflows/model_training/train_act.sh all

# 启用 wandb
WANDB_ENABLE=true WANDB_PROJECT=my_experiment \
    ./workflows/model_training/train_act.sh all

# 想用旧的纯训练版（无任何校验；已知数据集干净时用）
./workflows/model_training/train_act_simple.sh
```

**Phase 1 自动做的 6 件事**（train_act.sh 内部调用）：

1. **环境检查**（GPU / ffmpeg / lerobot-train）；
2. **数据集目录 + info.json 字段 + 交叉一致性**（info ↔ stats ↔ videos ↔ episodes parquet）；
3. **跑 `sanity_check.py`**（视频解码、stats 数值、ACT forward 占位）；
4. **跑 `check_timestamp_alignment.py`**—— 有 dirty episode 时弹选项让你选：
   - `1`：自动提取 CLEAN episode 列表，后续只训这部分；
   - `2`：使用全部数据 + `tolerance_s=1`（容忍更大漂移；视觉-本体会有错位）；
   - `3`：取消训练。
5. **跑 `clean_dirty_episodes.py --report-only`**—— 有脏数据时弹选项：
   - `1`：自动跑清洗脚本到 `<DATASET_ROOT>_cleaned`，并切换 DATASET_ROOT；
   - `2`：跳过清洗继续（不推荐）；
   - `3`：取消训练，打印手动清洗命令。
6. 完成。

**output_dir 规则**：`$OUTPUT_ROOT/act_<数据集 tag>_<时间戳>/`，自动加数据集后缀避免混淆。

**最后路径**写在 `$OUTPUT_ROOT/.last_act_run`，方便后续脚本读：

```bash
cat outputs/train/.last_act_run
# → /home/zzx23457/lerobot_vlahost/outputs/train/act_v2_20260717_205342
```

---

### 3.3 `deploy.py` — 策略部署主入口

**功能**：把训练好的策略部署到真机推理。Wrapper 负责：解析 yaml、覆盖 CLI 参数、找 `lerobot-rollout` 子进程、传递额外 flag、可选启动 `show_cameras.py` 子进程、退出后通过 `_robot_home.py` 送 home。

**使用流程**：

```bash
cd /home/zzx23457/lerobot_vlahost

# 最简：默认配置（chunk 模式 30Hz 部署到 192.168.10.123:8010）
python workflows/robot_interaction/deploy.py

# 换模型
python workflows/robot_interaction/deploy.py \
    --policy-path outputs/train/act_v2/checkpoints/140000/pretrained_model

# 慢速推理（调试）
python workflows/robot_interaction/deploy.py --fps 5

# 录制数据（sentry 策略：边推理边录）
python workflows/robot_interaction/deploy.py --strategy sentry

# RTC 模式
python workflows/robot_interaction/deploy.py --inference-type rtc --execution-horizon 8

# 打开相机预览窗口（同时 deploy + 实时看 4 路相机）
python workflows/robot_interaction/deploy.py --show-cameras

# 物理相机名 → 策略期望名的映射（VLA 模型必须）
python workflows/robot_interaction/deploy.py \
    --rename-map '{"left_eye":"camera1","left_wrist":"camera2","right_wrist":"camera3"}'
```

**关键 CLI 参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config PATH` | `deploy_config_chunk.yaml` | 配置文件 |
| `--policy-path PATH` | yaml 中 `policy.path` | 覆盖模型路径 |
| `--http-base-url URL` | yaml 中 `robot.http_base_url` | HTTP 服务端地址 |
| `--robot-id ID` | yaml 中 `robot.id` | 机器人实例 ID |
| `--safety-stats-path PATH` | yaml 中 `robot.safety_stats_path` | 训练数据集路径（启用数据驱动安全裁剪） |
| `--fps N` | yaml 中 `inference.fps` | 推理频率 |
| `--strategy {base,sentry,highlight,dagger,episodic}` | `base` | 推理+录制策略 |
| `--inference-type {sync,rtc,chunk}` | yaml 中 `inference.type` | 推理模式 |
| `--execution-horizon N` | yaml 中 `inference.rtc.execution_horizon` | RTC 模式每次执行步数 |
| `--max-guidance-weight W` | `10.0` | RTC 模式 VLA 引导权重 |
| `--duration SEC` | `0.0` | 运行时长（0=无限） |
| `--interpolation-multiplier N` | `1` | 动作插值倍数（chunk 模式忽略） |
| `--return-to-initial BOOL` | `true` | 结束后送 home |
| `--use-torch-compile` | `false` | 启用 torch.compile |
| `--rename-map JSON` | yaml 中 `rename_map` | 相机名重映射（VLA 必需） |
| `--show-cameras` | yaml 中 `inference.show_cameras` | 启动相机预览子进程 |
| `--cameras-override NAME ...` | — | 跳过 model 自动推导，显式指定相机 |

**核心参数替换指南**：

| 想做的事 | 改哪里 |
|---|---|
| 换模型 | `--policy-path <新 ckpt>` 或 yaml `policy.path` |
| 换机器人 | `--http-base-url <新 URL>` |
| 换推理频率 | `--fps <N>` 或 yaml `inference.fps` |
| 切推理模式（sync/rtc/chunk） | `--inference-type <mode>` 或 yaml `inference.type` |
| 启用/关闭安全裁剪 | yaml `robot.safety_stats_path` 指向训练数据集；`robot.action_clip_margin_deg` 调整裕量 |
| 录制数据 | `--strategy sentry` 等 + yaml `dataset.repo_id` / `dataset.root` |
| 打开相机预览 | `--show-cameras` 或 yaml `inference.show_cameras: true` |
| VLA 任务描述 | yaml `dataset.single_task: "..."` |

**进程管理（不容易被注意到但很重要）**：

- 子进程 `setsid()` + `PR_SET_PDEATHSIG = SIGTERM`：当 wrapper 被 SIGKILL 杀掉时，kernel 会自动 SIGTERM 子进程，保证 `try/finally: robot.disconnect()` 能跑完 → 不会留下"机械臂还通电但没人控制"的危险状态；
- wrapper 自己注册了 SIGINT/SIGTERM/SIGHUP 转发，用 `os.killpg` 发到子进程组，连同子进程 fork 出来的子进程一并杀掉。

**退出时 `_robot_home.py` 自动做的事**：

1. 优先调 `robot.go_home()`（走 ROS 服务）；
2. 失败则回退：连续 60 次 `@30Hz` 发送 home action（≈2 秒）；
3. 断连（HTTP 路径不下使能，机械臂仍通电；Hybrid 路径会下使能）；
4. 任意失败只打 warning，不影响 deploy 退出码。

---

### 3.4 UI 控制台（Gradio，可选）

> 给"不想记命令"的同事用——把 `deploy.py` / `replay.py` / `show_cameras.py` /
> 5 个数据处理脚本 / 3 个训练脚本全部包到一个 Web 控制台里。
>
> **核心原则**：**YAML 是真相源**。UI 内部就是「把当前 YAML 写到
> `tempfile/robot_config_*.yaml` → 用 `--config <tmp>` 调对应脚本」。
> 表单只是结构化视图，可以直接在 YAML Tab 里改文本。

#### 3.4.1 安装与启动

```bash
# 一次性安装（仅 UI 依赖）
uv sync --extra ui

# 启动（会自动清掉 ALL_PROXY 等 socks 代理以免 gradio 报错）
python workflows/robot_interaction/ui/launch.py

# 或
python -m workflows.robot_interaction.ui.launch
```

启动后终端会打印：

```
🚀 启动 LeRobot 统一控制界面（yaml-centric）
📍 访问地址: http://localhost:7860
🌐 局域网访问: http://<本机IP>:7860
```

> UI 默认监听 `0.0.0.0:7860`，**局域网别的电脑也能访问**——给机器人旁边
> 没用过终端的同事留浏览器就行。

#### 3.4.2 5 种操作模式

顶部"操作模式"切换：

| 模式 | 底层脚本 | 适合场景 |
|---|---|---|
| **部署** | `deploy.py --config <tmp>` | 训好的策略上真机（ACT / SmolVLA / 其他） |
| **回放** | `replay.py --config <tmp>` | 数据集里某条 episode 在真机上重放 |
| **相机预览** | `show_cameras.py --config <tmp>` | 不带 deploy 的相机实时预览（4 路切分） |
| **数据处理** | 5 个脚本分派 | 烟测 / 清洗 / 合并 / 时间戳对齐 / v2 转换 |
| **模型训练** | `train_act.sh` / `train_smolvla.sh` / `finetune_act.sh` | 在浏览器里启动训练 |

切换模式时，**"加载预设配置"下拉菜单会自动过滤**——只显示该模式可用的 yaml 模板与用户保存的预设。

#### 3.4.3 顶部控件（所有模式通用）

```
┌─────────────────────────┬──────────────────────────────────────┐
│ 操作模式  [部署▼]       │ 加载预设配置 [deploy_config_chunk▼] │
│                         │ 预设名称 [_________________]         │
│                         │ [💾 保存预设] [📥 导出 YAML]        │
└─────────────────────────┴──────────────────────────────────────┘
```

- **加载预设配置**：下拉里 = `workflows/robot_interaction/` 下所有 `.yaml`
  模板（`deploy_config_chunk` / `deploy_config_hybrid` / `replay_config` …）
  + 你之前用"💾 保存预设"存到 `ui/presets/<kind>/` 的自定义预设
  （带"(预设)"后缀）。切换操作模式时自动重新过滤。
- **💾 保存预设**：把当前 YAML 写到 `ui/presets/<kind>/<name>.yaml`，
  重启 UI 后会自动出现在下拉里。
- **📥 导出 YAML**：把当前最终配置复制到一个只读文本框，
  方便贴到 issue / commit message。

#### 3.4.4 部署模式详解

适用：训好的策略上真机推理。

**典型流程：部署 ACT**

1. 顶部模式选 **"部署"**；
2. "加载预设配置"下拉选 `deploy_config_chunk`（开环 chunk 模式，默认）；
3. **策略设置** 面板：
   - "选择训练目录"下拉选 `act_v2_20260701_181934` 之类的 run；
   - "选择 Checkpoint" 下拉选 `190000` / `last` 等；
   - 模型路径会自动填到 `policy.path`（也可手动覆盖）；
   - 推理设备默认 `cuda`。
4. **机器人设置** 面板：确认 `http_base_url = http://192.168.10.123:8010` /
   `robot.id` 与训练时一致；
5. **推理设置** 面板（默认折叠）：
   - `FPS`：默认 30；
   - `Strategy`：base / sentry（边推理边录）/ highlight / dagger / episodic；
   - `Inference Type`：sync / rtc / chunk；
   - 选 `rtc` 时，下方 RTC 子面板（Execution Horizon / Max Guidance Weight）自动展开；
   - `Show Camera Windows`：勾选 → 同 deploy 一起拉起 `show_cameras.py`；
   - `Camera Rename Map (Advanced)`：VLA 模型必填
     （如 `{"right_eye":"camera1", "left_wrist":"camera2", "right_wrist":"camera3"}`）。
6. **数据集设置（部署）** 面板（默认隐藏）：VLA 模型必填 `single_task`；
   `sentry` 等录制策略必填 `repo_id` / `root`。
7. 点 **🚀 启动** → UI 内部 = `deploy.py --config <tempfile>`；
   下方"实时日志"面板开始滚。
8. 点 **🛑 停止** → 杀进程组 + `_robot_home.py` 自动送 home。

**典型流程：部署 SmolVLA**

跟 ACT 几乎一样，区别只有：

- "Camera Rename Map (Advanced)"必填且通常更长；
- "数据集设置（部署）"必须填 `single_task`（自然语言任务描述）。
- `inference.type` 一般用 `rtc` 或 `chunk`。

> **YAML Tab vs 表单 Tab**：表单只是结构化视图。改完表单要
> 点 **"🔄 从表单刷新 → YAML"** 才会写入；改完 YAML 要点
> **"📋 YAML → 应用到表单"** 才会同步到表单。直接点启动会用 YAML 当前内容。

#### 3.4.5 回放模式详解

适用：把数据集里某条 episode 的动作在真机上重放。

1. 模式切到 **"回放"**；
2. 加载 `replay_config`；
3. YAML Tab 里改：
   - `dataset.repo_id`（或 `dataset.root` 走本地路径）；
   - `dataset.episode`（要回放的 episode 编号）；
   - `inference.fps`（回放频率）；
   - `return_to_initial_position`（退出后是否送 home）。
4. 点 **🚀 启动** → 调 `replay.py`。

#### 3.4.6 相机预览模式详解

适用：不带 deploy 的纯相机预览（4 路切分窗口）。

1. 模式切到 **"相机预览"**；
2. "相机设置"表单：选要显示的相机（默认 4 路全开）、FPS、窗口大小；
3. 点 **"🔄 从表单刷新 → YAML"** → 写入 yaml；
4. 点 **🚀 启动** → 调 `show_cameras.py`。
   - 物理相机名 / FPS / 窗口大小走 CLI；
   - `robot.http_base_url` / `policy.path` 走 yaml（policy 用于推导相机列表）。

#### 3.4.7 数据处理模式详解

切到 **"数据处理"** 后，"操作"下拉选 5 种之一：

| 操作 | 底层脚本 | 关键参数 | 注意点 |
|---|---|---|---|
| **数据集烟测** | `sanity_check.py` | `n_samples`（默认 5） | 退出码 0 = 通过；1 = 有错（请勿继续训练） |
| **清洗脏 Episode** | `clean_dirty_episodes.py` | `dry_run`（默认勾选）/ `report_only` / `zero_threshold` | **默认 dry-run**；输入与输出不能同路径 |
| **合并数据集** | `merge_two_datasets.py` | `source_roots_text`（每行一个，至少 2 个）/ `merge_repo_id` / `merge_video_size_mb` | **输出目录不能已存在**（防误覆盖） |
| **时间戳对齐检查** | `check_timestamp_alignment.py` | `video_key`（留空自动检测）/ `tolerance_ms` / `output_format` | 退出码 1 = 发现 drift（**业务结果，不是错误**，UI 视为成功） |
| **v2 Schema 转换** | `v2_convert.py` | `v2_variant`（standard / next_joint）/ 4 个相机复选框 / `v2_dry_run`（默认勾选） | v1 永不被修改；rollback = `rm -rf <v2_dir>` |

> **安全约束**：所有写操作（清洗 / 合并 / v2 转换）**默认 dry-run**。
> 实际写入前必须**手动取消勾选 Dry-run**——这是为了防止"点了启动就开干"。

#### 3.4.8 模型训练模式详解

切到 **"模型训练"** 后选训练脚本 + 阶段：

| 训练脚本 | 底层命令 | 专属参数 |
|---|---|---|
| **ACT** | `bash train_act.sh <phase>` | 通用超参（BATCH_SIZE / STEPS / EVAL_FREQ / SAVE_FREQ / LOG_FREQ）+ WANDB_ENABLE / PUSH_TO_HUB |
| **SmolVLA** | `bash train_smolvla.sh <phase>` | 同 ACT + POLICY_CHUNK_SIZE / POLICY_N_ACTION_STEPS / POLICY_LR / POLICY_PATH / LOAD_VLM_WEIGHTS / FREEZE_VISION_ENCODER / TRAIN_EXPERT_ONLY / HF_ENDPOINT / RENAME_MAP |
| **ACT Fine-tune** | `bash finetune_act.sh` | PRETRAINED_CKPT + NEW_DATASET（**没有 phase dispatch**，总是跑完整训练） |

| 阶段 | 含义 |
|---|---|
| `env` | 仅检查环境（GPU / 依赖），不启动训练 |
| `check` | 仅校验数据集（4 条规则），不启动 |
| `smoke` | 50 步烟测，建议第一次跑 |
| `train` | 正式训练（默认 400k 步） |
| `all` | env + check + smoke + train + eval（**一键 5 phase**） |
| `eval` | 仅打印评估命令（不评估模型） |

> **环境变量白名单**：训练通过 `bash` 子进程启动，环境变量走白名单注入
> （不会污染 Gradio 主进程）。所有训练日志写到
> `outputs/deploy_logs/model_training_<script>_<phase>_<ts>.log`。
>
> **点停止 = `killpg`**：会终止整个 bash 派生进程组（含 `lerobot-train` 后代）。

#### 3.4.9 YAML ↔ 表单 双向同步

UI 内部把 yaml 解析成 `UnifiedRobotConfig` dataclass（5 种 mode-aware），
再渲染成表单；改完表单再序列化成 yaml。

- **改完表单** → 必须点 **"🔄 从表单刷新 → YAML"** → 否则 yaml 不变；
- **改完 YAML** → 必须点 **"📋 YAML → 应用到表单"** → 否则表单与 yaml 不一致；
- **直接点 🚀 启动** → 用 YAML 当前内容（不看表单）。

`mode: deploy` vs `mode: replay` 的字段位置略有差异（`return_to_initial_position`
deploy 在 `inference.*`，replay 在 yaml 根）——UI 的 `save_yaml` / `load_yaml` 自动做
mode-aware 转换，无需手动调整。

#### 3.4.10 关键安全约束（架构层）

```
Gradio Web UI (app_zh.py)
   │  YAML 是真相源; 表单是结构化视图
   ▼
config_manager (yaml ⇄ dataclass 双向转换)
   │
   ▼
process_manager (写 yaml 到 tempfile + 启子进程)
   │  脚本路径 / 环境变量走白名单 (21 个 key)
   │  bash 训练用 setsid() 自建进程组, stop 用 killpg 杀整组
   ▼
deploy.py / replay.py / show_cameras.py / 5 数据处理脚本 / 3 训练脚本
```

- **白名单**：UI 不会注入任意命令或任意环境变量——脚本路径写死，训练 env 只注入
  21 个已批准的 key（`DATASET_ROOT` / `BATCH_SIZE` / `STEPS` / `WANDB_*` 等）。
- **资源隔离**：bash 训练用 `os.setsid()` 自建进程组，stop 用 `killpg` 杀整组。
- **dry-run 默认开**：清洗 / 合并 / v2 转换默认 `dry_run=True`。
- **YAML schema 不变**：旧 `deploy_config*.yaml` / `replay_config.yaml` 兼容加载（带 warning 容错过滤）。

#### 3.4.11 常见问题

**Q: UI 起不来？**

```bash
# 1) 检查 gradio 版本
pip list | grep gradio   # 需要 >=4.0,<6.0

# 2) socks 代理冲突（launch.py 已自动清，但有时环境变量还会被 set）
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY
python workflows/robot_interaction/ui/launch.py

# 3) 端口被占
lsof -i :7860   # 找到 PID 杀掉
```

**Q: 启动按钮按了但子脚本立刻挂了？**

看下方"实时日志"面板。常见：

- 模型路径不存在 → `FileNotFoundError`：检查"选择 Checkpoint"选了没；
- 机器人 HTTP 不通 → `Connection refused`：`curl http://192.168.10.123:8010/state` 验证；
- VLA 模型没 `single_task` → 报 schema 校验错：填上任务描述。

**Q: 表单改了但 yaml 不动？**

忘了点 **"🔄 从表单刷新 → YAML"**。改完表单再点这个按钮才生效。

**Q: yaml 改了但表单不变？**

忘了点 **"📋 YAML → 应用到表单"**。

**Q: 想用别的电脑访问？**

UI 默认监听 `0.0.0.0`，**同局域网**浏览器访问 `http://<本机IP>:7860` 即可。
跨网段需要端口转发或 VPN。

**Q: 训练中点停止进程没退干净？**

先等 5–10 秒——`killpg` 是把整个 bash 派生进程组（含 `lerobot-train` 子代）一起终止，
有时 GPU 释放需要几秒。如果 30 秒后还没退，开终端 `ps -ef | grep lerobot-train` 找 PID。

---

## 4. 端到端范例（复制粘贴就能跑）

### 4.1 完整链路：v1 数据集 → 转换 → 训练 → 部署

```bash
cd /home/zzx23457/lerobot_vlahost

# ==== 步骤 1：v1 → v2 转换 ====
# 编辑 workflows/data_processing/v2_convert_config.json：
#   "datasets": ["<你的 v1 数据集名>"]
python workflows/data_processing/v2_convert_next_joint.py
# 产出 datasets/<原名>_v2/

# ==== 步骤 2：训练（一键 5 phase） ====
DATASET_ROOT=datasets/<原名>_v2 \
BATCH_SIZE=4 STEPS=100000 WANDB_ENABLE=true \
    ./workflows/model_training/train_act.sh all

# 训完会打印 checkpoint 路径，自动写进 outputs/train/.last_act_run

# ==== 步骤 3：部署到真机 ====
CKPT=$(cat outputs/train/.last_act_run)/checkpoints/last/pretrained_model

python workflows/robot_interaction/deploy.py \
    --policy-path "$CKPT" \
    --fps 30

# Ctrl+C 退出，自动送 home
```

### 4.2 不接真机的开发循环（mock server）

```bash
# 终端 1：起假机器人（自动打印客户端的 payload）
python workflows/robot_interaction/mock_echo_server.py --port 8010

# 终端 2：慢速 deploy + 相机预览
python workflows/robot_interaction/deploy.py \
    --http-base-url http://127.0.0.1:8010 \
    --fps 2 \
    --show-cameras
```

### 4.3 跨数据集 fine-tune

```bash
# 假设有 act_v2 (140000 steps) 训好的 ckpt
PRETRAINED=outputs/train/act_v2_20260716_163013/checkpoints/140000/pretrained_model

# 用新数据集的 v2 版本
NEW_DATASET=datasets/26-07-22-15-45-02_v2

# 一键 fine-tune（自动校验 state/action 维度匹配）
PRETRAINED_CKPT="$PRETRAINED" NEW_DATASET="$NEW_DATASET" \
    ./workflows/model_training/finetune_act.sh
```

### 4.4 UI 控制台（给非命令行同事用）

```bash
# 启 UI（一次性安装）
uv sync --extra ui
python workflows/robot_interaction/ui/launch.py
# 浏览器打开 http://localhost:7860

# 流程：
# 1. 顶部下拉加载 deploy_config_chunk 模板
# 2. YAML Tab 改 policy.path / dataset.repo_id
# 3. 点 🚀 启动（UI 内部就是 deploy.py --config <tmp>）
# 4. 看下方日志
# 5. 退出时点 🛑 Stop（自动送 home）
```

> UI 内置 **5 种模式**（部署 / 回放 / 相机预览 / 数据处理 / 模型训练），
> 完整用法见 [§3.4 UI 控制台（Gradio）](#34-ui-控制台gradio可选)。

### 4.5 SmolVLA 训练（结构与 ACT 类似）

```bash
# 默认用 lerobot/smolvla_base 官方预训练起点
./workflows/model_training/train_smolvla.sh all

# 改预训练起点
POLICY_PATH=lerobot/smolvla_base \
    ./workflows/model_training/train_smolvla.sh train

# 关闭相机重命名（VLA 默认带 camera1/2/3 → right_eye/left_wrist/right_wrist）
RENAME_MAP="" \
    ./workflows/model_training/train_smolvla.sh train

# 切国内 HF 镜像（首次下 smolvla_base ~1GB 必备）
HF_ENDPOINT=https://hf-mirror.com \
    ./workflows/model_training/train_smolvla.sh train
```

> **ACT vs SmolVLA 关键差异**：
>
> | 维度 | ACT | SmolVLA |
> |---|---|---|
> | 模型类型 | 纯回归 | VLA（需要 task 文本） |
> | `tasks.parquet` ≥1 | 否 | **是**（train_smolvla.sh Phase 1 校验） |
> | chunk_size / n_action_steps | 100 | 50 |
> | 默认学习率 | 1e-5 | 1e-4 |
> | 显存 | 较小 | 较大（建议 ≥24GB） |
> | 部署必须 | — | 传 `single_task` 字符串 |
> | 预训练起点 | 不需要 | `lerobot/smolvla_base` 推荐 |

---

## 5. 其他子模块参考（被集成的子模块）

> 这一章是给"想了解 train_act.sh / deploy.py 内部到底跑了什么"或"想绕过 wrapper 直接调子模块"的同事看的。如果你只是想跑通流水线，看完第 3、4 章就够了。

### 5.1 数据处理子模块

#### `sanity_check.py` — train_act.sh Phase 1 调用

**功能**：6 阶段烟测——目录结构、`info.json`、`stats.json`、`tasks.parquet`、`LeRobotDataset` 加载 + 取样本、ACT 策略 forward（CPU 占位）。

```bash
python workflows/data_processing/sanity_check.py \
    --dataset-root datasets/my_dataset \
    --n-samples 10
```

退出码：`0 = 全部通过`，`1 = 有错误（请勿继续训练）`。

#### `check_timestamp_alignment.py` — train_act.sh Phase 1 调用

**功能**：对比 data parquet 的 `timestamp` 列与视频最后一帧的实际 `pts`，找出有显著漂移的 episode（>1ms 默认阈值）。

```bash
# 文本报告
python workflows/data_processing/check_timestamp_alignment.py \
    --dataset-root datasets/my_data_v2

# JSON 格式（便于脚本二次处理）
python workflows/data_processing/check_timestamp_alignment.py \
    --dataset-root datasets/my_data_v2 --format json

# 保存 clean episode 列表
python workflows/data_processing/check_timestamp_alignment.py \
    --dataset-root datasets/my_data_v2 --output clean_episodes.txt
```

退出码：`0 = 全 clean`，`1 = 有 dirty`，`2 = 脚本错误`。

#### `clean_dirty_episodes.py` — train_act.sh Phase 1 调用

**功能**：检测并删除 `observation.state[:7]` 全部为 0 的整段 episode（前 7 位是 A 臂 7 个关节，全零表示该子系统失效），重新索引数据集。

```bash
# 仅生成报告（不动文件）
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/my_dataset --report-only

# Dry-run（模拟所有操作，但不写）
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/my_dataset --dry-run

# 输出到新位置（推荐）
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/my_dataset \
    --output-path datasets/my_dataset_cleaned

# 就地覆盖（覆盖前会问 y/N；先备份！）
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/my_dataset
```

清洗范围：

| 处理对象 | 动作 |
|---|---|
| `data/chunk-*/file-*.parquet` | 删除 dirty 行，重映射 `episode_index`、全局 `index` |
| `meta/episodes/chunk-*/file-*.parquet` | 删除 dirty 行，重映射 `episode_index`、重算 `dataset_from/to_index` |
| `meta/tasks.parquet` | 原样复制 |
| `videos/` | 整体复制（dirty episode 的视频仍存在，需手动清理） |

清洗后自动验证：episode 索引连续、全局 index 连续、无剩余 dirty、行数匹配。

#### `merge_two_datasets.py` — 直接调用（不被 train_act.sh 集成）

**功能**：用 LeRobot 自带的 `merge_datasets()` 把两个 LeRobot v3 数据集合并成一个。

修改顶部 `SRC_A` / `SRC_B` / `DST` / `REPO_ID` 四个常量后直接 `python` 跑。

#### `v2_convert.py` — `v2_convert_next_joint.py` 的旧版

**区别**：arm_cmd 来自 `v1_actions[:, 14:28]`（不是下一帧 joint_pos）。保留作为参考。

#### `example_clean_dataset.py` — clean_dirty_episodes.py 的 Python 使用示例

演示 4 步流程（report → dry-run → clean → verify），改顶部 `dataset_path` / `cleaned_path` 常量即可。

---

### 5.2 真机交互子模块

#### `replay.py` — 数据集回放（deploy.py 间接用到）

**功能**：把数据集中某个 episode 的动作序列完整地在真机上重放。

```bash
python workflows/robot_interaction/replay.py \
    --repo-id datasets/26-07-20-06-08-43_v2 \
    --episode 0 \
    --fps 30

# 慢速回放
python workflows/robot_interaction/replay.py --fps 10
```

CLI 参数：`--config` / `--repo-id` / `--dataset-root` / `--episode` / `--fps` / `--http-base-url` / `--robot-id` / `--play-sounds` / `--no-sounds` / `--return-to-initial` / `--no-return-to-initial`。

#### `replay_chunk.py` — chunk 模式回放

按 chunk 批量发送动作（每次 POST 整 chunk 到 `/action_chunk`），适合高速回放或需要整段 action 的服务端。

```bash
python workflows/robot_interaction/replay_chunk.py \
    --episode 0 --chunk-size 100 --poll-interval 0.02
```

#### `show_cameras.py` — deploy.py --show-cameras 时自动启动

**功能**：用 OpenCV 窗口实时显示 quad_image 切分后的 4 路相机。

```bash
python workflows/robot_interaction/show_cameras.py \
    --policy-path outputs/train/act_xxx/checkpoints/10000/pretrained_model

# 显式指定相机
python workflows/robot_interaction/show_cameras.py \
    --cameras right_eye left_wrist right_wrist

# 额外显示未切分的原始 quad_image（调试象限用）
python workflows/robot_interaction/show_cameras.py --show-quad
```

相机列表推导顺序：
1. `--cameras` 显式列表；
2. 否则读 `<policy.path>/config.json` 的 `input_features` 过滤 `observation.images.*`，再用 yaml 的 `rename_map` 反查；
3. 都没拿到 → 回退到全部 4 路。

退出：`q` / `ESC` / `Ctrl+C`。

#### `capture_snapshot.py` — 直接调用（状态 + 图像截取）

```bash
python workflows/robot_interaction/capture_snapshot.py --save-images
```

比 `get_robot_state.py` 更重（会保存完整 JPEG），适合 bug 反馈时附带原始数据。

#### `mock_echo_server.py` — 无真机调试

```bash
python workflows/robot_interaction/mock_echo_server.py --port 8010
# 然后 deploy/replay 把 --http-base-url 指到 http://127.0.0.1:8010
```

POST `/action` / `/action_chunk` 会把 body 完整打印到终端——调试客户端真实上线 payload 用。

---

### 5.3 顶层工具（直接调用）

#### `get_robot_state.py` — 快速看机器人状态

```bash
python workflows/get_robot_state.py
python workflows/get_robot_state.py --save snapshot.json
python workflows/get_robot_state.py --url http://192.168.10.100:8010
```

只显示摘要（不保存图像）；要看完整状态 + 图像用 `capture_snapshot.py`。

#### `arm_control_http.py` — HTTP 交互式控制（CLI 菜单）

```bash
python workflows/arm_control_http.py
```

13 项菜单，**重点是选项 12/13：独立夹爪控制**（保持臂姿态不变，只动夹爪，便于隔离测试）。

#### `quickstart.sh` — 环境自检

```bash
bash workflows/quickstart.sh
```

#### `_robot_home_config.py` — home 姿态唯一真相源

```python
from workflows._robot_home_config import (
    HOME_LEFT_ARM,             # [7 floats, 度数]
    get_home_right_arm,        # () -> [7 floats]
    get_home_position,         # ('A'|'B'|None) -> 单臂 / (左,右)
    get_home_action_16joints,  # () -> [16 floats] 完整 home action
)
```

镜像规则：右臂 = 左臂的索引 0/2/4/6 取反，其他保持一致。

只改这一个文件就能让 deploy / replay / arm_control_http / _robot_home 全部同步更新。

#### `_robot_home.py` — deploy/replay 退出钩子（自动调用）

由 `deploy.py` / `replay.py` / `replay_chunk.py` 在子进程退出后自动调用，无需手动跑。

- HTTP 路径：送 home + 断连（**不下使能**，机械臂仍通电）；
- Hybrid 路径：送 home + 完整 SDK 断连路径（下使能夹爪 → set_state(0) 下伺服 → release SDK）。

#### `_config_loader.py` — 共享配置加载器（被 deploy.py 等 wrapper 引用）

按后缀自动分发 `.yaml` / `.json`，未知后缀先试 YAML 失败回落 JSON。自动剔除老版 `_comments` 字段。

```python
from workflows._config_loader import load_config
cfg = load_config(Path("deploy_config.yaml"))
```

### 5.4 UI 子模块（可选）

`workflows/robot_interaction/ui/` 是 Gradio Web 控制台，包装 `deploy.py` / `replay.py` / `show_cameras.py` + 5 个数据处理脚本 + 3 个训练脚本。架构：

```
Gradio Web UI (launch.py)
    │  YAML 是真相源;表单是结构化视图
    ▼
config_manager (yaml / dataclass 双向转换, 5 mode-aware)
    │
    ▼
process_manager (写 yaml 到 tempfile + 启子进程)
    │  脚本路径 / 环境变量走白名单 (21 个 key)
    │  bash 训练用 setsid() 自建进程组, stop 用 killpg 杀整组
    ▼
deploy.py / replay.py / show_cameras.py / 5 数据处理脚本 / 3 训练脚本
```

**用法见 [§3.4 UI 控制台（Gradio）](#34-ui-控制台gradio可选)**；底层实现详见 [`ui/快速开始.md`](robot_interaction/ui/快速开始.md)。

---

## 6. 参数替换速查表

### 6.1 CLI 参数 vs yaml 字段 vs 默认值

所有 wrapper 脚本优先级一致：**CLI 参数 > yaml 字段 > 脚本内置默认值**。

```bash
python workflows/robot_interaction/deploy.py --fps 5
# → 用 yaml 默认的 model / http url，但 fps 覆盖为 5
```

### 6.2 最常用的 5 个 CLI 参数

| 用途 | 参数 |
|---|---|
| 换模型 | `--policy-path <ckpt>` |
| 换机器人 | `--http-base-url <url>` |
| 调速 | `--fps <N>` |
| 录数据 | `--strategy sentry` |
| 打开相机预览 | `--show-cameras` |

### 6.3 yaml 配置"换路径"速查

| 想换 | yaml 字段 |
|---|---|
| 模型路径 | `policy.path` |
| HF repo | `dataset.repo_id` |
| 数据集本地路径 | `dataset.root` |
| HTTP 服务端 | `robot.http_base_url` |
| 机器人 ID | `robot.id` |
| 推理频率 | `inference.fps` |
| 推理模式 | `inference.type` |
| 录制策略 | `inference.strategy` |
| 安全裁剪数据集 | `robot.safety_stats_path` |
| 安全裕量 | `robot.action_clip_margin_deg` |
| 单步最大位移 | `robot.max_relative_target_deg` |
| 相机重命名 | `rename_map`（deploy）/ `inference.rename_map` |
| 相机预览开关 | `inference.show_cameras` |

### 6.4 train_act.sh 环境变量速查

| 变量 | 含义 |
|---|---|
| `DATASET_ROOT` | 数据集路径 |
| `OUTPUT_ROOT` | checkpoint 输出根 |
| `BATCH_SIZE` | batch size |
| `STEPS` | 训练总步数 |
| `EVAL_FREQ` | eval 频率 |
| `SAVE_FREQ` | checkpoint 频率 |
| `LOG_FREQ` | 日志频率 |
| `WANDB_PROJECT` / `WANDB_ENABLE` | wandb 配置 |
| `PUSH_TO_HUB` | 是否推 HF Hub |

### 6.5 v2_convert_config.json 字段

| 字段 | 含义 |
|---|---|
| `v2_suffix` | 输出目录后缀（默认 `_v2`） |
| `cameras` | `[left_eye, right_eye, left_wrist, right_wrist]` 中哪个 drop（0=drop） |
| `datasets` | 要转的 v1 数据集名列表（不含 `lerobot_datasets-` 前缀） |

### 6.6 "我换了 /xxx，但不起作用" checklist

1. CLI 参数拼写是否正确？tab 补全不会报错，手敲很容易打错；
2. yaml 字段名大小写？`policy.path` 不能写成 `Policy.Path`；
3. 路径是否绝对？相对路径以 `<repo_root>` 为基准；
4. 是否需要重启 wrapper？很多参数是 wrapper 启动时一次性读取的，运行时改 yaml 不生效；
5. 看日志第一行："✓ 加载配置: ..." 后面会打印实际生效的关键字段，确认 wrapper 读到了什么。

---

## 7. 常见故障排查

### 7.1 `ModuleNotFoundError: No module named 'lerobot'`

```bash
# 解决 1：用 uv run
uv run python workflows/robot_interaction/deploy.py

# 解决 2：手动设 PYTHONPATH
export PYTHONPATH=/home/zzx23457/lerobot_vlahost/src:$PYTHONPATH

# 解决 3：用仓库的 python
cd /home/zzx23457/lerobot_vlahost
python workflows/robot_interaction/deploy.py
```

### 7.2 HTTP 连接失败

```bash
# 1) 测试连通性
curl http://192.168.10.123:8010/state

# 2) 必须看到 joint_states.positions —— 缺了就跑不了 deploy
# {"joint_states": {"positions": [14 floats, radians], ...}, ...}

# 3) ping 检查网络
ping 192.168.10.123

# 4) 增加超时（改 yaml robot.timeout）
```

### 7.3 机器人不动，但 POST 返回 200

**最常见原因**：`/action` payload 字段名错了。**必须是 `jointcmd_left` / `jointcmd_right`**，用 `joint_left` / `joint_right` 会被服务端静默丢弃。

用 `arm_control_http.py` 选项 1 确认服务端确实收到了对的字段。

### 7.4 关节名称不匹配 `KeyError: 'left_arm_joint_1.pos'`

检查 yaml 的 `robot.joint_names` 是否与训练数据集 `meta/info.json` 里的 joint_names 完全一致：

```bash
cat outputs/train/my_model/pretrained_model/config.json | grep joint
cat datasets/my_dataset/meta/info.json | grep -A 20 joint_names
```

### 7.5 安全裁剪频繁触发

`WARNING: action clipped: joint 3 (left_arm_joint_4) 75.00° → 70.00°`

排查：
1. `robot.safety_stats_path` 是否指向训练数据集？
2. `action_clip_margin_deg` 是否太小（默认 5°）？临时调大到 10°；
3. 训练集本身关节范围太窄？考虑重新采集覆盖更广的姿态。

### 7.6 训练时 `FrameTimestampError`

几乎一定是 data parquet timestamp 与视频 pts 不一致：

```bash
# 先检查（train_act.sh Phase 1 已经自动跑了）
python workflows/data_processing/check_timestamp_alignment.py --dataset-root <ds>

# 选项 1: 过滤 DIRTY episodes（推荐）
# 选项 2: 调高 tolerance_s（容忍更大漂移；视觉-本体会有错位）
OVERRIDE_TOLERANCE_S=1 ./workflows/model_training/train_act.sh train
```

### 7.7 训练 loss=NaN 第一步

- 学习率过高 → `POLICY_LR=1e-5`；
- dtype 不稳 → 加 `--policy.dtype=float32`；
- 检查数据 stats.json 是否正常（用 `sanity_check.py`）。

### 7.8 v2 转换后视频打不开

```bash
# v2 的视频是软链接到 v1 的;v1 删了 v2 就废了
ls -la datasets/<v2>/videos/observation.images.right_eye/
# 应该是 -> ../../<v1>/videos/observation.images.right_eye/

# v1 还在吗?
ls datasets/<v1>/
```

### 7.9 UI 启动失败

```bash
# 1) 检查 gradio 版本
pip list | grep gradio   # 需要 >=4.0,<6.0

# 2) socks 代理问题（gradio 不支持 socks://）
# launch.py 已自动清掉 ALL_PROXY 等;如果还报错手动 unset
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY

# 3) 端口占用
lsof -i :7860
```

更详细的 UI 排查（含子脚本启动失败 / yaml-表单同步 / 跨网段访问）见
[§3.4.11 常见问题](#3411-常见问题)。

### 7.10 相机预览窗口是黑的

```bash
# 1) 看 quad_image 字段格式
curl -s http://192.168.10.123:8010/state | python3 -m json.tool | grep -A 3 quad_image
# 旧: {"format": "jpeg", "data": "<base64>"}
# 新: {"stream_url": "/stream/quad.mjpg"}  ← show_cameras 也支持

# 2) 服务端的某一象限没数据 → 用 --show-quad 看完整 quad 调象限
python workflows/robot_interaction/show_cameras.py --show-quad
```

### 7.11 训练后 deploy 但机器人动作乱

排查顺序：

1. **回放验证数据**：
   ```bash
   python workflows/robot_interaction/replay.py --episode 0 --fps 10
   ```
   不能复现 → 数据问题（跳 2）；能复现 → 策略问题（跳 3）。

2. **数据问题**：
   - `sanity_check.py`；
   - `check_timestamp_alignment.py`；
   - 看 `meta/stats.json` 有没有异常。

3. **策略问题**：
   - yaml `policy.path` 是否指向 `pretrained_model`（不是 `checkpoints/.../state`）；
   - `joint_names` 与训练时是否一致；
   - `rename_map`（VLA 模型必须正确）；
   - `dataset.single_task`（VLA 必须有）；
   - 慢速 `--fps 5` 推理观察。

---

## 8. 文档地图

| 文件 | 主题 |
|---|---|
| [`README.md`](README.md) | 项目总体 + HTTP 接口架构 |
| [`CHECKLIST.md`](CHECKLIST.md) | 使用前 5 步 checklist |
| [`GET_STATE_README.md`](GET_STATE_README.md) | `get_robot_state.py` 详细用法 |
| [`HTTP_CONTROL_README.md`](HTTP_CONTROL_README.md) | `arm_control_http.py` 详细用法 |
| [`README_HOME_CONFIG.md`](README_HOME_CONFIG.md) | home 姿态配置 |
| [`robot_interaction/ui/快速开始.md`](robot_interaction/ui/快速开始.md) | Gradio UI 底层实现参考（用法见 [§3.4](#34-ui-控制台gradio可选)） |
| [`data_processing/README.md`](data_processing/README.md) | 数据清洗工具详细文档 |
| [`data_processing/QUICKREF.md`](data_processing/QUICKREF.md) | 数据清洗快速参考 |
| [`data_processing/数据清洗分析总结.md`](data_processing/数据清洗分析总结.md) | 历史案例分析报告 |
| [`model_training/act_training_workflow.md`](model_training/act_training_workflow.md) | ACT 训练 5 phase 详解 |
| [`model_training/DIRTY_DATA_INTEGRATION.md`](model_training/DIRTY_DATA_INTEGRATION.md) | 训练 + 脏数据清洗集成 |

---

## 9. 一句话总结每个工具

| 工具 | 什么时候需要直接用 |
|---|---|
| `v2_convert_next_joint.py` | **每次有 v1 数据集时**（自动 train_act.sh 不做这一步） |
| `train_act.sh` | **每次要训练时** |
| `deploy.py` | **每次要真机推理时** |
| `v2_convert.py` | 几乎不用（保留作为 `_next_joint` 的对比参考） |
| `v2_convert_config.json` | 改要转的数据集列表时 |
| `sanity_check.py` | 几乎不用（train_act.sh 自动跑） |
| `check_timestamp_alignment.py` | 几乎不用（train_act.sh 自动跑） |
| `clean_dirty_episodes.py` | 几乎不用（train_act.sh 自动跑） |
| `merge_two_datasets.py` | 要合并两个数据集时 |
| `replay.py` | 想手动回放某 episode 验证数据 |
| `replay_chunk.py` | 同上，但要 chunk 模式 |
| `show_cameras.py` | 想单独看相机（不带 deploy） |
| `capture_snapshot.py` | 抓完整状态 + 图像用于 bug 反馈 |
| `mock_echo_server.py` | 无真机时调试 |
| `get_robot_state.py` | 日常巡检 / 抓初始姿态 |
| `arm_control_http.py` | 手动调位置 / 找 home / 测夹爪 |
| `quickstart.sh` | 新机器第一次自检 |
| `_robot_home_config.py` | home 姿态要更新时 |
| `_robot_home.py` | 不用手动跑（deploy/replay 自动调） |
| `_config_loader.py` | 不用手动调（wrapper 自动 import） |
| `train_act_simple.sh` | 已知干净的数据集 / CI 自动化 |
| `train_smolvla.sh` | 训 SmolVLA 时 |
| `finetune_act.sh` | 跨数据集 / 继续训 |
| `run_three_datasets.sh` | 临时脚本（用完删） |
| `ui/launch.py` | 想用 Gradio 控制台（用法见 [§3.4](#34-ui-控制台gradio可选)） |

---

**最后更新**：2026-07-23  
**对应代码版本**：lerobot_vlahost @ main  
**核心 3 脚本**：`v2_convert_next_joint.py` → `train_act.sh` → `deploy.py`