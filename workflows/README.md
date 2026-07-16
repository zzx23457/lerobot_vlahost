# Marvain M6 HTTP Interface Workflows

本项目提供了通过 HTTP API 控制 Marvain M6 双臂机器人的完整工作流程，包括策略部署（deploy）和数据集回放（replay）功能。

## 架构概述

```
HTTP Server (192.168.10.123:8010)
    ↓ HTTP REST API (JSON + base64 / MJPEG)
    ↓ GET  /state   → {joint_states.positions, gripper_*, quad_image, ...}
    ↓ POST /action  ← {jointcmd_left, jointcmd_right, gripper_left, gripper_right}
    ↓
MarvainM6HttpRobot (MarvainM6HttpRobot)
    ↓ 单位转换 (radians ↔ degrees)
    ↓ 安全裁剪 (基于训练数据边界)
    ↓ quad_image 自动分割为 4 路相机
    ↓
LeRobot Rollout / Replay Infrastructure
```

> **注意**：服务端 `/action` 的合法字段是 `jointcmd_left` / `jointcmd_right`
> （不是 `joint_left` / `joint_right`）。用错字段名会被服务端静默丢弃，
> 机器人不会动。详见 `src/lerobot/robots/marvain_m6_http/marvain_m6_http.py:_prepare_action`。

## 目录结构

```
src/lerobot/robots/marvain_m6_http/
├── __init__.py                    # 包导出
├── config_marvain_m6_http.py      # 机器人配置类（含 HttpCameraConfig）
└── marvain_m6_http.py             # HTTP 机器人实现

workflows/
├── _config_loader.py              # 共享配置加载器 (YAML/JSON)
├── _robot_home.py                 # 返回 home 位置工具（HTTP/Hybrid 共用）
├── _robot_home_config.py          # Home 位置中心配置
├── get_robot_state.py             # 状态查看工具
├── arm_control_http.py            # HTTP 接口交互式控制脚本
└── robot_interaction/
    ├── deploy.py                  # 部署脚本
    ├── deploy_config.yaml         # 部署配置（sync 模式）
    ├── deploy_config_chunk.yaml   # 部署配置（默认：chunk 模式）
    ├── deploy_config_hybrid.yaml  # 部署配置（Hybrid：HTTP 观测 + SDK 控制）
    ├── replay.py                  # 回放脚本
    ├── replay_config.yaml         # 回放配置
    ├── mock_echo_server.py        # 假机器人 HTTP server（无真机时调试用）
    ├── capture_snapshot.py        # 状态 + 图像截取
    └── show_cameras.py            # 推理时并行预览相机窗口
```

`workflows/quickstart.sh` 里有一段引用了已不存在的
`workflows/robot_interaction/test_http_robot.py`，建议改用下面的命令替代：
`uv run python -c "from lerobot.robots.marvain_m6_http import MarvainM6HttpRobotConfig, MarvainM6HttpRobot; print('import ok')"`。

## 快速开始

### 1. 前置条件

- HTTP 服务器运行在 `http://192.168.10.123:8010`
- Python 3.10+ 环境
- 已安装项目依赖：
  ```bash
  # 使用 uv（推荐）
  uv sync --locked
  
  # 或使用 pip
  pip install -e .
  ```

### 2. 测试 HTTP 连接

```bash
# 轻量：只验证模块能正常 import
uv run python -c "from lerobot.robots.marvain_m6_http import MarvainM6HttpRobotConfig, MarvainM6HttpRobot; print('import ok')"

# 或在没有真机时启动一个 mock echo server
python workflows/robot_interaction/mock_echo_server.py --port 8010
# 然后另开终端把 --http-base-url 指到 http://127.0.0.1:8010 跑 deploy / replay
```

### 3. 部署策略（Deploy）

```bash
# 使用默认配置
python workflows/robot_interaction/deploy.py

# 指定策略路径
python workflows/robot_interaction/deploy.py \
    --policy-path outputs/train/my_model/pretrained_model

# 自定义推理频率
python workflows/robot_interaction/deploy.py --fps 15.0

# 使用 sentry 策略录制数据
python workflows/robot_interaction/deploy.py --strategy sentry

# 切换 HTTP 服务器地址
python workflows/robot_interaction/deploy.py \
    --http-base-url http://192.168.10.100:8010

# 启用安全裁剪（指向训练数据集 meta/stats.json）
python workflows/robot_interaction/deploy.py \
    --safety-stats-path datasets/my_training_dataset

# 推理时并行打开 OpenCV 预览窗口（policy 当前看到哪些相机）
python workflows/robot_interaction/deploy.py --show-cameras

# 把相机名映射到训练时的名字（如把 left_eye 重命名为 front_camera）
python workflows/robot_interaction/deploy.py \
    --rename-map '{"left_eye": "front_camera"}'
```

主要参数（完整列表见 `deploy.py --help`）：
- `--config PATH` — 配置文件（默认 `workflows/robot_interaction/deploy_config_chunk.yaml`，
  即整 chunk 下发到 `/action_chunk`；另有 `deploy_config.yaml`（sync）、`deploy_config_hybrid.yaml`）
- `--policy-path PATH` / `--http-base-url URL` / `--robot-id ID`
- `--safety-stats-path PATH` — 覆盖 `robot.safety_stats_path`
- `--fps` / `--strategy {base,sentry,highlight,dagger,episodic}`
- `--inference-type {sync,rtc,chunk}` / `--execution-horizon N`（RTC 模式）
- `--rename-map JSON` — 传给 `RenameObservationsProcessorStep`
- `--show-cameras` — 启动 `show_cameras.py` 子进程开 OpenCV 窗口
- `--cameras-override NAME ...` — 仅 `--show-cameras` 时生效，覆盖自动推导的相机列表

### 4. 回放数据集（Replay）

```bash
# 回放第 0 个 episode
python workflows/robot_interaction/replay.py --episode 0

# 指定数据集
python workflows/robot_interaction/replay.py \
    --repo-id username/my_dataset \
    --episode 3 \
    --fps 30

# 慢速回放（调试用）
python workflows/robot_interaction/replay.py --fps 10
```

## 配置文件

### Deploy 配置 (`deploy_config.yaml`)

主要配置项：

```yaml
policy:
  path: outputs/train/latest/pretrained_model  # 模型路径
  device: cuda                                  # 推理设备

robot:
  type: marvain_m6_http
  http_base_url: http://192.168.10.123:8010   # HTTP API 地址
  timeout: 5.0                                  # 请求超时（秒）
  joint_names: [...]                            # 16 个关节名称
  cameras: {...}                                # 相机配置
  safety_stats_path: datasets/training_dataset  # 安全边界（可选）
  action_clip_margin_deg: 5.0                   # 安全裕量（度）
  max_relative_target_deg: 10.0                 # 最大单步运动（度）

inference:
  type: sync              # sync 或 rtc
  fps: 30.0               # 推理频率
  strategy: base          # base/sentry/highlight/dagger/episodic
  return_to_initial_position: true  # 结束后返回 home
```

### Replay 配置 (`replay_config.yaml`)

主要配置项：

```yaml
dataset:
  repo_id: username/my_dataset  # 数据集 HuggingFace repo
  root: datasets/my_dataset     # 本地数据集路径（可选）
  episode: 0                    # 回放的 episode 索引
  fps: 30                       # 回放帧率

robot:
  type: marvain_m6_http
  http_base_url: http://192.168.10.123:8010
  joint_names: [...]            # 必须与数据集匹配

play_sounds: true               # 开始时语音播报
return_to_initial_position: true  # 结束后返回 home
```

## HTTP API 规范

### GET /state

返回当前机器人观测（HTTP 端点是 `/state`，不是 `/observation`）：

```json
{
  "stamp": 1782461906138432782,
  "joint_states": {
    "positions":  [0.0, -0.523, ..., 1.047],   // 14 个臂关节（弧度），前 7 个 = 左臂，后 7 个 = 右臂
    "velocities": [14 floats],                  // 可选
    "efforts":    [14 floats]                   // 可选
  },
  "gripper_left":  {"position": <rad>, "velocity": ..., ...},   // 夹爪格式历史上变化过（list / dict 都兼容）
  "gripper_right": {"position": <rad>, ...},
  "eef_left":  null,
  "eef_right": null,
  "quad_image": {
    // 旧格式：base64 内嵌单帧 JPEG
    "format": "jpeg",
    "data":   "base64_encoded_image_string..."
    // 新格式：MJPEG 流 URL（驱动器从流拉单帧）
    // "stream_url": "/stream/quad.mjpg"
  }
}
```

### POST /action

发送目标关节位置。**合法字段是 `jointcmd_left` / `jointcmd_right`**，不是 `joint_left` / `joint_right`：

```json
{
  "jointcmd_left":  [0.0, -0.523, 0.0, 1.047, 0.0, 0.0, 0.0],   // 左臂 7 关节（弧度）
  "jointcmd_right": [0.0,  0.523, 0.0, 1.047, 0.0, 0.0, 0.0],   // 右臂 7 关节（弧度）
  "gripper_left":   0.0,    // 单个浮点数（弧度），不是数组
  "gripper_right":  0.0
}
```

### POST /action_chunk（可选，RTC / 整 chunk 下发）

驱动器还有一个 `send_action_chunk()`，把整段 action chunk 一次 POST 到 `config.action_chunk_path`（默认 `/action_chunk`）：

```json
{
  "actions": [
    {"jointcmd_left":  [7 rad], "jointcmd_right": [7 rad],
     "gripper_left": <rad>, "gripper_right": <rad>},
    ...
  ]
}
```

### quad_image 相机布局

服务端返回的单张 1280×960 JPEG（或 MJPEG 流首帧）由驱动器 `_split_quad_image()` 自动
按 `_QUAD_CELL_OF_CAMERA` 映射切成 4 路 640×480。当前默认映射：

```
┌─────────────┬─────────────┐
│ left_eye    │ left_wrist  │  顶行
│  (左上)      │  (右上)      │  0:480
├─────────────┼─────────────┤
│ right_wrist │ right_eye   │  底行
│  (左下)      │  (右下)      │  480:960
└─────────────┴─────────────┘
   0:640        640:1280
```

如果硬件服务端换了相机的安装位置，只需修改 `MarvainM6HttpRobot._QUAD_CELL_OF_CAMERA`
（在 `src/lerobot/robots/marvain_m6_http/marvain_m6_http.py`），不用改其他文件。
如果某一路是空的（只有一台前视相机），代码会自动用 `left_eye` 兜底，避免模型看到全黑图。

## 单位转换说明

- **HTTP API**: 使用**弧度** (radians) 作为关节位置单位
- **LeRobot 内部**: 使用**角度** (degrees) 作为关节位置单位
- **自动转换**: `MarvainM6HttpRobot` 类在 HTTP 边界自动进行单位转换
  - `get_observation()`: 弧度 → 角度
  - `send_action()`: 角度 → 弧度

## 安全特性

### 1. 数据驱动的安全裁剪

设置 `safety_stats_path` 指向训练数据集：

```yaml
robot:
  safety_stats_path: datasets/my_training_dataset
  action_clip_margin_deg: 5.0
```

驱动器会自动：
- 从 `meta/stats.json` 加载训练时的关节位置范围（优先 `action`，回退 `observation.state`）
- 把动作裁剪到 `[min - margin, max + margin]` 范围内（仅作用于前 14 个臂关节，夹爪不裁剪）
- 缺字段 / shape 不匹配时打 WARNING 但不阻断（这时安全检查自动关掉）

### 2. 最大相对运动限制

```yaml
robot:
  max_relative_target_deg: 10.0
```

限制相邻两次 `send_action` 之间同一关节的位移不超过该角度（仅作用于前 14 个臂关节）。
驱动器内部缓存 `_last_sent_pos`，整 chunk 下发时会沿 chunk 链式应用，所以 chunk
里逐步逼近目标的限速也是正确的。

### 3. 观测范围警告

```yaml
robot:
  warn_on_observation_out_of_range: true
```

当读取的臂关节位置超出训练范围时，按关节**一次性**记一条 WARNING（避免长跑时刷屏）。
观察侧不会阻断推理，仅用于提示。

## 故障排查

### 导入错误

```python
ModuleNotFoundError: No module named 'lerobot'
```

**解决方法**：
```bash
# 设置 PYTHONPATH
export PYTHONPATH=/home/zzx23457/lerobot_vlahost/src:$PYTHONPATH

# 或使用 uv run
uv run python workflows/robot_interaction/deploy.py
```

### HTTP 连接失败

```
Failed to connect to HTTP server at http://192.168.10.123:8010
```

**检查项**：
1. HTTP 服务器是否运行？
2. 防火墙是否允许 8010 端口？
3. 网络连接是否正常？

```bash
# 测试连接（端点是 /state）
curl http://192.168.10.123:8010/state | python3 -m json.tool | head -30
```

### 关节名称不匹配

```
KeyError: 'left_arm_joint_1.pos'
```

**解决方法**：确保配置文件中的 `joint_names` 与策略训练时使用的名称完全一致。

检查训练数据集的 `meta/info.json`:
```bash
cat datasets/my_dataset/meta/info.json | grep -A 20 joint_names
```

### 机器人不动，但 HTTP 返回 200

最常见原因是 `/action` payload 字段名错了。**必须是 `jointcmd_left` /
`jointcmd_right`**，用 `joint_left` / `joint_right` 会被服务端静默丢弃。代码侧用
的字段名见 `src/lerobot/robots/marvain_m6_http/marvain_m6_http.py:_prepare_action`。

### `joint_states.positions` 缺失

```
RuntimeError: HTTP /state did not return joint_states.positions ...
```

新版本驱动器对 `/state` 的关节位置做了硬要求（不允许用假值蒙混）。如果服务端还在用
旧版 `get_observation` 接口或没返回 `joint_states`，需要服务端升级。

## 与 SDK 版本的区别

| 特性 | SDK 版本 | HTTP 版本 |
|------|---------|-----------|
| 通信方式 | TCP/IP 直连 | HTTP REST API |
| 单位 | 度 (degrees) | 弧度 (radians) |
| 相机 | 直接读取 | HTTP 返回 base64 |
| 电机控制 | 直接控制 | 不支持 |
| 下使能 | 支持 | 不支持 |
| 依赖 | Marvin SDK | requests |

## 开发注意事项

### 关节名称约定

必须与策略训练配置完全一致（默认在 `MarvainM6HttpRobotConfig` 里）：

```python
joint_names = [
    "left_arm_joint_1", "left_arm_joint_2", ..., "left_arm_joint_7",   # 0..6
    "right_arm_joint_1", "right_arm_joint_2", ..., "right_arm_joint_7", # 7..13
    "left_gripper", "right_gripper"                                     # 14, 15
]
```

`MarvainM6HttpRobotConfig.__post_init__` 会校验长度必须 = 16。

### Home 位置定义

中心配置文件是 `workflows/_robot_home_config.py`（参见 [README_HOME_CONFIG.md](README_HOME_CONFIG.md)）：

- `HOME_LEFT_ARM` — 左臂（A 臂）的 7 个 home 关节（度数）
- `get_home_right_arm()` — 按镜像规则（索引 0/2/4/6 取反）生成右臂
- `get_home_action_16joints()` — 拼出完整的 16 关节 home action（左+右+两夹爪）

`_robot_home.py` 的 `return_to_home_and_disable()`（被 deploy / replay 用作退出钩子）
会从这个中心文件导入。HTTP-only 路径只送 home + 断连，不下使能；Hybrid 路径会下使能。

## 参考资源

- LeRobot 文档: https://github.com/huggingface/lerobot
- 隔壁项目参考实现: `/home/zzx23457/lerobot/workflows/`
- 驱动器源码: `src/lerobot/robots/marvain_m6_http/marvain_m6_http.py`

## 常见问题

**Q: 为什么 HTTP 接口使用弧度而不是角度？**

A: 这是服务器端的设计决定。客户端（`MarvainM6HttpRobot`）在 HTTP 边界自动转换，对调用方透明。

**Q: 可以同时运行多个 deploy 实例吗？**

A: 不建议。HTTP 服务器通常一次只能服务一个客户端，并发控制需要服务器端支持。

**Q: 如何添加新的相机？**

A: 相机由 HTTP 服务器管理。客户端会根据 `/state` 返回的 `quad_image` 自动发现相机。
如果你要新增/重命名一路相机：

1. 改服务端让它在 `quad_image` 的某一格输出新相机；
2. 改 `MarvainM6HttpRobot._QUAD_CELL_OF_CAMERA` 把新名字映射到对应角落；
3. 在 `deploy_config.yaml` 的 `robot.cameras` 里给新相机补一个 `type: http` 的配置（用于声明 feature shape）。

**Q: 返回 home 位置为什么不下使能？**

A: HTTP 接口不暴露电机控制功能。`_robot_home.py` 的 `return_to_home_and_disable()` 在
HTTP 路径下只能把机械臂送到 home 姿态并断连，不会下使能（Hybrid 路径才会下使能）。

**Q: gripper_left/right 现在是什么格式？**

A: 服务端历史上用过两种格式：
- 早期：`[number]`（list，取 `[0]`）
- 当前：`{"position": number, ...}`（dict，取 `["position"]`）

驱动器 `_extract_gripper_pos()` 两种都兼容，识别不出来就回落到 `default_gripper_pos`
（默认 0.0°）。
