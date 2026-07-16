# Marvain M6 HTTP Interface Workflows

本项目提供了通过 HTTP API 控制 Marvain M6 双臂机器人的完整工作流程，包括策略部署（deploy）和数据集回放（replay）功能。

## 架构概述

```
HTTP Server (192.168.10.123:8010)
    ↓ HTTP REST API (JSON + base64 images)
    ↓ GET /observation → {joints: [radians], images: {base64...}}
    ↓ POST /action ← {joints: [radians]}
    ↓
MarvainM6Http Robot Class
    ↓ 单位转换 (radians ↔ degrees)
    ↓ 安全裁剪 (基于训练数据边界)
    ↓
LeRobot Rollout/Replay Infrastructure
```

## 目录结构

```
src/lerobot/robots/marvain_m6_http/
├── __init__.py                    # 包导出
├── config_marvain_m6_http.py      # 机器人配置类
└── marvain_m6_http.py             # HTTP 机器人实现

workflows/
├── _config_loader.py              # 共享配置加载器 (YAML/JSON)
├── _robot_home.py                 # 返回 home 位置工具
└── robot_interaction/
    ├── deploy.py                  # 部署脚本
    ├── deploy_config.yaml         # 部署配置
    ├── replay.py                  # 回放脚本
    ├── replay_config.yaml         # 回放配置
    └── test_http_robot.py         # 测试脚本
```

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
# 测试机器人连接和基本功能
python workflows/robot_interaction/test_http_robot.py
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
```

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

### GET /observation

返回当前机器人观测：

```json
{
  "joints": [0.0, -0.523, 0.0, 1.047, ...],  // 16 个关节位置（弧度）
  "images": {
    "right_eye": "base64_encoded_image_string...",
    "left_wrist": "base64_encoded_image_string...",
    "right_wrist": "base64_encoded_image_string..."
  }
}
```

### POST /action

发送目标关节位置：

```json
{
  "joints": [0.0, -0.523, 0.0, 1.047, ...]  // 16 个目标位置（弧度）
}
```

## 单位转换说明

- **HTTP API**: 使用**弧度** (radians) 作为关节位置单位
- **LeRobot 内部**: 使用**角度** (degrees) 作为关节位置单位
- **自动转换**: `MarvainM6Http` 类在 HTTP 边界自动进行单位转换
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

系统会自动：
- 从 `meta/stats.json` 加载训练时的关节位置范围
- 将动作裁剪到 `[min - margin, max + margin]` 范围内
- 记录裁剪事件到日志

### 2. 最大相对运动限制

```yaml
robot:
  max_relative_target_deg: 10.0
```

限制单步关节运动不超过 10 度，防止突然跳变。

### 3. 观测范围警告

```yaml
robot:
  warn_on_observation_out_of_range: true
```

当读取的关节位置超出训练范围时，记录一次性警告（每个关节只警告一次）。

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
# 测试连接
curl http://192.168.10.123:8010/observation
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

必须与策略训练配置完全一致：

```python
joint_names = [
    "left_arm_joint_1", "left_arm_joint_2", ..., "left_arm_joint_7",
    "right_arm_joint_1", "right_arm_joint_2", ..., "right_arm_joint_7",
    "left_gripper", "right_gripper"
]
```

### Home 位置定义

在 `workflows/_robot_home.py` 中定义：

```python
home_left = [0.0, -30.0, 0.0, 60.0, 0.0, 30.0, 0.0]  # 左臂
home_right = [0.0, 30.0, 0.0, -60.0, 0.0, -30.0, 0.0]  # 右臂（镜像）
home_grippers = [0.0, 0.0]  # 夹爪
```

单位为**角度** (degrees)。

## 参考资源

- LeRobot 文档: https://github.com/huggingface/lerobot
- 隔壁项目参考实现: `/home/zzx23457/lerobot/workflows/`
- HTTP 服务器代码: (待填写)

## 常见问题

**Q: 为什么 HTTP 接口使用弧度而不是角度？**

A: 这是服务器端的设计决定，可能为了与某些标准机器人接口保持一致。客户端会自动处理转换。

**Q: 可以同时运行多个 deploy 实例吗？**

A: 不建议。HTTP 服务器通常一次只能服务一个客户端。并发控制需要服务器端支持。

**Q: 如何添加新的相机？**

A: 相机由 HTTP 服务器管理。客户端会自动发现服务器返回的所有相机。只需在配置文件中添加相机的尺寸信息即可。

**Q: 返回 home 位置为什么不下使能？**

A: HTTP 接口不暴露电机控制功能。如需下使能，请在服务器端实现或手动操作。
