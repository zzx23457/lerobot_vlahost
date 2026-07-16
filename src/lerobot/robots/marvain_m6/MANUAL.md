# Marvain M6 操作手册

> 工厂自研的 Marvin M6 双臂机械臂在 lerobot 中的接口实现。
> **本文档只讲怎么用**——不涉及硬件装配、SDK 编译、电机接线等。

---

## 目录

0. [一句话总结](#0-一句话总结)
1. [安装前置](#1-安装前置)
2. [三种用法](#2-三种用法)
3. [`MarvainM6RobotConfig` 全部字段](#3-marvainm6robotconfig-全部字段)
4. [`MarvainM6` 全部公开 API](#4-marvainm6-全部公开-api)
5. [安全机制](#5-安全机制)
6. [运行时切换控制模式](#6-运行时切换控制模式)
7. [部署脚本 `workflows/deploy_marvain_m6.py`](#7-部署脚本-workflowsdeploy_marvain_m6py)
8. [故障排查](#8-故障排查)
9. [已知限制](#9-已知限制)
10. [文件清单与相关链接](#10-文件清单与相关链接)

---

## 0. 一句话总结

把 Marvin SDK 的 `MarvinRobotWrapper` 包成 lerobot 标准 `Robot` 接口，让你能用 `make_robot_from_config` + 标准方法（`connect` / `disconnect` / `get_observation` / `send_action`）跟其他 lerobot 机器人（SO-100、Koch 等）**完全一样**地调用。

- **关节数**：16（左臂 7 + 右臂 7 + 左夹爪 + 右夹爪）
- **通信**：TCP/IP（`robot_ip`）+ 相机走 USB
- **相机数**：当前部署数据集 3 路（`right_eye` / `left_wrist` / `right_wrist`），可通过 `config.cameras` 改
- **单位**：14 个机械臂关节 = **度**；2 个夹爪内部 = **弧度**（IO 边界自动换算，对外统一是度）

---

## 1. 安装前置

```bash
# 1. 确保 Marvin SDK 在 lerobot 同级
ls src/lerobot/Marvin_sdk_pro/   # 应该看到 libMarvinSDK.so + marvin_robot_wrapper.py

# 2. 加载 .so 到 LD_LIBRARY_PATH（wrapper 用 ctypes load）
export LD_LIBRARY_PATH=$PWD/src/lerobot/Marvin_sdk_pro:$LD_LIBRARY_PATH

# 3. lerobot 本身装好
uv sync --locked --extra dev

# 4. 4 个 USB 摄像头接到主机（当前部署用 3 个）
ls /dev/video*  # 应该看到 video0, video1, video2, ...
```

---

## 2. 三种用法

### 2.1 真机推理（最常见）

仓库根目录有现成脚本 [`workflows/deploy_marvain_m6.py`](../../../workflows/deploy_marvain_m6.py)：

```bash
python workflows/deploy_marvain_m6.py \
    --policy-path outputs/train/act_v2_*/checkpoints/last/pretrained_model \
    --dataset-root lerobot_datasets-26-06-09-09-34-54_v2 \
    --robot-ip 192.168.15.190 \
    --max-steps 10000 \
    --fps 10
```

会做的事：
1. 自动从 `--dataset-root` 读 `meta/stats.json` 加载安全区间
2. 自动从 `meta/tasks.parquet` 加载任务文本喂给 policy
3. 连机器人 + 4 路相机
4. 10 Hz 跑 policy 推理循环，**自动 clip + max_relative_target 限幅**
5. Ctrl-C 安全断连（默认锁住模式）

加 `--dry-run` 不连真机，只验证配置。

### 2.2 写自己的 Python 代码（自定义循环）

```python
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "src/lerobot/Marvin_sdk_pro")

import torch
from lerobot.robots import make_robot_from_config
from lerobot.robots.marvain_m6.config_marvain_m6 import MarvainM6RobotConfig
from lerobot.policies import make_policy
from lerobot.configs.pretrained import PreTrainedConfig
from lerobot.policies.utils import prepare_observation_for_inference, make_robot_action
from lerobot.utils.device_utils import get_safe_torch_device

# 1. 加载 policy
policy, preprocessor, postprocessor = make_policy(
    config=PreTrainedConfig.from_pretrained("outputs/train/act_v2_xxx/pretrained_model"),
    pretrained_path="outputs/train/act_v2_xxx/pretrained_model",
)
device = get_safe_torch_device("cuda")
policy.to(device).eval()

# 2. 构造 robot config (注意 safety_stats_path 启用数据驱动安全)
cfg = MarvainM6RobotConfig(
    id="arm01",
    robot_ip="192.168.15.190",
    control_mode="impedance",  # 默认阻抗(柔顺),可改 "position"
    safety_stats_path="lerobot_datasets-26-06-09-09-34-54_v2",  # 启用 safety
    disable_torque_on_disconnect=False,                          # 锁住(Q5)
)

# 3. 实例化 + 连接
robot = make_robot_from_config(cfg)
robot.connect()

# 4. 推理循环
task = "Use left hand to grasp..."  # 或从 dataset 自动读
with torch.inference_mode():
    while True:
        obs = robot.get_observation()
        obs_t = prepare_observation_for_inference(obs, device, task, robot.name)
        action_t = policy.select_action(obs_t)
        action_dict = make_robot_action(action_t, {"action": {"dtype": "float32", "shape": [16]}})
        robot.send_action(action_dict)   # 内部自动 clip + delta cap + deg→rad

# 5. 断连
robot.disconnect()
```

### 2.3 数据回放（验证 SDK + 数据流）

```bash
lerobot-replay \
    --robot.type=marvain_m6 \
    --robot.robot_ip=192.168.15.190 \
    --robot.id=arm01 \
    --dataset.repo_id=local \
    --dataset.root=lerobot_datasets-26-06-09-09-34-54_v2 \
    --dataset.episode=0
```

⚠️ 注意：这是**回放**——把录好的 action 一帧帧发给机器人，不调 policy。

---

## 3. `MarvainM6RobotConfig` 全部字段

> 路径：[`src/lerobot/robots/marvain_m6/config_marvain_m6.py`](config_marvain_m6.py)
>
> 父类：[`RobotConfig`](../../config.py) — 所有字段都可以用 draccus CLI（如 `--robot.robot_ip=...`）覆盖。

### 3.1 网络 / 控制

| 字段 | 类型 | 默认值 | 必填？ | 含义 |
|------|------|--------|--------|------|
| `id` | `str` | `None` | ❌ | 机器人实例标识（`calibration_dir` 用作文件名），多台同型机器人时必须唯一 |
| `robot_ip` | `str` | `"192.168.15.190"` | ❌ | Marvin 控制器 IP（TCP/IP），真机必改 |
| `control_mode` | `Literal["position", "impedance"]` | `"impedance"` | ❌ | 启动时的运动控制模式；运行时用 `set_control_mode()` 切换。默认 **阻抗(柔顺)**——跑 ACT policy 时机械臂对外力有缓冲,更稳 |
| `vel_ratio` | `int` | `20` | ❌ | 速度百分比 1-100（wrapper 级，作用于 set_vel_acc）|
| `acc_ratio` | `int` | `20` | ❌ | 加速度百分比 1-100 |

### 3.2 阻抗模式调参（仅 `control_mode="impedance"` 时生效）

| 字段 | 类型 | 默认值 | 必填？ | 含义 |
|------|------|--------|--------|------|
| `impedance_k` | `list[float] \| None` | `None` | ❌ | 14 个刚度值（左臂 7 + 右臂 7）。`None` = wrapper 默认 `[2,2,2,1.6,0.5,1,1] × 2` |
| `impedance_d` | `list[float] \| None` | `None` | ❌ | 14 个阻尼值。`None` = wrapper 默认 `[0.6,0.6,0.6,0.4,0.2,0.2,0.2] × 2` |

> 💡 只给 K 不给 D 也合法——D 自动用默认，反之亦然。

### 3.3 释放行为

| 字段 | 类型 | 默认值 | 必填？ | 含义 |
|------|------|--------|--------|------|
| `disable_torque_on_disconnect` | `bool` | `False` | ❌ | `False`=锁住（保持扭矩，断电不松）；`True`=释放（down-servo + 释放夹爪）|

### 3.4 关节命名（**端到端契约，改了要重训**）

| 字段 | 类型 | 默认值 | 必填？ | 含义 |
|------|------|--------|--------|------|
| `joint_names` | `list[str]` | 16 个 `left_arm_joint_*` / `right_arm_joint_*` / `left_gripper` / `right_gripper` | ❌ | 顺序与训练数据 `observation.state` / `action` 的 16 维必须**逐字符一致** |

**默认列表**（顺序敏感）：

```
left_arm_joint_1,   left_arm_joint_2,   left_arm_joint_3,
left_arm_joint_4,   left_arm_joint_5,   left_arm_joint_6,   left_arm_joint_7,
right_arm_joint_1,  right_arm_joint_2,  right_arm_joint_3,
right_arm_joint_4,  right_arm_joint_5,  right_arm_joint_6,  right_arm_joint_7,
left_gripper,       right_gripper
```

### 3.5 相机

| 字段 | 类型 | 默认值 | 必填？ | 含义 |
|------|------|--------|--------|------|
| `cameras` | `dict[str, CameraConfig]` | 3 路 `OpenCVCameraConfig` | ❌ | key = 相机名（如 `right_eye`），value = 相机配置 |

**默认相机配置**（必须与训练数据集 `info.json` 的 `observation.images.*` key 完全一致）：

| key | 类型 | width | height | fps | index_or_path |
|-----|------|------:|-------:|----:|--------------:|
| `right_eye` | OpenCVCameraConfig | 960 | 540 | 10 | 0（**占位**）|
| `left_wrist` | OpenCVCameraConfig | 960 | 540 | 10 | 1（**占位**）|
| `right_wrist` | OpenCVCameraConfig | 960 | 540 | 10 | 2（**占位**）|

**`OpenCVCameraConfig` 全部可配字段**（来自 [`lerobot.cameras.opencv`](../../cameras/opencv/)）：

| 字段 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `index_or_path` | `int \| str` | — | **必填**。`int` = `/dev/videoN` 编号；`str` = RTSP/HTTP URL（如 `"rtsp://..."`）|
| `width` | `int` | 1920 | 帧宽（**CameraConfig 必填**，见 [robot.py:29-36](../../config.py)）|
| `height` | `int` | 1080 | 帧高（必填）|
| `fps` | `int` | 30 | 帧率（必填）|
| `warmup_s` | `float` | 1.0 | 启动时预热时间（秒）|
| `fourcc` | `str \| None` | `None` | 视频编解码器（如 `"MJPG"`、`"YUYV"`）|

> ⚠️ 训练用的 width/height/fps 必须和真机一致，**否则 policy 收到的图像尺寸不匹配**。当前训练是 540×960 @ 10fps，所以默认配置也是这个。

**示例：换相机 index**

```python
cfg = MarvainM6RobotConfig(
    ...,
    cameras={
        "right_eye":  OpenCVCameraConfig(index_or_path=2, width=960, height=540, fps=10),
        "left_wrist": OpenCVCameraConfig(index_or_path=4, width=960, height=540, fps=10),
        "right_wrist":OpenCVCameraConfig(index_or_path=6, width=960, height=540, fps=10),
    },
)
```

### 3.6 数据驱动安全

| 字段 | 类型 | 默认值 | 必填？ | 含义 |
|------|------|--------|--------|------|
| `safety_stats_path` | `Path \| None` | `None` | ❌ | 指向数据集根目录，**启用后**才生效 3 道安全闸。`None` = 不启用（**真机不推荐**）|
| `action_clip_margin_deg` | `float` | `5.0` | ❌ | clip 范围比 `stats.json` 的 [min, max] 多留的余量（度）|
| `max_relative_target_deg` | `float \| None` | `10.0` | ❌ | 单 tick 每关节最多转多少度（相对上一帧）。`None` = 不限 |
| `warn_on_observation_out_of_range` | `bool` | `True` | ❌ | 读到越界值时打 WARNING（只一次/关节/会话）|

### 3.7 完整 config 速查（默认构造）

```python
MarvainM6RobotConfig(
    id=None,                              # 可选；不传 = "marvain_m6_xxx"
    # 网络 / 控制
    robot_ip="192.168.15.190",
    control_mode="impedance",  # 部署默认阻抗
    vel_ratio=20,
    acc_ratio=20,
    # 阻抗调参
    impedance_k=None,
    impedance_d=None,
    # 释放
    disable_torque_on_disconnect=False,
    # 关节名（不要改）
    joint_names=[...16个...],
    # 相机（真机必改 index_or_path）
    cameras={
        "right_eye":  OpenCVCameraConfig(index_or_path=0, width=960, height=540, fps=10),
        "left_wrist": OpenCVCameraConfig(index_or_path=1, width=960, height=540, fps=10),
        "right_wrist":OpenCVCameraConfig(index_or_path=2, width=960, height=540, fps=10),
    },
    # 数据驱动安全（强烈建议启用）
    safety_stats_path=None,
    action_clip_margin_deg=5.0,
    max_relative_target_deg=10.0,
    warn_on_observation_out_of_range=True,
)
```

---

## 4. `MarvainM6` 全部公开 API

> 路径：[`src/lerobot/robots/marvain_m6/marvain_m6.py`](marvain_m6.py)
>
> 父类：[`Robot`](../../config.py) — 所有方法签名都和 SO-100 等其他 robot 一致

### 4.1 类属性

| 属性 | 值 | 说明 |
|------|-----|------|
| `config_class` | `MarvainM6RobotConfig` | draccus 用的配置类 |
| `name` | `"marvain_m6"` | draccus 注册字符串，对应 CLI 的 `--robot.type=marvain_m6` |
| `config` | `MarvainM6RobotConfig` | 实例属性，构造时设入 |
| `cameras` | `dict[str, Camera]` | 实例属性，由 `make_cameras_from_configs` 构造 |

### 4.2 `connect(calibrate: bool = True) -> None`

建立与机器人和所有相机的连接。

| 参数 | 类型 | 默认 | 必填？ | 含义 |
|------|------|------|--------|------|
| `calibrate` | `bool` | `True` | ❌ | 兼容 lerobot 标准签名；marvain_m6 的 `calibrate()` 是 no-op，所以这个参数目前**不起作用**。保留它是为了和 SO-100 签名一致 |

**做的事**：
1. 加载 safety bounds（`safety_stats_path` 启用时）
2. `wrapper.connect()`：建立 TCP/IP，验证数据流，切换到 `control_mode`
3. 如果是 `impedance` 模式且 `impedance_k/d` 非 None，**覆盖 wrapper 默认 K/D**
4. 连接所有相机
5. `self.configure()`（no-op）
6. 重置 `_last_sent_pos`

**装饰器**：`@check_if_already_connected` — 已连接时抛 `DeviceAlreadyConnectedError`

**耗时**：~3-5 秒（含 TCP/IP 握手 + 状态机切换 + 相机预热）

**示例**：

```python
robot.connect()                    # 默认
robot.connect(calibrate=False)    # 等价，calibrate 参数无作用
```

### 4.3 `disconnect() -> None`

断开与机器人和所有相机的连接。

**做的事**：
1. 断开所有相机
2. 根据 `disable_torque_on_disconnect`：
   - `False`（默认，**锁住**）：仅调 `wrapper.robot.release_robot()` + 翻 wrapper 内部 flag（**电机扭矩保持**）
   - `True`（**释放**）：走 wrapper 完整 `disconnect()`（下伺服 + 释放夹爪 + release SDK）

**装饰器**：`@check_if_not_connected` — 未连接时抛 `DeviceNotConnectedError`

**示例**：

```python
robot.disconnect()    # 按 config 行为锁住或释放
```

### 4.4 `is_connected: bool` (property)

连接状态检查。

| 返回 | 含义 |
|------|------|
| `True` | `wrapper.is_connected() and all(cam.is_connected for cam in ...)` 都 True |
| `False` | 任一组件未连接 |

**已知盲点**：TCP/IP 真断了 `is_connected` 仍可能返回 `True`（看的是 wrapper 内部 flag，未做 socket 心跳）

### 4.5 `is_calibrated: bool` (property)

校准状态。**恒为 `True`**（marvain_m6 没有真校准状态查询）。

### 4.6 `calibrate() -> None`

**no-op + WARNING 日志**。SDK 内部处理归位，外部不暴露。

**副作用**：打一条 WARNING

```python
robot.calibrate()
# WARNING: marvain_m6 ... calibrate() is a no-op: SDK handles homing internally.
```

### 4.7 `configure() -> None`

**no-op**。wrapper 不暴露 PID/限流配置接口。

这个方法会在 `connect()` 末尾被自动调用（`@check_if_already_connected` 之后），用户不需要手动调。

### 4.8 `set_control_mode(mode, k=None, d=None) -> None`

运行时切换运动控制模式。详见 [§6](#6-运行时切换控制模式)。

| 参数 | 类型 | 默认 | 必填？ | 含义 |
|------|------|------|--------|------|
| `mode` | `Literal["position", "impedance"]` | — | ✅ | 目标模式 |
| `k` | `list[float] \| None` | `None` | ❌ | 仅 `mode="impedance"` 时使用；长度 14（7+7）。`None` = wrapper 默认 |
| `d` | `list[float] \| None` | `None` | ❌ | 同上 |

**异常**：
- `DeviceNotConnectedError`：未连接
- `ValueError`：mode 不在 `("position", "impedance")` 或 K/D 长度 != 14

### 4.9 `get_observation() -> RobotObservation`

读一帧状态。

**返回**：`dict`，19 个 key（默认）：

```python
{
    # 16 个关节（度）
    "left_arm_joint_1.pos":  <float>,
    ...
    "left_arm_joint_7.pos":  <float>,
    "right_arm_joint_1.pos": <float>,
    ...
    "right_arm_joint_7.pos": <float>,
    "left_gripper.pos":      <float>,   # 度（内部自动 rad→deg）
    "right_gripper.pos":     <float>,   # 度
    # 3 路相机（numpy H×W×3 uint8）
    "right_eye":   np.ndarray[H, W, 3],
    "left_wrist":  np.ndarray[H, W, 3],
    "right_wrist": np.ndarray[H, W, 3],
}
```

**装饰器**：`@check_if_not_connected` — 未连接时抛 `DeviceNotConnectedError`

**异常**：
- `DeviceNotConnectedError`
- `RuntimeError`：wrapper 返回的关节数与 `joint_names` 数量不匹配

**副作用**：
- 如果 `warn_on_observation_out_of_range=True` 且有 joint 越界，每个**只打一次** WARNING

### 4.10 `send_action(action) -> RobotAction`

下发一帧动作。

| 参数 | 类型 | 必填？ | 含义 |
|------|------|--------|------|
| `action` | `dict[str, float]` | ✅ | 16 个 key，格式 `{"<joint_name>.pos": <float, 度>}` |

**返回**：实际下发的 action（**已 clip + cap 后的值，度**）。如果触发了安全机制，返回的可能与传入不同。

**做的事**（按顺序）：
1. 按 `joint_names` 顺序把 dict 拆成 16-vec
2. **Safety 1**：如果 `safety_stats_path` 启用，clip 到 `[stats.min - margin, stats.max + margin]`
3. **Safety 2**：如果 `max_relative_target_deg` 非 None，cap 到相对上一帧 ±N°
4. 夹爪 indices 转弧度
5. `wrapper.set_joint_positions(vel_ratio, acc_ratio)` 发给 SDK

**异常**：
- `DeviceNotConnectedError`
- `KeyError`：action 缺 key（`f"{joint_name}.pos"` 不全）

**示例**：

```python
action = {f"{n}.pos": 0.0 for n in robot.config.joint_names}
action["right_gripper.pos"] = 30.0
action["left_arm_joint_1.pos"] = 45.0
sent = robot.send_action(action)   # sent 是 clip/cap 后的值
```

### 4.11 `observation_features: dict` (cached_property)

声明机器人能产生的观测量 schema。

**返回**：

```python
{
    **{f"{j}.pos": float for j in self.config.joint_names},  # 16 个 float
    **{cam: (height, width, 3) for cam in self.cameras},    # 3 个 (H,W,3)
}
```

### 4.12 `action_features: dict` (cached_property)

声明机器人接受的动作 schema。

**返回**：

```python
{f"{j}.pos": float for j in self.config.joint_names}  # 16 个 float
```

---

## 5. 安全机制

### 5.1 三道安全闸（启用条件：`safety_stats_path` 非 None）

```
policy 输出 (action, 度)
   ↓
[1] action range clip    → 截到 [stats.min - 5°, stats.max + 5°]
   ↓
[2] per-tick delta cap   → 截到 ±10° / tick（相对上一帧）
   ↓
deg → rad 转换（仅夹爪 14, 15）
   ↓
wrapper.set_joint_positions()  → TCP/IP → 真机
```

每次 clip / cap 都会打 WARNING 告诉你哪几个关节被截、截了多少。

### 5.2 观测越界报警（`get_observation()` 时）

```
get_observation() → 读到 joint 0 = 250°（stats 范围 [-20.9°, 131.2°]）
              ↓
WARNING: observation left_arm_joint_1.pos = 250.000 outside training range [-20.88, 131.22]
         (margin 5.0°). Possible causes: (1) wrong units (rad vs deg),
         (2) wrong joint_names order, (3) sensor not homed.
```

每个关节**只警告一次**（不会 spam）。

⚠️ **这条 WARNING 是 unit-bug 早期发现机制**——如果看到 16 个关节值都在 ±π 附近（~3.14），说明 SDK 整体返回弧度；如果是单一关节越界，可能是 joint_names 顺序错。

### 5.3 释放策略（`disable_torque_on_disconnect`）

| 值 | 行为 | 适用场景 |
|----|------|---------|
| `False`（默认）| 仅释放 TCP/IP 资源，**电机扭矩保持**，机械臂锁在最后位姿 | 调试中突然断电不希望臂倒下 |
| `True` | 走 wrapper 完整 `disconnect()`：下伺服 + 释放夹爪 + release SDK | 正常结束 / E-stop |

### 5.4 越界 config 的容错

| 配置错误 | 行为 |
|---------|------|
| `safety_stats_path` 指向不存在的目录 | WARNING，3 道闸全部 disabled |
| `safety_stats_path` 存在但 `stats.json` 里 action/state 维度不是 16 | WARNING，3 道闸全部 disabled |
| `impedance_k` 长度 ≠ 14 | `set_control_mode("impedance", k=...)` 抛 `ValueError` |
| `impedance_k` 类型不是 `list[float]` | TypeError（draccus 校验）|

---

## 6. 运行时切换控制模式

### 6.1 三种用法

```python
# 1) 启动时（config 决定）
cfg = MarvainM6RobotConfig(
    control_mode="impedance",
    impedance_k=[1.0]*7 + [1.0]*7,
    impedance_d=[0.3]*7 + [0.3]*7,
)
robot = make_robot_from_config(cfg)
robot.connect()  # 自动用 config 的 K/D 覆盖 wrapper 默认

# 2) 运行时切回刚性格
robot.set_control_mode("position")

# 3) 运行时切到阻抗（用 wrapper 默认 K/D）
robot.set_control_mode("impedance")

# 4) 运行时切到阻抗 + 自定义 K/D
robot.set_control_mode("impedance",
                       k=[1.0]*7 + [1.0]*7,
                       d=[0.3]*7 + [0.3]*7)
```

### 6.2 `set_control_mode(mode, k=None, d=None)` 参数详解

| 参数 | 类型 | 默认 | 必填？ | 含义 |
|------|------|------|--------|------|
| `mode` | `Literal["position", "impedance"]` | — | ✅ | 目标模式 |
| `k` | `list[float] \| None` | `None` | ❌ | 仅 `mode="impedance"` 时用。14 元素（7+7）。`None` = wrapper 默认 |
| `d` | `list[float] \| None` | `None` | ❌ | 同上 |

**返回值**：无（修改 `self.config.control_mode` 反映新状态）

**异常**：
- `DeviceNotConnectedError`：未连接
- `ValueError`：`mode` 不在白名单 / `k` 长度 != 14 / `d` 长度 != 14

**耗时**：
- `"position"`：~0.5 秒
- `"impedance"`：~1.9 秒（5 步 + 多个 sleep）

### 6.3 底层调用流程

```
set_control_mode("impedance", K, D) 内部执行：
  1. set_state(arm, 0) × 2  → 双臂 IDLE
  2. sleep 0.5s
  3. set_state(arm, 3) × 2  → 切扭矩模式
  4. sleep 0.5s
  5. set_impedance_type(arm, 1) × 2
  6. sleep 0.3s
  7. set_joint_kd_params(arm, K, D) × 2
  8. sleep 0.3s
  9. set_vel_acc(arm, velRatio, AccRatio) × 2
  10. sleep 0.3s
总计约 1.9s，期间机械臂完全失力
```

> 内部还有个 `_apply_impedance_kd(k, d)` 私有方法：只重写 K/D，不做 IDLE→TORQ dance，约 0.2 秒。一般用户不直接调，`connect()` 启动时若给了 config K/D 走的就是这条。

### 6.4 ⚠️ 注意事项

1. **阻塞操作**：`set_control_mode` 是 1-2 秒阻塞，**不要在 policy 10Hz 推理循环里调**
2. **丢帧**：切换期间机械臂**短暂 IDLE**（下伺服），会丢一帧 action
3. **K/D 顺序**：14 元素 = A 臂 7 + B 臂 7，顺序必须与 `joint_names` 一致
4. **状态同步**：切完后 `self.config.control_mode` 会同步更新成新值

---

## 7. 部署脚本 `workflows/deploy_marvain_m6.py`

### 7.1 CLI 参数全表

| 参数 | 类型 | 默认值 | 必填？ | 含义 |
|------|------|--------|--------|------|
| `--policy-path` | `str` | — | ✅ | policy 目录路径（**支持 glob**，如 `outputs/train/act_v2_*/checkpoints/last/pretrained_model`，多匹配选 mtime 最新）|
| `--robot-ip` | `str` | `"192.168.15.190"` | ❌ | Marvin 控制器 IP |
| `--robot-id` | `str` | `"arm01"` | ❌ | 机器人实例 ID（多台同时运行时区分）|
| `--dataset-root` | `Path` | `None` | ❌ | 训练数据集根目录；提供后**自动加载** task 文本 + safety 区间（不传 = 不用 safety）|
| `--device` | `str` | `"cuda"` | ❌ | policy 推理 device（`"cpu"` / `"cuda"` / `"cuda:0"` 等）|
| `--max-steps` | `int` | `10000` | ❌ | 最多跑多少 step（Ctrl-C 提前结束）|
| `--fps` | `float` | `10.0` | ❌ | 推理循环目标频率（Hz）|
| `--dry-run` | flag | `False` | ❌ | 不连真机，只验证 policy + config 装载 |

### 7.2 完整调用示例

```bash
# 完整推理
python workflows/deploy_marvain_m6.py \
    --policy-path outputs/train/act_v2_*/checkpoints/last/pretrained_model \
    --dataset-root  lerobot_datasets-26-06-09-09-34-54_v2 \
    --robot-ip 192.168.15.190 \
    --robot-id arm01 \
    --device cuda \
    --max-steps 10000 \
    --fps 10

# 只验证配置不连真机
python workflows/deploy_marvain_m6.py \
    --policy-path outputs/train/act_v2_*/checkpoints/last/pretrained_model \
    --dataset-root  lerobot_datasets-26-06-09-09-34-54_v2 \
    --dry-run

# 慢速试运行
python workflows/deploy_marvain_m6.py \
    --policy-path outputs/train/act_v2_*/checkpoints/last/pretrained_model \
    --dataset-root  lerobot_datasets-26-06-09-09-34-54_v2 \
    --fps 5 --max-steps 100
```

### 7.3 内部行为详解

1. 解析 `--policy-path`（glob 多匹配选最新）
2. 如果有 `--dataset-root`：
   - 读 `meta/stats.json` → 给 `MarvainM6RobotConfig(safety_stats_path=...)`
   - 读 `meta/tasks.parquet` → 第一行 task 文本喂给 policy
3. `make_policy(...)` 加载 policy
4. `make_robot_from_config(cfg)` 构造 robot
5. `robot.connect()`
6. 10Hz 循环：
   - `obs = robot.get_observation()`（带越界 warn）
   - `obs_t = prepare_observation_for_inference(...)`（归一化 → tensor）
   - `action_t = policy.select_action(obs_t)`
   - `action_dict = make_robot_action(action_t, dataset_features)`
   - `robot.send_action(action_dict)`（带 clip + delta cap + deg→rad）
7. Ctrl-C / KeyboardInterrupt → `robot.disconnect()`（**锁住模式**）

---

## 8. 故障排查

| 症状 | 原因 | 处理 |
|------|------|------|
| `ImportError: cannot import name 'MarvinRobotWrapper'` | `Marvin_sdk_pro/__init__.py` 没建 | 看 § 1，复制 `__ini__.py` → `__init__.py` |
| `OSError: libMarvinSDK.so: cannot open shared object file` | LD_LIBRARY_PATH 没设 | `export LD_LIBRARY_PATH=$PWD/src/lerobot/Marvin_sdk_pro:$LD_LIBRARY_PATH` |
| `connect()` 卡在 `机器人数据流未激活` | 网络/控制器未就绪 | 检查网线 + 控制器开机 + `ping $robot_ip` |
| 16 路都报 `outside training range` | SDK 返回弧度（bug） | 检查 `get_observation` 转换逻辑 / 硬件版本 |
| 单个关节越界 | joint_names 顺序错 | 对比 stats.json 的 min/max 与 `joint_names` |
| `clipped N/16 joint commands` WARNING | policy 输出越界 | 正常，**别忽略**——说明 policy 见过训练分布外的状态 |
| `capped N/16 joints' per-tick motion` | policy 输出跳变 | 正常，被 10° 限幅截了；如频繁出现可能 policy 没收敛 |
| TCP/IP 断了 `is_connected` 仍 True | 内部 flag 没刷 | **已知盲点**，需加心跳（未实现）|
| E-stop 后再连不上 | 控制器需复位 | 急停按钮复位后重试，或重启 SDK |
| `RuntimeError: Wrapper returned X joints but config expects 16` | SDK 返回关节数与 `joint_names` 数量不匹配 | 检查 `joint_names` 顺序 / SDK 版本 |
| 相机画面是黑屏 | `index_or_path` 错 / 摄像头被占 | 跑 `v4l2-ctl --list-devices` 查实际编号；改 `config.cameras` |
| 相机帧率和训练不一致 | 训练 10fps 真机 30fps | `config.cameras[*].fps = 10` |
| `set_control_mode` 报 `K must be length 14` | K 列表长度不是 14 | 检查 K = `[左臂7个] + [右臂7个]` |
| `set_control_mode` 报 `not connected` | robot 未 connect | 先 `robot.connect()` |

---

## 9. 已知限制

| 项 | 状态 | 备注 |
|----|------|------|
| `calibrate()` | no-op | SDK 内部归位，外部不暴露 |
| `configure()` | no-op | wrapper 不暴露 PID/限流配置接口 |
| TCP/IP 断线心跳 | **未实现** | `is_connected` 只看内部 flag |
| Action chunking 队列 | 未实现 | `set_joint_positions` 一次只发 1 个；走 chunked 流式需要本地队列 |
| 关节速度/扭矩/温度透传 | 未实现 | SDK 有 `fb_joint_vel` / `fb_joint_sToq` / `fb_joint_them` 但没暴露给 policy |
| Action chunking (`AsyncInference` 走 gRPC) | **不推荐** | 走 `async_inference/robot_client.py` 也行，但单步同步循环更简单 |
| 双臂 leader (主手) | **不在 scope** | 仓库 `lerobot/teleoperators/` 下没有 |
| `set_control_mode` 内部重力补偿参数 | hardcoded | 来自 wrapper 硬编码 `[5, 0, 0, 50, 0.004, ...]`，未暴露 config |

---

## 10. 文件清单与相关链接

### 10.1 本目录文件

```
src/lerobot/robots/marvain_m6/
├── __init__.py              # import 触发 draccus 注册
├── config_marvain_m6.py     # 16 关节命名 + 3 相机 + safety + impedance 配置
├── marvain_m6.py            # Robot 子类（核心，含 set_control_mode 等）
└── MANUAL.md                # ← 你正在读
```

### 10.2 相关文件

| 路径 | 用途 |
|------|------|
| `src/lerobot/robots/utils.py` | 工厂分发 `elif "marvain_m6"` 分支 |
| `src/lerobot/robots/robot.py` | `Robot` 抽象基类 |
| `src/lerobot/robots/config.py` | `RobotConfig` 基类 |
| `src/lerobot/Marvin_sdk_pro/` | 硬件部门给的 SDK（含 `marvin_robot_wrapper.py`）|
| `src/lerobot/Marvin_sdk_pro/marvin_robot_wrapper.py` | 我们的 wrapper 类 |
| `src/lerobot/Marvin_sdk_pro/fx_robot.py` | SDK 底层 ctypes 绑定 |
| `workflows/deploy_marvain_m6.py` | 一键部署脚本 |

### 10.3 相关数据集和工作流

- 训练数据集：[`lerobot_datasets-26-06-09-09-34-54_v2`](../../../lerobot_datasets-26-06-09-09-34-54_v2) （188 ep，3 相机，当前部署用）
- 历史训练数据集：[`lerobot_datasets-26-06-09-10-23-51_v2`](../../../lerobot_datasets-26-06-09-10-23-51_v2) （300 ep，4 相机）
- 训练工作流：[`workflows/train_act.sh`](../../../workflows/train_act.sh) + [`workflows/act_training_workflow.md`](../../../workflows/act_training_workflow.md)
- 怎么新增一个 robot（参考骨架）：[`ADDING_A_NEW_ROBOT.md`](../../ADDING_A_NEW_ROBOT.md)
- SO-100 对照实现：[`src/lerobot/robots/so_follower/`](../so_follower/)

### 10.4 相关辅助函数 / 工具

| 路径 | 用途 |
|------|------|
| `lerobot.policies.utils.prepare_observation_for_inference` | 把 obs dict 转成 policy 期望的 tensor batch |
| `lerobot.policies.utils.make_robot_action` | 把 policy 输出的 tensor 拆回 dict（按 `dataset_features`）|
| `lerobot.utils.device_utils.get_safe_torch_device` | 安全获取 torch device（自动检测 cuda/mps/cpu）|
| `lerobot.utils.decorators.check_if_already_connected` | 装饰器，重复 connect 抛异常 |
| `lerobot.utils.decorators.check_if_not_connected` | 装饰器，未 connect 时操作抛异常 |
| `lerobot.utils.errors.DeviceNotConnectedError` | 异常类 |
