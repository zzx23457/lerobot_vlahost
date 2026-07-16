# 配置文件说明

## 项目对比

### 原始项目（/home/zzx23457/lerobot）
- **接口**: SDK 直接连接（IP: 192.168.10.190）
- **控制**: 阻抗模式、速度控制、力矩控制
- **相机**: Intel RealSense（通过序列号识别）
- **功能**: 完整的底层控制

### 本项目（/home/zzx23457/lerobot_vlahost）
- **接口**: HTTP API（URL: http://192.168.10.123:8010）
- **控制**: 基础位置控制（阻抗/速度等由服务器端处理）
- **相机**: 从 HTTP 获取 quad_image（1280x960），自动分割
- **功能**: 简化的高层控制

## 配置文件结构

### 相同的部分 ✅

#### 1. Policy 配置
```yaml
policy:
  path: outputs/train/model/pretrained_model  # 模型路径
  device: cuda  # 推理设备
```

#### 2. Dataset 配置
```yaml
dataset:
  root: datasets/26-06-26-10-35-39_v2  # 本地数据集
  repo_id: datasets/26-06-26-10-35-39_v2  # HF repo ID
  single_task: "manipulation task"  # 任务描述
```

### 不同的部分 ❌

#### Robot 配置

**原始项目（SDK）**:
```yaml
robot:
  ip: 192.168.10.190
  control_mode: impedance  # 阻抗模式
  vel_ratio: 20  # 速度限制
  acc_ratio: 20  # 加速度限制
  impedance_k: [...]  # 刚度参数
  impedance_d: [...]  # 阻尼参数
  cameras:
    right_eye:
      type: intelrealsense  # 真实相机
      serial_number_or_name: "239722073373"
```

**本项目（HTTP）**:
```yaml
robot:
  type: marvain_m6_http
  http_base_url: http://192.168.10.123:8010
  timeout: 5.0
  # ❌ 没有 control_mode（HTTP不支持）
  # ❌ 没有 vel_ratio（HTTP不支持）
  # ❌ 没有 impedance 参数（HTTP不支持）
  cameras:
    right_eye:
      type: opencv  # 占位类型
      # 实际图像来自 HTTP quad_image
```

## 为什么不能直接复制原始配置？

### 1. HTTP 接口限制

HTTP 接口只提供：
- ✅ 读取关节位置（14臂关节）
- ✅ 发送目标位置（joint_left + joint_right）
- ✅ 读取夹爪位置
- ✅ 发送夹爪目标
- ✅ 获取相机图像（quad_image）

HTTP 接口**不支持**：
- ❌ 阻抗模式设置
- ❌ 速度/加速度控制
- ❌ 力矩控制
- ❌ 状态查询
- ❌ 错误清除
- ❌ 单独的相机访问

### 2. 相机差异

**原始项目**:
- 3个 Intel RealSense 相机
- 通过序列号直接访问
- 每个相机独立配置

**本项目**:
- 1个合成图像（quad_image: 1280x960）
- 自动分割为4个相机（640x480）
- 相机配置只是占位符

### 3. 控制方式差异

**原始项目**:
```python
# 可以设置阻抗模式
robot.set_impedance_mode(K=[2,2,2,...], D=[0.6,0.6,...])

# 可以控制速度
robot.set_vel_ratio(20)
```

**本项目**:
```python
# 只能发送目标位置
robot.send_action({
    "joint_left": [7个目标],
    "joint_right": [7个目标],
    ...
})
# 速度、阻抗等由HTTP服务器端处理
```

## 当前配置（已更新）

### deploy_config.yaml
```yaml
dataset:
  root: datasets/26-06-26-10-35-39_v2
  repo_id: datasets/26-06-26-10-35-39_v2
  single_task: "manipulation task"

robot:
  type: marvain_m6_http
  http_base_url: http://192.168.10.123:8010
  safety_stats_path: datasets/26-06-26-10-35-39_v2  # 启用安全裁剪
  action_clip_margin_deg: 5.0
  max_relative_target_deg: 10.0
```

### replay_config.yaml
```yaml
dataset:
  root: datasets/26-06-26-10-35-39_v2
  repo_id: datasets/26-06-26-10-35-39_v2
  episode: 0
  fps: 30

robot:
  type: marvain_m6_http
  http_base_url: http://192.168.10.123:8010
```

## 使用方法

### Deploy（部署策略）
```bash
# 使用配置文件
python workflows/robot_interaction/deploy.py

# 覆盖配置
python workflows/robot_interaction/deploy.py \
    --policy-path path/to/model \
    --fps 30
```

### Replay（回放数据）
```bash
# 使用配置文件
python workflows/robot_interaction/replay.py

# 覆盖配置
python workflows/robot_interaction/replay.py \
    --episode 5 \
    --fps 20
```

## 数据集兼容性

### 格式要求

**observation.images** (来自 meta/info.json):
- right_eye: (480, 640, 3)
- left_eye: (480, 640, 3)
- left_wrist: (480, 640, 3)
- right_wrist: (480, 640, 3)

**observation.state**:
- 16个值：14臂关节 + 2夹爪

**action**:
- 16个值：14臂关节 + 2夹爪

### 你的数据集（26-06-26-10-35-39_v2）

```bash
datasets/26-06-26-10-35-39_v2/
├── data/
│   └── chunk-000/
│       └── episode_*.parquet
├── meta/
│   ├── info.json
│   ├── stats.json
│   └── tasks.jsonl
└── videos/
    └── chunk-000/
        ├── observation.images.right_eye/
        ├── observation.images.left_eye/
        ├── observation.images.left_wrist/
        └── observation.images.right_wrist/
```

✅ **格式完全兼容！**

## 总结

| 配置项 | 原始项目 | 本项目 | 说明 |
|--------|---------|--------|------|
| policy | ✅ 相同 | ✅ 相同 | 直接复用 |
| dataset | ✅ 相同 | ✅ 相同 | 直接复用 |
| robot.ip | ✅ | ❌ | HTTP用URL |
| robot.http_base_url | ❌ | ✅ | HTTP专用 |
| robot.control_mode | ✅ | ❌ | HTTP不支持 |
| robot.vel_ratio | ✅ | ❌ | HTTP不支持 |
| robot.impedance_* | ✅ | ❌ | HTTP不支持 |
| robot.cameras.type | intelrealsense | opencv | 不同方式 |
| safety_stats_path | ✅ | ✅ | 都支持 |

**结论**: 
- ✅ policy 和 dataset 配置可以参考原始项目
- ❌ robot 配置**不能**复制（接口完全不同）
- ✅ 当前配置已针对 HTTP 接口优化，**不需要改动**
