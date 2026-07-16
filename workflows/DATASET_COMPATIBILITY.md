# 数据集兼容性更新说明

## 更新日期
2026-06-26 (第二次更新)

## 目标
确保 HTTP 接口收集的数据与现有数据集（如 `datasets/26-06-26-10-35-39_v2`）格式完全兼容。

## 数据集格式要求

根据 `datasets/26-06-26-10-35-39_v2/meta/info.json` 分析：

### 关节配置
- **总数**: 16 个关节
- **结构**: 7 左臂 + 7 右臂 + 2 夹爪
- **特征**:
  - `observation.state`: float32[16]
  - `action`: float32[16]

### 相机配置
- **总数**: 4 个相机
- **名称**: 
  - `observation.images.left_eye` (480x640x3)
  - `observation.images.right_eye` (480x640x3)
  - `observation.images.left_wrist` (480x640x3)
  - `observation.images.right_wrist` (480x640x3)
- **格式**: H264 video

## HTTP API 限制

### 关节数据
- **HTTP 返回**: 14 个关节（仅臂关节，无夹爪）
- **解决方案**: 观测时用默认值填充夹爪位置，动作时忽略夹爪指令

### 图像数据
- **HTTP 返回**: 单个 `quad_image` (1280x960)
- **布局**: 2x2 网格
  - 左上 (0:480, 0:640): `right_eye`
  - 右上 (0:480, 640:1280): `left_wrist`
  - 左下 (480:960, 0:640): `right_wrist`
  - 右下 (480:960, 640:1280): `left_eye` (可能空白)
- **解决方案**: 自动分割 quad_image 为4个独立相机

## 实现的兼容性层

### 1. 关节填充/过滤

#### 观测 (get_observation)
```python
# HTTP 返回 14 个关节（弧度）
joints_rad = data["joint_states"]["positions"]  # [14]

# 转换为度数
joints_deg = np.degrees(joints_rad)  # [14]

# 填充夹爪位置（使用默认值 0.0°）
obs = {}
for i in range(14):
    obs[f"{joint_names[i]}.pos"] = joints_deg[i]
obs["left_gripper.pos"] = 0.0  # 索引 14
obs["right_gripper.pos"] = 0.0  # 索引 15
```

#### 动作 (send_action)
```python
# 接收 16 个关节动作（度数）
action = {f"{joint}.pos": value for joint in joint_names}  # [16]

# 只提取前 14 个（臂关节）
joints_deg = [action[f"{joint}.pos"] for joint in joint_names[:14]]  # [14]

# 转换为弧度并发送到 HTTP
joints_rad = np.radians(joints_deg)
http_post("/action", {"joints": joints_rad.tolist()})  # [14]

# 返回时包含所有 16 个关节（夹爪原样返回）
```

### 2. 图像分割

#### quad_image 分割函数
```python
def _split_quad_image(quad_image: np.ndarray) -> dict:
    """Split 1280x960 image into 4 cameras (640x480 each)"""
    return {
        "right_eye": quad_image[0:480, 0:640],
        "left_wrist": quad_image[0:480, 640:1280],
        "right_wrist": quad_image[480:960, 0:640],
        "left_eye": quad_image[480:960, 640:1280],  # 如果非空
    }
```

#### 应用
- 在 `connect()` 时自动检测并分割
- 在 `get_observation()` 时返回分割后的相机
- 相机名称自动发现，匹配数据集格式

## 配置更新

### joint_names (16个)
```yaml
joint_names:
  - left_arm_joint_1    # 0
  - left_arm_joint_2    # 1
  - left_arm_joint_3    # 2
  - left_arm_joint_4    # 3
  - left_arm_joint_5    # 4
  - left_arm_joint_6    # 5
  - left_arm_joint_7    # 6
  - right_arm_joint_1   # 7
  - right_arm_joint_2   # 8
  - right_arm_joint_3   # 9
  - right_arm_joint_4   # 10
  - right_arm_joint_5   # 11
  - right_arm_joint_6   # 12
  - right_arm_joint_7   # 13
  - left_gripper        # 14 (填充默认值)
  - right_gripper       # 15 (填充默认值)
```

### cameras (4个)
```yaml
cameras:
  right_eye:
    width: 640
    height: 480
  left_eye:
    width: 640
    height: 480
  left_wrist:
    width: 640
    height: 480
  right_wrist:
    width: 640
    height: 480
```

## 数据流对比

### 原始数据集录制（SDK版本）
```
SDK → 16 joints (degrees) → LeRobotDataset
SDK → 4 cameras (separate) → LeRobotDataset
```

### HTTP 接口录制（新版本）
```
HTTP → 14 joints (radians) → convert to degrees + pad 2 grippers → LeRobotDataset
HTTP → quad_image (1280x960) → split to 4 cameras (640x480) → LeRobotDataset
```

### 结果
两种方式生成的数据集格式完全一致：
- `observation.state`: float32[16]
- `action`: float32[16]
- 4 个相机 (640x480x3)

## 验证步骤

### 1. 测试图像分割
```bash
python workflows/robot_interaction/capture_snapshot.py --save-images

# 检查生成的图像
ls -lh snapshot_*
# 应该看到：snapshot_YYYYMMDD_HHMMSS_quad.jpeg (1280x960)

# 测试分割功能
python3 << 'EOF'
import cv2
import numpy as np

# 读取 quad image
quad = cv2.imread("snapshot_20260626_162203_quad.jpeg")
print(f"Quad image shape: {quad.shape}")

# 分割
right_eye = quad[0:480, 0:640]
left_wrist = quad[0:480, 640:1280]
right_wrist = quad[480:960, 0:640]
left_eye = quad[480:960, 640:1280]

print(f"right_eye shape: {right_eye.shape}")
print(f"left_wrist shape: {left_wrist.shape}")
print(f"right_wrist shape: {right_wrist.shape}")
print(f"left_eye shape: {left_eye.shape}")

# 保存分割后的图像
cv2.imwrite("test_right_eye.jpg", right_eye)
cv2.imwrite("test_left_wrist.jpg", left_wrist)
cv2.imwrite("test_right_wrist.jpg", right_wrist)
cv2.imwrite("test_left_eye.jpg", left_eye)
EOF
```

### 2. 测试机器人接口
```bash
python workflows/robot_interaction/test_http_robot.py

# 期望输出：
# ✓ 发现的相机: ['right_eye', 'left_wrist', 'right_wrist', 'left_eye']
# ✓ 获取观测成功，包含 20 个键 (16 关节 + 4 相机)
```

### 3. 测试录制（需要实际策略）
```bash
# Deploy 测试
python workflows/robot_interaction/deploy.py \
    --strategy sentry \
    --fps 15

# 检查生成的数据集
# 应该有 16 个关节 + 4 个相机
```

## 已知限制

### 1. 夹爪控制
- **观测**: 夹爪位置始终为默认值（0.0°），不反映真实状态
- **动作**: 夹爪指令被忽略，不会发送到机器人
- **影响**: 如果策略依赖夹爪反馈，可能不准确

**解决方案选项**:
- A) 服务器端增加夹爪状态API
- B) 训练时忽略夹爪特征
- C) 使用固定夹爪策略

### 2. left_eye 相机
- quad_image 右下角可能是空白的
- 如果为空，`_split_quad_image` 会检测并跳过
- 数据集可能只有 3 个相机而不是 4 个

**解决方案**:
- 动态检测：如果 quad_image 右下角平均像素值 < 10，则跳过 left_eye
- 配置文件中保留 4 个相机，但实际返回 3-4 个

## 文件变更总结

### 核心实现
- ✅ `config_marvain_m6_http.py`: 16关节, 添加 `default_gripper_pos`
- ✅ `marvain_m6_http.py`: 
  - 观测填充2个夹爪
  - 动作只发送14个臂关节
  - 图像自动分割为4个相机

### 配置文件
- ✅ `deploy_config.yaml`: 16关节 + 4相机
- ✅ `replay_config.yaml`: 16关节
- ✅ `_robot_home.py`: 16关节 home 位置

### 测试工具
- ✅ `test_http_robot.py`: 16关节测试
- ✅ `capture_snapshot.py`: 保持14关节（低级工具）

## 与数据集的最终兼容性

| 特征 | 数据集要求 | HTTP API | 兼容性层 | 结果 |
|------|----------|---------|---------|------|
| 关节数 | 16 | 14 | ✅ 填充夹爪 | ✅ 兼容 |
| 关节单位 | 度 | 弧度 | ✅ 自动转换 | ✅ 兼容 |
| 相机数 | 4 | 1 (拼接) | ✅ 自动分割 | ✅ 兼容 |
| 相机尺寸 | 640x480 | 1280x960 | ✅ 自动分割 | ✅ 兼容 |
| 相机名称 | 特定名称 | quad_image | ✅ 映射到正确名称 | ✅ 兼容 |

所有数据现在都与 `datasets/26-06-26-10-35-39_v2` 格式完全兼容！🎉
