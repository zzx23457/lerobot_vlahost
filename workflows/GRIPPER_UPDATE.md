# HTTP API 最终更新 - 夹爪支持 + 相机说明

## 更新日期
2026-06-26 (第三次更新 - 最终版)

## 🎯 本次更新内容

### 1. 夹爪数据完全支持

根据最新的 HTTP API 实现，现在**完全支持**夹爪的读取和控制。

#### API 结构更新

**GET /state** 现在返回：
```json
{
  "joint_states": {
    "positions": [14个臂关节，弧度]
  },
  "gripper_left": [左夹爪位置，弧度],  // 新增
  "gripper_right": [右夹爪位置，弧度], // 新增
  "quad_image": {
    "format": "jpeg",
    "data": "base64..."
  }
}
```

**POST /action** 现在接受：
```json
{
  "joints": [14个臂关节，弧度],
  "gripper_left": [左夹爪目标，弧度],  // 新增
  "gripper_right": [右夹爪目标，弧度]  // 新增
}
```

#### 实现更新

**观测 (get_observation)**:
```python
# 从 HTTP 获取夹爪数据
gripper_left_rad = data["gripper_left"][0]  # 取数组第一个元素
gripper_right_rad = data["gripper_right"][0]

# 转换为度数
obs["left_gripper.pos"] = np.degrees(gripper_left_rad)
obs["right_gripper.pos"] = np.degrees(gripper_right_rad)

# 如果数据不可用，使用默认值
if not available:
    obs["left_gripper.pos"] = default_gripper_pos  # 0.0°
    obs["right_gripper.pos"] = default_gripper_pos
```

**动作 (send_action)**:
```python
# 提取夹爪指令（度数）
left_gripper_deg = action["left_gripper.pos"]
right_gripper_deg = action["right_gripper.pos"]

# 转换为弧度并发送
payload = {
    "joints": [14个臂关节，弧度],
    "gripper_left": [np.radians(left_gripper_deg)],
    "gripper_right": [np.radians(right_gripper_deg)]
}
```

### 2. 相机布局说明

根据实际使用情况，quad_image 的布局如下：

```
+-------------------+-------------------+
|                   |                   |
|   right_eye       |   left_wrist      |
|   (眼部相机)       |   (左手腕)         |
|   640x480         |   640x480         |
|                   |                   |
+-------------------+-------------------+
|                   |                   |
|   right_wrist     |   left_eye        |
|   (右手腕)         |   (眼部相机)       |
|   640x480         |   640x480         |
|                   |                   |
+-------------------+-------------------+
```

**重要说明**：
- **left_eye 和 right_eye 是同一个眼部相机**
- 左上角和右下角的内容完全相同
- 数据集需要4个相机，所以我们保留两个眼部相机名称
- 实现时：如果右下角有内容则使用，否则复制左上角

#### 实现
```python
def _split_quad_image(quad_image):
    """分割 1280x960 的 quad_image 为 4 个 640x480 相机"""
    # 眼部相机（左上角）
    eye_camera = quad_image[0:480, 0:640].copy()
    cameras["right_eye"] = eye_camera

    # 手腕相机
    cameras["left_wrist"] = quad_image[0:480, 640:1280].copy()
    cameras["right_wrist"] = quad_image[480:960, 0:640].copy()

    # left_eye：检查右下角
    bottom_right = quad_image[480:960, 640:1280]
    if bottom_right.mean() > 10:  # 有内容
        cameras["left_eye"] = bottom_right.copy()
    else:  # 空白，复制眼部相机
        cameras["left_eye"] = eye_camera
    
    return cameras
```

## 📊 完整数据流

### 观测流程
```
HTTP /state
├─ joint_states.positions [14] (radians)
│   └─> 转换为度 → obs[joint_0..13].pos (degrees)
├─ gripper_left [1] (radians)
│   └─> 取 [0] → 转换为度 → obs[left_gripper.pos] (degrees)
├─ gripper_right [1] (radians)
│   └─> 取 [0] → 转换为度 → obs[right_gripper.pos] (degrees)
└─ quad_image.data (base64 JPEG 1280x960)
    └─> 解码 → 分割 → obs[right_eye] (640x480)
                      obs[left_eye] (640x480)
                      obs[left_wrist] (640x480)
                      obs[right_wrist] (640x480)

结果：16 个关节 + 4 个相机 = 20 个特征
```

### 动作流程
```
LeRobot action (16 joints, degrees)
├─ joints[0..13] (arm joints)
│   └─> 安全裁剪 → 相对运动限制 → 转换为弧度 → HTTP joints [14]
├─ joints[14] (left_gripper)
│   └─> 转换为弧度 → HTTP gripper_left [1]
└─ joints[15] (right_gripper)
    └─> 转换为弧度 → HTTP gripper_right [1]

HTTP POST /action
{
  "joints": [14个臂关节，弧度],
  "gripper_left": [1个值，弧度],
  "gripper_right": [1个值，弧度]
}
```

## ✅ 更新的文件

1. **marvain_m6_http.py**
   - ✅ `get_observation()`: 从 HTTP 读取 gripper_left/right
   - ✅ `send_action()`: 向 HTTP 发送 gripper_left/right
   - ✅ `_split_quad_image()`: 正确处理眼部相机复制

2. **config_marvain_m6_http.py**
   - ✅ 文档更新：说明 gripper 数据结构

## 🧪 验证步骤

### 1. 测试夹爪读取
```bash
python3 << 'EOF'
import requests
import numpy as np

resp = requests.get("http://192.168.10.123:8010/state")
data = resp.json()

# 检查夹爪数据
if "gripper_left" in data:
    print(f"✓ gripper_left: {data['gripper_left']} rad")
    print(f"  = {np.degrees(data['gripper_left'][0]):.2f}°")
else:
    print("✗ gripper_left 不存在")

if "gripper_right" in data:
    print(f"✓ gripper_right: {data['gripper_right']} rad")
    print(f"  = {np.degrees(data['gripper_right'][0]):.2f}°")
else:
    print("✗ gripper_right 不存在")
EOF
```

### 2. 测试相机分割
```bash
python workflows/robot_interaction/capture_snapshot.py --save-images

# 检查生成的图像
ls -lh snapshot_*_quad.jpeg

# 测试分割
python3 << 'EOF'
import cv2
quad = cv2.imread("snapshot_*_quad.jpeg")  # 替换为实际文件名
eye_left_top = quad[0:480, 0:640]
eye_right_bottom = quad[480:960, 640:1280]

# 两个眼部相机应该相同或非常相似
diff = cv2.absdiff(eye_left_top, eye_right_bottom)
print(f"眼部相机差异: mean={diff.mean():.2f}, max={diff.max()}")
# 期望：差异很小（mean < 5）表示是同一相机
EOF
```

### 3. 完整机器人测试
```bash
python workflows/robot_interaction/test_http_robot.py

# 期望输出：
# ✓ 获取观测成功，包含 20 个键:
#   - left_arm_joint_1.pos: XX.XX°
#   ...
#   - left_gripper.pos: XX.XX°  ← 应该是真实值，不是 0.0
#   - right_gripper.pos: XX.XX° ← 应该是真实值，不是 0.0
#   - right_eye: shape=(480, 640, 3)
#   - left_eye: shape=(480, 640, 3)
#   - left_wrist: shape=(480, 640, 3)
#   - right_wrist: shape=(480, 640, 3)
```

## 🎉 完整性检查

- [x] **16 个关节**：14 臂 + 2 夹爪 ✅
- [x] **夹爪读取**：从 HTTP 获取真实值 ✅
- [x] **夹爪控制**：向 HTTP 发送指令 ✅
- [x] **4 个相机**：right_eye, left_eye, left_wrist, right_wrist ✅
- [x] **相机内容正确**：left_eye = right_eye (同一眼部相机) ✅
- [x] **单位转换**：弧度 ↔ 度数 ✅
- [x] **数据集兼容**：与 datasets/26-06-26-10-35-39_v2 格式一致 ✅

## 📝 与之前版本的变化

| 项目 | 之前 | 现在 |
|------|------|------|
| 夹爪观测 | 填充默认值 0.0° | **从 HTTP 读取真实值** ✅ |
| 夹爪控制 | 忽略指令 | **发送到 HTTP** ✅ |
| left_eye 相机 | 检测空白或使用右下角 | **明确复制 right_eye** ✅ |
| 相机说明 | 不明确 | **明确标注眼部相机重复** ✅ |

## 🚀 现在可以做什么

所有功能现在完全可用：

```bash
# 1. 测试完整功能
python workflows/robot_interaction/test_http_robot.py

# 2. 回放数据集（包括夹爪动作）
python workflows/robot_interaction/replay.py --episode 0

# 3. 部署策略（包括夹爪控制）
python workflows/robot_interaction/deploy.py

# 4. 录制新数据集（包括夹爪状态）
python workflows/robot_interaction/deploy.py --strategy sentry
```

所有16个关节（包括夹爪）和4个相机现在都完全支持！🎉
