# HTTP API 实际结构更新说明

## 更新日期
2026-06-26

## 发现的实际 API 结构

通过实际测试 `http://192.168.10.123:8010`，发现 API 结构与初始假设不同：

### 实际 vs 假设

| 项目 | 假设 | 实际 |
|------|------|------|
| 端点 | `/observation` | `/state` |
| 关节数量 | 16 (14臂+2夹爪) | 14 (仅臂) |
| 关节位置路径 | `data['joints']` | `data['joint_states']['positions']` |
| 图像数据 | `data['images']` (多相机字典) | `data['quad_image']` (单个拼接图) |
| 图像格式 | base64 字符串字典 | `{'format': 'jpeg', 'data': 'base64...'}` |
| 图像尺寸 | 未知 | 1280x960 (可能是4相机拼接) |

### 实际 API 响应结构

```json
{
  "stamp": 1782461906138432782,
  "joint_states": {
    "positions": [14个关节位置，弧度],
    "velocities": [14个速度],
    "efforts": [14个力矩],
    "est_joint_force": [14个估计力]
  },
  "eef_left": null,
  "eef_right": null,
  "quad_image": {
    "format": "jpeg",
    "data": "base64编码的JPEG图像..."
  }
}
```

## 已更新的文件

### 1. 核心实现

#### `src/lerobot/robots/marvain_m6_http/config_marvain_m6_http.py`
- ✅ 关节数量：16 → 14
- ✅ 默认 joint_names：移除 left_gripper 和 right_gripper
- ✅ 验证逻辑：检查 14 个关节而非 16 个

#### `src/lerobot/robots/marvain_m6_http/marvain_m6_http.py`
- ✅ 端点：`/observation` → `/state`
- ✅ 数据路径：`data['joints']` → `data['joint_states']['positions']`
- ✅ 图像处理：`data['images']` → `data['quad_image']`
- ✅ 相机发现：改为检测单个 quad_image
- ✅ 安全边界：(16, 2) → (14, 2)
- ✅ 所有循环：range(16) → range(14)

### 2. 配置文件

#### `workflows/robot_interaction/deploy_config.yaml`
- ✅ joint_names 列表：16项 → 14项
- ✅ 注释更新：说明无夹爪

#### `workflows/robot_interaction/replay_config.yaml`
- ✅ joint_names 列表：16项 → 14项
- ✅ 注释更新：说明无夹爪

### 3. 工具脚本

#### `workflows/_robot_home.py`
- ✅ home 动作：16关节 → 14关节
- ✅ 移除夹爪位置 (home_grippers)
- ✅ 验证逻辑：检查 14 个关节

#### `workflows/robot_interaction/test_http_robot.py`
- ✅ 配置创建：16关节 → 14关节
- ✅ 测试动作：16关节 → 14关节

#### `workflows/robot_interaction/capture_snapshot.py` (新增)
- ✅ 专门用于从实际 API 截取状态的脚本
- ✅ 支持保存 JSON 数据和图像
- ✅ 单位转换（弧度→角度）显示

## 图像处理说明

### quad_image 可能的含义

根据名称和尺寸推测，`quad_image` (1280x960) 可能是4个相机的拼接图：
- 4个 640x480 的相机拼接成 1280x960
- 布局可能是 2x2 网格

### 未来改进方向

如果需要单独访问各个相机，可以考虑：

1. **服务器端改进**：提供单独的相机端点
   ```
   GET /camera/right_eye
   GET /camera/left_wrist
   GET /camera/right_wrist
   GET /camera/head
   ```

2. **客户端分割**：在 `MarvainM6Http.get_observation()` 中分割 quad_image
   ```python
   # 假设是 2x2 布局
   quad = self._decode_image(quad_img_data["data"])
   obs["right_eye"] = quad[:480, :640]  # 左上
   obs["left_wrist"] = quad[:480, 640:]  # 右上
   obs["right_wrist"] = quad[480:, :640]  # 左下
   obs["head"] = quad[480:, 640:]  # 右下
   ```

3. **保持现状**：如果策略训练时使用的就是拼接图，那么当前实现已经足够

## 使用说明

### 截取当前状态

```bash
# 基本用法
python workflows/robot_interaction/capture_snapshot.py

# 保存图像
python workflows/robot_interaction/capture_snapshot.py --save-images

# 自定义服务器
python workflows/robot_interaction/capture_snapshot.py --url http://192.168.10.100:8010
```

### 测试连接

```bash
# 简单 curl 测试
curl http://192.168.10.123:8010/state | python3 -m json.tool | head -50

# 完整机器人接口测试
python workflows/robot_interaction/test_http_robot.py
```

## 待确认事项

1. **关节名称映射**：当前使用通用名称 `left_arm_joint_X`，需要从实际训练数据集确认准确名称
2. **quad_image 分割**：是否需要将拼接图分割为单独的相机？取决于策略训练配置
3. **POST /action 端点**：假设存在但未测试，格式待确认：
   ```json
   {"joints": [14个目标位置，弧度]}
   ```

## 验证步骤

在完整环境中运行以下测试：

```bash
# 1. 基础连接测试
python workflows/robot_interaction/test_http_robot.py

# 2. 截取状态测试
python workflows/robot_interaction/capture_snapshot.py --save-images

# 3. (需要实际策略) Deploy 测试
# python workflows/robot_interaction/deploy.py --fps 15

# 4. (需要实际数据集) Replay 测试  
# python workflows/robot_interaction/replay.py --episode 0 --fps 10
```

## 回退方案

如果需要恢复到 16 关节版本（假设服务器升级支持夹爪）：

1. Git 查看历史：`git log workflows/`
2. 恢复特定文件：`git checkout <commit> -- <file>`
3. 或查看本次更新前的版本

所有更改都是向后兼容的 - 只要更新配置文件中的 `joint_names` 列表即可。
