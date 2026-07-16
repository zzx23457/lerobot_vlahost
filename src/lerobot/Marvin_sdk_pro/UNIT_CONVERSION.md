# 单位转换说明

## 概述

为了让 policy 模型使用统一的**角度（度）**单位，wrapper 在内部自动进行单位转换。

## 单位转换实现

### 外部接口（Policy 视角）

所有关节（包括夹爪）都使用 **度 (°)**：

```python
# 读取关节位置 - 返回 16 个角度（全部为度）
positions = robot.get_joint_positions()
# [A臂7关节(°), B臂7关节(°), 左夹爪(°), 右夹爪(°)]

# 设置关节位置 - 接收 16 个角度（全部为度）
robot.set_joint_positions(positions)
```

### 内部实现（SDK 层）

Wrapper 内部处理不同的单位：

- **机械臂关节**: SDK 原生使用度 `fb_joint_pos` → 无需转换
- **夹爪**: SDK 原生使用弧度 `motor.getPosition()` → **需要转换**

## 转换逻辑

### 1. 读取时（get_joint_positions）

```python
# 获取夹爪位置（弧度）
left_gripper_rad = self._motor_left.getPosition()   # 弧度
right_gripper_rad = self._motor_right.getPosition() # 弧度

# 弧度 → 度
left_gripper_deg = math.degrees(left_gripper_rad)
right_gripper_deg = math.degrees(right_gripper_rad)

# 返回统一单位（度）
return a_joints + b_joints + [left_gripper_deg, right_gripper_deg]
```

### 2. 控制时（set_joint_positions）

```python
# 接收统一单位（度）
left_gripper_deg = positions[14]
right_gripper_deg = positions[15]

# 度 → 弧度
left_gripper_rad = math.radians(left_gripper_deg)
right_gripper_rad = math.radians(right_gripper_deg)

# 发送到夹爪（弧度）
self._gripper.controlMIT(self._motor_left, stiffness, damping, left_gripper_rad, 0.0, 0.0)
self._gripper.controlMIT(self._motor_right, stiffness, damping, right_gripper_rad, 0.0, 0.0)
```

## 转换公式

- **弧度 → 度**: `degrees = radians × 180 / π`
- **度 → 弧度**: `radians = degrees × π / 180`

Python 实现：
```python
import math

# 弧度 → 度
degrees = math.degrees(radians)

# 度 → 弧度
radians = math.radians(degrees)
```

## 示例值对照

| 弧度 (rad) | 度 (°) | 说明 |
|-----------|--------|------|
| 0.0 | 0.0 | 闭合 |
| 0.2 | 11.46 | 微开 |
| π/4 (0.785) | 45.0 | 四分之一圈 |
| π/2 (1.571) | 90.0 | 四分之二圈 |
| π (3.142) | 180.0 | 半圈 |

## 测试验证

```bash
python marvin_robot_wrapper.py
```

预期输出：
```
步骤 3: 读取当前关节位置...
左夹爪 (度): 0.00    ← 统一使用度
右夹爪 (度): 0.00

步骤 4: 移动关节...
目标位置:
  左夹爪: 0.00° → 11.46°   ← 11.46度 ≈ 0.2弧度
  右夹爪: 0.00° → 11.46°
```

## Policy 集成注意事项

### ✅ 优点
- **统一单位**: Policy 无需关心内部实现差异
- **直观**: 度数比弧度更直观
- **兼容**: 与机械臂关节保持一致

### ⚠️ 注意
- **数据集**: 如果训练数据使用弧度，需要转换
- **归一化**: 注意夹爪的角度范围（通常 0-90° 或 0-180°）
- **精度**: 度数和弧度转换可能有微小精度损失

## 与 LeRobot 集成

在创建 LeRobot Robot 子类时：

```python
from lerobot.robots import Robot

class MarvinRobot(Robot):
    def get_observation(self) -> RobotObservation:
        # 获取 16 个关节角度（全部为度）
        positions = self.wrapper.get_joint_positions()
        
        # 构建观测字典
        obs = {}
        for i, name in enumerate(self.joint_names):
            obs[f"{name}.pos"] = positions[i]  # 单位：度
        
        return obs
    
    def send_action(self, action: RobotAction) -> RobotAction:
        # 从动作字典提取 16 个关节角度（全部为度）
        positions = [action[f"{name}.pos"] for name in self.joint_names]
        
        # 发送到机器人（内部自动转换夹爪单位）
        self.wrapper.set_joint_positions(positions)
        
        return action
```

## 更新日志

- **2026-06-12**: 添加夹爪单位自动转换（度 ↔ 弧度）
- 外部接口统一使用度数
- 内部自动处理 SDK 层弧度转换
