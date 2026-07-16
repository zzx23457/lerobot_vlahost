# Marvin Robot Wrapper 使用说明

## 概述

这个封装类实现了 Marvin 双臂机器人的 LeRobot 兼容接口，支持两种控制模式：
- **位置模式 (position)**: 刚性位置控制，精确跟踪目标位置
- **关节阻抗模式 (impedance)**: 柔顺控制，适合与环境交互

## 机器人配置

- **A臂（左臂）**: 7个关节
- **B臂（右臂）**: 7个关节  
- **左夹爪**: 1个关节（对应A臂）
- **右夹爪**: 1个关节（对应B臂）
- **总计**: 16个关节

关节顺序: `[A臂7关节, B臂7关节, 左夹爪, 右夹爪]`

## 核心接口

### 1. 初始化

```python
from marvin_robot_wrapper import MarvinRobotWrapper

# 位置模式（默认）
robot = MarvinRobotWrapper(robot_ip='192.168.15.190', control_mode='position')

# 阻抗模式
robot = MarvinRobotWrapper(robot_ip='192.168.15.190', control_mode='impedance')
```

### 2. 连接机器人

```python
if robot.connect():
    print("连接成功")
else:
    print("连接失败")
```

### 3. 检查连接状态

```python
if robot.is_connected():
    print("机器人和夹爪已连接")
```

### 4. 读取关节位置

```python
positions = robot.get_joint_positions()
# 返回 16 个关节角度的列表
# 单位: 度（机械臂）+ 弧度（夹爪）
```

### 5. 设置关节位置

```python
# 准备16个关节的目标位置
target_positions = [
    # A臂 7 个关节（度）
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # B臂 7 个关节（度）
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # 左夹爪（弧度）
    0.0,
    # 右夹爪（弧度）
    0.0
]

# 发送目标位置
robot.set_joint_positions(
    target_positions,
    vel_ratio=20,   # 速度百分比 (1-100)
    acc_ratio=20    # 加速度百分比 (1-100)
)
```

### 6. 断开连接

```python
robot.disconnect()
```

## 控制模式对比

| 特性 | 位置模式 (position) | 阻抗模式 (impedance) |
|------|-------------------|---------------------|
| 刚度 | 高（刚性） | 低（柔顺） |
| 精度 | 高 | 中 |
| 安全性 | 需要避障 | 可安全接触 |
| 适用场景 | 精确定位、快速运动 | 接触任务、拖动示教 |
| 响应速度 | 快 | 较慢 |

## 测试脚本

### 完整测试

```bash
# 测试所有功能
python marvin_robot_wrapper.py
```

### 模式测试

```bash
# 测试位置模式
python test_control_modes.py position

# 测试阻抗模式
python test_control_modes.py impedance
```

## 示例：简单移动

```python
import time
from marvin_robot_wrapper import MarvinRobotWrapper

# 创建机器人实例（位置模式）
robot = MarvinRobotWrapper(robot_ip='192.168.15.190', control_mode='position')

try:
    # 连接
    robot.connect()
    
    # 读取当前位置
    current_pos = robot.get_joint_positions()
    print(f"当前位置: {current_pos}")
    
    # 修改目标位置（两臂最后一个关节各转15度）
    target_pos = current_pos.copy()
    target_pos[6] += 15.0   # A臂关节7
    target_pos[13] += 15.0  # B臂关节7
    
    # 移动到目标位置
    robot.set_joint_positions(target_pos, vel_ratio=20, acc_ratio=20)
    
    # 等待运动完成
    time.sleep(3.0)
    
    # 返回初始位置
    robot.set_joint_positions(current_pos, vel_ratio=20, acc_ratio=20)
    time.sleep(3.0)
    
finally:
    # 断开连接
    robot.disconnect()
```

## 阻抗模式参数

阻抗模式使用默认参数：
- **刚度 K**: `[2, 2, 2, 1, 1, 1, 1]`（适合拖动）
- **阻尼 D**: `[0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3]`

如需调整，可修改 `_enable_impedance_mode()` 方法中的默认值。

## 注意事项

1. **单位转换**: 机械臂使用角度（度），夹爪使用弧度
2. **连接顺序**: 必须先 `connect()` 再调用其他方法
3. **异常处理**: 建议使用 try-finally 确保正确断开连接
4. **速度限制**: vel_ratio 和 acc_ratio 范围为 1-100
5. **阻抗模式**: 更柔顺但定位精度略低，适合接触任务

## 集成到 LeRobot

这个 wrapper 提供了 LeRobot 所需的核心接口：
- `connect()` / `disconnect()`
- `is_connected()`
- `get_joint_positions()` - 对应 LeRobot 的 `get_observation()`
- `set_joint_positions()` - 对应 LeRobot 的 `send_action()`

下一步可以创建 LeRobot Robot 子类，继承 `lerobot.robots.Robot` 并使用这个 wrapper。
