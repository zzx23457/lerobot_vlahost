# ⚠️ 重要发现：HTTP API 实际格式

## 问题发现
2026-06-26 下午发现：我们一直使用的 API 格式是**错误的**！

## 错误的格式（不工作）❌

```json
POST /action
{
  "joints": [14个关节，弧度],      // 合并的14个关节
  "gripper_left": 单个浮点数,
  "gripper_right": 单个浮点数
}
```

**结果**: 服务器返回 `{"success": true}`，但**机器人不动**！

## 正确的格式（工作）✅

```json
POST /action
{
  "joint_left": [7个关节，弧度],   // 左臂（A臂）
  "joint_right": [7个关节，弧度],  // 右臂（B臂）
  "gripper_left": 单个浮点数,
  "gripper_right": 单个浮点数
}
```

**结果**: 机器人**真的移动了**！

## 验证方法

### 测试代码
```python
import requests
import numpy as np

# 获取当前状态
resp = requests.get("http://192.168.10.123:8010/state")
data = resp.json()
current_joints = data["joint_states"]["positions"]

# 准备动作（左臂第一个关节 +5度）
target_joints = list(current_joints)
target_joints[0] = current_joints[0] + np.radians(5.0)

# ✅ 正确格式：分离 joint_left 和 joint_right
payload = {
    "joint_left": target_joints[:7],      # 前7个
    "joint_right": target_joints[7:14],   # 后7个
    "gripper_left": data["gripper_left"][0],
    "gripper_right": data["gripper_right"][0]
}

# 发送
resp = requests.post("http://192.168.10.123:8010/action", json=payload)
print(resp.json())  # {"success": true}

# 等待并验证
import time
time.sleep(2)
resp2 = requests.get("http://192.168.10.123:8010/state")
new_joints = resp2.json()["joint_states"]["positions"]

# 关节0应该增加了约5度
print(f"关节0变化: {np.degrees(current_joints[0]):.2f}° → {np.degrees(new_joints[0]):.2f}°")
```

### 实测结果
```
关节0变化: 69.00° → 74.31° ✓ 成功！
```

## 为什么会这样？

查看网页源码（`curl http://192.168.10.123:8010`）发现：

```javascript
// 网页的表单提交代码
const action = {
  joint_left: parseVec7(form.get("joint_left")),    // 7个值
  joint_right: parseVec7(form.get("joint_right")),  // 7个值
  gripper_left: Number(form.get("gripper_left")),
  gripper_right: Number(form.get("gripper_right")),
};
```

服务器端只认这个格式！

## 已修复的文件

✅ `src/lerobot/robots/marvain_m6_http/marvain_m6_http.py`
✅ `workflows/arm_control_http.py`

## 需要注意

### 关节索引对应

**joint_left（左臂，A臂）**:
- 索引 0-6 对应 joint_states.positions[0:7]

**joint_right（右臂，B臂）**:
- 索引 0-6 对应 joint_states.positions[7:14]

### 完整的 Python 示例

```python
def send_action_correct(arm_joints_14_deg):
    """正确发送14个关节的动作"""
    import numpy as np
    import requests
    
    # 转换为弧度
    arm_joints_rad = np.radians(arm_joints_14_deg)
    
    # 分离左右臂
    joint_left = arm_joints_rad[:7].tolist()
    joint_right = arm_joints_rad[7:14].tolist()
    
    # 获取当前夹爪位置
    state = requests.get("http://192.168.10.123:8010/state").json()
    gripper_left = state["gripper_left"][0]
    gripper_right = state["gripper_right"][0]
    
    # 构建正确的payload
    payload = {
        "joint_left": joint_left,
        "joint_right": joint_right,
        "gripper_left": float(gripper_left),
        "gripper_right": float(gripper_right)
    }
    
    # 发送
    resp = requests.post("http://192.168.10.123:8010/action", json=payload)
    return resp.json()
```

## 经验教训

1. **不要假设 API 格式** - 先看文档或网页源码
2. **测试实际效果** - 不要只看 HTTP 200 就认为成功
3. **物理验证** - 检查机器人是否真的移动了

---

**状态**: ✅ 已修复并验证工作正常

**修复日期**: 2026-06-26 下午
