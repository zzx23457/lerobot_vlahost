# arm_control_http.py 单臂/双臂控制说明

## 修复的问题

### 之前的问题 ❌
```python
# 旧版 move_to_home(arm='B')
# 总是同时移动两个臂到 home
target_joints = home_left + home_right  # ← 问题！
```

**结果**: 无论选择 A 还是 B，两个臂都会移动

### 修复后 ✅
```python
# 新版 move_to_home(arm='B')
if arm == 'A':
    # 只移动左臂，右臂保持当前位置
    target_joints = home_left + current_joints[7:14]
elif arm == 'B':
    # 只移动右臂，左臂保持当前位置
    target_joints = current_joints[:7] + home_right
```

**结果**: 
- 选择 A → 只有 A 臂移动到 home，B 臂保持不动
- 选择 B → 只有 B 臂移动到 home，A 臂保持不动

## 新增功能

### 同时移动两臂
```python
def move_both_arms_to_home(self):
    """同时移动两个臂到 home 位置"""
    home_left = self.get_default_home('A')
    home_right = self.get_default_home('B')
    target_joints = home_left + home_right
    ...
```

## 菜单选项（更新后）

```
请选择操作:
  1. 查看当前位置（16关节）
  2. A臂回到 home 位置（B臂保持不动）  ← 修复
  3. B臂回到 home 位置（A臂保持不动）  ← 修复
  4. 两个臂同时回到 home 位置         ← 新增
  5. 移动到自定义位置（输入14个臂关节）
  6. 保存当前位置
  7. 加载保存的位置
  8. 列出所有保存的位置
  9. 删除保存的位置
 10. 保存位置到文件
 11. 从文件加载位置
  0. 退出程序
```

## 使用场景

### 场景 1: 单臂调试
```
当前: A臂在任意位置，B臂在工作位置
目标: 只让 A 臂回到 home，B 臂继续保持

操作: 选择 2 (A臂回到 home)
结果: ✓ A臂 → home，B臂 → 保持不动
```

### 场景 2: 双臂复位
```
当前: 两个臂都在任意位置
目标: 两个臂都回到 home

操作: 选择 4 (两个臂同时回到 home)
结果: ✓ A臂 → home，B臂 → home
```

### 场景 3: 交替操作
```
1. 选择 2 → A臂到 home，B臂保持
2. 调整 B 臂到某个位置
3. 选择 6 → 保存当前位置（A在home，B在新位置）
4. 选择 3 → B臂到 home，A臂保持
```

## 技术细节

### HTTP 接口的限制

HTTP 接口要求**同时发送14个关节**（7左臂 + 7右臂），无法只发送单臂。

**解决方案**: 
- 读取当前位置
- 只修改需要移动的臂
- 另一臂使用当前位置
- 一起发送

```python
# 示例：只移动 B 臂
current = get_current_state()  # 获取当前14个关节
home_b = get_default_home('B')  # B臂的home（7个）

target = current[:7] + home_b  # 前7个用当前（A臂保持），后7个用home（B臂移动）
send_action(target)
```

### 为什么会有"保持不动"的效果？

1. **读取当前位置**: `current_joints = [A当前, B当前]`
2. **构建目标**: `target = [A当前, B_home]` 
3. **发送指令**: 机器人收到14个目标
4. **结果**: 
   - A 臂目标 = A 当前 → 不动（目标等于当前）
   - B 臂目标 = B home → 移动

## 测试

### 测试单臂 home
```bash
python workflows/arm_control_http.py

# 1. 查看当前位置
选择: 1
输出: A臂 [69.0, -20.1, ...], B臂 [-68.6, -20.1, ...]

# 2. 移动 B 臂到 home
选择: 3
输出: 
  目标: A臂→保持, B臂→home
  ✓ 指令已发送

# 3. 再次查看位置
选择: 1
输出: A臂 [69.0, -20.1, ...] (未变), B臂 [-66.0, -19.0, ...] (已到home)
```

### 测试双臂 home
```bash
# 1. 两个臂同时回 home
选择: 4
输出:
  目标位置:
    A臂: [66.05, -19.0, ...]
    B臂: [-66.05, -19.0, ...]
  ✓ 指令已发送（两个臂同时移动）
```

## Default Home 位置

```python
DEFAULT_HOME_LEFT = [
    66.04866790771484,    # A臂 joint 0
    -18.997726440429688,  # A臂 joint 1
    -80.62322998046875,   # A臂 joint 2
    -84.70333862304688,   # A臂 joint 3
    -47.016021728515625,  # A臂 joint 4
    31.47335433959961,    # A臂 joint 5
    -40.16086959838867,   # A臂 joint 6
]

# B臂（右臂）通过镜像规则生成
# 索引 0,2,4,6 取反，1,3,5 保持
DEFAULT_HOME_RIGHT = [
    -66.05,  # joint 0: 取反
    -19.00,  # joint 1: 保持
    80.62,   # joint 2: 取反
    -84.70,  # joint 3: 保持
    47.02,   # joint 4: 取反
    31.47,   # joint 5: 保持
    40.16,   # joint 6: 取反
]
```

## 总结

✅ **修复**: 单臂 home 现在只移动指定的臂
✅ **新增**: 双臂同时 home 选项
✅ **原理**: 利用"目标=当前"实现"保持不动"
✅ **灵活**: 可以单独控制每个臂

**关键**: HTTP 接口必须发送14个关节，但通过巧妙设置目标值，可以实现单臂移动的效果。
