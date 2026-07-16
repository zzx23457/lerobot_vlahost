# 机械臂 Home 位置配置说明

## 概述

所有机械臂的 home 位置现在集中定义在 `_robot_home_config.py` 文件中。这确保了整个项目中 home 位置的一致性。

## 中心配置文件

**`workflows/_robot_home_config.py`** - 唯一定义 home 位置的文件

包含：
- `HOME_LEFT_ARM` - 左臂（A臂）的 home 位置（7个关节，单位：度）
- `get_home_position(arm)` - 获取指定臂的 home 位置
- `get_home_action_16joints()` - 获取完整的 16 关节 home 动作
- 历史 home 位置备份（作为注释）

## 使用这个配置的文件

以下文件都从中心配置导入 home 位置：

1. **`workflows/_robot_home.py`**
   - 用于 deploy.py 在推理结束后归位
   - 使用 `get_home_action_16joints()`

2. **`workflows/arm_control_http.py`**
   - HTTP 接口的交互式控制脚本
   - 使用 `HOME_LEFT_ARM` 作为 `HTTPArmController.DEFAULT_HOME_LEFT`

## 如何更新 Home 位置

### 方法 1: 使用交互式脚本（推荐）

```bash
# 1. 启动交互式控制脚本
python3 workflows/arm_control_http.py

# 2. 选择选项 1 查看当前位置
# 3. 记录下当前的左臂（A臂）位置

# 4. 手动编辑 _robot_home_config.py
# 5. 将 HOME_LEFT_ARM 更新为当前位置
# 6. 旧的值会自动保留在文件末尾的注释中
```

### 方法 2: 使用 Python 脚本读取

```python
import sys
sys.path.insert(0, 'workflows')
from arm_control_http import HTTPArmController

# 连接并获取当前位置（默认 http://192.168.10.123:8010，必要时覆盖）
controller = HTTPArmController(http_url='http://192.168.10.123:8010')
if controller.connect():
    state = controller.get_current_state()
    if state and state['arm_joints']:
        left_arm = state['arm_joints'][:7]
        print("当前左臂位置:", left_arm)
        print("\n将这个值更新到 workflows/_robot_home_config.py 的 HOME_LEFT_ARM")
    else:
        print("未能获取关节位置，检查 /state 是否返回 joint_states.positions")
```

### 更新步骤

1. 读取当前机械臂位置
2. 编辑 `workflows/_robot_home_config.py`
3. 将当前位置写入 `HOME_LEFT_ARM`
4. 在文件末尾的注释区域添加旧值的备份，格式如下：

```python
# 历史 home 位置（作为备份参考）
# --------------------------------------------------
# 2026-07-02 更新:
#   HOME_LEFT_ARM = [66.04866790771484, -18.997726440429688, ...]
# 
# 初始定义 (日期未知):
#   HOME_LEFT_ARM = [66.04866790771484, -18.997726440429688, ...]
```

## 镜像规则

- 左臂（A臂）是主定义
- 右臂（B臂）通过镜像规则自动生成：索引 0, 2, 4, 6 的关节取反，其他保持一致
- 镜像逻辑在 `_robot_home_config.py` 的 `get_home_right_arm()` 函数中实现

## 注意事项

1. **只修改一个文件**：只需要修改 `_robot_home_config.py`，其他文件会自动使用新值
2. **保留历史记录**：更新时请在文件末尾的注释区域保留旧值作为备份
3. **验证一致性**：更新后可以运行测试脚本验证所有文件都使用了新值
4. **单位统一**：所有 home 位置都使用度数（degree），不是弧度

## 验证配置

运行以下命令验证所有文件都正确引用了中心配置：

```bash
python3 -c "
import sys
sys.path.insert(0, 'workflows')

from _robot_home_config import HOME_LEFT_ARM, get_home_position
from arm_control_http import HTTPArmController

print('中心配置:', HOME_LEFT_ARM)
print('HTTP控制器:', HTTPArmController.DEFAULT_HOME_LEFT)
print('是否一致:', HOME_LEFT_ARM == HTTPArmController.DEFAULT_HOME_LEFT)
"
```

## 相关文件

- `workflows/_robot_home_config.py` - 中心配置（唯一真相源）
- `workflows/_robot_home.py` - deploy / replay 退出钩子（HTTP-only 路径不
  下使能；Hybrid 路径会下使能）
- `workflows/arm_control_http.py` - HTTP 交互式控制
- `workflows/robot_interaction/deploy.py` - 部署主入口
