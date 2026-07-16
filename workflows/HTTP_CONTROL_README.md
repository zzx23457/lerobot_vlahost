# HTTP 接口控制脚本说明

## 文件对比

### 原始版本（SDK）
- **文件**: `arm_clear_and_impedance_fixed.py`
- **接口**: 原生 SDK (`fx_robot`, `Marvin_Robot`, `DCSS`)
- **功能**: 完整（错误清除、阻抗模式、拖动模式、扭矩控制）

### HTTP 版本（简化）
- **文件**: `arm_control_http.py`
- **接口**: HTTP API (`http://192.168.10.123:8010`)
- **功能**: 基础位置控制

## 功能对比

| 功能 | SDK 版本 | HTTP 版本 | 说明 |
|------|----------|-----------|------|
| 连接测试 | ✅ | ✅ | HTTP: 简单GET请求 |
| 查看关节位置 | ✅ | ✅ | 16关节（14臂+2夹爪） |
| 移动到位置 | ✅ | ✅ | HTTP: 直接目标，无速度控制 |
| Home 位置 | ✅ | ✅ | 使用相同的默认值 |
| 保存/加载位置 | ❌ | ✅ | HTTP版本新增 |
| 错误检查/清除 | ✅ | ❌ | HTTP不支持 |
| 关节阻抗模式 | ✅ | ❌ | HTTP不支持 |
| 笛卡尔阻抗模式 | ✅ | ❌ | HTTP不支持 |
| 拖动模式 | ✅ | ❌ | HTTP不支持 |
| 扭矩控制 | ✅ | ❌ | HTTP不支持 |
| 速度/加速度设置 | ✅ | ❌ | HTTP不支持 |
| 状态查询 | ✅ | ❌ | HTTP不支持 |
| 伺服错误码 | ✅ | ❌ | HTTP不支持 |

## HTTP 版本使用方法

### 基本使用

```bash
# 运行脚本
python workflows/arm_control_http.py

# 输入HTTP服务器地址（或直接回车使用默认）
HTTP 服务器地址 (默认 http://192.168.10.123:8010): 
```

### 主要功能

#### 1. 查看当前位置
```
选项 1: 查看当前位置（16关节）
输出:
  A臂 (左臂): [66.05, -19.00, -80.62, -84.70, -47.02, 31.47, -40.16]
  B臂 (右臂): [-66.05, -19.00, 80.62, -84.70, 47.02, 31.47, 40.16]
  左夹爪: -1.34°
  右夹爪: -0.30°
```

#### 2. 回到 Home 位置
```
选项 2/3: A臂/B臂回到 home 位置
注意: HTTP接口会立即移动，无法控制速度
```

#### 3. 移动到自定义位置
```
选项 4: 移动到自定义位置
输入14个臂关节角度，用逗号或空格分隔:
  前7个 = A臂（左臂），后7个 = B臂（右臂）

示例:
60, -20, -75, -80, -45, 30, -35, -60, -20, 75, -80, 45, 30, 35
```

#### 4. 保存/加载位置
```
选项 5: 保存当前位置
输入名称: my_position_1

选项 6: 加载保存的位置
输入位置名称: my_position_1

选项 9/10: 保存/加载到文件
文件名: saved_positions.json
```

## HTTP API 限制

### ❌ 不支持的功能

由于 HTTP 接口只提供基础的位置读取和控制，以下功能**无法实现**：

1. **阻抗控制**: 无法设置刚度和阻尼参数
2. **拖动模式**: 无法手动拖动机械臂
3. **扭矩控制**: 只能位置控制，无法直接控制扭矩
4. **错误处理**: 无法查询或清除错误码
5. **状态管理**: 无法查询当前控制模式（下伺服/位置/扭矩等）
6. **速度控制**: 动作直接执行，无法设置速度和加速度
7. **限位处理**: 无法检测或处理限位问题

### 解决方案

如果需要这些高级功能，有几个选择：

#### 方案 1: HTTP 服务器端增强（推荐）
让同事在 HTTP 服务器端添加这些功能的 API 端点：

```python
# 建议的新端点
POST /mode/impedance    # 设置阻抗模式
POST /mode/drag         # 设置拖动模式
POST /clear_error       # 清除错误
GET  /status           # 获取详细状态
POST /action_with_vel  # 带速度的动作
```

#### 方案 2: 继续使用 SDK 版本
对于需要高级控制的场景（拖动示教、阻抗控制），继续使用原始 SDK 版本：

```bash
python workflows/arm_clear_and_impedance_fixed.py
```

#### 方案 3: 混合使用
- **简单位置控制**: 使用 HTTP 版本（`arm_control_http.py`）
- **高级功能**: 使用 SDK 版本（`arm_clear_and_impedance_fixed.py`）

## 使用建议

### 适合 HTTP 版本的场景
✅ 查看当前位置
✅ 移动到预定义位置
✅ 测试关节运动范围
✅ 简单的位置序列
✅ 数据收集（读取位置）

### 需要 SDK 版本的场景
❌ 手动拖动示教
❌ 柔顺控制
❌ 错误排查
❌ 精细速度控制
❌ 碰撞检测和安全

## 安全注意事项

⚠️ **HTTP 版本的安全限制**：

1. **无速度控制**: 机械臂会以最大速度移动到目标
2. **无错误反馈**: 如果出错，无法从 HTTP 接口得知
3. **无碰撞检测**: 需要手动确保路径安全
4. **无限位保护**: 可能超出安全范围

**使用建议**：
- 首次测试时使用小幅度移动
- 确保周围无障碍物
- 准备好急停按钮
- 不要在生产环境使用

## 示例：典型工作流

### 场景 1: 位置示教
```bash
# 1. 使用 SDK 版本进入拖动模式
python workflows/arm_clear_and_impedance_fixed.py
# 选择: 8. B臂进入拖动模式
# 手动拖动到目标位置

# 2. 使用 HTTP 版本保存位置
python workflows/arm_control_http.py
# 选择: 5. 保存当前位置
# 输入名称: position_1

# 3. 回放位置
# 选择: 6. 加载保存的位置
```

### 场景 2: 错误恢复
```bash
# 1. 使用 SDK 版本清除错误
python workflows/arm_clear_and_impedance_fixed.py
# 选择: 2. 检查并清除B臂错误

# 2. 使用 HTTP 版本回到 home
python workflows/arm_control_http.py
# 选择: 3. B臂回到 home 位置
```

## 未来改进

如果 HTTP 服务器端添加了更多功能，可以扩展 `arm_control_http.py`：

```python
# 可能的扩展
def enter_impedance_mode(self, arm='B'):
    response = self.session.post(
        f"{self.http_url}/mode/impedance",
        json={"arm": arm, "K": [...], "D": [...]}
    )

def enter_drag_mode(self, arm='B'):
    response = self.session.post(
        f"{self.http_url}/mode/drag",
        json={"arm": arm, "type": 1}
    )

def get_detailed_status(self):
    response = self.session.get(f"{self.http_url}/status")
    return response.json()  # 包含错误码、状态等
```

---

**总结**：HTTP 版本适合基础操作，高级功能仍需 SDK 版本。根据实际需求选择合适的工具。
