# 机器人状态查看工具使用说明

## 工具文件

**`workflows/get_robot_state.py`** - 快速查看 HTTP 机器人当前状态

## 功能

- ✅ 查看 14 个臂关节位置（A臂 + B臂）
- ✅ 查看 2 个夹爪位置
- ✅ 查看关节速度和力矩
- ✅ 查看末端执行器位置
- ✅ 查看相机图像信息
- ✅ 保存状态摘要到 JSON 文件

## 使用方法

### 1. 基本使用（查看当前状态）

```bash
python workflows/get_robot_state.py
```

**输出示例**（不同服务端版本的夹爪格式可能不同，脚本会自动适配）：
```
============================================================
HTTP 机器人状态
============================================================
获取时间: 2026-06-26 17:16:06
时间戳: 1782465366901431670

────────────────────────────────────────────────────────────
关节位置（14个臂关节）
────────────────────────────────────────────────────────────

  A臂（左臂）:
    关节 0:   1.2043 rad =   69.00°
    关节 1:  -0.3510 rad =  -20.11°
    ...
    关节 6:  -0.6893 rad =  -39.50°

  B臂（右臂）:
    关节 7:  -1.1981 rad =  -68.64°
    ...
    关节 13:   0.7013 rad =   40.18°

  最大速度: 0.001497 rad/s    # 若 /state 含 velocities 字段

  关节力矩:
    A臂: [-18.48, 20.06, ...]
    B臂: [18.29, 20.16, ...]

────────────────────────────────────────────────────────────
夹爪位置
────────────────────────────────────────────────────────────
  左夹爪:  -0.0235 rad =   -1.34°
  右夹爪:  -0.0051 rad =   -0.30°

────────────────────────────────────────────────────────────
末端执行器
────────────────────────────────────────────────────────────
  左臂末端: None   # 当前服务端 eef_*/pose_* 字段为 null
  右臂末端: None

────────────────────────────────────────────────────────────
相机图像
────────────────────────────────────────────────────────────
  格式: jpeg
  Base64 长度: 176,712 字符       # 旧格式：内嵌 base64 帧
  预估大小: 129.4 KB
  # 新格式（MJPEG 流）：脚本只会显示 'stream_url' 字段长度，元数据更少

============================================================
```

> **如果 `/state` 缺少 `joint_states.positions`**，脚本会直接显示 "❌ 获取失败"
> 或不打印臂关节段；这通常意味着服务端接口变了，需要升级服务端。

### 2. 保存状态到文件

```bash
python workflows/get_robot_state.py --save robot_state.json
```

保存的文件包含：
- 时间戳
- 所有关节位置、速度、力矩
- 夹爪位置
- 末端执行器信息
- 图像元数据（**不包含实际图像数据**）

### 3. 指定服务器地址

```bash
python workflows/get_robot_state.py --url http://192.168.10.100:8010
```

### 4. 组合使用

```bash
# 从不同服务器获取状态并保存
python workflows/get_robot_state.py \
    --url http://192.168.10.100:8010 \
    --save robot1_state.json \
    --timeout 10
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--url` | HTTP 服务器地址 | `http://192.168.10.123:8010` |
| `--save` | 保存到文件 | 不保存 |
| `--timeout` | 请求超时（秒） | 5.0 |

## 应用场景

### 场景 1: 快速检查机器人状态

```bash
# 快速查看当前位置
python workflows/get_robot_state.py
```

### 场景 2: 记录训练数据的初始位置

```bash
# 记录当前位置用于后续回放
python workflows/get_robot_state.py --save training_start.json

# 训练完成后记录结束位置
python workflows/get_robot_state.py --save training_end.json
```

### 场景 3: 对比不同时刻的状态

```bash
# 记录位置 1
python workflows/get_robot_state.py --save pos1.json

# 移动机器人...

# 记录位置 2
python workflows/get_robot_state.py --save pos2.json

# 对比两个位置
diff pos1.json pos2.json
```

### 场景 4: 监控机器人状态

```bash
# 每5秒记录一次状态
while true; do
    python workflows/get_robot_state.py --save "state_$(date +%H%M%S).json"
    sleep 5
done
```

## 与其他工具配合

### 配合 capture_snapshot.py

`capture_snapshot.py` 也可以截取状态，但功能更强大：

```bash
# get_robot_state.py: 快速查看（友好格式）
python workflows/get_robot_state.py

# capture_snapshot.py: 完整数据 + 图像
python workflows/robot_interaction/capture_snapshot.py --save-images
```

**对比**：

| 工具 | 输出格式 | 图像 | 速度 | 用途 |
|------|---------|------|------|------|
| `get_robot_state.py` | 友好摘要 | 仅元数据 | 快 | 快速查看 |
| `capture_snapshot.py` | 完整数据 | 完整图像 | 慢 | 完整记录 |

### 配合 arm_control_http.py

```bash
# 1. 查看当前位置
python workflows/get_robot_state.py

# 2. 使用控制工具移动
python workflows/arm_control_http.py
# 选择: 5. 保存当前位置

# 3. 再次查看确认
python workflows/get_robot_state.py
```

## 输出数据说明

### 关节位置
- **单位**: 弧度（rad）和度数（°）同时显示
- **数量**: 14 个臂关节（7 A臂 + 7 B臂）
- **格式**: `关节 X: Y.YYYY rad = ZZ.ZZ°`

### 夹爪位置
- **单位**: 弧度（rad）和度数（°）
- **数量**: 2 个（左夹爪 + 右夹爪）
- **来源**: HTTP API 的 `gripper_left` 和 `gripper_right`。服务端历史上用过两种
  格式：
  - **list 格式**：`[number]`（取 `[0]`）—— 旧服务端
  - **dict 格式**：`{"position": number, ...}`（取 `["position"]`）—— 当前服务端

  脚本本身只对 list 格式做了 `gripper_left[0]` 抽取；对 dict 格式会原样打印（因为
  没有 `position` 字段的兜底抽取）。**实际驱动（`MarvainM6HttpRobot._extract_gripper_pos`）**
  对两种格式都兼容。

### 关节速度
- **单位**: rad/s
- **显示**: 仅显示最大绝对速度
- **用途**: 判断机器人是否在运动

### 关节力矩
- **单位**: N·m（牛顿米）
- **显示**: 7 个关节一组（A臂 / B臂）
- **用途**: 判断负载情况

### 相机图像
- **格式**: 服务端历史上用过两种：
  - 旧：`{"format": "jpeg", "data": "<base64>"}`（内嵌单帧 JPEG）
  - 新：`{"stream_url": "/stream/quad.mjpg"}`（MJPEG 流，工具无法解码帧内容）
- **大小**: 旧 base64 格式约 129 KB（脚本会打印长度估算）
- **分辨率**: 1280x960（quad_image，2×2 的 640×480 拼接）
- **注意**: 工具只显示元数据，不解码图像；要保存原始图像请用
  `workflows/robot_interaction/capture_snapshot.py --save-images`

## 故障排查

### 连接失败

```bash
❌ 获取失败: Connection refused
```

**解决**：检查服务器地址和端口是否正确

```bash
# 测试连接
curl http://192.168.10.123:8010/state

# 如果失败，尝试 ping
ping 192.168.10.123
```

### 超时

```bash
❌ 获取失败: Read timed out
```

**解决**：增加超时时间

```bash
python workflows/get_robot_state.py --timeout 15
```

### 数据格式错误

如果输出显示 `N/A` 或异常值，可能是 API 格式变化。请查看原始数据：

```bash
curl -s http://192.168.10.123:8010/state | python3 -m json.tool | less
```

## 总结

**快速查看** → 使用 `get_robot_state.py`

**完整记录** → 使用 `capture_snapshot.py --save-images`

**控制机器人** → 使用 `arm_control_http.py`

三个工具各有侧重，根据需求选择！
