# 使用检查清单

在运行 deploy 或 replay 之前，请确认以下各项：

## 环境准备

- [ ] Python 3.10+ 已安装
- [ ] 项目依赖已安装（运行 `uv sync --locked` 或 `pip install -e .`）
- [ ] HTTP 服务器运行在 `http://192.168.10.123:8010`
- [ ] 网络连接正常（可以 ping 通 192.168.10.123）

## HTTP 服务器测试

```bash
# 测试服务器是否响应（端点是 /state，不是 /observation）
curl http://192.168.10.123:8010/state | python3 -m json.tool | head -30

# 应该返回类似（不同服务端版本字段略有差异，但 joint_states.positions 必须存在）：
# {
#   "joint_states": {"positions": [14 floats, radians], ...},
#   "gripper_left":  ...,
#   "gripper_right": ...,
#   "quad_image":    {"format": "jpeg", "data": "..."} 或 {"stream_url": "..."}
# }
```

## 配置文件检查

### Deploy 配置 (`workflows/robot_interaction/deploy_config.yaml`)

- [ ] `policy.path` 指向有效的 pretrained_model 目录
- [ ] `robot.http_base_url` 设置正确
- [ ] `robot.joint_names` 与策略训练配置一致
- [ ] `robot.cameras` 定义了正确的相机尺寸
- [ ] `inference.fps` 设置合理（推荐 15-30 Hz）

### Replay 配置 (`workflows/robot_interaction/replay_config.yaml`)

- [ ] `dataset.repo_id` 指向有效的 HuggingFace 数据集
- [ ] `dataset.episode` 是有效的 episode 索引
- [ ] `robot.http_base_url` 设置正确
- [ ] `robot.joint_names` 与数据集配置一致

## 测试步骤

### 1. 基础连接测试

```bash
cd /home/zzx23457/lerobot_vlahost

# 1a. 轻量导入验证（不需要真机）
uv run python -c "from lerobot.robots.marvain_m6_http import MarvainM6HttpRobotConfig, MarvainM6HttpRobot; print('import ok')"

# 1b. 无真机时启动 mock echo server（默认监听 0.0.0.0:8010）
python workflows/robot_interaction/mock_echo_server.py --port 8010

# 1c. 端到端冒烟测试（向真实 / mock 服务端拉一次状态）
curl -s http://192.168.10.123:8010/state | python3 -m json.tool | head -30
```

**预期 1a 输出**：
```
import ok
```

**预期 1b 输出**：
```
mock echo server on http://0.0.0.0:8010  (GET /state, POST /action, POST /action_chunk)
INFO:     Started server process [...]
INFO:     Uvicorn running on http://0.0.0.0:8010
```

**预期 1c 输出**（必须有 `joint_states.positions`，否则驱动无法工作）：
```json
{
  "joint_states": {
    "positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  },
  "gripper_left":  ...,
  "gripper_right": ...,
  "quad_image":    ...
}
```

### 2. 回放测试（推荐先做这个）

回放比 deploy 更安全，因为它执行的是已验证的示教数据。

```bash
# 慢速回放（便于观察）
python workflows/robot_interaction/replay.py \
    --repo-id username/my_dataset \
    --episode 0 \
    --fps 10
```

**观察项**：
- [ ] 机器人动作流畅，无突然跳变
- [ ] 无关节位置警告（超出范围）
- [ ] 无 HTTP 连接错误
- [ ] 回放结束后机器人返回 home 位置

### 3. Deploy 测试（推理部署）

确认回放正常后，再进行 deploy 测试。

```bash
# 低速推理（便于观察）
python workflows/robot_interaction/deploy.py \
    --policy-path outputs/train/my_model/pretrained_model \
    --fps 15 \
    --strategy base
```

**观察项**：
- [ ] 推理频率稳定（检查终端输出的 FPS）
- [ ] 机器人动作合理（与训练任务一致）
- [ ] 无异常动作（突然停止、震荡等）
- [ ] 无安全裁剪警告（如果有，检查 safety_stats_path）

## 常见问题检查

### 问题 1: 导入错误

```
ModuleNotFoundError: No module named 'lerobot'
```

**解决**：
```bash
# 检查当前目录
pwd
# 应该在：/home/zzx23457/lerobot_vlahost

# 设置 PYTHONPATH
export PYTHONPATH=/home/zzx23457/lerobot_vlahost/src:$PYTHONPATH

# 或使用绝对路径运行
cd /home/zzx23457/lerobot_vlahost
python workflows/robot_interaction/deploy.py
```

### 问题 2: HTTP 连接超时

```
Failed to connect to HTTP server: ReadTimeout
```

**检查**：
1. 服务器是否运行？`ps aux | grep http`
2. 防火墙设置：`sudo ufw status`
3. 网络连接：`ping 192.168.10.123`
4. 端口监听：`netstat -tuln | grep 8010`

**解决**：
```yaml
# 增加超时时间
robot:
  timeout: 10.0  # 从 5.0 增加到 10.0
```

### 问题 3: 关节名称不匹配

```
KeyError: 'left_arm_joint_1.pos'
```

**检查策略训练配置**：
```bash
# 查看策略配置
cat outputs/train/my_model/pretrained_model/config.json | grep joint

# 查看数据集配置
cat datasets/my_dataset/meta/info.json | grep -A 20 joint_names
```

**解决**：确保 `deploy_config.yaml` 和 `replay_config.yaml` 中的 `joint_names` 与策略/数据集完全一致。

### 问题 4: 安全裁剪频繁

```
WARNING: action clipped: joint 3 (left_arm_joint_4) 75.00° → 70.00°
```

**原因**：策略输出超出训练数据范围。

**解决**：
1. 检查 `safety_stats_path` 是否指向正确的训练数据集
2. 增加安全裕量：
   ```yaml
   robot:
     action_clip_margin_deg: 10.0  # 从 5.0 增加到 10.0
   ```
3. 重新训练策略以改善动作分布

### 问题 5: 图像解码失败

```
ValueError: Failed to decode image from base64 string
```

**检查**：
1. HTTP 服务器返回的图像格式（JPEG/PNG）
2. base64 编码是否正确

**调试**：
```python
# 保存原始 base64 数据进行检查
import base64
with open("/tmp/image.b64", "w") as f:
    f.write(base64_string)

# 手动解码测试
import cv2
import numpy as np
img_bytes = base64.b64decode(base64_string)
img_array = np.frombuffer(img_bytes, dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
print(img.shape)  # 应该是 (H, W, 3)
```

## 性能基准

### 预期性能指标

- **网络延迟**: < 10ms (本地网络)
- **观测获取**: 20-50ms (包括图像解码)
- **动作发送**: 5-10ms
- **总推理延迟**: 50-100ms @ 30 Hz
- **CPU 使用率**: 10-30% (推理)
- **内存使用**: 500MB-2GB (取决于策略大小)

### 性能监控

```bash
# 监控推理频率
# 终端输出会显示实际 FPS，应接近配置的 fps 值

# 监控系统资源
htop  # 查看 CPU 和内存
nvidia-smi -l 1  # 查看 GPU 使用率
```

## 安全注意事项

⚠️ **重要安全提示**：

1. **首次运行**：始终以低速（fps=5-10）开始，观察机器人行为
2. **紧急停止**：准备好物理急停按钮，或 `Ctrl+C` 终止程序
3. **工作空间**：确保机器人周围没有障碍物和人员
4. **数据验证**：使用 replay 验证数据集质量后再进行 deploy
5. **安全边界**：始终设置 `safety_stats_path` 启用安全裁剪
6. **限制运动**：设置合理的 `max_relative_target_deg`（推荐 5-15°）

## 下一步

检查清单全部完成后：

- [ ] 运行基础连接测试
- [ ] 运行低速回放测试
- [ ] 运行正常速度回放
- [ ] 运行低速 deploy 测试
- [ ] 运行正常速度 deploy
- [ ] 根据需要调整配置
- [ ] 记录任何异常行为

## 反馈和改进

如有问题或改进建议，请记录：

- 错误信息和完整日志
- 配置文件内容
- HTTP 服务器版本和状态
- 复现步骤

---

**检查清单版本**: 1.0  
**最后更新**: 2026-06-26
