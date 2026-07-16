# 配置对比分析：record vs deploy

## 同事的 record 命令参数

```bash
lerobot-record \
  --robot.type=marvin \
  --robot.id=marvin_pro_001 \
  --robot.robot_ip=192.168.10.190 \
  --robot.control_mode=impedance \
  --robot.cameras='...' \
  --policy.path=/media/marvin/80E88E30E88E250E/zzx23457/200000/pretrained_model \
  --dataset.repo_id=hukewei/eval_pro_act5 \
  --dataset.single_task="eval_pro_act5" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=660 \
  --dataset.encoder_threads=2 \
  --dataset.root=/home/marvin/.cache/huggingface/lerobot/hukewei/eval_pro_act5 \
  --resume=true
```

## 你的 deploy_config.yaml

```yaml
policy:
  path: outputs/train/act_v2_20260701_181934/checkpoints/200000/pretrained_model
  device: cuda

dataset:
  root: datasets/deploy
  repo_id: datasets/deploy
  single_task: "manipulation task"

robot:
  type: marvain_m6_http
  id: marvin_http_001
  http_base_url: http://192.168.10.123:8010
  timeout: 5.0
  safety_stats_path: datasets/26-06-26-10-35-39_v2
  action_clip_margin_deg: 5.0
  max_relative_target_deg: 10.0
  warn_on_observation_out_of_range: true

inference:
  type: sync
  fps: 30.0
  duration: 0.0
  max_steps: 10000
  strategy: base
  interpolation_multiplier: 1
  return_to_initial_position: True
  use_torch_compile: false
```

## 关键差异及建议

### 1. **控制模式 (Critical)**
- **同事**: `--robot.control_mode=impedance`
- **你**: 未显式设置
- **建议**: 添加 `control_mode: impedance` 到你的 robot 配置中

### 2. **插值倍数 (Important)**
- **同事**: 未设置（使用 lerobot 默认值 = 1）
- **你**: `interpolation_multiplier: 1`
- **状态**: 已对齐 ✓

### 3. **FPS (Important)**
- **同事**: 未设置（使用默认值 = 30）
- **你**: `fps: 30.0`
- **状态**: 已对齐 ✓

### 4. **安全裁剪参数 (May Affect Performance)**
- **同事**: 未设置（不启用安全裁剪）
- **你**: 
  - `safety_stats_path: datasets/26-06-26-10-35-39_v2`
  - `action_clip_margin_deg: 5.0`
  - `max_relative_target_deg: 10.0`
- **建议**: 尝试禁用安全裁剪，设置 `safety_stats_path: null`

### 5. **Duration / Episode Time**
- **同事**: `--dataset.episode_time_s=660` (11分钟)
- **你**: `duration: 0.0` (无限)
- **状态**: 不影响效果，仅控制运行时长

### 6. **Torch Compile**
- **同事**: 未设置（默认 false）
- **你**: `use_torch_compile: false`
- **状态**: 已对齐 ✓

### 7. **数据集配置**
- **同事**: 
  - `repo_id: hukewei/eval_pro_act5`
  - `single_task: "eval_pro_act5"`
  - `encoder_threads: 2`
- **你**: 
  - `repo_id: datasets/deploy`
  - `single_task: "manipulation task"`
- **状态**: 不影响推理效果

## 推荐修改

### 高优先级（可能影响效果）

1. **添加 control_mode**:
```yaml
robot:
  type: marvain_m6_http
  control_mode: impedance  # 添加这一行
```

2. **禁用安全裁剪**（至少测试一下）:
```yaml
robot:
  safety_stats_path: null  # 或者注释掉这一行
  # action_clip_margin_deg: 5.0  # 注释掉
  # max_relative_target_deg: 10.0  # 注释掉
```

### 中优先级（可能影响稳定性）

3. **检查 max_relative_target_deg**:
   - 同事没有设置此参数，默认应该是不限制或者很大的值
   - 你设置的 `10.0` 度可能限制了机器人的动作幅度
   - 建议设置为 `null` 或更大的值（如 30.0）

### 其他检查项

4. **确认相机配置一致**: 同事使用 3 个相机，你也使用了相同的相机配置 ✓
5. **确认模型路径**: 确保你们使用的是同一个 checkpoint ✓
6. **确认机器人 IP**: 同事是 192.168.10.190，你是 192.168.10.123 - 确认这是同一台机器人

## LeRobot 默认值参考

根据源码，以下是关键参数的默认值：
- `fps`: 30
- `interpolation_multiplier`: 1
- `use_torch_compile`: false
- `return_to_initial_position`: true
- `duration`: 0.0 (无限)
- `max_steps`: 10000
- `strategy`: base
- `inference.type`: sync
