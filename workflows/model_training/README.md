# ACT Training Workflow

端到端的工作流:用 `26-06-17-11-32-27_v2` 训练一个 ACT 策略。

## 目录

| 文件 | 用途 |
|---|---|
| [`act_training_workflow.md`](act_training_workflow.md) | **主文档**——完整步骤、决策点、故障排查 |
| [`train_act.sh`](train_act.sh) | 一键运行训练的可执行 shell 脚本 (含 5 个 phase) |
| [`DIRTY_DATA_INTEGRATION.md`](DIRTY_DATA_INTEGRATION.md) | 脏数据检测与清洗集成说明（训练脚本内置）|
| [`sanity_check.py`](../data_processing/sanity_check.py) | 数据集 + 环境烟测脚本(训练前必跑) |
| [`README.md`](README.md) | 本文件 |

## 快速开始(给赶时间的人)

```bash
cd /home/zzx23457/lerobot
./workflows/model_training/train_act.sh   # 一键跑完: 校验 → 烟测 → 训练 → 提示评估
```

**注意**: 训练脚本会自动检测脏数据（observation.state 前 7 位异常全零的 episode）。如果发现脏数据，会提示你选择：
1. 自动清洗并使用清洗后的数据集（推荐）
2. 跳过清洗继续训练（不推荐，可能影响训练质量）
3. 取消训练，手动处理

详见 [DIRTY_DATA_INTEGRATION.md](DIRTY_DATA_INTEGRATION.md)

## 数据集速览

| 字段 | 值 |
|---|---|
| 路径 | `/home/zzx23457/lerobot/datasets/26-06-17-11-32-27_v2` |
| 格式 | LeRobot v3.0 |
| Episodes | 100 |
| Frames | 21,092 |
| FPS | 25 |
| State / Action 维度 | 16 (= 14 关节 + 2 夹爪,单位度) |
| 相机 | right_eye / left_wrist / right_wrist (640×480) |
| Task | 详见 `meta/tasks.parquet`(由录制端写入) |
| 算力 | RTX 4090 (48 GB) — batch=4 起步,batch=8 视显存占用 |

## 阅读顺序

1. **第一次跑**: 先看 [act_training_workflow.md](act_training_workflow.md) 的 Phase 0 / 1 / 2(环境、校验、烟测)
2. **正式训练**: 看 Phase 3 的命令与时长预估
3. **训练后**: Phase 4(评估)+ Phase 5(常见故障)
