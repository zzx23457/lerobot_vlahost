# 训练脚本中的脏数据检测与清洗集成

## 概述

`train_act.sh` 在 Phase 1 数据校验阶段已集成脏数据自动检测功能。

## 工作流程

### 1. 自动检测（必须执行）

训练脚本在数据校验阶段会自动运行脏数据检测：

```bash
./workflows/model_training/train_act.sh check
```

或运行完整训练流程时：

```bash
./workflows/model_training/train_act.sh
```

### 2. 交互式清洗（可选，需用户确认）

如果检测到脏数据，脚本会显示报告并提供三个选项：

```
⚠ 发现脏数据 episode!

SUMMARY:
  Total episodes:        100
  Dirty episodes:        7 (7.0%)
  Clean episodes:        93 (93.0%)
  Total rows:            21,092
  Dirty rows:            1,301 (6.2%)
  Clean rows:            19,791 (93.8%)

脏数据 episode 的特征: observation.state 前 7 位全为 0
建议清洗这些 episode 以提高训练质量

选项:
  1) 清洗数据集 (自动创建 ${DATASET_ROOT}_cleaned 并使用)
  2) 跳过清洗,继续使用原数据集 (不推荐,可能影响训练质量)
  3) 取消训练,手动处理

请选择 [1/2/3]:
```

### 3. 用户选择说明

**选项 1: 自动清洗（推荐）**
- 自动创建清洗后的数据集副本（原数据集名称 + `_cleaned` 后缀）
- 如果清洗后的数据集已存在，会询问是否覆盖或直接使用
- 清洗完成后，训练脚本会自动切换到清洗后的数据集
- 后续所有 phase（烟测、训练）都使用清洗后的数据集

**选项 2: 跳过清洗**
- 继续使用原数据集进行训练
- **警告**: 脏数据可能影响训练质量
- 仅在你确认这些 episode 不影响训练时选择此选项

**选项 3: 取消训练**
- 退出训练流程
- 显示手动清洗命令，供用户自行处理
- 适合需要检查脏数据或执行自定义清洗策略的情况

## 示例场景

### 场景 1: 首次发现脏数据

```bash
$ ./workflows/model_training/train_act.sh

[17:30:00] Phase 0: 环境检查
[17:30:01] ✓ Phase 0 通过
[17:30:01] Phase 1: 数据校验
[17:30:02] ✓ 数据集目录存在
[17:30:02] ✓ data/ meta/ videos/ 都在
[17:30:03] ✓ info.json 字段符合预期
[17:30:04] ✓ 交叉一致性通过
[17:30:05] ✓ sanity_check.py 通过
[17:30:06] 检测脏数据 episode (observation.state 前 7 位异常全零) ...
[17:30:07] ⚠ 发现脏数据 episode!

SUMMARY:
  Total episodes:        100
  Dirty episodes:        7 (7.0%)
  Clean episodes:        93 (93.0%)
  ...

请选择 [1/2/3]: 1

[17:30:10] 开始清洗数据集...
[17:30:15] ✓ 数据清洗完成,已切换到: datasets/26-06-17-11-32-27_v2_cleaned
[17:30:16] ✓ Phase 1 通过
[17:30:16] Phase 2: 烟测 (50 步,batch=2)
...
```

### 场景 2: 清洗后的数据集已存在

```bash
$ ./workflows/model_training/train_act.sh

...
[17:30:06] 检测脏数据 episode ...
[17:30:07] ⚠ 发现脏数据 episode!
...
请选择 [1/2/3]: 1

[17:30:10] ⚠ 清洗后的数据集已存在: datasets/26-06-17-11-32-27_v2_cleaned
是否删除并重新清洗? (y/N) n

[17:30:12] 使用现有清洗后的数据集
[17:30:12] ✓ 已切换到清洗后的数据集: datasets/26-06-17-11-32-27_v2_cleaned
[17:30:12] ✓ Phase 1 通过
...
```

### 场景 3: 数据集干净，无需清洗

```bash
$ ./workflows/model_training/train_act.sh

...
[17:30:06] 检测脏数据 episode ...
[17:30:07] ✓ 未发现脏数据 episode
[17:30:08] 检查数据集时间戳对齐 ...
[17:30:09] ✓ Phase 1 通过
[17:30:09] Phase 2: 烟测 (50 步,batch=2)
...
```

### 场景 4: 手动处理

```bash
$ ./workflows/model_training/train_act.sh

...
[17:30:06] 检测脏数据 episode ...
[17:30:07] ⚠ 发现脏数据 episode!
...
请选择 [1/2/3]: 3

[17:30:10] 用户取消训练

手动清洗命令:
  python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/26-06-17-11-32-27_v2 \
    --output-path datasets/26-06-17-11-32-27_v2_cleaned
```

## 技术细节

### 检测时机

脏数据检测在 Phase 1 的以下位置执行：

1. ✓ 环境检查
2. ✓ 数据集目录结构验证
3. ✓ info.json 字段校验
4. ✓ 交叉一致性校验
5. ✓ sanity_check.py
6. **→ 脏数据检测（这里）** ← 新增
7. ✓ 时间戳对齐检查

### 数据集路径切换

选择自动清洗后，脚本会：

1. 创建 `${DATASET_ROOT}_cleaned` 目录
2. 运行清洗脚本，输出到新目录
3. 更新 `DATASET_ROOT` 变量指向清洗后的数据集
4. 导出 `DATASET_ROOT` 环境变量供后续 phase 使用

这意味着后续的烟测、训练、评估都会自动使用清洗后的数据集。

### 幂等性

如果清洗后的数据集已存在：
- 脚本会询问是否重新清洗
- 选择 "N" 会直接使用现有的清洗数据集
- 选择 "y" 会删除并重新清洗

这允许在多次运行训练脚本时复用清洗结果，避免重复清洗。

## 环境变量控制

可以通过环境变量预先指定数据集路径：

```bash
# 直接使用清洗后的数据集，跳过清洗步骤
export DATASET_ROOT="datasets/26-06-17-11-32-27_v2_cleaned"
./workflows/model_training/train_act.sh
```

## 非交互式使用

如果需要在 CI/CD 或自动化脚本中使用，建议预先清洗数据集：

```bash
# 步骤 1: 预先清洗
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/my_dataset \
    --output-path datasets/my_dataset_cleaned

# 步骤 2: 使用清洗后的数据集训练
export DATASET_ROOT="datasets/my_dataset_cleaned"
./workflows/model_training/train_act.sh
```

## 注意事项

1. **磁盘空间**: 清洗会创建数据集的完整副本（不包括视频软链接），确保有足够的磁盘空间

2. **原数据集保留**: 清洗不会修改原数据集，原始数据仍然完整保留

3. **视频文件**: 清洗后的数据集会复制视频目录的软链接，脏 episode 对应的视频仍然存在

4. **重复运行**: 如果多次运行训练脚本，清洗后的数据集会被复用，不会重复清洗

5. **手动验证**: 首次使用时，建议先用 `--report-only` 手动查看脏数据报告，了解数据质量

## 相关文档

- 数据清洗工具详细文档: [workflows/data_processing/README.md](../data_processing/README.md)
- 快速参考: [workflows/data_processing/QUICKREF.md](../data_processing/QUICKREF.md)
- 分析总结: [workflows/data_processing/数据清洗分析总结.md](../data_processing/数据清洗分析总结.md)
