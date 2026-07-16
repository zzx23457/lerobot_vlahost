# 数据清洗工具快速参考

## 一句话描述

检测并删除 observation.state 前 7 位异常全零的 episode，并重新索引数据集。

## 最小使用示例

```bash
# 查看问题
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/my_dataset \
    --report-only

# 清洗
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/my_dataset \
    --output-path datasets/my_dataset_cleaned
```

## 所有选项

```bash
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path PATH              # 必需：数据集路径
    [--output-path PATH]            # 可选：输出路径（默认：就地清洗）
    [--report-only]                 # 可选：仅生成报告
    [--dry-run]                     # 可选：模拟清洗不写入
    [--zero-threshold N]            # 可选：零值位数阈值（默认：7）
```

## 输出示例

```
================================================================================
DIRTY EPISODE DETECTION REPORT
================================================================================
Dataset: datasets/26-06-17-11-32-27_v2

SUMMARY:
  Total episodes:        100
  Dirty episodes:        7 (7.0%)
  Clean episodes:        93 (93.0%)

  Total rows:            21,092
  Dirty rows:            1,301 (6.2%)
  Clean rows:            19,791 (93.8%)

DIRTY EPISODES:
  [33, 35, 37, 55, 63, 67, 73]
================================================================================

CLEANING COMPLETE ✓
Cleaned dataset: datasets/26-06-17-11-32-27_v2_cleaned
Episodes: 100 → 93 (-7)
Rows: 21,092 → 19,791 (-1,301)
```

## 脏数据特征

```python
# 脏数据模式：observation.state 前 7 位全为 0
dirty_obs = [0, 0, 0, 0, 0, 0, 0, -66.126, -18.976, ...]

# 正常数据：所有位置都有非零值
clean_obs = [66.126, -19.039, -80.546, -84.680, -46.944, ...]
```

## 安全性

✓ `--report-only` 不修改任何文件  
✓ `--dry-run` 模拟所有操作  
✓ 就地清洗前需用户确认  
✓ 自动验证清洗后数据完整性  
✓ 详细日志记录每个操作  

## 清洗内容

1. **数据文件** (`data/chunk-*/file-*.parquet`)
   - 删除脏 episode 的所有行
   - 重新映射 episode_index (0, 1, 2, ...)
   - 重新索引全局 index

2. **元数据** (`meta/episodes/chunk-*/file-*.parquet`)
   - 删除脏 episode 的元数据行
   - 更新 episode_index
   - 重算 dataset_from_index 和 dataset_to_index

3. **其他文件**
   - 复制 meta/tasks.parquet
   - 复制 videos/ 目录（脏 episode 的视频仍存在）

## 验证检查

清洗后自动验证：
- ✓ Episode 索引连续（0 到 N-1）
- ✓ 全局 index 连续
- ✓ 没有剩余脏数据
- ✓ 行数匹配预期

## 文档位置

完整文档：[workflows/data_processing/README.md](README.md)  
分析总结：[workflows/data_processing/数据清洗分析总结.md](数据清洗分析总结.md)  
使用示例：[workflows/data_processing/example_clean_dataset.py](example_clean_dataset.py)

## 故障排查

**"Dataset structure validation failed"**  
→ 检查目录结构：需要 `data/chunk-*/` 和 `meta/episodes/chunk-*/`

**"Output path already exists"**  
→ 删除现有输出目录或用 `--output-path` 指定其他路径

**"No dirty episodes found"**  
→ 数据集已经干净，无需清洗

**"Verification failed"**  
→ 内部错误，不要使用清洗后的数据集，请报告 bug
