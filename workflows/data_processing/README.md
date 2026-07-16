# Data Processing Workflows

This directory contains scripts for processing and cleaning LeRobot datasets.

## Scripts

### `clean_dirty_episodes.py`

Detects and removes episodes with anomalous zero values in `observation.state` arrays. This typically indicates sensor or data collection failures.

**Dirty Data Pattern:**
- First 7 positions of `observation.state` array are all zeros across all frames in the episode
- Remaining positions (7-15) contain normal sensor readings
- Affects both raw data files and episode metadata statistics

**Usage:**

```bash
# 1. Detection only - see which episodes would be removed
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/my_dataset \
    --report-only

# 2. Dry run - simulate cleaning without making changes
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/my_dataset \
    --dry-run

# 3. Clean to a new location (recommended for first-time use)
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/my_dataset \
    --output-path datasets/my_dataset_cleaned

# 4. Clean in-place (overwrites original dataset - use with caution!)
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/my_dataset
```

**Example Output:**

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

DIRTY EPISODE STATISTICS:
  Min length:            167 rows
  Max length:            237 rows
  Mean length:           185.9 rows
  Median length:         174.0 rows

CLEAN EPISODE STATISTICS:
  Min length:            172 rows
  Max length:            321 rows
  Mean length:           212.8 rows
  Median length:         209.0 rows

DIRTY EPISODES:
  [33, 35, 37, 55, 63, 67, 73]
================================================================================
```

**What Gets Cleaned:**

1. **Data files** (`data/chunk-*/file-*.parquet`):
   - Removes all rows belonging to dirty episodes
   - Remaps `episode_index` to sequential values (0, 1, 2, ...)
   - Reindexes global `index` column to be sequential
   
2. **Episode metadata** (`meta/episodes/chunk-*/file-*.parquet`):
   - Removes metadata rows for dirty episodes
   - Updates `episode_index` to match data files
   - Recomputes `dataset_from_index` and `dataset_to_index` based on cleaned data

3. **Other files**:
   - Copies `meta/tasks.parquet` unchanged
   - Copies `videos/` directory (videos for dirty episodes remain - manual cleanup if needed)

**Safety Features:**

- **Report-only mode** shows what would be removed without making changes
- **Dry-run mode** simulates all operations without writing files
- **Backup warning** for in-place cleaning
- **Automatic verification** after cleaning:
  - Episode indices are sequential (0 to N-1)
  - Global index is sequential
  - No dirty episodes remain
  - Row counts match expected values

**Advanced Options:**

```bash
# Custom zero threshold (default: 7)
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/my_dataset \
    --zero-threshold 5 \
    --report-only
```

**Workflow Recommendations:**

1. **First run:** Use `--report-only` to understand the extent of dirty data
2. **Validation:** Use `--dry-run` to simulate cleaning and review changes
3. **Initial clean:** Use `--output-path` to create a cleaned copy
4. **Verify:** Test the cleaned dataset with your training pipeline
5. **Replace:** Once verified, replace the original dataset or use cleaned version

**Integration with LeRobot:**

The cleaned dataset maintains full compatibility with LeRobot's dataset loaders:

```python
from lerobot.datasets import LeRobotDataset

# Load cleaned dataset
dataset = LeRobotDataset("datasets/my_dataset_cleaned")

# Episode indices are sequential starting from 0
print(f"Episodes: {dataset.episode_indices}")
print(f"Total frames: {len(dataset)}")
```

**Example: Clean the provided example dataset**

```bash
# 1. See the report
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/26-06-17-11-32-27_v2 \
    --report-only

# 2. Create cleaned copy
python workflows/data_processing/clean_dirty_episodes.py \
    --dataset-path datasets/26-06-17-11-32-27_v2 \
    --output-path datasets/26-06-17-11-32-27_v2_cleaned

# Output:
# Episodes: 100 → 93 (-7)
# Rows: 21,092 → 19,791 (-1,301)
```

## Directory Structure

```
workflows/data_processing/
├── README.md                    # This file
└── clean_dirty_episodes.py      # Dirty episode cleaning script
```

## Requirements

- pandas
- numpy
- pyarrow (for parquet support)

These are already included in the LeRobot environment.

## Notes

- **Videos:** The script copies the entire `videos/` directory. Videos for dirty episodes will remain and take up disk space. If this is a concern, manually identify and remove video files associated with dirty episode indices after cleaning.

- **In-place cleaning:** When cleaning in-place without `--output-path`, the script modifies the original dataset. Create a backup first or use version control to track the original state.

- **Performance:** The script processes datasets efficiently by reading/writing parquet files in chunks. Typical cleaning time is a few seconds for datasets with ~100 episodes.

## Troubleshooting

**"Dataset structure validation failed"**
- Ensure the dataset has the expected structure: `data/chunk-*/*.parquet` and `meta/episodes/chunk-*/*.parquet`
- Check that parquet files are not corrupted

**"Output path already exists"**
- Remove the existing output directory or choose a different path
- Or use in-place cleaning (with caution)

**"Verification failed"**
- This indicates an internal error in the cleaning logic
- Check the log for details and file a bug report
- Do not use the cleaned dataset if verification fails

**"No dirty episodes found"**
- Your dataset is already clean!
- No action needed

## Future Enhancements

Potential improvements for future versions:

- Support for multiple dirty data patterns (not just leading zeros)
- Video file cleanup based on episode indices
- Support for multi-chunk datasets with complex file organization
- Configurable detection thresholds and patterns
- Integration with LeRobot CLI (`lerobot-clean-dataset`)
