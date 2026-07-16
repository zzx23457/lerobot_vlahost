#!/usr/bin/env python3
"""
Clean Dirty Episodes from LeRobot Dataset

This script detects and removes episodes with anomalous zero values in observation.state arrays.

Dirty data pattern:
- observation.state[:7] are all zeros across all frames in the episode
- This indicates sensor or data collection failure
- Affects both raw data files and episode metadata

Usage:
    # Detect and report only
    python clean_dirty_episodes.py --dataset-path datasets/my_dataset --report-only

    # Dry run (show what would be removed)
    python clean_dirty_episodes.py --dataset-path datasets/my_dataset --dry-run

    # Clean in-place (overwrites original files)
    python clean_dirty_episodes.py --dataset-path datasets/my_dataset

    # Clean to new location
    python clean_dirty_episodes.py --dataset-path datasets/my_dataset --output-path datasets/my_dataset_cleaned

Example:
    python clean_dirty_episodes.py --dataset-path datasets/26-06-17-11-32-27_v2 --dry-run
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_dataset_structure(dataset_path: Path) -> bool:
    """
    Validate that the dataset has the expected structure.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        True if valid, False otherwise
    """
    required_dirs = ['data', 'meta']
    required_meta_subdirs = ['meta/episodes']

    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if not dir_path.exists():
            logger.error(f"Missing required directory: {dir_path}")
            return False

    for subdir in required_meta_subdirs:
        subdir_path = dataset_path / subdir
        if not subdir_path.exists():
            logger.error(f"Missing required subdirectory: {subdir_path}")
            return False

    # Check for at least one data chunk
    data_chunks = list((dataset_path / 'data').glob('chunk-*'))
    if not data_chunks:
        logger.error("No data chunks found in data/ directory")
        return False

    # Check for at least one metadata chunk
    meta_chunks = list((dataset_path / 'meta' / 'episodes').glob('chunk-*'))
    if not meta_chunks:
        logger.error("No episode metadata chunks found in meta/episodes/ directory")
        return False

    logger.info(f"Dataset structure validated: {len(data_chunks)} data chunks, {len(meta_chunks)} metadata chunks")
    return True


def detect_dirty_episodes(dataset_path: Path, zero_threshold: int = 7) -> Set[int]:
    """
    Detect episodes with anomalous zero patterns in observation.state.

    An episode is considered dirty if ALL rows have observation.state[:zero_threshold] == 0.

    Args:
        dataset_path: Path to dataset directory
        zero_threshold: Number of leading positions that must be zero (default: 7)

    Returns:
        Set of dirty episode indices
    """
    logger.info("Starting dirty episode detection...")
    dirty_episodes = set()

    # Find all data chunks
    data_dir = dataset_path / 'data'
    chunk_dirs = sorted(data_dir.glob('chunk-*'))

    total_episodes_checked = 0

    for chunk_dir in chunk_dirs:
        chunk_name = chunk_dir.name
        parquet_files = sorted(chunk_dir.glob('file-*.parquet'))

        logger.info(f"Scanning {chunk_name} ({len(parquet_files)} files)...")

        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)

                # Group by episode
                for episode_idx in df['episode_index'].unique():
                    if episode_idx in dirty_episodes:
                        # Already identified as dirty
                        continue

                    episode_data = df[df['episode_index'] == episode_idx]
                    total_episodes_checked += 1

                    # Check if ALL rows have zeros in first zero_threshold positions
                    is_dirty = True
                    for _, row in episode_data.iterrows():
                        obs_state = row['observation.state']
                        if not np.all(obs_state[:zero_threshold] == 0):
                            is_dirty = False
                            break

                    if is_dirty:
                        dirty_episodes.add(episode_idx)
                        logger.warning(
                            f"  → Found dirty episode {episode_idx} in {chunk_name}/{parquet_file.name} "
                            f"({len(episode_data)} rows)"
                        )

            except Exception as e:
                logger.error(f"Error reading {parquet_file}: {e}")
                raise

    logger.info(
        f"Detection complete: {len(dirty_episodes)} dirty episodes found "
        f"out of {total_episodes_checked} total episodes"
    )

    return dirty_episodes


def calculate_episode_stats(dataset_path: Path, dirty_episodes: Set[int]) -> Dict:
    """
    Calculate statistics about dirty vs clean episodes.

    Args:
        dataset_path: Path to dataset directory
        dirty_episodes: Set of dirty episode indices

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_episodes': 0,
        'dirty_episodes': len(dirty_episodes),
        'clean_episodes': 0,
        'total_rows': 0,
        'dirty_rows': 0,
        'clean_rows': 0,
        'dirty_episode_lengths': [],
        'clean_episode_lengths': [],
    }

    data_dir = dataset_path / 'data'
    chunk_dirs = sorted(data_dir.glob('chunk-*'))

    episodes_seen = set()

    for chunk_dir in chunk_dirs:
        parquet_files = sorted(chunk_dir.glob('file-*.parquet'))

        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            stats['total_rows'] += len(df)

            for episode_idx in df['episode_index'].unique():
                if episode_idx in episodes_seen:
                    continue
                episodes_seen.add(episode_idx)
                stats['total_episodes'] += 1

                episode_data = df[df['episode_index'] == episode_idx]
                episode_length = len(episode_data)

                if episode_idx in dirty_episodes:
                    stats['dirty_rows'] += episode_length
                    stats['dirty_episode_lengths'].append(episode_length)
                else:
                    stats['clean_rows'] += episode_length
                    stats['clean_episode_lengths'].append(episode_length)

    stats['clean_episodes'] = stats['total_episodes'] - stats['dirty_episodes']

    return stats


def generate_report(dataset_path: Path, dirty_episodes: Set[int]) -> str:
    """
    Generate a detailed report about dirty episodes.

    Args:
        dataset_path: Path to dataset directory
        dirty_episodes: Set of dirty episode indices

    Returns:
        Report string
    """
    stats = calculate_episode_stats(dataset_path, dirty_episodes)

    report_lines = [
        "=" * 80,
        "DIRTY EPISODE DETECTION REPORT",
        "=" * 80,
        f"Dataset: {dataset_path}",
        "",
        "SUMMARY:",
        f"  Total episodes:        {stats['total_episodes']}",
        f"  Dirty episodes:        {stats['dirty_episodes']} ({stats['dirty_episodes']/stats['total_episodes']*100:.1f}%)",
        f"  Clean episodes:        {stats['clean_episodes']} ({stats['clean_episodes']/stats['total_episodes']*100:.1f}%)",
        "",
        f"  Total rows:            {stats['total_rows']:,}",
        f"  Dirty rows:            {stats['dirty_rows']:,} ({stats['dirty_rows']/stats['total_rows']*100:.1f}%)",
        f"  Clean rows:            {stats['clean_rows']:,} ({stats['clean_rows']/stats['total_rows']*100:.1f}%)",
        "",
    ]

    if stats['dirty_episode_lengths']:
        report_lines.extend([
            "DIRTY EPISODE STATISTICS:",
            f"  Min length:            {min(stats['dirty_episode_lengths'])} rows",
            f"  Max length:            {max(stats['dirty_episode_lengths'])} rows",
            f"  Mean length:           {np.mean(stats['dirty_episode_lengths']):.1f} rows",
            f"  Median length:         {np.median(stats['dirty_episode_lengths']):.1f} rows",
            "",
        ])

    if stats['clean_episode_lengths']:
        report_lines.extend([
            "CLEAN EPISODE STATISTICS:",
            f"  Min length:            {min(stats['clean_episode_lengths'])} rows",
            f"  Max length:            {max(stats['clean_episode_lengths'])} rows",
            f"  Mean length:           {np.mean(stats['clean_episode_lengths']):.1f} rows",
            f"  Median length:         {np.median(stats['clean_episode_lengths']):.1f} rows",
            "",
        ])

    report_lines.extend([
        "DIRTY EPISODES:",
        f"  {sorted(dirty_episodes)}",
        "=" * 80,
    ])

    return "\n".join(report_lines)


def create_episode_mapping(dirty_episodes: Set[int], total_episodes: int) -> Dict[int, int]:
    """
    Create mapping from old episode indices to new sequential indices.

    Args:
        dirty_episodes: Set of dirty episode indices to remove
        total_episodes: Total number of episodes before cleaning

    Returns:
        Dictionary mapping old_index -> new_index (dirty episodes map to -1)
    """
    mapping = {}
    new_idx = 0

    for old_idx in range(total_episodes):
        if old_idx in dirty_episodes:
            mapping[old_idx] = -1  # Mark for removal
        else:
            mapping[old_idx] = new_idx
            new_idx += 1

    return mapping


def clean_data_files(
    dataset_path: Path,
    dirty_episodes: Set[int],
    episode_mapping: Dict[int, int],
    output_path: Path,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Clean data files by removing dirty episodes and reindexing.

    Args:
        dataset_path: Source dataset path
        dirty_episodes: Set of episode indices to remove
        episode_mapping: Mapping from old to new episode indices
        output_path: Output dataset path
        dry_run: If True, don't write files

    Returns:
        Tuple of (files_processed, rows_removed)
    """
    logger.info("Cleaning data files...")

    data_dir = dataset_path / 'data'
    output_data_dir = output_path / 'data'

    if not dry_run:
        output_data_dir.mkdir(parents=True, exist_ok=True)

    chunk_dirs = sorted(data_dir.glob('chunk-*'))
    files_processed = 0
    total_rows_removed = 0
    global_index = 0

    for chunk_dir in chunk_dirs:
        chunk_name = chunk_dir.name
        output_chunk_dir = output_data_dir / chunk_name

        if not dry_run:
            output_chunk_dir.mkdir(parents=True, exist_ok=True)

        parquet_files = sorted(chunk_dir.glob('file-*.parquet'))

        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            original_rows = len(df)

            # Filter out dirty episodes
            df_clean = df[~df['episode_index'].isin(dirty_episodes)].copy()
            rows_removed = original_rows - len(df_clean)
            total_rows_removed += rows_removed

            if len(df_clean) == 0:
                logger.info(f"  {chunk_name}/{parquet_file.name}: All rows removed (skipping file)")
                continue

            # Remap episode indices
            df_clean['episode_index'] = df_clean['episode_index'].map(episode_mapping)

            # Reindex global index
            df_clean['index'] = range(global_index, global_index + len(df_clean))
            global_index += len(df_clean)

            # Write output
            output_file = output_chunk_dir / parquet_file.name

            if not dry_run:
                df_clean.to_parquet(output_file, index=False)

            logger.info(
                f"  {chunk_name}/{parquet_file.name}: "
                f"{original_rows} → {len(df_clean)} rows (-{rows_removed})"
            )

            files_processed += 1

    logger.info(f"Data files cleaned: {files_processed} files, {total_rows_removed} rows removed")
    return files_processed, total_rows_removed


def clean_metadata_files(
    dataset_path: Path,
    dirty_episodes: Set[int],
    episode_mapping: Dict[int, int],
    output_path: Path,
    dry_run: bool = False
) -> Tuple[int, int]:
    """
    Clean episode metadata files by removing dirty episodes and updating references.

    Args:
        dataset_path: Source dataset path
        dirty_episodes: Set of episode indices to remove
        episode_mapping: Mapping from old to new episode indices
        output_path: Output dataset path
        dry_run: If True, don't write files

    Returns:
        Tuple of (files_processed, episodes_removed)
    """
    logger.info("Cleaning metadata files...")

    meta_dir = dataset_path / 'meta' / 'episodes'
    output_meta_dir = output_path / 'meta' / 'episodes'

    if not dry_run:
        output_meta_dir.mkdir(parents=True, exist_ok=True)

    chunk_dirs = sorted(meta_dir.glob('chunk-*'))
    files_processed = 0
    total_episodes_removed = 0

    # Build cumulative row count mapping for dataset_from_index / dataset_to_index
    # We need to recompute these based on cleaned data
    data_dir = output_path / 'data' if not dry_run else dataset_path / 'data'
    episode_row_ranges = {}
    current_index = 0

    # Scan cleaned data files to get actual row ranges
    if not dry_run:
        for chunk_dir in sorted((output_path / 'data').glob('chunk-*')):
            for parquet_file in sorted(chunk_dir.glob('file-*.parquet')):
                df = pd.read_parquet(parquet_file)
                for ep_idx in df['episode_index'].unique():
                    ep_data = df[df['episode_index'] == ep_idx]
                    from_idx = current_index
                    to_idx = current_index + len(ep_data)
                    episode_row_ranges[ep_idx] = (from_idx, to_idx)
                    current_index = to_idx

    for chunk_dir in chunk_dirs:
        chunk_name = chunk_dir.name
        output_chunk_dir = output_meta_dir / chunk_name

        if not dry_run:
            output_chunk_dir.mkdir(parents=True, exist_ok=True)

        parquet_files = sorted(chunk_dir.glob('file-*.parquet'))

        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            original_episodes = len(df)

            # Filter out dirty episodes
            df_clean = df[~df['episode_index'].isin(dirty_episodes)].copy()
            episodes_removed = original_episodes - len(df_clean)
            total_episodes_removed += episodes_removed

            if len(df_clean) == 0:
                logger.info(f"  {chunk_name}/{parquet_file.name}: All episodes removed (skipping file)")
                continue

            # Remap episode indices
            df_clean['episode_index'] = df_clean['episode_index'].map(episode_mapping)

            # Update dataset_from_index and dataset_to_index
            if not dry_run and episode_row_ranges:
                for idx, row in df_clean.iterrows():
                    ep_idx = row['episode_index']
                    if ep_idx in episode_row_ranges:
                        from_idx, to_idx = episode_row_ranges[ep_idx]
                        df_clean.at[idx, 'dataset_from_index'] = from_idx
                        df_clean.at[idx, 'dataset_to_index'] = to_idx

            # Write output
            output_file = output_chunk_dir / parquet_file.name

            if not dry_run:
                df_clean.to_parquet(output_file, index=False)

            logger.info(
                f"  {chunk_name}/{parquet_file.name}: "
                f"{original_episodes} → {len(df_clean)} episodes (-{episodes_removed})"
            )

            files_processed += 1

    logger.info(f"Metadata files cleaned: {files_processed} files, {total_episodes_removed} episodes removed")
    return files_processed, total_episodes_removed


def copy_other_files(dataset_path: Path, output_path: Path, dry_run: bool = False):
    """
    Copy other files (tasks.parquet, videos, etc.) to output directory.

    Args:
        dataset_path: Source dataset path
        output_path: Output dataset path
        dry_run: If True, don't copy files
    """
    logger.info("Copying other dataset files...")

    # Copy tasks.parquet if it exists
    tasks_file = dataset_path / 'meta' / 'tasks.parquet'
    if tasks_file.exists():
        output_tasks_file = output_path / 'meta' / 'tasks.parquet'
        if not dry_run:
            output_tasks_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tasks_file, output_tasks_file)
        logger.info(f"  Copied {tasks_file.name}")

    # Copy videos directory if it exists (note: videos for dirty episodes will remain)
    videos_dir = dataset_path / 'videos'
    if videos_dir.exists():
        output_videos_dir = output_path / 'videos'
        if not dry_run:
            if output_videos_dir.exists():
                shutil.rmtree(output_videos_dir)
            # Use symlinks=True to preserve symlinks instead of following them
            try:
                shutil.copytree(videos_dir, output_videos_dir, symlinks=True)
                logger.info(f"  Copied videos/ directory")
                logger.warning(
                    "  Note: Videos for dirty episodes are still present. "
                    "Manual cleanup may be needed if disk space is a concern."
                )
            except Exception as e:
                logger.warning(f"  Could not copy videos/ directory: {e}")
                logger.warning("  Video files will need to be copied manually if needed.")
        else:
            logger.info(f"  Would copy videos/ directory")


def verify_cleaned_dataset(output_path: Path, expected_clean_episodes: int) -> bool:
    """
    Verify the cleaned dataset integrity.

    Args:
        output_path: Path to cleaned dataset
        expected_clean_episodes: Expected number of clean episodes

    Returns:
        True if verification passes, False otherwise
    """
    logger.info("Verifying cleaned dataset...")

    try:
        # Check episode indices are sequential
        data_dir = output_path / 'data'
        all_episodes = set()

        for chunk_dir in sorted(data_dir.glob('chunk-*')):
            for parquet_file in sorted(chunk_dir.glob('file-*.parquet')):
                df = pd.read_parquet(parquet_file)
                all_episodes.update(df['episode_index'].unique())

        sorted_episodes = sorted(all_episodes)
        expected_episodes = list(range(expected_clean_episodes))

        if sorted_episodes != expected_episodes:
            logger.error(
                f"Episode indices are not sequential! "
                f"Found: {sorted_episodes[:10]}... Expected: {expected_episodes[:10]}..."
            )
            return False

        logger.info(f"  ✓ Episode indices are sequential (0 to {expected_clean_episodes - 1})")

        # Check global index is sequential
        global_indices = []
        for chunk_dir in sorted(data_dir.glob('chunk-*')):
            for parquet_file in sorted(chunk_dir.glob('file-*.parquet')):
                df = pd.read_parquet(parquet_file)
                global_indices.extend(df['index'].tolist())

        expected_indices = list(range(len(global_indices)))
        if global_indices != expected_indices:
            logger.error("Global index is not sequential!")
            return False

        logger.info(f"  ✓ Global index is sequential (0 to {len(global_indices) - 1})")

        # Re-detect dirty episodes (should be empty)
        remaining_dirty = detect_dirty_episodes(output_path)
        if remaining_dirty:
            logger.error(f"Found {len(remaining_dirty)} dirty episodes after cleaning: {remaining_dirty}")
            return False

        logger.info("  ✓ No dirty episodes remain")

        logger.info("Verification passed ✓")
        return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Detect and clean dirty episodes from LeRobot dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to dataset directory'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output path for cleaned dataset (default: overwrite input dataset)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Only generate detection report, do not clean'
    )

    parser.add_argument(
        '--zero-threshold',
        type=int,
        default=7,
        help='Number of leading observation.state positions that must be zero to flag as dirty (default: 7)'
    )

    args = parser.parse_args()

    # Validate paths
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return 1

    # Validate dataset structure
    if not validate_dataset_structure(dataset_path):
        logger.error("Dataset structure validation failed")
        return 1

    # Detect dirty episodes
    dirty_episodes = detect_dirty_episodes(dataset_path, args.zero_threshold)

    if not dirty_episodes:
        logger.info("No dirty episodes found! Dataset is clean.")
        return 0

    # Generate report
    report = generate_report(dataset_path, dirty_episodes)
    print("\n" + report + "\n")

    # If report-only, stop here
    if args.report_only:
        logger.info("Report-only mode: exiting without cleaning")
        return 0

    # Determine output path
    if args.output_path:
        output_path = Path(args.output_path)
        if output_path.exists() and not args.dry_run:
            logger.error(f"Output path already exists: {output_path}")
            logger.error("Please remove it first or choose a different path")
            return 1
    else:
        # In-place cleaning
        output_path = dataset_path
        if not args.dry_run:
            logger.warning("IN-PLACE CLEANING: Original dataset will be modified!")
            logger.warning("Consider making a backup first.")
            response = input("Continue? [y/N]: ")
            if response.lower() != 'y':
                logger.info("Aborted by user")
                return 0

    # Dry run notification
    if args.dry_run:
        logger.info("DRY RUN MODE: No files will be modified")

    # Calculate episode mapping
    stats = calculate_episode_stats(dataset_path, dirty_episodes)
    episode_mapping = create_episode_mapping(dirty_episodes, stats['total_episodes'])

    # Clean data files
    clean_data_files(dataset_path, dirty_episodes, episode_mapping, output_path, args.dry_run)

    # Clean metadata files
    clean_metadata_files(dataset_path, dirty_episodes, episode_mapping, output_path, args.dry_run)

    # Copy other files
    if output_path != dataset_path:
        copy_other_files(dataset_path, output_path, args.dry_run)

    # Verify if not dry run
    if not args.dry_run:
        if not verify_cleaned_dataset(output_path, stats['clean_episodes']):
            logger.error("Verification failed!")
            return 1

        logger.info(f"\n{'='*80}")
        logger.info("CLEANING COMPLETE ✓")
        logger.info(f"Cleaned dataset: {output_path}")
        logger.info(f"Episodes: {stats['total_episodes']} → {stats['clean_episodes']} (-{stats['dirty_episodes']})")
        logger.info(f"Rows: {stats['total_rows']:,} → {stats['clean_rows']:,} (-{stats['dirty_rows']:,})")
        logger.info(f"{'='*80}\n")
    else:
        logger.info("\nDRY RUN COMPLETE - No files were modified")

    return 0


if __name__ == '__main__':
    exit(main())
