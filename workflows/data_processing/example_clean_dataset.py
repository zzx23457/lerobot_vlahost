#!/usr/bin/env python3
"""
Example: Using the dirty episode cleaner

This script demonstrates how to use the clean_dirty_episodes.py script
programmatically or as part of an automated pipeline.
"""

import subprocess
import sys
from pathlib import Path


def run_cleaner(dataset_path: str, mode: str = "report", output_path: str = None):
    """
    Run the dirty episode cleaner.

    Args:
        dataset_path: Path to dataset to clean
        mode: One of "report", "dry-run", or "clean"
        output_path: Optional output path for cleaned dataset
    """
    cmd = [
        sys.executable,
        "workflows/data_processing/clean_dirty_episodes.py",
        "--dataset-path", dataset_path
    ]

    if mode == "report":
        cmd.append("--report-only")
    elif mode == "dry-run":
        cmd.append("--dry-run")
    elif mode == "clean":
        if output_path:
            cmd.extend(["--output-path", output_path])
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'report', 'dry-run', or 'clean'")

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    return result.returncode


def main():
    """Example workflow for cleaning a dataset."""

    # Dataset to clean
    dataset_path = "datasets/26-06-17-11-32-27_v2"
    cleaned_path = "datasets/26-06-17-11-32-27_v2_cleaned"

    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please update the dataset_path variable in this script.")
        return 1

    print("=" * 80)
    print("STEP 1: Generate Report")
    print("=" * 80)
    result = run_cleaner(dataset_path, mode="report")
    if result != 0:
        print("Report generation failed!")
        return result

    print("\n" + "=" * 80)
    print("STEP 2: Dry Run (simulate cleaning)")
    print("=" * 80)
    result = run_cleaner(dataset_path, mode="dry-run")
    if result != 0:
        print("Dry run failed!")
        return result

    print("\n" + "=" * 80)
    print("STEP 3: Clean to new location")
    print("=" * 80)

    # Check if output already exists
    if Path(cleaned_path).exists():
        print(f"Output path already exists: {cleaned_path}")
        response = input("Remove it and continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
        import shutil
        shutil.rmtree(cleaned_path)

    result = run_cleaner(dataset_path, mode="clean", output_path=cleaned_path)
    if result != 0:
        print("Cleaning failed!")
        return result

    print("\n" + "=" * 80)
    print("SUCCESS! Cleaned dataset ready at:")
    print(cleaned_path)
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
