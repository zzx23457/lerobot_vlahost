#!/usr/bin/env python3
"""检查数据集中的时间戳对齐情况。

扫描所有 episode，对比 data parquet 的 timestamp 列和视频最后一帧的实际 pts，
找出时间戳漂移的 episode。

用法:
    python check_timestamp_alignment.py --dataset-root /path/to/dataset
    python check_timestamp_alignment.py --dataset-root /path/to/dataset --tolerance-ms 1.0
    python check_timestamp_alignment.py --dataset-root /path/to/dataset --output clean_episodes.txt
"""

import argparse
import glob
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


def get_video_last_pts(video_path: Path) -> float | None:
    """获取视频最后一帧的实际 pts（单位：秒）。

    Args:
        video_path: 视频文件路径

    Returns:
        最后一帧的 pts（秒），如果视频无法读取返回 None
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time",
            "-of", "csv=p=0",
            str(video_path),
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        pts_times = [float(line) for line in output.split("\n") if line]
        return max(pts_times) if pts_times else None
    except Exception as e:
        print(f"Warning: Failed to read {video_path}: {e}", file=sys.stderr)
        return None


def check_dataset(dataset_root: Path, video_key: str | None = None) -> dict:
    """检查数据集的时间戳对齐情况。

    Args:
        dataset_root: 数据集根目录
        video_key: 视频特征名。如果为 None, 自动从 info.json 选择第一个视频流

    Returns:
        字典，包含 clean 和 dirty 的 episode 列表及详细信息
    """
    # 0. 自动检测视频流 (如果未指定)
    if video_key is None:
        info_path = dataset_root / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            video_features = [
                k for k, v in info["features"].items()
                if v.get("dtype") == "video"
            ]
            if video_features:
                video_key = video_features[0]
                print(f"Auto-detected video stream: {video_key}", file=sys.stderr)
            else:
                # fallback: 用 videos/ 目录下第一个子目录
                videos_dir = dataset_root / "videos"
                if videos_dir.exists():
                    subdirs = [d.name for d in videos_dir.iterdir() if d.is_dir()]
                    if subdirs:
                        video_key = sorted(subdirs)[0]
                        print(f"Fallback to first video dir: {video_key}", file=sys.stderr)
        if video_key is None:
            raise FileNotFoundError(
                f"No video stream found. Tried info.json and videos/ directory."
            )

    # 1. 加载所有 data parquet，读取每个 episode 的最后时间戳
    data_dir = dataset_root / "data" / "chunk-000"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(f"Loading data from {len(parquet_files)} parquet files...", file=sys.stderr)
    df = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "timestamp"]) for f in parquet_files],
        ignore_index=True,
    )
    data_max_ts = df.groupby("episode_index")["timestamp"].max().to_dict()
    total_episodes = len(data_max_ts)
    print(f"Found {total_episodes} episodes in data", file=sys.stderr)

    # 2. 并行读取视频最后一帧的 pts
    video_dir = dataset_root / "videos" / video_key / "chunk-000"
    if not video_dir.exists():
        # 列出可用的视频流, 给出明确错误
        videos_dir = dataset_root / "videos"
        available = sorted([d.name for d in videos_dir.iterdir() if d.is_dir()]) if videos_dir.exists() else []
        raise FileNotFoundError(
            f"Video directory not found: {video_dir}\n"
            f"  Available video streams: {available}\n"
            f"  用 --video-key <name> 指定一个存在的流"
        )

    print(f"Reading video timestamps from {video_dir}...", file=sys.stderr)
    video_paths = [video_dir / f"file-{ep:03d}.mp4" for ep in range(total_episodes)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        video_last_pts = list(executor.map(get_video_last_pts, video_paths))

    # 3. 对比 data 和 video 时间戳, 记录所有 gap 用于统计
    results = {
        "total": total_episodes,
        "clean": [],
        "dirty": [],
        "all_gaps_ms": [],  # 所有 episode 的 gap_ms, 用于统计
    }

    for ep in range(total_episodes):
        if ep not in data_max_ts:
            continue

        data_ts = data_max_ts[ep]
        video_pts = video_last_pts[ep]

        if video_pts is None:
            print(f"Warning: Episode {ep} video unreadable, skipping", file=sys.stderr)
            continue

        gap_ms = (data_ts - video_pts) * 1000
        results["all_gaps_ms"].append(gap_ms)

        if abs(gap_ms) < 1.0:  # 容差 1ms
            results["clean"].append({
                "episode": ep,
                "data_max_ts": round(data_ts, 4),
                "video_last_pts": round(video_pts, 4),
                "gap_ms": round(gap_ms, 4),  # 保留更多小数位
            })
        else:
            results["dirty"].append({
                "episode": ep,
                "data_max_ts": round(data_ts, 4),
                "video_last_pts": round(video_pts, 4),
                "gap_ms": round(gap_ms, 2),
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="检查 LeRobot 数据集的时间戳对齐情况",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查数据集，打印结果
  python check_timestamp_alignment.py --dataset-root datasets/my_data_v2

  # 指定容差（毫秒）
  python check_timestamp_alignment.py --dataset-root datasets/my_data_v2 --tolerance-ms 50.0

  # 将 clean episode 列表保存到文件
  python check_timestamp_alignment.py --dataset-root datasets/my_data_v2 --output clean_episodes.txt

  # 输出为 JSON 格式
  python check_timestamp_alignment.py --dataset-root datasets/my_data_v2 --format json
        """,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="数据集根目录（包含 data/, videos/, meta/ 的目录）",
    )
    parser.add_argument(
        "--video-key",
        type=str,
        default=None,
        help="用于检查的视频流名称（默认从 info.json 自动检测第一个视频流）",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=1.0,
        help="时间戳对齐容差（毫秒），默认 1.0ms",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="保存 clean episode 列表的文件路径（每行一个 episode index）",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="输出格式（text 或 json）",
    )

    args = parser.parse_args()

    if not args.dataset_root.exists():
        print(f"Error: Dataset root not found: {args.dataset_root}", file=sys.stderr)
        sys.exit(2)  # exit 2 = 脚本错误 (数据集路径不存在)

    # 检查
    try:
        results = check_dataset(args.dataset_root, args.video_key)
    except FileNotFoundError as e:
        # 数据集结构问题 (如视频流不存在)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)  # exit 2 = 脚本错误
    except Exception as e:
        # 其他未预期错误
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)

    # 输出
    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        import numpy as np

        print(f"\n{'='*75}")
        print(f"Dataset: {args.dataset_root}")
        print(f"Total episodes: {results['total']}")
        print(f"CLEAN: {len(results['clean'])} episodes")
        print(f"DIRTY: {len(results['dirty'])} episodes")
        print(f"{'='*75}\n")

        # 显示所有 episode 的 gap 统计 (包括 CLEAN 的)
        all_gaps = results.get("all_gaps_ms", [])
        if all_gaps:
            gaps_arr = np.array(all_gaps)
            print(f"All {len(gaps_arr)} episodes gap 统计 (ms):")
            print(f"  mean:    {gaps_arr.mean():+.4f}")
            print(f"  std:     {gaps_arr.std():.4f}")
            print(f"  min:     {gaps_arr.min():+.4f}")
            print(f"  max:     {gaps_arr.max():+.4f}")
            print(f"  |max|:   {np.abs(gaps_arr).max():.4f}")
            print()
            print(f"绝对值分布:")
            print(f"  == 0.0000:    {(gaps_arr == 0).sum():>4}")
            print(f"  < 0.001ms:    {(np.abs(gaps_arr) < 0.001).sum():>4}  (1μs 内)")
            print(f"  < 0.01ms:     {(np.abs(gaps_arr) < 0.01).sum():>4}  (10μs 内)")
            print(f"  < 0.1ms:      {(np.abs(gaps_arr) < 0.1).sum():>4}  (100μs 内)")
            print(f"  < 1.0ms:      {(np.abs(gaps_arr) < 1.0).sum():>4}  (1ms 内, CLEAN 阈值)")
            print()

        if results["clean"]:
            print(f"CLEAN episodes ({len(results['clean'])}):")
            # 显示前 10 个的精确 gap, 方便验证
            print(f"  前 10 个的精确 gap (μs):")
            for c in results["clean"][:10]:
                gap_us = c["gap_ms"] * 1000
                print(
                    f"    ep {c['episode']:>3}: data={c['data_max_ts']:.6f}  "
                    f"video={c['video_last_pts']:.6f}  gap={gap_us:+.2f}μs"
                )
            if len(results["clean"]) > 10:
                print(f"    ... (共 {len(results['clean'])} 个 CLEAN)")
            print()

        if results["dirty"]:
            print(f"DIRTY episodes ({len(results['dirty'])}):")
            print(f"  {'ep':>3} | {'data_max':>10} | {'video_last':>11} | {'gap_ms':>9}")
            print(f"  {'-'*50}")
            for d in results["dirty"]:
                print(
                    f"  {d['episode']:>3} | {d['data_max_ts']:>10.4f} | "
                    f"{d['video_last_pts']:>11.4f} | {d['gap_ms']:>+9.2f}"
                )
            print()

    # 保存 clean episode 列表
    if args.output:
        with open(args.output, "w") as f:
            for ep in results["clean"]:
                f.write(f"{ep}\n")
        print(f"Clean episode list saved to: {args.output}", file=sys.stderr)

    # 退出码
    sys.exit(0 if not results["dirty"] else 1)


if __name__ == "__main__":
    main()
