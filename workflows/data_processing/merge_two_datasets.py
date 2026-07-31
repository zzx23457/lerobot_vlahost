"""合并 2 个或更多 source dataset 到一个新的 dataset root.

关键参数: video_files_size_in_mb=0.001 (沿用源数据集), 让每个 episode 保留独立 mp4。
否则默认 200MB 会把几十个 episode 拼到同一 mp4, 破坏 "file-NN.mp4 == episode N"
的对应关系, 同时让训练 sanity_check 报 timestamp drift (data parquet timestamp
与视频 pts 不同步)。

CLI 模式 (新增) 取代原本的硬编码 SRC_ROOTS / DST / REPO_ID。`main()` 现在接受
argparse 参数；旧硬编码作为默认值保留以便手测，但仍优先用 CLI 传入值。
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# 兼容旧硬编码调用 —— 仅在 --source-root 未提供时使用
SRC_ROOTS_DEFAULT = [
    Path("/home/zzx23457/lerobot_vlahost/datasets/26-07-21+22+23+25-merged"),
    Path("/home/zzx23457/lerobot_vlahost/datasets/26-07-25-01-19-04"),
]
DST_DEFAULT = Path("/home/zzx23457/lerobot_vlahost/datasets/26-07-21+22+23+25+25-merged")
REPO_ID_DEFAULT = "26-07-21+22+23+25+25-merged"

VIDEO_FILES_SIZE_IN_MB = 0.001


def fix_meta_episodes_index(root: Path) -> None:
    """把 `root/meta/episodes/` 下每个 parquet 文件内部的
    `meta/episodes/chunk_index` / `meta/episodes/file_index` 两列归一化到
    与所在文件路径一致——即 `chunk-N/file-M.parquet` 文件里所有行的
    `(chunk_index, file_index) = (N, M)`。

    lerobot `aggregate_datasets` 在 `meta/episodes` parquet 走 append 分片
    时（旧默认行为），这两列的值是按"源文件 layout 镜像到目标"算出来的
    偏移加法，并不会跟随实际写入的目标文件归一化。当 size 阈值未触发
    rotate、多个源文件的行被塞进同一个目标文件时，文件内部的
    `file_index` 列就会指向并不存在的 `file-NN.parquet`，下次合并 / 读取
    时直接 `FileNotFoundError`。

    这个函数作为 script 层的兜底：跑完 `aggregate_datasets` 后扫一遍
    输出，把所有分片内的索引列统一刷成 `(N, M)`，让内部声明和磁盘事实
    对齐。
    """
    episodes_root = root / "meta" / "episodes"
    if not episodes_root.exists():
        return

    fixed_files = 0
    for path in sorted(episodes_root.glob("chunk-*/file-*.parquet")):
        chunk_n = int(path.parent.name.split("-")[1])
        file_m = int(path.stem.split("-")[1])

        df = pd.read_parquet(path)
        if (
            (df["meta/episodes/chunk_index"] == chunk_n).all()
            and (df["meta/episodes/file_index"] == file_m).all()
        ):
            continue

        df["meta/episodes/chunk_index"] = chunk_n
        df["meta/episodes/file_index"] = file_m
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        fixed_files += 1

    if fixed_files:
        print(f"  fix_meta_episodes_index: rewrote {fixed_files} file(s) under {episodes_root}")
    else:
        print(f"  fix_meta_episodes_index: all files already consistent ({episodes_root})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合并多个 LeRobot 数据集 (CLI 模式)。"
        "默认参数保留原硬编码值以方便手测。",
    )
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        help="源数据集 root 路径，可重复传入多个。至少 2 个。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="合并后的输出数据集 root。",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="输出数据集的 repo_id。",
    )
    parser.add_argument(
        "--video-files-size-mb",
        type=float,
        default=VIDEO_FILES_SIZE_IN_MB,
        help="视频分片大小阈值 (MB)。默认 0.001 强制每个 episode 独立 mp4。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划，不实际写入。",
    )
    return parser.parse_args(argv)


def merge(
    src_roots: list[Path],
    dst: Path,
    repo_id: str,
    video_files_size_mb: float,
) -> None:
    """实际执行合并。CLI / 程序化调用共用。"""
    # 加载所有源并打印 summary
    datasets = []
    for i, root in enumerate(src_roots):
        ds = LeRobotDataset(repo_id=f"local/{root.name}", root=root)
        print(f"  [{i}] {root.name}: {len(ds)} frames, {ds.meta.total_episodes} episodes")
        datasets.append(ds)

    aggregate_datasets(
        repo_ids=[ds.repo_id for ds in datasets],
        aggr_repo_id=repo_id,
        roots=[ds.root for ds in datasets],
        aggr_root=dst,
        video_files_size_in_mb=video_files_size_mb,
        data_files_size_in_mb=datasets[0].meta.data_files_size_in_mb,
    )

    # 兜底：aggregate.py 在 meta/episodes 走 append 分片时，内部 chunk/file_index
    # 列会和实际文件路径脱节。下面这一步把每个分片里的两列归一化到 (N, M)，
    # 让内部声明和磁盘事实对齐。
    print("fix_meta_episodes_index (post-merge):")
    fix_meta_episodes_index(dst)

    merged = LeRobotDataset(repo_id=repo_id, root=dst)

    print(
        f"Merged -> {dst} | "
        f"episodes={merged.meta.total_episodes} frames={merged.meta.total_frames}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # 优先级：CLI > 默认硬编码
    src_roots = args.source_root if args.source_root else SRC_ROOTS_DEFAULT
    dst = args.output_root if args.output_root else DST_DEFAULT
    repo_id = args.repo_id if args.repo_id else REPO_ID_DEFAULT

    if len(src_roots) < 2:
        print(
            "ERROR: at least 2 source datasets required (got "
            f"{len(src_roots)}). Pass --source-root multiple times.",
            file=sys.stderr,
        )
        return 1

    # 检查所有源存在
    for src in src_roots:
        if not src.exists():
            print(f"ERROR: source dataset not found: {src}", file=sys.stderr)
            return 1

    # 拒绝覆盖已有输出
    if dst.exists():
        print(f"ERROR: output dir already exists: {dst}", file=sys.stderr)
        return 1

    print("Plan:")
    print(f"  sources ({len(src_roots)}):")
    for s in src_roots:
        print(f"    - {s}")
    print(f"  output : {dst}")
    print(f"  repo_id: {repo_id}")
    print(f"  video_files_size_mb = {args.video_files_size_mb}")
    print(f"  dry_run = {args.dry_run}")
    print()

    if args.dry_run:
        print("(dry-run) nothing written.")
        return 0

    merge(src_roots, dst, repo_id, args.video_files_size_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())