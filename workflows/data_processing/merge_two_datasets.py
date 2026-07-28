"""合并 2 个 source dataset（其中一个是上一次的合并产物）:
  - datasets/26-07-21+22+23+25+27-merged     (上次合并，499 eps)
  - datasets/26-07-25-01-19-04               (原始数据集，100 eps)
输出到 datasets/26-07-21+22+23+25+27+25-merged。

关键参数: video_files_size_in_mb=0.001 (沿用源数据集), 让每个 episode 保留独立 mp4。
否则默认 200MB 会把几十个 episode 拼到同一 mp4, 破坏 "file-NN.mp4 == episode N"
的对应关系, 同时让训练 sanity_check 报 timestamp drift (data parquet timestamp
与视频 pts 不同步)。
"""

from pathlib import Path

import pandas as pd

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


SRC_ROOTS = [
    Path("/home/zzx23457/lerobot_vlahost/datasets/26-07-21+22+23+25+27-merged"),
    Path("/home/zzx23457/lerobot_vlahost/datasets/26-07-25-01-19-04"),
]
DST = Path("/home/zzx23457/lerobot_vlahost/datasets/26-07-21+22+23+25+27+25-merged")

REPO_ID = "26-07-21+22+23+25+27+25-merged"

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


def main() -> None:
    if DST.exists():
        raise FileExistsError(f"Output dir already exists: {DST}")

    # 加载所有源并打印 summary
    datasets = []
    for i, root in enumerate(SRC_ROOTS):
        ds = LeRobotDataset(repo_id=f"local/{root.name}", root=root)
        print(f"  [{i}] {root.name}: {len(ds)} frames, {ds.meta.total_episodes} episodes")
        datasets.append(ds)

    aggregate_datasets(
        repo_ids=[ds.repo_id for ds in datasets],
        aggr_repo_id=REPO_ID,
        roots=[ds.root for ds in datasets],
        aggr_root=DST,
        video_files_size_in_mb=VIDEO_FILES_SIZE_IN_MB,
        data_files_size_in_mb=datasets[0].meta.data_files_size_in_mb,
    )

    # 兜底：aggregate.py 在 meta/episodes 走 append 分片时，内部 chunk/file_index
    # 列会和实际文件路径脱节。下面这一步把每个分片里的两列归一化到 (N, M)，
    # 让内部声明和磁盘事实对齐。
    print("fix_meta_episodes_index (post-merge):")
    fix_meta_episodes_index(DST)

    merged = LeRobotDataset(repo_id=REPO_ID, root=DST)

    print(
        f"Merged -> {DST} | "
        f"episodes={merged.meta.total_episodes} frames={merged.meta.total_frames}"
    )


if __name__ == "__main__":
    main()