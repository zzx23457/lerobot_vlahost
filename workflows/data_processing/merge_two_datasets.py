"""合并 datasets/26-07-21-14-14-22 和 datasets/26-07-22-15-45-02.

输出到 datasets/26-07-21+22-merged.
"""

from pathlib import Path

from lerobot.datasets import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


SRC_A = Path("/home/zzx23457/lerobot/datasets/26-07-21-14-14-22")
SRC_B = Path("/home/zzx23457/lerobot/datasets/26-07-22-15-45-02")
DST = Path("/home/zzx23457/lerobot/datasets/26-07-21+22-merged")

REPO_ID = "local/26-07-21+22-merged"


def main() -> None:
    ds_a = LeRobotDataset(repo_id="local/26-07-21-14-14-22", root=SRC_A)
    ds_b = LeRobotDataset(repo_id="local/26-07-22-15-45-02", root=SRC_B)

    print(f"A: {len(ds_a)} frames, {ds_a.meta.total_episodes} episodes")
    print(f"B: {len(ds_b)} frames, {ds_b.meta.total_episodes} episodes")

    if DST.exists():
        raise FileExistsError(f"Output dir already exists: {DST}")

    merged = merge_datasets(
        datasets=[ds_a, ds_b],
        output_repo_id=REPO_ID,
        output_dir=DST,
    )

    print(
        f"Merged -> {DST} | "
        f"episodes={merged.meta.total_episodes} frames={merged.meta.total_frames}"
    )


if __name__ == "__main__":
    main()