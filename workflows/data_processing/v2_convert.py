"""Convert v1 LeRobot v3 datasets to v2 schema (cleaned, joint-angle + gripper state).

支持两种调用模式:
1. **legacy 模式** (无 CLI): 读取同目录 `v2_convert_config.json` 处理全部声明数据集。
   - 旧调用: `python3 workflows/v2_convert.py`
2. **CLI 单任务模式** (新增, 非交互): 由 UI 或脚本驱动。
   - `python3 workflows/v2_convert.py --dataset-root <v1_path> --output-root <v2_path> [--v2-suffix _v2] [--camera-enabled 0,1,1,1] [--dry-run]`
   - 完全不修改 `v2_convert_config.json`。

v1 → v2 transforms:
    - `action` (56,) → (16,): arm_command (14) + left_grip_next (1) + right_grip_next (1)
    - `observation.state` (26,) → (16,): joint_pos (14) + left_grip_angle (1) + right_grip_angle (1)
    - v1 angles are in **radians**; v2 angles are in **degrees** (× 180/π).
    - New column `action_is_pad` (bool) added (all False).
    - Videos are **symlinked** from v1 to save space.
    - Cameras listed in --camera-enabled as 0 are dropped from the v2 output
      (videos dir, info.json features entry, stats.json top-level key, and the
      4 videos/<cam>/* columns in the episodes parquet). Missing key keeps
      all cameras (backward compatible). Order is fixed:
      [left_eye, right_eye, left_wrist, right_wrist].

v1 is **never modified**; rollback for one dataset is `rm -rf <v2_dir>`.
The v2 `.cache/` directory is removed on every successful run.

Run from anywhere:
    python3 workflows/v2_convert.py                                    # legacy: read JSON config
    python3 workflows/v2_convert.py --dataset-root datasets/foo --output-root datasets/foo_v2 \
        --camera-enabled 0,1,1,1 --dry-run
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Sequence, Value

# Force offline (don't hit the Hub)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# --- paths ---
THIS_DIR  = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent.parent
DATASETS_ROOT = REPO_ROOT / "datasets"
CONFIG_PATH = THIS_DIR / "v2_convert_config.json"


# --- camera config ---
# Order is fixed and matches the config["cameras"] array. Index 0 -> "left_eye".
CAMERA_KEYS = ["left_eye", "right_eye", "left_wrist", "right_wrist"]


def _full_cam_key(short: str) -> str:
    """Map a short camera name from CAMERA_KEYS to its full LeRobot feature key."""
    return f"observation.images.{short}"


# ----------------------------------------------------------------------------
# core conversion (one v1 → one v2)
# ----------------------------------------------------------------------------
def convert_dataset(v1_root: Path, v2_root: Path, disabled_cameras: set[str] | None = None) -> None:
    """Convert one v1 dataset at v1_root to v2_root.

    Assumes v1_root is a valid LeRobot v3 dataset produced by the KM converter
    (action 56-dim, observation.state 26-dim, both in radians).

    disabled_cameras: set of full camera keys (e.g. {"observation.images.left_eye"})
        to drop from the v2 output. None or empty set means keep all cameras.
    """
    disabled_cameras = disabled_cameras or set()
    DATA_V1     = v1_root / "data"   / "chunk-000"
    META_V1     = v1_root / "meta"
    VIDEOS_V1   = v1_root / "videos"
    DATA_V2     = v2_root / "data"   / "chunk-000"
    META_V2     = v2_root / "meta"
    EPISODES_V2 = META_V2 / "episodes" / "chunk-000"
    VIDEOS_V2   = v2_root / "videos"

    RAD_TO_DEG = 180.0 / np.pi

    # ---- 0. safety: don't blow away v1 by accident ----
    assert v2_root != v1_root, f"v2_root == v1_root ({v1_root}); refusing to overwrite"
    assert v2_root.parent == v1_root.parent, (
        f"v2_root ({v2_root}) and v1_root ({v1_root}) must be siblings "
        f"so that videos/ can be symlinked back to v1"
    )

    # ---- 1. scaffold ----
    print(f"[1/7] Scaffolding {v2_root}")
    for d in [DATA_V2, META_V2, EPISODES_V2, VIDEOS_V2]:
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy2(META_V1 / "tasks.parquet", META_V2 / "tasks.parquet")
    print(f"      copied tasks.parquet")

    v1_cam_dirs = sorted(os.listdir(VIDEOS_V1))

    # Per-camera log lines so users can verify the camera decision up-front.
    for cam_dir in v1_cam_dirs:
        if cam_dir in disabled_cameras:
            print(f"      [camera] skipping {cam_dir} (disabled in config)")
        else:
            print(f"      [camera] keeping  {cam_dir}")

    # Warn if v1 has a camera outside CAMERA_KEYS: it is always kept.
    for cam_dir in v1_cam_dirs:
        short = cam_dir.removeprefix("observation.images.")
        if short not in CAMERA_KEYS:
            print(
                f"      [camera] WARNING: v1 has unknown camera '{cam_dir}' "
                f"(not in CAMERA_KEYS); keeping it. Update CAMERA_KEYS if you "
                f"want to control this one via config.",
                file=sys.stderr,
            )

    # Symlink loop with disabled-camera handling.
    def _clear_dst(dst: Path) -> None:
        """Remove whatever is at dst: symlink (unlink) or real dir (rmtree)."""
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst, ignore_errors=True)

    for cam_dir in v1_cam_dirs:
        src = VIDEOS_V1 / cam_dir
        dst = VIDEOS_V2 / cam_dir
        _clear_dst(dst)
        if cam_dir in disabled_cameras:
            continue
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)

    kept_n = sum(1 for c in v1_cam_dirs if c not in disabled_cameras)
    print(
        f"      symlinked {kept_n} video keys → v1 "
        f"({len(v1_cam_dirs) - kept_n} disabled)"
    )

    # ---- 2. load v1 data ----
    print(f"[2/7] Loading v1 data parquet files")
    v1_files = sorted(DATA_V1.glob("file-*.parquet"))
    print(f"      {len(v1_files)} files")
    dfs = [pd.read_parquet(p) for p in v1_files]
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"      {len(df_all)} rows total")

    v1_actions   = np.stack(df_all["action"].values).astype(np.float32)            # (T, 56)
    v1_obs_state = np.stack(df_all["observation.state"].values).astype(np.float32) # (T, 26)
    ep_indices   = df_all["episode_index"].values
    assert v1_actions.shape[0] == v1_obs_state.shape[0] == len(ep_indices)

    # ---- 3. assemble (rad → deg) ----
    print(f"[3/7] Assembling obs.state (16-dim) and action (16-dim)  [rad → deg]")

    joint_pos  = v1_actions[:, 42:56]   * RAD_TO_DEG   # (T, 14)
    left_grip  = v1_obs_state[:, 14:15] * RAD_TO_DEG   # (T,  1)
    right_grip = v1_obs_state[:, 20:21] * RAD_TO_DEG   # (T,  1)
    new_obs_state = np.concatenate([joint_pos, left_grip, right_grip], axis=1)
    assert new_obs_state.shape == (len(df_all), 16)

    arm_cmd    = v1_actions[:, 14:28]    * RAD_TO_DEG   # (T, 14)
    left_next  = left_grip.copy()
    right_next = right_grip.copy()
    for ep_id in np.unique(ep_indices):
        mask = ep_indices == ep_id
        for src, dst in [(left_grip, left_next), (right_grip, right_next)]:
            ep_slice = src[mask]
            if len(ep_slice) >= 2:
                dst[mask] = np.concatenate([ep_slice[1:], ep_slice[-1:]], axis=0)
    new_action = np.concatenate([arm_cmd, left_next, right_next], axis=1)
    assert new_action.shape == (len(df_all), 16)

    # ---- 4. write 17 data parquets ----
    print(f"[4/7] Writing {len(v1_files)} new data parquets to {DATA_V2}")
    SPLIT = [len(d) for d in dfs]
    FEATURES = Features({
        "observation.state": Sequence(length=16, feature=Value("float32")),
        "action":            Sequence(length=16, feature=Value("float32")),
        "action_is_pad":     Value("bool"),
        "timestamp":         Value("float32"),
        "frame_index":       Value("int64"),
        "episode_index":     Value("int64"),
        "index":             Value("int64"),
        "task_index":        Value("int64"),
    })

    cursor = 0
    for file_idx, n in enumerate(SPLIT):
        sl = slice(cursor, cursor + n)
        cursor += n
        rec = {
            "observation.state": new_obs_state[sl].tolist(),
            "action":            new_action[sl].tolist(),
            "action_is_pad":     [False] * n,
            "timestamp":     df_all["timestamp"].values[sl].astype(np.float32).tolist(),
            "frame_index":   df_all["frame_index"].values[sl].astype(np.int64).tolist(),
            "episode_index": df_all["episode_index"].values[sl].astype(np.int64).tolist(),
            "index":         df_all["index"].values[sl].astype(np.int64).tolist(),
            "task_index":    df_all["task_index"].values[sl].astype(np.int64).tolist(),
        }
        table = pa.Table.from_pydict(rec).cast(FEATURES.arrow_schema)
        out = DATA_V2 / f"file-{file_idx:03d}.parquet"
        pq.write_table(table, out, compression="snappy", use_dictionary=True)
    print(f"      wrote {len(SPLIT)} shards ({sum(SPLIT)} rows total)")

    # ---- 5. info.json ----
    print(f"[5/7] Writing {META_V2 / 'info.json'}")
    with open(META_V1 / "info.json") as f:
        info = json.load(f)
    vid_info = {
        k: v for k, v in info["features"].items()
        if k.startswith("observation.images.") and k not in disabled_cameras
    }

    # Define joint names with .pos suffix (Marvain M6: 7+7 arms + 2 grippers)
    joint_names = [
        "left_arm_joint_1.pos", "left_arm_joint_2.pos", "left_arm_joint_3.pos",
        "left_arm_joint_4.pos", "left_arm_joint_5.pos", "left_arm_joint_6.pos", "left_arm_joint_7.pos",
        "right_arm_joint_1.pos", "right_arm_joint_2.pos", "right_arm_joint_3.pos",
        "right_arm_joint_4.pos", "right_arm_joint_5.pos", "right_arm_joint_6.pos", "right_arm_joint_7.pos",
        "left_gripper.pos", "right_gripper.pos",
    ]

    info["features"] = {
        "observation.state": {"dtype": "float32", "shape": [16], "names": joint_names},
        "action":            {"dtype": "float32", "shape": [16], "names": joint_names},
        "action_is_pad":     {"dtype": "bool",    "shape": [1],  "names": None},
        "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
        "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
        "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
        "index":         {"dtype": "int64",   "shape": [1], "names": None},
        "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        **vid_info,
    }
    with open(META_V2 / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # ---- 6. compute per-episode stats (no LeRobotDataset load to avoid Hub fallback) ----
    print(f"[6/7] Computing per-episode stats via lerobot.datasets.compute_stats")
    from lerobot.datasets.compute_stats import compute_episode_stats, aggregate_stats

    FEATURES_FOR_STATS = {
        "observation.state": {"dtype": "float32", "shape": [16]},
        "action":            {"dtype": "float32", "shape": [16]},
    }
    per_ep_stats = []
    for ep_id in np.unique(ep_indices):
        mask = ep_indices == ep_id
        per_ep_stats.append(compute_episode_stats(
            {"observation.state": new_obs_state[mask], "action": new_action[mask]},
            FEATURES_FOR_STATS,
        ))
    agg = aggregate_stats(per_ep_stats)

    def _to_json(x):
        a = np.asarray(x)
        return a.item() if a.ndim == 0 else a.tolist()
    new_stats = {
        "observation.state": {k: _to_json(v) for k, v in agg["observation.state"].items()},
        "action":            {k: _to_json(v) for k, v in agg["action"].items()},
    }

    with open(META_V1 / "stats.json") as f:
        stats_v1 = json.load(f)
    # Drop disabled-camera top-level keys inherited from v1.
    for cam_key in disabled_cameras:
        stats_v1.pop(cam_key, None)
    stats_v1["observation.state"] = new_stats["observation.state"]
    stats_v1["action"]            = new_stats["action"]
    with open(META_V2 / "stats.json", "w") as f:
        json.dump(stats_v1, f, indent=2)

    # ---- 7. regenerate episodes parquet ----
    print(f"[7/7] Regenerating episodes parquet with new per-episode stats")
    ep_v1_files = sorted((META_V1 / "episodes" / "chunk-000").glob("file-*.parquet"))
    ep_v1_dfs = [pd.read_parquet(p) for p in ep_v1_files]
    ep_v1 = pd.concat(ep_v1_dfs, ignore_index=True)
    STAT_KEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
    new_ep = ep_v1.copy()
    # Drop the 4 videos/<cam>/* columns per disabled camera.
    drop_cols: list[str] = []
    for cam_key in disabled_cameras:
        for sub in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            drop_cols.append(f"videos/{cam_key}/{sub}")
    if drop_cols:
        new_ep = new_ep.drop(columns=drop_cols, errors="ignore")
    for i, ep_id in enumerate(np.unique(ep_indices)):
        eps = per_ep_stats[i]
        row_idx = new_ep.index[new_ep["episode_index"] == ep_id][0]
        for feat in ["observation.state", "action"]:
            for stat in STAT_KEYS:
                col = f"stats/{feat}/{stat}"
                if col not in new_ep.columns:
                    continue
                val = eps[feat][stat]
                a = np.asarray(val)
                new_ep.at[row_idx, col] = a.item() if a.ndim == 0 else a.tolist()

    ep_v1_splits = [len(d) for d in ep_v1_dfs]
    ep_cursor = 0
    for file_idx, n in enumerate(ep_v1_splits):
        sl = slice(ep_cursor, ep_cursor + n)
        ep_cursor += n
        out = EPISODES_V2 / f"file-{file_idx:03d}.parquet"
        new_ep.iloc[sl].to_parquet(out, index=False)

    # ---- 8. clean .cache/ to match the canonical v2 example ----
    cache_dir = v2_root / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"      removed {cache_dir}")


# ----------------------------------------------------------------------------
# CLI / config resolution
# ----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args. If --dataset-root is omitted, falls back to JSON config mode."""
    parser = argparse.ArgumentParser(
        description="v1 → v2 schema conversion. CLI 单任务模式 (新增) 或 JSON 配置模式 (旧兼容)。",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="v1 数据集 root (CLI 单任务模式)。与 --output-root 配套使用。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="v2 数据集 root (CLI 单任务模式)。必须与 dataset-root 同 parent。",
    )
    parser.add_argument(
        "--v2-suffix",
        default="_v2",
        help="当仅提供 --dataset-root 而未给 --output-root 时, 自动在 dataset-root 同级创建 "
             "<name><suffix> 目录。默认 _v2。",
    )
    parser.add_argument(
        "--camera-enabled",
        type=str,
        help="4 个相机的 0/1 序列 (按 CAMERA_KEYS 顺序)。例: '1,1,1,1'。0 表示丢弃该相机。"
             "未提供则全部保留。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划，不实际写入。",
    )
    return parser.parse_args(argv)


def _parse_camera_enabled(raw: str | None) -> list[int] | None:
    """Parse '0,1,1,1' style into [0,1,1,1], validating length and entries."""
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if len(parts) != len(CAMERA_KEYS):
        print(
            f"ERROR: --camera-enabled must have exactly {len(CAMERA_KEYS)} "
            f"comma-separated entries (got {len(parts)}: {raw!r})",
            file=sys.stderr,
        )
        sys.exit(1)
    out: list[int] = []
    for p in parts:
        if p not in ("0", "1"):
            print(
                f"ERROR: --camera-enabled entries must be 0 or 1, got {raw!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        out.append(int(p))
    return out


def _resolve_jobs(args: argparse.Namespace) -> list[tuple[Path, Path, set[str]]]:
    """Return [(v1_root, v2_root, disabled_cameras), ...] based on CLI or JSON config.

    CLI 单任务模式优先级最高: 如果给了 --dataset-root 就只用它, 忽略 v2_convert_config.json。
    """
    cameras = _parse_camera_enabled(args.camera_enabled)
    if cameras is None:
        disabled_cameras: set[str] = set()
    else:
        disabled_cameras = {
            _full_cam_key(name) for name, flag in zip(CAMERA_KEYS, cameras) if flag == 0
        }

    if args.dataset_root:
        # CLI 单任务模式
        v1 = args.dataset_root
        if args.output_root:
            v2 = args.output_root
        else:
            v2 = v1.with_name(v1.name + args.v2_suffix)

        if v1 == v2:
            print(f"ERROR: --dataset-root and --output-root resolve to the same path: {v1}", file=sys.stderr)
            sys.exit(1)
        if v2.parent != v1.parent:
            print(
                f"ERROR: --dataset-root ({v1}) and --output-root ({v2}) "
                f"must be siblings so videos/ can be symlinked back to v1.",
                file=sys.stderr,
            )
            sys.exit(1)
        return [(v1, v2, disabled_cameras)]

    # Legacy JSON config 模式
    if not CONFIG_PATH.exists():
        print(f"ERROR: config not found: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    v2_suffix = config.get("v2_suffix", "_v2")
    datasets = config.get("datasets", [])
    cfg_cameras = config.get("cameras", None)
    if cfg_cameras is not None:
        # Legacy mode uses JSON, which always reflects all four cameras.
        # CLI --camera-enabled has already been applied to `disabled_cameras` above.
        # If user gave both, prefer CLI.
        if cameras is None:
            disabled_cameras = {
                _full_cam_key(name) for name, flag in zip(CAMERA_KEYS, cfg_cameras) if flag == 0
            }

    if not datasets:
        print(f"ERROR: config['datasets'] is empty in {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    return [(DATASETS_ROOT / ds, (DATASETS_ROOT / ds).with_name(ds + v2_suffix), disabled_cameras)
            for ds in datasets]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    jobs = _resolve_jobs(args)

    print(f"Config: REPO_ROOT     = {REPO_ROOT}")
    print(f"        DATASETS_ROOT = {DATASETS_ROOT}")
    cameras_repr = _parse_camera_enabled(args.camera_enabled)
    print(f"        cameras       = {cameras_repr!r}  (None = all enabled)")
    if any(j[2] for j in jobs):
        print(f"        disabled      = {sorted(jobs[0][2])}")
    print(f"        datasets      = {len(jobs)}")
    for v1, v2, _ in jobs:
        print(f"          - {v1.name} -> {v2.name}")
    print()

    if args.dry_run:
        print("(dry-run) nothing written.")
        return 0

    rc = 0
    for v1_root, v2_root, disabled_cameras in jobs:
        print("=" * 72)
        print(f" Converting: {v1_root.name}")
        print(f"   v1 = {v1_root}")
        print(f"   v2 = {v2_root}")
        print("=" * 72)
        if not v1_root.exists():
            print(f"SKIP: v1 not found: {v1_root}", file=sys.stderr)
            rc = 2
            continue
        if v2_root.exists():
            print(f"SKIP: v2 already exists: {v2_root} (refusing to overwrite)", file=sys.stderr)
            rc = 2
            continue
        try:
            convert_dataset(v1_root, v2_root, disabled_cameras=disabled_cameras)
        except Exception as e:
            print(f"FAILED on {v1_root.name}: {type(e).__name__}: {e}", file=sys.stderr)
            rc = 3
            continue

        # Final summary per dataset
        print()
        print(f"   ✓ {v2_root.name}  ready at {v2_root}")
        print(f"     rollback: rm -rf {v2_root}")
        print()

    print("=" * 72)
    print(f" Done. exit code = {rc}")
    print("=" * 72)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())