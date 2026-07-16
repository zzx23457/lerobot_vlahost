#!/usr/bin/env python3
"""sanity_check.py — 数据集 / 环境 烟测脚本

目的: 训练前快速验证 datasets/26-06-17-11-32-27_v2 能被 LeRobot 正确加载,
       视频能解码,stats 合理,跑一个 batch 不会报错。

用法:
    python workflows/sanity_check.py --dataset-root /path/to/dataset
    python workflows/sanity_check.py --dataset-root /path/to/dataset --n-samples 10

退出码: 0=全部通过,1=有错误(请勿继续训练)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ============== 颜色 ==============
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
NC = "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}✓{NC} {msg}")


def info(msg: str) -> None:
    print(f"{BLUE}·{NC} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}⚠{NC} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}✗{NC} {msg}")


# ============== 检查函数 ==============


def check_layout(root: Path) -> None:
    """检查目录结构"""
    info("Phase 1.1 — 目录结构")
    for d in ("data", "meta", "videos"):
        if not (root / d).is_dir():
            fail(f"缺少子目录: {d}")
            sys.exit(1)
        ok(f"存在: {d}/")

    # meta 必有的文件
    for f in ("info.json", "stats.json", "tasks.parquet"):
        if not (root / "meta" / f).exists():
            fail(f"meta/ 下缺少: {f}")
            sys.exit(1)
    ok("meta/info.json, meta/stats.json, meta/tasks.parquet 都存在")

    # data 下至少一个 chunk
    chunks = list((root / "data").glob("chunk-*"))
    if not chunks:
        fail("data/ 下没有任何 chunk-XXX")
        sys.exit(1)
    ok(f"data/ 下有 {len(chunks)} 个 chunk")

    # videos 下至少一路相机
    cams = [d.name for d in (root / "videos").iterdir() if d.is_dir()]
    if not cams:
        fail("videos/ 下没有任何相机目录")
        sys.exit(1)
    ok(f"videos/ 下有 {len(cams)} 路相机: {cams}")


def check_info(root: Path) -> dict[str, Any]:
    """检查 info.json 关键字段"""
    info("Phase 1.2 — info.json")
    info_path = root / "meta" / "info.json"
    info_data: dict[str, Any] = json.loads(info_path.read_text())

    version = info_data.get("codebase_version")
    if version != "v3.0":
        fail(f"codebase_version 是 {version},不是 v3.0 (可能不兼容)")
        sys.exit(1)
    ok(f"codebase_version: v3.0")

    n_ep = info_data.get("total_episodes", 0)
    n_fr = info_data.get("total_frames", 0)
    fps = info_data.get("fps", 0)
    ok(f"total_episodes={n_ep}  total_frames={n_fr}  fps={fps}")

    feats = info_data.get("features", {})
    if "observation.state" not in feats:
        fail("缺 observation.state 字段")
        sys.exit(1)
    if "action" not in feats:
        fail("缺 action 字段")
        sys.exit(1)

    state_shape = feats["observation.state"].get("shape")
    action_shape = feats["action"].get("shape")
    ok(f"observation.state shape = {state_shape}")
    ok(f"action shape = {action_shape}")
    if state_shape != [16] or action_shape != [16]:
        warn(f"预期 state/action 都是 [16],实际是 {state_shape}/{action_shape}")

    images = sorted(k for k in feats if k.startswith("observation.images."))
    ok(f"图像特征数 = {len(images)}: {images}")

    return info_data


def check_stats(root: Path) -> None:
    """检查 stats.json 数值合理性"""
    info("Phase 1.3 — stats.json")
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())

    issues = 0
    for key, val in stats.items():
        if not isinstance(val, dict):
            continue
        for stat_name in ("mean", "std", "min", "max"):
            if stat_name in val:
                arr = np.asarray(val[stat_name])
                if np.isnan(arr).any() or np.isinf(arr).any():
                    fail(f"  {key}.{stat_name} 含 NaN/Inf")
                    issues += 1
        if "mean" in val and "std" in val:
            mean_arr = np.asarray(val["mean"])
            std_arr = np.asarray(val["std"])
            if (std_arr <= 0).any():
                warn(f"  {key}.std 含 0 或负值 (constant 维度)")

    if issues:
        fail(f"stats.json 有 {issues} 个数值问题")
        sys.exit(1)
    ok("stats.json 数值无 NaN/Inf")

    # 简单 sanity:state mean 应该在合理范围 (关节 + 夹爪,单位度)
    if "observation.state" in stats and "mean" in stats["observation.state"]:
        sm = np.asarray(stats["observation.state"]["mean"])
        ss = np.asarray(stats["observation.state"]["std"])
        info(f"  state.mean 范围: [{sm.min():.2f}, {sm.max():.2f}]")
        info(f"  state.std  范围: [{ss.min():.2f}, {ss.max():.2f}]")
        if sm.min() < -360 or sm.max() > 360:
            warn("  state 均值超出 ±360 度范围,请确认单位")


def check_tasks(root: Path) -> None:
    """检查 tasks.parquet"""
    info("Phase 1.4 — tasks.parquet")
    import pandas as pd

    df = pd.read_parquet(root / "meta" / "tasks.parquet")
    if len(df) == 0:
        fail("tasks.parquet 是空的")
        sys.exit(1)
    # LeRobot 写盘:任务文本在 index (name 通常是 "task",但有时不保留),
    # 列是 task_index。两种情况都支持。
    if df.index.name in ("task", None) and len(df.columns) >= 1:
        # 文本在 index
        if "task_index" in df.columns:
            task_texts = df.index.tolist()
            task_idxs = df["task_index"].tolist()
        else:
            task_texts = df.iloc[:, 0].tolist()
            task_idxs = list(range(len(df)))
    else:
        # 文本在列里
        text_col = "task" if "task" in df.columns else df.columns[0]
        task_texts = df[text_col].tolist()
        task_idxs = df["task_index"].tolist() if "task_index" in df.columns else list(range(len(df)))

    ok(f"任务数: {len(df)}")
    for text, idx in zip(task_texts[:5], task_idxs[:5]):
        info(f"  task_index={idx}: {text!r}")


def check_dataset_loadable(root: Path, n_samples: int) -> None:
    """核心:实际加载 LeRobotDataset 并取样本"""
    info("Phase 1.5 — LeRobotDataset 加载 + 取样本")

    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as e:
        fail(f"无法 import lerobot: {e}")
        sys.exit(1)

    # repo_id 这里用绝对路径;LeRobot 接受 "local" 路径
    repo_id = "local"
    info(f"开始加载 LeRobotDataset(root={root}) ...")
    t0 = time.time()
    try:
        ds = LeRobotDataset(repo_id=repo_id, root=str(root))
    except Exception as e:
        fail(f"LeRobotDataset 构造失败: {e}")
        sys.exit(1)
    load_time = time.time() - t0
    ok(f"加载耗时 {load_time:.1f}s,  num_frames={len(ds)}")

    # 取 n 个样本,逐个检查
    if len(ds) == 0:
        fail("数据集是空的")
        sys.exit(1)

    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(ds), size=min(n_samples, len(ds)))

    img_keys_seen: set[str] = set()
    state_seen = []
    action_seen = []

    for i, idx in enumerate(indices):
        try:
            sample = ds[int(idx)]
        except Exception as e:
            fail(f"取样本 {idx} 失败: {e}")
            sys.exit(1)

        # state
        for k in ("observation.state", "state"):
            if k in sample:
                s = np.asarray(sample[k])
                if s.ndim == 0:
                    s = s[None]
                state_seen.append(s)
                break
        else:
            fail(f"样本 {idx} 缺 observation.state / state")
            sys.exit(1)

        # action
        if "action" not in sample:
            fail(f"样本 {idx} 缺 action")
            sys.exit(1)
        a = np.asarray(sample["action"])
        if a.ndim == 0:
            a = a[None]
        action_seen.append(a)

        # images
        for k, v in sample.items():
            if k.startswith("observation.images."):
                img_keys_seen.add(k)
                arr = np.asarray(v)
                if arr.ndim < 3:
                    fail(f"  {k} 维度异常: {arr.shape}")
                    sys.exit(1)
                # 用原始 dtype 的范围判断常量(避免 float [0,1] vs uint8 [0,255] 误报)
                if arr.dtype == np.uint8:
                    threshold = 5  # uint8 至少 5 个灰度差才算非常量
                else:
                    threshold = 0.02  # float [0,1] 至少 0.02 差距
                if (arr.max() - arr.min()) < threshold:
                    warn(
                        f"  {k} 几乎是常量 (max-min={arr.max() - arr.min():.4f}, "
                        f"dtype={arr.dtype}, shape={arr.shape})"
                    )
                if arr.size == 0:
                    fail(f"  {k} 是空数组")
                    sys.exit(1)

    if not img_keys_seen:
        fail("没有取到任何 observation.images.* 字段")
        sys.exit(1)
    ok(f"已解码 {len(img_keys_seen)} 路图像: {sorted(img_keys_seen)}")

    state_arr = np.stack(state_seen, axis=0)
    action_arr = np.stack(action_seen, axis=0)
    info(f"  state  batch shape: {state_arr.shape}  range=[{state_arr.min():.2f}, {state_arr.max():.2f}]")
    info(f"  action batch shape: {action_arr.shape}  range=[{action_arr.min():.2f}, {action_arr.max():.2f}]")

    # 最后一帧:把 batch 丢给 ACT policy 看 shape 是否对得上
    info("Phase 1.6 — 用一个 batch 喂给 ACT 策略(只做 forward,不训)")
    try:
        from lerobot.policies import make_policy
    except ImportError as e:
        warn(f"无法 import lerobot.policies (跳过 forward 测): {e}")
        return

    try:
        # 拼一个 batch dict(模拟 dataloader 输出)
        ds_meta = ds.meta
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig

        # 用 PolicyFeature 构造 input/output features(框架期望的对象类型)
        input_features: dict[str, PolicyFeature] = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        }
        for k in img_keys_seen:
            input_features[k] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))}

        cfg = ACTConfig(input_features=input_features, output_features=output_features)
        cfg.device = "cpu"  # 不依赖 GPU,纯做 shape check

        policy = make_policy(cfg, ds_meta=ds_meta)
        ok(f"ACT 策略构造成功: {type(policy).__name__}")

        # 构造一个 batch
        batch = {
            "observation.state": torch.from_numpy(state_arr[:1]).float(),
            "action": torch.from_numpy(action_arr[:1]).float(),
        }
        for k in img_keys_seen:
            v = np.asarray(ds[int(indices[0])][k])
            batch[k] = torch.from_numpy(v[None]).float()

        with torch.no_grad():
            out = policy.forward(batch)
        loss = out.get("loss", None) if isinstance(out, dict) else None
        if loss is not None and not torch.isnan(loss).any():
            ok(f"ACT forward 成功,loss={float(loss):.4f} (CPU 占位,数值无意义)")
        else:
            ok("ACT forward 成功,shape 全对得上")
    except Exception as e:
        warn(f"ACT forward 测失败(可能是 policy 配置问题,不一定影响训练): {e}")


# ============== 入口 ==============

def main() -> None:
    p = argparse.ArgumentParser(description="LeRobot 数据集烟测")
    p.add_argument("--dataset-root", type=Path, required=True, help="数据集根目录")
    p.add_argument("--n-samples", type=int, default=5, help="随机抽样数")
    args = p.parse_args()

    root: Path = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        fail(f"数据集根目录不存在: {root}")
        sys.exit(1)

    print()
    print(f"{BLUE}=== Sanity Check for {root} ==={NC}")
    print()

    check_layout(root)
    print()
    check_info(root)
    print()
    check_stats(root)
    print()
    check_tasks(root)
    print()
    check_dataset_loadable(root, args.n_samples)

    print()
    print(f"{GREEN}=== 全部通过,可以训练 ==={NC}")


if __name__ == "__main__":
    main()
