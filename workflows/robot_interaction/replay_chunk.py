#!/usr/bin/env python3
"""replay_chunk.py — HTTP 接口真机回放主入口（chunk 模式）

把数据集中某个 episode 的动作序列按 chunk 方式在真实机器人上回放。
与 replay.py 的区别：使用 send_action_chunk 批量发送，等待 need_new_chunk 信号。

用法:
    # 使用默认配置
    python workflows/robot_interaction/replay_chunk.py

    # 使用自定义配置文件
    python workflows/robot_interaction/replay_chunk.py --config my_replay_chunk_config.yaml

    # 临时覆盖参数
    python workflows/robot_interaction/replay_chunk.py --repo-id username/my_dataset --episode 0
    python workflows/robot_interaction/replay_chunk.py --chunk-size 50

支持的命令行参数:
    --config PATH           配置文件路径（默认：workflows/robot_interaction/replay_chunk_config.yaml）
    --repo-id REPO          数据集 HuggingFace repo ID（覆盖配置文件）
    --dataset-root PATH     数据集本地根目录（覆盖配置文件）
    --episode N             要回放的 episode 索引（覆盖配置文件）
    --chunk-size N          每个 chunk 的 action 数量（覆盖配置文件）
    --poll-interval FLOAT   轮询 need_new_chunk 的间隔秒数（覆盖配置文件）
    --http-base-url URL     HTTP API 地址（覆盖配置文件）
    --robot-id ID           机器人 ID（覆盖配置文件）
    --play-sounds BOOL      是否在开始时语音播报（覆盖配置文件）
    --no-sounds             关闭语音播报
    --return-to-initial     回放结束后把机械臂送回 episode 第一帧（覆盖配置文件）
    --no-return-to-initial  回放结束后不回家（覆盖配置文件）

示例:
    # 回放第 0 个 episode，chunk_size=100
    python workflows/robot_interaction/replay_chunk.py --episode 0 --chunk-size 100

    # 指定不同数据集
    python workflows/robot_interaction/replay_chunk.py \
        --repo-id username/my_dataset \
        --episode 3 \
        --chunk-size 50

    # 调整轮询间隔
    python workflows/robot_interaction/replay_chunk.py --poll-interval 0.02
"""
import argparse
import atexit
import ctypes
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path

# 共享配置加载器（支持 .json / .yaml / .yml）
sys.path.insert(0, str(Path(__file__).parent.parent))
from _config_loader import load_config as _load_config_raw  # noqa: E402
from _robot_home import return_to_home_and_disable  # noqa: E402


def _resolve_dataset_root(value: str) -> str:
    """解析 dataset.root 配置项，使其与 wrapper 的 CWD 解耦"""
    p = Path(value)
    if p.is_absolute():
        return value
    repo_root = Path(__file__).parent.parent.parent
    return str((repo_root / value).resolve())


def load_config(config_path: Path) -> dict:
    """加载配置文件，保留原异常处理（文件不存在直接 sys.exit(1)）"""
    if not config_path.is_file():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    return _load_config_raw(config_path)


# Linux-only: when our parent dies, the kernel will deliver SIGTERM to
# this child.
_PR_SET_PDEATHSIG = 1


def _child_preexec() -> None:
    """Run in the child process immediately after fork, before exec."""
    os.setsid()
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except OSError:
        pass


def find_lerobot_replay_chunk():
    """查找 lerobot-replay-chunk 命令"""
    # 方式 1: 检查是否在 PATH 中
    replay_path = shutil.which("lerobot-replay-chunk")
    if replay_path:
        return replay_path

    # 方式 2: 检查 conda 环境
    conda_env = os.environ.get("CONDA_PREFIX")
    if conda_env:
        conda_replay = Path(conda_env) / "bin" / "lerobot-replay-chunk"
        if conda_replay.is_file():
            return str(conda_replay)

    # 方式 3: 尝试 uv run
    if shutil.which("uv"):
        return ["uv", "run", "lerobot-replay-chunk"]

    # 方式 4: 尝试 python -m
    return [sys.executable, "-m", "lerobot.scripts.lerobot_replay_chunk"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "replay_chunk_config.yaml",
        help="配置文件路径（默认：workflows/robot_interaction/replay_chunk_config.yaml）",
    )
    parser.add_argument(
        "--repo-id",
        help="数据集 HuggingFace repo ID（覆盖配置文件中的 dataset.repo_id）",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="数据集本地根目录（覆盖配置文件中的 dataset.root）",
    )
    parser.add_argument(
        "--episode",
        type=int,
        help="要回放的 episode 索引（覆盖配置文件中的 dataset.episode）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="每个 chunk 的 action 数量（覆盖配置文件中的 chunk.size）",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        help="轮询 need_new_chunk 的间隔秒数（覆盖配置文件中的 chunk.poll_interval）",
    )
    parser.add_argument(
        "--http-base-url",
        help="HTTP API 地址（覆盖配置文件中的 robot.http_base_url）",
    )
    parser.add_argument(
        "--robot-id",
        help="机器人 ID（覆盖配置文件中的 robot.id）",
    )
    parser.add_argument(
        "--play-sounds",
        dest="play_sounds",
        action="store_true",
        help="启用语音播报（覆盖配置文件）",
    )
    parser.add_argument(
        "--no-sounds",
        dest="play_sounds",
        action="store_false",
        help="关闭语音播报（覆盖配置文件）",
    )
    parser.add_argument(
        "--return-to-initial",
        dest="return_to_initial",
        action="store_true",
        help="回放结束后把机械臂平滑送回 episode 第一帧（覆盖配置文件）",
    )
    parser.add_argument(
        "--no-return-to-initial",
        dest="return_to_initial",
        action="store_false",
        help="回放结束后不回家（覆盖配置文件）",
    )
    parser.set_defaults(play_sounds=None, return_to_initial=None)

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    print(f"✓ 加载配置: {args.config}")

    # CLI 参数覆盖配置文件
    if args.repo_id:
        config["dataset"]["repo_id"] = args.repo_id
    if args.dataset_root:
        config["dataset"]["root"] = str(args.dataset_root)
    if args.episode is not None:
        config["dataset"]["episode"] = args.episode
    if args.chunk_size is not None:
        config["chunk"]["size"] = args.chunk_size
    if args.poll_interval is not None:
        config["chunk"]["poll_interval"] = args.poll_interval
    if args.http_base_url:
        config["robot"]["http_base_url"] = args.http_base_url
    if args.robot_id:
        config["robot"]["id"] = args.robot_id
    if args.play_sounds is not None:
        config["play_sounds"] = args.play_sounds
    if args.return_to_initial is not None:
        config["return_to_initial_position"] = args.return_to_initial

    # 校验必要字段
    if not config.get("dataset", {}).get("repo_id"):
        print("❌ 数据集 repo_id 必填（请在配置文件中设置 dataset.repo_id 或使用 --repo-id）")
        sys.exit(1)
    if config.get("dataset", {}).get("episode") is None:
        print("❌ episode 必填（请在配置文件中设置 dataset.episode 或使用 --episode）")
        sys.exit(1)

    repo_id = config["dataset"]["repo_id"]
    episode = config["dataset"]["episode"]
    chunk_size = config["chunk"]["size"]
    poll_interval = config["chunk"]["poll_interval"]

    print(f"✓ Dataset: {repo_id}")
    print(f"✓ Episode: {episode}")
    print(f"✓ Chunk size: {chunk_size}")
    print(f"✓ Poll interval: {poll_interval}s")
    print(f"✓ Robot: {config['robot']['id']} @ {config['robot']['http_base_url']}")
    print("✓ Cameras: disabled (replay 不需要相机)")

    # 查找 lerobot-replay-chunk
    replay_cmd = find_lerobot_replay_chunk()
    if isinstance(replay_cmd, list):
        cmd = replay_cmd.copy()
    else:
        cmd = [replay_cmd]

    # 构造 lerobot-replay-chunk 命令参数（HTTP 版本）
    cmd.extend([
        "--robot.type=marvain_m6_http",
        f"--robot.id={config['robot']['id']}",
        f"--robot.http_base_url={config['robot']['http_base_url']}",
        f"--robot.timeout={config['robot'].get('timeout', 5.0)}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.episode={episode}",
        f"--chunk_size={chunk_size}",
        f"--poll_interval={poll_interval}",
    ])

    # 添加 joint_names 参数
    if config["robot"].get("joint_names"):
        import json
        joint_names_json = json.dumps(config["robot"]["joint_names"])
        cmd.append(f"--robot.joint_names={joint_names_json}")

    # dataset.root（可选，指定本地数据集路径可避免重新下载）
    if config["dataset"].get("root"):
        cmd.append(f"--dataset.root={_resolve_dataset_root(config['dataset']['root'])}")

    # 语音播报
    play_sounds = config.get("play_sounds", True)
    cmd.append(f"--play_sounds={'true' if play_sounds else 'false'}")

    env = os.environ.copy()

    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}\n")

    # Run the child with robust process management
    proc = subprocess.Popen(
        cmd,
        env=env,
        preexec_fn=_child_preexec,
    )

    def _forward_to_child(signum: int) -> None:
        try:
            os.killpg(proc.pid, signum)
        except ProcessLookupError:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda s, _frm: _forward_to_child(s))

    def _ensure_child_dead() -> None:
        """Idempotent: best-effort TERM, then KILL. Safe to call repeatedly."""
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

    atexit.register(_ensure_child_dead)

    # 记录是否要做 return-to-home（在 _child 已经 disconnect 后才执行）
    return_to_initial = config.get("return_to_initial_position", True)
    if return_to_initial:
        print(f"🟢 Return-to-home: enabled（回放结束后送回 episode 第一帧）")
    else:
        print(f"⚪ Return-to-home: disabled")

    returncode = 0
    try:
        returncode = proc.wait()
    except KeyboardInterrupt:
        try:
            returncode = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _ensure_child_dead()
            returncode = 130
    finally:
        _ensure_child_dead()

    # 回放结束后（child 进程已 disconnect）把机械臂送回 home（HTTP 版本）
    if return_to_initial:
        return_to_home_and_disable(config, returncode)

    return returncode


if __name__ == "__main__":
    sys.exit(main())
