#!/usr/bin/env python3
"""replay.py — HTTP 接口真机回放主入口

把数据集中某个 episode 的动作序列完整地在真实机器人上回放一遍（通过 HTTP API）。
常用于：复现某次示教、验证数据是否可执行、对比策略与原示教轨迹。

用法:
    # 使用默认配置
    python workflows/robot_interaction/replay.py

    # 使用自定义配置文件
    python workflows/robot_interaction/replay.py --config my_replay_config.yaml

    # 临时覆盖参数
    python workflows/robot_interaction/replay.py --repo-id username/my_dataset --episode 0
    python workflows/robot_interaction/replay.py --http-base-url http://192.168.10.123:8010
    python workflows/robot_interaction/replay.py --fps 30

支持的命令行参数:
    --config PATH           配置文件路径（默认：workflows/robot_interaction/replay_config.yaml）
    --repo-id REPO          数据集 HuggingFace repo ID（覆盖配置文件）
    --dataset-root PATH     数据集本地根目录（覆盖配置文件）
    --episode N             要回放的 episode 索引（覆盖配置文件）
    --fps FPS               回放帧率 Hz（覆盖配置文件）
    --http-base-url URL     HTTP API 地址（覆盖配置文件）
    --robot-id ID           机器人 ID（覆盖配置文件）
    --play-sounds BOOL      是否在开始时语音播报（覆盖配置文件）
    --no-sounds             关闭语音播报
    --return-to-initial     回放结束后把机械臂送回 episode 第一帧（覆盖配置文件）
    --no-return-to-initial  回放结束后不回家（覆盖配置文件）

与 deploy.py 的区别:
    deploy.py   用训练好的 policy 推理 → 让机器人"按策略做"
    replay.py   直接重放数据集中的 action → 让机器人"按示教做"
    - replay 不需要 policy 路径
    - replay 不需要选择策略 (strategy)
    - replay 不需要相机配置（重放只发关节位置 action）
    - replay 必须指定 episode 索引

示例:
    # 回放第 0 个 episode
    python workflows/robot_interaction/replay.py --episode 0

    # 指定不同数据集
    python workflows/robot_interaction/replay.py \
        --repo-id username/my_dataset \
        --episode 3 \
        --fps 30

    # 慢速回放（调试用）
    python workflows/robot_interaction/replay.py --fps 10

    # 切换 HTTP 服务器地址
    python workflows/robot_interaction/replay.py --http-base-url http://192.168.10.100:8010
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
    """解析 dataset.root 配置项，使其与 wrapper 的 CWD 解耦：

    - 绝对路径：原样返回
    - 相对路径（无论是否含 ``/``）：统一解析为
      ``<REPO_ROOT>/<value>``，其中 ``REPO_ROOT = workflows/..``。
      这样无论用户从哪个目录启动 wrapper，``root: datasets/<name>``
      和 ``root: <name>`` 都会指向仓库内的 ``datasets/<name>``。
    """
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
# this child. This is the only mechanism that survives the parent being
# killed with SIGKILL — it gives the child a chance to run its
# `try/finally: robot.disconnect()` cleanup.
_PR_SET_PDEATHSIG = 1


def _child_preexec() -> None:
    """Run in the child process immediately after fork, before exec.

    1. ``os.setsid()`` puts the child in a new session & process group
       so the wrapper can signal the entire subtree (not just the
       direct child) with a single ``os.killpg`` call.
    2. ``PR_SET_PDEATHSIG = SIGTERM`` asks the kernel to send SIGTERM
       to the child when this child is reaped because its parent died.
       This is the only mechanism that survives the parent being killed
       with SIGKILL — without it, killing the wrapper would leave the
       replay process orphaned and the arm still energized.
    """
    os.setsid()
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except OSError:
        # Non-Linux: best-effort only, the wrapper-side signal
        # forwarding is the best we can do there.
        pass


def find_lerobot_replay():
    """查找 lerobot-replay 命令"""
    # 方式 1: 检查是否在 PATH 中
    replay_path = shutil.which("lerobot-replay")
    if replay_path:
        return replay_path

    # 方式 2: 检查 conda 环境
    conda_env = os.environ.get("CONDA_PREFIX")
    if conda_env:
        conda_replay = Path(conda_env) / "bin" / "lerobot-replay"
        if conda_replay.is_file():
            return str(conda_replay)

    # 方式 3: 尝试 uv run
    if shutil.which("uv"):
        return ["uv", "run", "lerobot-replay"]

    # 方式 4: 尝试 python -m
    return [sys.executable, "-m", "lerobot.scripts.lerobot_replay"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "replay_config.yaml",
        help="配置文件路径（默认：workflows/robot_interaction/replay_config.yaml）",
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
        "--fps",
        type=float,
        help="回放帧率 Hz（覆盖配置文件中的 dataset.fps）",
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
    if args.fps:
        config["dataset"]["fps"] = args.fps
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
    print(f"✓ Dataset: {repo_id}")
    print(f"✓ Episode: {episode}")
    print(f"✓ Robot: {config['robot']['id']} @ {config['robot']['http_base_url']}")
    print("✓ Cameras: disabled (replay 不需要相机)")

    # 查找 lerobot-replay
    replay_cmd = find_lerobot_replay()
    if isinstance(replay_cmd, list):
        cmd = replay_cmd.copy()
    else:
        cmd = [replay_cmd]

    # 构造 lerobot-replay 命令参数（HTTP 版本）
    cmd.extend([
        "--robot.type=marvain_m6_http",
        f"--robot.id={config['robot']['id']}",
        f"--robot.http_base_url={config['robot']['http_base_url']}",
        f"--robot.timeout={config['robot'].get('timeout', 5.0)}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.episode={episode}",
    ])

    # 添加 joint_names 参数
    if config["robot"].get("joint_names"):
        import json
        joint_names_json = json.dumps(config["robot"]["joint_names"])
        cmd.append(f"--robot.joint_names={joint_names_json}")

    # dataset.root（可选，指定本地数据集路径可避免重新下载）
    if config["dataset"].get("root"):
        cmd.append(f"--dataset.root={_resolve_dataset_root(config['dataset']['root'])}")

    # 回放帧率（默认使用数据集 fps，CLI 或配置文件可覆盖）
    fps = config["dataset"].get("fps")
    if fps:
        cmd.append(f"--dataset.fps={int(fps)}")

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
