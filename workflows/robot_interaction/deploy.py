#!/usr/bin/env python3
"""deploy.py — HTTP 接口真机部署主入口

这是 HTTP 接口版本的部署脚本，通过 HTTP API 与机器人通信。

用法:
    # 使用默认配置
    python workflows/robot_interaction/deploy.py

    # 使用自定义配置文件
    python workflows/robot_interaction/deploy.py --config my_config.yaml

    # 临时覆盖某些参数
    python workflows/robot_interaction/deploy.py --policy-path outputs/train/latest/pretrained_model
    python workflows/robot_interaction/deploy.py --http-base-url http://192.168.10.123:8010
    python workflows/robot_interaction/deploy.py --fps 15.0
    python workflows/robot_interaction/deploy.py --strategy sentry

支持的命令行参数:
    --config PATH               配置文件路径（默认：workflows/robot_interaction/deploy_config.yaml）
    --policy-path PATH          模型路径（覆盖配置文件）
    --http-base-url URL         HTTP API 地址（覆盖配置文件）
    --device DEVICE             推理设备，cuda/cpu（覆盖配置文件）
    --fps FPS                   推理频率 Hz（覆盖配置文件）
    --strategy STRATEGY         推理策略（覆盖配置文件）
                                可选值：base, sentry, highlight, dagger, episodic
    --inference-type TYPE       推理模式（覆盖配置文件）
                                可选值：sync（默认）, rtc（Real-Time Chunking）
    --execution-horizon N       RTC模式：每次执行N步（覆盖配置文件）

推理模式说明:
    sync       同步模式（默认）：推理整个chunk，全部执行完再推理
    rtc        Real-Time Chunking：推理chunk，只执行前N步就重新推理
               更快响应环境变化，适合动态场景

推理策略说明:
    base       只推理，不录制（默认，最常用）
    sentry     边推理边录制，持续保存所有数据
    highlight  环形缓冲，只保存按键触发的片段
    dagger     人工干预模式，可随时接管控制
    episodic   分段录制，每个 episode 单独保存

示例:
    # 测试新模型
    python workflows/robot_interaction/deploy.py \
        --policy-path outputs/train/act_v2_new/checkpoints/100000/pretrained_model

    # 降低推理频率（慢速调试）
    python workflows/robot_interaction/deploy.py --fps 5.0

    # 使用 sentry 策略录制数据
    python workflows/robot_interaction/deploy.py --strategy sentry

    # 切换 HTTP 服务器地址
    python workflows/robot_interaction/deploy.py --http-base-url http://192.168.10.100:8010
"""
import argparse
import atexit
import ctypes
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

# 共享配置加载器（支持 .json / .yaml / .yml）
sys.path.insert(0, str(Path(__file__).parent.parent))
from _config_loader import load_config as _load_config_raw  # noqa: E402
from _robot_home import return_to_home_and_disable  # noqa: E402


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
       rollout process orphaned and the arm still energized.
    """
    os.setsid()
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except OSError:
        # Non-Linux: best-effort only, the wrapper-side signal
        # forwarding is the best we can do there.
        pass


def resolve_policy_path(path: str) -> Path:
    """解析 policy 路径"""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(__file__).parent.parent.parent / p


def find_lerobot_rollout():
    """查找 lerobot-rollout 命令"""
    import shutil

    # 方式 1: 检查是否在 PATH 中
    rollout_path = shutil.which("lerobot-rollout")
    if rollout_path:
        return rollout_path

    # 方式 2: 检查 conda 环境
    conda_env = os.environ.get("CONDA_PREFIX")
    if conda_env:
        conda_rollout = Path(conda_env) / "bin" / "lerobot-rollout"
        if conda_rollout.is_file():
            return str(conda_rollout)

    # 方式 3: 尝试 uv run
    if shutil.which("uv"):
        return ["uv", "run", "lerobot-rollout"]

    # 方式 4: 尝试 python -m
    return [sys.executable, "-m", "lerobot.scripts.lerobot_rollout"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "deploy_config_chunk.yaml",
        help="配置文件路径（默认：workflows/robot_interaction/deploy_config.yaml）"
    )
    parser.add_argument(
        "--policy-path",
        help="模型路径（覆盖配置文件中的 policy.path）"
    )
    parser.add_argument(
        "--http-base-url",
        help="HTTP API 地址（覆盖配置文件中的 robot.http_base_url）"
    )
    parser.add_argument(
        "--robot-id",
        help="机器人 ID（覆盖配置文件中的 robot.id）"
    )
    parser.add_argument(
        "--safety-stats-path",
        help="安全统计文件路径（覆盖配置文件中的 robot.safety_stats_path）"
    )
    parser.add_argument(
        "--device",
        help="推理设备 cuda/cpu（覆盖配置文件中的 policy.device）"
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="推理频率 Hz（覆盖配置文件中的 inference.fps）"
    )
    parser.add_argument(
        "--strategy",
        choices=["base", "sentry", "highlight", "dagger", "episodic"],
        help="推理策略（覆盖配置文件中的 inference.strategy）"
    )
    parser.add_argument(
        "--inference-type",
        choices=["sync", "rtc", "chunk"],
        help="推理模式 sync/rtc/chunk（覆盖配置文件中的 inference.type）"
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        help="RTC模式：每次执行的步数（覆盖配置文件中的 inference.rtc.execution_horizon）"
    )
    parser.add_argument(
        "--max-guidance-weight",
        type=float,
        help="RTC模式：最大引导权重（覆盖配置文件中的 inference.rtc.max_guidance_weight）"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="运行时长（秒，0=无限）（覆盖配置文件中的 inference.duration）"
    )
    parser.add_argument(
        "--interpolation-multiplier",
        type=int,
        help="动作插值倍数（覆盖配置文件中的 inference.interpolation_multiplier）"
    )
    parser.add_argument(
        "--return-to-initial",
        type=bool,
        help="断连前是否回到初始位置（覆盖配置文件中的 inference.return_to_initial_position）"
    )
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="启用 torch.compile 编译加速（覆盖配置文件中的 inference.use_torch_compile）"
    )
    parser.add_argument(
        "--rename-map",
        type=str,
        help=(
            "JSON 字符串，例如 '{\"right_eye\":\"camera1\",\"left_wrist\":\"camera2\"}'。"
            "将 robot 输出的 obs key (left_eye/right_eye/left_wrist/right_wrist) 重映射"
            "到 policy 期待的 key（由模型训练 schema 决定，例如 SmolVLA 用 camera1/2/3）。"
            "留空或传 '{}' 等于不重命名。"
        ),
    )
    parser.add_argument(
        "--show-cameras",
        action="store_true",
        help=(
            "推理时并行启动 show_cameras.py，打开 OpenCV 预览窗口显示模型"
            "当前正在使用的相机（按 model.config.json 自动推导，或被 "
            "--cameras-override / 配置 inference.show_cameras 覆盖）。"
            "无 DISPLAY/WAYLAND 时该子进程会以非零退出码退出，deploy 仍继续。"
        ),
    )
    parser.add_argument(
        "--cameras-override",
        nargs="+",
        metavar="NAME",
        help=(
            "仅当 --show-cameras 时生效。显式指定要预览的物理相机名裸 key，"
            "跳过从 model.config.json 自动推导。"
            "例: --cameras-override right_eye left_wrist right_wrist"
        ),
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    print(f"✓ 加载配置: {args.config}")

    # CLI 参数覆盖配置文件
    if args.policy_path:
        config["policy"]["path"] = args.policy_path
    if args.http_base_url:
        config["robot"]["http_base_url"] = args.http_base_url
    if args.robot_id:
        config["robot"]["id"] = args.robot_id
    if args.safety_stats_path:
        config["robot"]["safety_stats_path"] = args.safety_stats_path
    if args.device:
        config["policy"]["device"] = args.device
    if args.fps:
        config["inference"]["fps"] = args.fps
    if args.strategy:
        config["inference"]["strategy"] = args.strategy
    if args.inference_type:
        config["inference"]["type"] = args.inference_type
    if args.execution_horizon:
        if "rtc" not in config["inference"]:
            config["inference"]["rtc"] = {}
        config["inference"]["rtc"]["execution_horizon"] = args.execution_horizon
    if args.max_guidance_weight is not None:
        if "rtc" not in config["inference"]:
            config["inference"]["rtc"] = {}
        config["inference"]["rtc"]["max_guidance_weight"] = args.max_guidance_weight
    if args.duration:
        config["inference"]["duration"] = args.duration
    if args.interpolation_multiplier:
        config["inference"]["interpolation_multiplier"] = args.interpolation_multiplier
    if args.return_to_initial is not None:
        config["inference"]["return_to_initial_position"] = args.return_to_initial
    if args.use_torch_compile:
        config["inference"]["use_torch_compile"] = True

    # CLI --rename-map 覆盖（JSON 字符串）；空串视为不改写
    if args.rename_map is not None and args.rename_map.strip() != "":
        try:
            parsed = json.loads(args.rename_map)
        except json.JSONDecodeError as e:
            print(f"❌ --rename-map 不是合法 JSON: {e}")
            sys.exit(1)
        if not isinstance(parsed, dict):
            print(f"❌ --rename-map 必须解析成 dict，实际是 {type(parsed).__name__}")
            sys.exit(1)
        config["rename_map"] = parsed

    # CLI --show-cameras / --cameras-override 覆盖
    if args.show_cameras:
        config.setdefault("inference", {})["show_cameras"] = True
    if args.cameras_override:
        config.setdefault("inference", {})["cameras_override"] = args.cameras_override

    # 解析 policy 路径
    policy_path = resolve_policy_path(config["policy"]["path"])
    if not policy_path.is_dir():
        print(f"❌ Policy 路径不存在: {policy_path}")
        sys.exit(1)
    print(f"✓ Policy: {policy_path}")

    # 构造相机配置
    cameras_json = json.dumps(config["robot"]["cameras"])

    # 获取任务
    task = config.get("dataset", {}).get("single_task", "")
    if task:
        print(f"✓ Task: {task}")

    # 查找 lerobot-rollout
    rollout_cmd = find_lerobot_rollout()
    if isinstance(rollout_cmd, list):
        cmd = rollout_cmd.copy()
    else:
        cmd = [rollout_cmd]

    # 构造 lerobot-rollout 命令参数（robot.type 决定使用哪一套后端）
    # 默认仍是 marvain_m6_http（向后兼容），配置里写 marvain_m6_hybrid 时
    # 走 SDK 直接下发动作的路径（HTTP 只用于观测）。
    robot_type = config["robot"].get("type", "marvain_m6_http")
    cmd.extend([
        f"--robot.type={robot_type}",
        f"--robot.id={config['robot']['id']}",
        f"--robot.http_base_url={config['robot']['http_base_url']}",
        f"--robot.timeout={config['robot'].get('timeout', 5.0)}",
        f"--robot.cameras={cameras_json}",
        f"--policy.path={policy_path}",
        f"--policy.device={config['policy']['device']}",
        f"--fps={config['inference']['fps']}",
        f"--strategy.type={config['inference']['strategy']}",
    ])

    # Hybrid 机器人（HTTP 观测 + SDK 下发动作）：补 SDK 侧参数
    if robot_type == "marvain_m6_hybrid":
        cmd.append(f"--robot.robot_ip={config['robot'].get('ip', '192.168.10.190')}")
        cmd.append(f"--robot.control_mode={config['robot'].get('control_mode', 'impedance')}")
        cmd.append(f"--robot.vel_ratio={config['robot'].get('vel_ratio', 20)}")
        cmd.append(f"--robot.acc_ratio={config['robot'].get('acc_ratio', 20)}")
        cmd.append(
            f"--robot.disable_torque_on_disconnect="
            f"{str(config['robot'].get('disable_torque_on_disconnect', False)).lower()}"
        )

    # 顶层 rename_map（robot obs keys → policy input slots）
    # YAML 里写在 robot: 下面，deploy 给 key/value 加 "observation.images."
    # 前缀后序列化成 JSON 字符串给 draccus；draccus 内部 cfgparsing.parse_string
    # 走 YAML parser，会识别 JSON 风格的 {...} 并解析回 dict。
    #
    # 为什么加 "observation.images." 前缀：SmolVLA 的 saved preprocessor 把
    # rename_observations_processor 放在 to_transition→steps→to_output 之间
    # 运行，step 看到的是已经带前缀的 transition key（policy_preprocessor.json
    # 里就是 "observation.images.right_eye" → "observation.images.camera1"）。
    yaml_rename_map = (config.get("rename_map")
                       or config.get("robot", {}).get("rename_map")
                       or {})
    if yaml_rename_map:
        prefixed = {
            f"observation.images.{k}": f"observation.images.{v}"
            for k, v in yaml_rename_map.items()
        }
        cmd.append(f"--rename_map={json.dumps(prefixed)}")

    # 添加 joint_names 参数
    if config["robot"].get("joint_names"):
        joint_names_json = json.dumps(config["robot"]["joint_names"])
        cmd.append(f"--robot.joint_names={joint_names_json}")

    if task:
        cmd.append(f"--task={task}")

    # 推理参数
    if config["inference"].get("duration", 0.0) > 0:
        cmd.append(f"--duration={config['inference']['duration']}")

    interpolation = config["inference"].get("interpolation_multiplier", 1)
    if interpolation != 1:
        cmd.append(f"--interpolation_multiplier={interpolation}")

    if not config["inference"].get("return_to_initial_position", True):
        cmd.append("--return_to_initial_position=false")

    if config["inference"].get("use_torch_compile", False):
        cmd.append("--use_torch_compile=true")
        cmd.append(f"--torch_compile_backend={config['inference'].get('torch_compile_backend', 'inductor')}")
        cmd.append(f"--torch_compile_mode={config['inference'].get('torch_compile_mode', 'default')}")
        cmd.append(f"--compile_warmup_inferences={config['inference'].get('compile_warmup_inferences', 2)}")

    # RTC 模式参数
    inference_type = config["inference"].get("type", "sync")
    if inference_type == "rtc":
        cmd.append("--inference.type=rtc")
        rtc_config = config["inference"].get("rtc", {})
        if rtc_config.get("execution_horizon"):
            cmd.append(f"--inference.rtc.execution_horizon={rtc_config['execution_horizon']}")
        if rtc_config.get("max_guidance_weight"):
            cmd.append(f"--inference.rtc.max_guidance_weight={rtc_config['max_guidance_weight']}")
    elif inference_type == "chunk":
        # 开环 chunk 模式：每 chunk_interval_s 推理一次，把前 n_action_steps 个 action
        # 一次性 POST 给机器人（send_action_chunk）。
        cmd.append("--inference.type=chunk")
        if config["inference"].get("n_action_steps") is not None:
            cmd.append(f"--inference.n_action_steps={config['inference']['n_action_steps']}")
        if config["inference"].get("chunk_interval_s") is not None:
            cmd.append(f"--inference.chunk_interval_s={config['inference']['chunk_interval_s']}")

    # 非 base 模式需要 dataset 参数
    if config["inference"]["strategy"] != "base":
        cmd.append(f"--dataset.repo_id={config['dataset']['repo_id']}")
        if config["dataset"].get("root"):
            # 与 replay.py 的 _resolve_dataset_root 行为一致：相对路径
            # 一律解析为 <REPO_ROOT>/<value>，与 wrapper 的 CWD 解耦。
            _root = config["dataset"]["root"]
            _root_p = Path(_root)
            if not _root_p.is_absolute():
                _root = str((Path(__file__).parent.parent.parent / _root).resolve())
            cmd.append(f"--dataset.root={_root}")
            cmd.append(f"--dataset.fps={int(config['inference']['fps'])}")

    # 添加 safety 参数（如果配置了）
    if config["robot"].get("safety_stats_path"):
        cmd.append(f"--robot.safety_stats_path={config['robot']['safety_stats_path']}")
    if config["robot"].get("action_clip_margin_deg"):
        cmd.append(f"--robot.action_clip_margin_deg={config['robot']['action_clip_margin_deg']}")
    if config["robot"].get("max_relative_target_deg"):
        cmd.append(f"--robot.max_relative_target_deg={config['robot']['max_relative_target_deg']}")

    env = os.environ.copy()

    # Hybrid 机器人：rollout 进程要直接 import MarvinRobotWrapper，
    # 需要把 libMarvinSDK.so / libKine.so 所在的目录加进 LD_LIBRARY_PATH。
    if robot_type == "marvain_m6_hybrid":
        so_dir = Path(__file__).parent.parent.parent / "src" / "lerobot" / "Marvin_sdk_pro"
        if so_dir.is_dir():
            env["LD_LIBRARY_PATH"] = f"{so_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}\n")

    # Run the child with robust process management
    proc = subprocess.Popen(
        cmd,
        env=env,
        preexec_fn=_child_preexec,
    )

    # Optionally launch show_cameras.py alongside rollout.
    # It's a separate process that polls the same HTTP /state endpoint.
    show_cameras_proc: subprocess.Popen | None = None
    show_cameras_script = Path(__file__).parent / "show_cameras.py"
    if config.get("inference", {}).get("show_cameras", False):
        show_cmd = [sys.executable, str(show_cameras_script), "--config", str(args.config)]
        cams_override = config.get("inference", {}).get("cameras_override")
        if cams_override:
            show_cmd.extend(["--cameras", *cams_override])
        print(f"\n📷 Launching camera viewer: {' '.join(shlex.quote(c) for c in show_cmd)}\n")
        # Force unbuffered stdout/stderr so any cv2.error traceback from the
        # viewer appears immediately rather than getting lost in pipe buffers.
        # Silence Qt's harmless "no fonts" warning (OpenCV's bundled Qt can't
        # find a font dir on most distros; window still renders correctly).
        viewer_env = env.copy()
        viewer_env["PYTHONUNBUFFERED"] = "1"
        viewer_env.setdefault("QT_LOGGING_RULES", "qt.text.font.db.warning=false")
        try:
            show_cameras_proc = subprocess.Popen(
                show_cmd,
                env=viewer_env,
                preexec_fn=_child_preexec,
            )
        except FileNotFoundError as e:
            print(f"⚠️  Failed to launch show_cameras.py: {e}（continuing without preview）")

    # 所有受 wrapper 管理的子进程：rollout + 可选的 preview viewer
    managed_procs: list[subprocess.Popen] = [p for p in (proc, show_cameras_proc) if p is not None]

    def _forward_to_child(signum: int) -> None:
        for p in managed_procs:
            try:
                os.killpg(p.pid, signum)
            except ProcessLookupError:
                pass

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda s, _frm: _forward_to_child(s))

    def _ensure_child_dead() -> None:
        """Idempotent: best-effort TERM, then KILL. Safe to call repeatedly."""
        for p in managed_procs:
            if p.poll() is not None:
                continue
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                continue
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

    atexit.register(_ensure_child_dead)

    # 记录是否要在子进程结束后做 wrapper 层的 return-to-home
    return_to_initial = config.get("inference", {}).get("return_to_initial_position", True)
    if return_to_initial:
        print(f"🟢 Return-to-home: enabled（rollout 结束后送回标准 home）")
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

    # 子进程结束后把机械臂送回 home（HTTP 版本）
    if return_to_initial:
        return_to_home_and_disable(config, returncode)

    return returncode


if __name__ == "__main__":
    sys.exit(main())
