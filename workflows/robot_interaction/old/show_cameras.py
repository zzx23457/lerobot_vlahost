#!/usr/bin/env python3
"""show_cameras.py — 实时相机预览（OpenCV 窗口）

用法:
    # 使用默认 deploy_config.yaml
    python workflows/robot_interaction/show_cameras.py

    # 显式指定相机（覆盖 yaml/model 推导）
    python workflows/robot_interaction/show_cameras.py \\
        --cameras right_eye left_wrist right_wrist

    # 自定义 HTTP 端点 / 模型路径
    python workflows/robot_interaction/show_cameras.py \\
        --http-base-url http://192.168.10.100:8010 \\
        --policy-path outputs/train/act_xxx/checkpoints/10000/pretrained_model

支持的命令行参数:
    --config PATH               配置文件路径（默认：workflows/robot_interaction/deploy_config.yaml）
    --http-base-url URL         HTTP API 地址（覆盖配置文件中 robot.http_base_url）
    --policy-path PATH          模型路径（覆盖配置文件中 policy.path，用来推导相机列表）
    --cameras cam1 cam2 ...     显式指定要显示的物理相机名裸 key（覆盖配置文件/model 推导）
    --fps N                     轮询频率 Hz（默认 30）
    --show-quad                 额外显示未切分的原始 quad_image，调试象限用
    --window-size W H           单个相机窗口的宽高（默认 640 480）

显示逻辑:
    1. 优先用 --cameras 给定的物理相机名；
    2. 否则从 <policy.path>/config.json 的 input_features 里过滤
       observation.images.* 得到模型期望的 model-side 相机名，
       再用配置文件的 rename_map 反查回物理相机名；
    3. 都没拿到就回退到全部 4 个物理相机（左/右眼 + 左/右腕部）。

    窗口标题: "<idx>: <physical_name> → <model_name>"（rename 时显示后者，否则省略）

按 q 或 ESC 退出；Ctrl+C 也会被捕获。
"""
>>> OLD 备份：本会话开始时仓库里的版本。
>>> 与 ../show_cameras.py 当前版本的差异：
>>>   1. 没有 _grab_mjpeg_frame() 模块函数
>>>   2. quad_image 分支只处理 base64 `data` 字段（不识别 stream_url MJPEG 流）
>>>
>>> 回滚：cp old/show_cameras.py ../show_cameras.py
>>> ⚠ 这个版本跟当前服务端不兼容（quad_image 用的是 MJPEG stream_url 不是 base64），
>>>   跑起来相机窗口会是黑的。

按 q 或 ESC 退出；Ctrl+C 也会被捕获。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

# 允许 `python workflows/robot_interaction/show_cameras.py` 直接调用
sys.path.insert(0, str(Path(__file__).parent.parent))
from _config_loader import load_config  # noqa: E402
from lerobot.robots.marvain_m6_http.marvain_m6_http import (  # noqa: E402
    MarvainM6HttpRobot,
)
from lerobot.utils.keyboard_input import is_headless  # noqa: E402


# 4 个物理相机（HTTP quad_image 切分后的 key），用作回退列表
DEFAULT_PHYSICAL_CAMERAS = ("left_eye", "right_eye", "left_wrist", "right_wrist")


def resolve_path(p: str | Path) -> Path:
    """相对路径解析为项目根下的绝对路径，与 deploy.py 一致。"""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (Path(__file__).parent.parent.parent / pp).resolve()


def derive_model_cameras(policy_path: Path) -> list[str]:
    """从 model config.json 读 observation.images.* 的 model-side 相机名列表。

    例：ACT → ['right_eye', 'left_wrist', 'right_wrist']
        SmolVLA → ['camera1', 'camera2', 'camera3']
    """
    cfg_file = policy_path / "config.json"
    if not cfg_file.is_file():
        return []
    with open(cfg_file) as f:
        cfg = json.load(f)
    names = []
    for k in cfg.get("input_features", {}):
        if k.startswith("observation.images."):
            names.append(k.removeprefix("observation.images."))
    return names


def model_to_physical(
    model_names: list[str], rename_map: dict[str, str]
) -> list[tuple[str, str]]:
    """把 model-side 相机名通过 rename_map 反查回物理相机名。

    rename_map 的语义是 {物理名: 模型名}（见 deploy.py），所以反向查表即可。
    ACT 的 rename_map 为空 → 直接返回 [(name, name), ...]
    SmolVLA rename_map={right_eye: camera1, ...} → [(camera1, right_eye), ...]
    """
    inverse = {v: k for k, v in rename_map.items()}
    return [(m, inverse.get(m, m)) for m in model_names]


def make_window_title(idx: int, model_name: str, phys_name: str) -> str:
    if model_name == phys_name:
        return f"{idx}: {phys_name}"
    return f"{idx}: {phys_name} -> {model_name}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "deploy_config.yaml",
        help="配置文件路径（默认：workflows/robot_interaction/deploy_config.yaml）",
    )
    parser.add_argument(
        "--http-base-url",
        help="HTTP API 地址（覆盖配置文件中的 robot.http_base_url）",
    )
    parser.add_argument(
        "--policy-path",
        help="模型路径（覆盖配置文件中的 policy.path，用于推导相机列表）",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        metavar="NAME",
        help=(
            "显式指定要显示的物理相机名裸 key（覆盖配置文件 / model 推导）。"
            "例: --cameras right_eye left_wrist right_wrist"
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="HTTP 轮询频率 Hz（默认 30）",
    )
    parser.add_argument(
        "--show-quad",
        action="store_true",
        help="额外显示未切分的原始 quad_image（1280x960），用于调试象限切分是否正确",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=[640, 480],
        help="单个相机窗口的宽高（默认 640 480）",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. 加载配置（可选）
    # ------------------------------------------------------------------
    config: dict = {}
    if args.config.is_file():
        config = load_config(args.config)
        print(f"✓ Loaded config: {args.config}")
    else:
        print(f"⚠️  Config not found: {args.config}（仅使用 --http-base-url / --policy-path）")

    http_url = (
        args.http_base_url
        or config.get("robot", {}).get("http_base_url")
    )
    policy_path_str = (
        args.policy_path
        or config.get("policy", {}).get("path")
    )
    if not http_url:
        print("❌ --http-base-url 或配置文件里的 robot.http_base_url 是必填项")
        return 2
    if not policy_path_str:
        print("❌ --policy-path 或配置文件里的 policy.path 是必填项")
        return 2

    policy_path = resolve_path(policy_path_str)
    rename_map: dict = (
        config.get("rename_map")
        or config.get("robot", {}).get("rename_map")
        or {}
    )

    # ------------------------------------------------------------------
    # 2. 推导要显示的相机列表
    # ------------------------------------------------------------------
    if args.cameras:
        # 显式列表：不应用 rename_map（用户给的就是物理 key）
        cams_to_show: list[tuple[str, str]] = [(c, c) for c in args.cameras]
        source = "explicit --cameras"
    else:
        model_names = derive_model_cameras(policy_path) if policy_path.is_dir() else []
        if model_names:
            cams_to_show = model_to_physical(model_names, rename_map)
            source = f"{policy_path.name}/config.json"
        else:
            cams_to_show = [(c, c) for c in DEFAULT_PHYSICAL_CAMERAS]
            source = "fallback (all 4 physical cameras)"

    print(f"✓ HTTP server: {http_url}")
    print(f"✓ Policy:      {policy_path}")
    print(f"✓ Rename map:  {rename_map or '{}'}")
    print(f"✓ Source:      {source}")
    print(f"✓ Showing {len(cams_to_show)} camera(s):")
    for idx, (model_name, phys_name) in enumerate(cams_to_show):
        if model_name == phys_name:
            print(f"   [{idx}] {phys_name}")
        else:
            print(f"   [{idx}] {phys_name}  -> {model_name}")
    if args.show_quad:
        print(f"   [quad] raw quad_image (1280x960)")

    # ------------------------------------------------------------------
    # 3. 无显示器检测
    # ------------------------------------------------------------------
    if is_headless():
        print(
            "❌ No display server detected (DISPLAY / WAYLAND_DISPLAY not set).\n"
            "   Cannot open OpenCV windows. Tip: use X11 forwarding over SSH,"
            " or set DISPLAY=:0"
        )
        return 1

    # ------------------------------------------------------------------
    # 4. 创建窗口
    # ------------------------------------------------------------------
    # is_headless() 只看 DISPLAY/WAYLAND 环境变量；当 DISPLAY 已设置但
    # X server 不可达 / 认证过期 / OpenCV 编译没带 Qt / 装的是 headless 变体
    # 时，cv2.namedWindow 会直接抛 cv2.error。这里做一次显式 sanity check。

    # 检测当前 cv2 是不是 headless 变体：headless 包的 cv2.getBuildInformation()
    # 里 "GUI:" 那行会是 "NONE"，完整包会是 "QT5" / "GTK" / "WIN32" / "Cocoa"。
    # （注意 cv2.highgui 不是 Python 子模块，不能用来探测）
    _build_info = cv2.getBuildInformation()
    _gui_backend = ""
    for _line in _build_info.splitlines():
        _s = _line.strip()
        if _s.startswith("GUI:"):
            _gui_backend = _s.split(":", 1)[1].strip()
            break
    _has_gui = bool(_gui_backend) and _gui_backend.upper() != "NONE"

    titles: list[str] = []
    if _has_gui:
        try:
            probe_title = "__show_cameras_probe__"
            cv2.namedWindow(probe_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(probe_title, 32, 32)
            # imshow + waitKey 才真正把窗口连到 X server；某些环境下
            # namedWindow 成功但 waitKey 才挂
            cv2.imshow(probe_title, np.zeros((32, 32, 3), dtype=np.uint8))
            cv2.waitKey(1)
            cv2.destroyWindow(probe_title)
        except cv2.error as e:
            print(
                f"❌ OpenCV window creation failed: {e}\n"
                f"   is_headless() = {is_headless()}, "
                f"DISPLAY={os.environ.get('DISPLAY')!r}, "
                f"WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY')!r}\n"
                f"   cv2 GUI backend: {_gui_backend!r}\n"
                "   常见原因：X server 不可达 / .Xauthority 过期 / Qt 缺字体。\n"
                "   建议：\n"
                "     - 用 ssh -Y 转发（不是 -X），或直接在有桌面的本机跑\n"
                "     - 或 unset DISPLAY 重新运行，脚本会自动 headless 退出\n"
                "     - 或用 xvfb-run: xvfb-run -a python .../show_cameras.py ..."
            )
            return 1
    else:
        # 命中 headless 变体 → 给出针对性修复命令
        print(
            "❌ 当前 cv2 是 headless 变体（getBuildInformation 报告 GUI=NONE），\n"
            "   它无法创建 GUI 窗口——和 X server / DISPLAY 无关。\n"
            f"   当前: cv2.__version__ = {cv2.__version__}, file = {cv2.__file__}\n"
            "\n"
            "   修复方法（在 lerobot_vlahost 这个 conda env 里跑）：\n"
            f"     {sys.executable} -m pip install --upgrade 'opencv-python>=4.9.0,<4.14.0'\n"
            f"     {sys.executable} -m pip uninstall -y opencv-python-headless\n"
            "\n"
            "   两条都要跑：opencv-python-headless 和 opencv-python 装的是同一个\n"
            "   'cv2' 包名，pip install 不会自动卸载旧的，得手动 uninstall。\n"
            "   lerobot 实际只用 cv2.imdecode / cvtColor / imencode 这一类非 GUI API，\n"
            "   换包后所有推理 / 训练 / 数据加载功能不受影响。\n"
            "\n"
            "   装完之后还需要把 pyproject.toml:65 的依赖约束从\n"
            "     opencv-python-headless>=4.9.0,<4.14.0\n"
            "   改成\n"
            "     opencv-python>=4.9.0,<4.14.0\n"
            "   （否则下次 uv sync 会再次装回头less 变体）。\n"
        )
        return 1

    for idx, (model_name, phys_name) in enumerate(cams_to_show):
        title = make_window_title(idx, model_name, phys_name)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, args.window_size[0], args.window_size[1])
        titles.append(title)
    if args.show_quad:
        cv2.namedWindow("quad_image (raw)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("quad_image (raw)", 1280, 960)

    # ------------------------------------------------------------------
    # 5. 轮询循环
    # ------------------------------------------------------------------
    session = requests.Session()
    dt_ms = max(1, int(1000.0 / max(args.fps, 0.1)))
    blank = np.zeros((args.window_size[1], args.window_size[0], 3), dtype=np.uint8)
    n_ok = 0
    n_err = 0
    t_start = time.monotonic()
    last_stats_print = t_start

    try:
        while True:
            try:
                resp = session.get(f"{http_url}/state", timeout=2.0)
                resp.raise_for_status()
                data = resp.json()
                quad_field = data.get("quad_image")
                if quad_field is not None and "data" in quad_field:
                    quad_rgb = MarvainM6HttpRobot._decode_image(quad_field["data"])
                    if args.show_quad:
                        cv2.imshow(
                            "quad_image (raw)",
                            cv2.cvtColor(quad_rgb, cv2.COLOR_RGB2BGR),
                        )
                    cams = MarvainM6HttpRobot._split_quad_image(quad_rgb)
                    for title, (_, phys_name) in zip(titles, cams_to_show):
                        img = cams.get(phys_name)
                        if img is not None:
                            cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        else:
                            # 物理相机不在 quad 里 → 显示红字 N/A
                            cv2.putText(
                                blank,
                                f"{phys_name}: NOT IN QUAD",
                                (30, args.window_size[1] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2,
                            )
                            cv2.imshow(title, blank)
                    n_ok += 1
                else:
                    n_err += 1
            except Exception as e:
                n_err += 1
                if n_err <= 3 or n_err % 30 == 0:
                    print(f"⚠️  Polling error: {type(e).__name__}: {e}")

            try:
                key = cv2.waitKey(dt_ms) & 0xFF
            except cv2.error as e:
                # X 连接中途断（比如 X server 被 kill）
                print(f"\n❌ X server connection lost: {e}")
                break
            if key in (ord("q"), 27):  # q or ESC
                break

            now = time.monotonic()
            if n_ok > 0 and now - last_stats_print > 3.0:
                elapsed = now - t_start
                print(
                    f"   fps: {n_ok / elapsed:.1f} "
                    f"(ok={n_ok}, err={n_err}, elapsed={elapsed:.1f}s)"
                )
                last_stats_print = now
    except KeyboardInterrupt:
        print("\n⚠️  KeyboardInterrupt — closing windows")
    finally:
        cv2.destroyAllWindows()
        elapsed = time.monotonic() - t_start
        print(
            f"\n✓ Stopped after {n_ok + n_err} polls "
            f"({n_ok} ok, {n_err} errors, {elapsed:.1f}s)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())