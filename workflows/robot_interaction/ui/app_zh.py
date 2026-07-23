"""中文优化版 Gradio UI 主应用

为中文用户优化的机器人控制界面
"""

import os

# gradio 通过 httpx 拉资源时不支持 socks:// 代理 scheme。
# 如果用户 shell 里有 ALL_PROXY=socks://... 或类似设置，
# 会在 import gradio 时直接 ValueError。本 UI 只服务本地 0.0.0.0:7860，
# 不需要任何代理，所以先清掉再 import gradio。
for _proxy_var in (
    "ALL_PROXY", "all_proxy",
    "HTTPS_PROXY", "https_proxy",
    "HTTP_PROXY", "http_proxy",
):
    os.environ.pop(_proxy_var, None)

# 支持 `python workflows/robot_interaction/ui/app_zh.py` 直接调用：
# 把仓库根加进 sys.path，让相对 import 找得到 workflow package。
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import json
import gradio as gr
from pathlib import Path

from .config_manager import (
    UnifiedRobotConfig,
    PolicyConfig,
    RobotConfig,
    InferenceConfig,
    RTCConfig,
    DatasetConfig,
    RuntimeConfig,
    validate,
    save_yaml,
    load_yaml,
    list_presets,
    load_preset,
)
from .process_manager import get_process_manager

# 获取全局进程管理器
pm = get_process_manager()


def _list_train_dirs():
    """列出 outputs/train/ 下的所有训练目录"""
    base_path = Path("outputs/train")
    if not base_path.exists():
        return []

    train_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            train_dirs.append(item.name)

    return sorted(train_dirs, reverse=True)  # 最新的在前


def _list_checkpoints(train_dir_name: str):
    """列出指定训练目录下的所有 checkpoints"""
    if not train_dir_name:
        return []

    train_path = Path("outputs/train") / train_dir_name
    checkpoints_dir = train_path / "checkpoints"

    if not checkpoints_dir.exists():
        # 检查是否有直接的 pretrained_model
        pretrained = train_path / "pretrained_model"
        if pretrained.exists() and (pretrained / "config.json").exists():
            return ["pretrained_model"]
        return []

    checkpoints = []
    for ckpt_dir in checkpoints_dir.iterdir():
        if not ckpt_dir.is_dir():
            continue
        pretrained = ckpt_dir / "pretrained_model"
        if pretrained.exists() and (pretrained / "config.json").exists():
            checkpoints.append(f"checkpoints/{ckpt_dir.name}/pretrained_model")

    # 排序：last 在最前，然后数字从大到小
    def sort_key(ckpt):
        if "last" in ckpt:
            return (0, 0)
        try:
            num = int(ckpt.split("/")[1])
            return (1, -num)
        except (ValueError, IndexError):
            return (2, ckpt)

    return sorted(checkpoints, key=sort_key)


def _list_models():
    """列出 outputs/train/ 下的所有模型（包括 checkpoints 中的）"""
    base_path = Path("outputs/train")
    if not base_path.exists():
        return []

    models = []
    for train_dir in base_path.iterdir():
        if not train_dir.is_dir():
            continue

        # 检查直接在训练目录下的 pretrained_model
        pretrained = train_dir / "pretrained_model"
        if pretrained.exists() and (pretrained / "config.json").exists():
            models.append(str(pretrained))

        # 检查 checkpoints/ 下的所有 checkpoint
        checkpoints_dir = train_dir / "checkpoints"
        if checkpoints_dir.exists():
            for ckpt_dir in sorted(checkpoints_dir.iterdir(), reverse=True):  # 最新的在前
                if not ckpt_dir.is_dir():
                    continue
                pretrained = ckpt_dir / "pretrained_model"
                if pretrained.exists() and (pretrained / "config.json").exists():
                    models.append(str(pretrained))

    return models


def _list_datasets():
    """列出 datasets/ 下的所有数据集"""
    base_path = Path("datasets")
    if not base_path.exists():
        return []

    datasets = []
    for item in base_path.iterdir():
        if item.is_dir():
            datasets.append(str(item))

    return sorted(datasets)


def build_config_from_ui(
    mode: str,
    # Policy
    policy_path: str,
    policy_device: str,
    # Robot
    http_url: str,
    robot_id: str,
    safety_stats_path: str,
    # Inference
    fps: float,
    strategy: str,
    inference_type: str,
    execution_horizon: float,
    max_guidance_weight: float,
    duration: float,
    interpolation_multiplier: float,
    use_torch_compile: bool,
    show_cameras_inf: bool,
    rename_map_json: str,
    # Dataset
    repo_id: str,
    dataset_root: str,
    episode: float,
    single_task: str,
    dataset_fps: float,
    # Camera
    camera_list: list,
    camera_fps: float,
    show_quad: bool,
    window_width: float,
    window_height: float,
    # Runtime
    return_to_home: bool,
    play_sounds: bool,
) -> UnifiedRobotConfig:
    """从 UI 组件值构建 UnifiedRobotConfig"""

    mode_key = mode.lower().replace(" ", "_").replace("部署", "deploy").replace("回放", "replay").replace("相机预览", "camera_preview")

    # 解析 rename_map JSON
    rename_map = {}
    if rename_map_json and rename_map_json.strip():
        try:
            rename_map = json.loads(rename_map_json)
        except json.JSONDecodeError:
            pass  # 将在验证中捕获

    # 构建配置
    config = UnifiedRobotConfig(
        mode=mode_key,
        robot=RobotConfig(
            http_base_url=http_url,
            id=robot_id,  # RobotConfig field is `id` (matches YAML/deploy.py contract)
            safety_stats_path=safety_stats_path if safety_stats_path else None,
        ),
        runtime=RuntimeConfig(
            return_to_initial_position=return_to_home,
            play_sounds=play_sounds,
            camera_list=camera_list if camera_list else ["right_eye", "left_eye", "left_wrist", "right_wrist"],
            camera_fps=camera_fps,
            show_quad=show_quad,
            window_width=int(window_width) if window_width else 640,
            window_height=int(window_height) if window_height else 480,
        ),
    )

    if "deploy" in mode_key or mode == "部署":
        config.policy = PolicyConfig(
            path=policy_path,
            device=policy_device,
        )
        config.inference = InferenceConfig(
            type=inference_type,
            strategy=strategy,
            fps=fps,
            duration=duration,
            interpolation_multiplier=int(interpolation_multiplier),
            use_torch_compile=use_torch_compile,
            show_cameras=show_cameras_inf,
            rename_map=rename_map,
            rtc=RTCConfig(
                execution_horizon=int(execution_horizon) if execution_horizon else None,
                max_guidance_weight=max_guidance_weight,
            ),
        )
        config.dataset = DatasetConfig(
            repo_id=repo_id if repo_id else "",
            root=dataset_root if dataset_root else None,
            single_task=single_task if single_task else None,
        )

    elif "replay" in mode_key or mode == "回放":
        config.dataset = DatasetConfig(
            repo_id=repo_id,
            episode=int(episode) if episode is not None else None,
            root=dataset_root if dataset_root else None,
            fps=dataset_fps,
        )

    elif "camera" in mode_key or mode == "相机预览":
        # show_cameras.py reads policy.path to derive camera list (or honor
        # explicit --cameras). Without this, save_yaml skips the policy block
        # and show_cameras.py fails with "policy.path 是必填项".
        config.policy = PolicyConfig(
            path=policy_path,
            device=policy_device,
        )

    return config


def create_app_zh():
    """创建中文优化的 Gradio 应用"""

    with gr.Blocks(title="🤖 LeRobot 机器人控制中心") as app:
        gr.Markdown(
            """
            # 🤖 LeRobot 统一控制界面

            一站式机器人推理工作流：部署策略、回放数据集、预览相机
            """
        )

        # ====================================================================
        # 顶部控制
        # ====================================================================

        with gr.Row():
            mode = gr.Radio(
                choices=["部署", "回放", "相机预览"],
                value="部署",
                label="操作模式",
                scale=2,
            )

            with gr.Column(scale=1):
                preset_dropdown = gr.Dropdown(
                    label="加载预设配置",
                    choices=list_presets(),
                    value=None,
                    interactive=True,
                )
                save_preset_name = gr.Textbox(
                    label="预设名称",
                    placeholder="my_config",
                    info="输入名称后点击保存预设按钮",
                )
                with gr.Row():
                    save_preset_btn = gr.Button("💾 保存预设", size="sm")
                    export_btn = gr.Button("📥 导出 YAML", size="sm")

                # 导出的 YAML 内容显示区域
                exported_yaml = gr.Textbox(
                    label="导出的配置（可复制）",
                    lines=10,
                    visible=False,
                    interactive=False,
                )

        # ====================================================================
        # 策略设置（仅部署模式）
        # ====================================================================

        with gr.Accordion("策略设置", open=True, visible=True) as policy_panel:
            with gr.Row():
                train_dir_dropdown = gr.Dropdown(
                    label="选择训练目录",
                    choices=_list_train_dirs(),
                    allow_custom_value=False,
                    interactive=True,
                    scale=2,
                )
                refresh_train_dirs_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)

            checkpoint_dropdown = gr.Dropdown(
                label="选择 Checkpoint",
                choices=[],
                allow_custom_value=False,
                interactive=True,
                visible=False,
            )

            policy_path = gr.Textbox(
                label="最终模型路径（或手动输入完整路径）",
                placeholder="outputs/train/act_v2_20260701_181934/checkpoints/190000/pretrained_model",
                interactive=True,
            )

            policy_device = gr.Radio(
                choices=["cuda", "cpu"],
                value="cuda",
                label="推理设备",
                interactive=True,
            )

            # 训练目录选择后，更新 checkpoint 下拉菜单
            def on_train_dir_change(train_dir):
                if not train_dir:
                    return gr.update(choices=[], visible=False), ""
                checkpoints = _list_checkpoints(train_dir)
                if not checkpoints:
                    return gr.update(choices=[], visible=False), f"outputs/train/{train_dir}"
                return gr.update(choices=checkpoints, visible=True, value=checkpoints[0]), ""

            train_dir_dropdown.change(
                fn=on_train_dir_change,
                inputs=[train_dir_dropdown],
                outputs=[checkpoint_dropdown, policy_path],
            )

            # checkpoint 选择后，更新最终路径
            def on_checkpoint_change(train_dir, checkpoint):
                if not train_dir or not checkpoint:
                    return ""
                return f"outputs/train/{train_dir}/{checkpoint}"

            checkpoint_dropdown.change(
                fn=on_checkpoint_change,
                inputs=[train_dir_dropdown, checkpoint_dropdown],
                outputs=[policy_path],
            )

            # 刷新训练目录列表
            refresh_train_dirs_btn.click(
                fn=_list_train_dirs,
                outputs=[train_dir_dropdown],
            )

        # ====================================================================
        # 机器人设置（所有模式）
        # ====================================================================

        with gr.Accordion("机器人设置", open=False) as robot_panel:
            http_url = gr.Textbox(
                label="HTTP 基础 URL",
                value="http://192.168.10.123:8010",
                placeholder="http://192.168.10.123:8010",
                info="机器人 HTTP API 地址",
                interactive=True,
            )
            robot_id = gr.Textbox(
                label="机器人 ID",
                value="marvain_m6_01",
                interactive=True,
            )
            safety_stats_path = gr.Textbox(
                label="安全统计路径（可选）",
                placeholder="datasets/stats/safety_bounds.json",
                info="动作裁剪边界文件",
                interactive=True,
            )

        # ====================================================================
        # 推理设置（仅部署模式）
        # ====================================================================

        with gr.Accordion("推理设置", open=False, visible=True) as inference_panel:
            fps = gr.Slider(
                minimum=1,
                maximum=60,
                value=30,
                step=1,
                label="控制频率 (Hz)",
                interactive=True,
            )
            strategy = gr.Dropdown(
                choices=["base", "sentry", "highlight", "dagger", "episodic"],
                value="base",
                label="录制策略",
                info="base: 仅推理 | sentry: 持续录制 | highlight: 按键保存 | dagger: 人工干预 | episodic: 分段录制",
                interactive=True,
            )
            inference_type = gr.Radio(
                choices=["sync", "rtc", "chunk"],
                value="sync",
                label="推理类型",
                info=(
                    "sync: 完整 chunk 执行 | "
                    "rtc: 实时分块（更快响应） | "
                    "chunk: 开环 chunk（每 chunk_interval_s 推理一次、一次发 n_action_steps）"
                ),
                interactive=True,
            )

            # RTC 特定设置（条件显示）
            with gr.Group(visible=False) as rtc_group:
                gr.Markdown("### Real-Time Chunking 设置")
                execution_horizon = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=4,
                    step=1,
                    label="执行步长",
                    info="每次推理后执行的动作数",
                    interactive=True,
                )
                max_guidance_weight = gr.Slider(
                    minimum=0,
                    maximum=20,
                    value=10.0,
                    step=0.5,
                    label="最大引导权重",
                    interactive=True,
                )

            duration = gr.Slider(
                minimum=0,
                maximum=600,
                value=0,
                step=10,
                label="运行时长（秒）",
                info="0 = 无限",
                interactive=True,
            )

            interpolation_multiplier = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                label="插值倍数",
                interactive=True,
            )

            use_torch_compile = gr.Checkbox(
                label="使用 Torch 编译",
                value=False,
                info="PyTorch 2.0+ 编译加速",
                interactive=True,
            )

            show_cameras_inf = gr.Checkbox(
                label="显示相机窗口",
                value=False,
                interactive=True,
            )

            # 相机重命名映射编辑器
            with gr.Accordion("相机重命名映射（高级）", open=False):
                gr.Markdown("将物理相机名称映射到模型观测键")
                rename_map_json = gr.Textbox(
                    label="重命名映射 (JSON)",
                    placeholder='{"right_eye": "image", "left_wrist": "wrist_image"}',
                    lines=3,
                    info="留空如果不需要",
                    interactive=True,
                )

        # ====================================================================
        # 数据集设置（部署模式可选，回放模式必需）
        # ====================================================================

        with gr.Accordion("数据集设置（VLA 和录制需要）", open=False, visible=False) as deploy_dataset_panel:
            repo_id_deploy = gr.Textbox(
                label="HuggingFace Repo ID",
                placeholder="username/dataset_name",
                info="VLA 模型和非 base 策略需要",
                interactive=True,
            )

            with gr.Row():
                dataset_dropdown_deploy = gr.Dropdown(
                    label="快速选择本地数据集",
                    choices=_list_datasets(),
                    allow_custom_value=True,
                    interactive=True,
                    scale=2,
                )
                refresh_datasets_deploy_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)

            dataset_root_deploy = gr.Textbox(
                label="本地数据集根目录（或手动输入）",
                placeholder="datasets/my_dataset",
                info="本地缓存目录",
                interactive=True,
            )

            single_task = gr.Textbox(
                label="任务描述（仅 VLA）",
                placeholder="拿起红色方块放到蓝色碗里",
                lines=2,
                info="VLA 模型的自然语言任务描述",
                interactive=True,
            )

            # 同步下拉菜单到文本框
            dataset_dropdown_deploy.change(
                fn=lambda x: x if x else "",
                inputs=[dataset_dropdown_deploy],
                outputs=[dataset_root_deploy],
            )

            # 刷新数据集列表
            refresh_datasets_deploy_btn.click(
                fn=_list_datasets,
                outputs=[dataset_dropdown_deploy],
            )

        with gr.Accordion("数据集设置", open=True, visible=False) as replay_dataset_panel:
            repo_id_replay = gr.Textbox(
                label="HuggingFace Repo ID",
                placeholder="username/dataset_name",
                info="必需：要回放的数据集",
                interactive=True,
            )

            episode = gr.Number(
                label="轨迹索引",
                value=0,
                precision=0,
                minimum=0,
                info="要回放哪个轨迹",
                interactive=True,
            )

            with gr.Row():
                dataset_dropdown_replay = gr.Dropdown(
                    label="快速选择本地数据集",
                    choices=_list_datasets(),
                    allow_custom_value=True,
                    interactive=True,
                    scale=2,
                )
                refresh_datasets_replay_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)

            dataset_root_replay = gr.Textbox(
                label="本地数据集根目录（或手动输入）",
                placeholder="datasets/my_dataset",
                info="本地缓存避免重新下载",
                interactive=True,
            )

            # 同步下拉菜单到文本框
            dataset_dropdown_replay.change(
                fn=lambda x: x if x else "",
                inputs=[dataset_dropdown_replay],
                outputs=[dataset_root_replay],
            )

            # 刷新数据集列表
            refresh_datasets_replay_btn.click(
                fn=_list_datasets,
                outputs=[dataset_dropdown_replay],
            )

            dataset_fps = gr.Slider(
                minimum=1,
                maximum=60,
                value=30,
                step=1,
                label="回放频率 (Hz)",
                interactive=True,
            )

        # ====================================================================
        # 相机设置（仅相机预览模式）
        # ====================================================================

        with gr.Accordion("相机设置", open=True, visible=False) as camera_panel:
            camera_list = gr.CheckboxGroup(
                choices=["right_eye", "left_eye", "left_wrist", "right_wrist"],
                value=["right_eye", "left_eye", "left_wrist", "right_wrist"],
                label="要显示的相机",
                info="选择要显示哪些相机",
                interactive=True,
            )

            camera_fps = gr.Slider(
                minimum=1,
                maximum=60,
                value=30,
                step=1,
                label="显示频率 (Hz)",
                interactive=True,
            )

            show_quad = gr.Checkbox(
                label="四宫格布局",
                value=True,
                info="在 2x2 网格中显示所有 4 个相机",
                interactive=True,
            )

            with gr.Row():
                window_width = gr.Number(
                    label="窗口宽度",
                    value=640,
                    precision=0,
                    minimum=320,
                    interactive=True,
                )
                window_height = gr.Number(
                    label="窗口高度",
                    value=480,
                    precision=0,
                    minimum=240,
                    interactive=True,
                )

        # ====================================================================
        # 运行时设置（所有模式）
        # ====================================================================

        with gr.Accordion("运行时设置", open=False) as runtime_panel:
            return_to_home = gr.Checkbox(
                label="执行后返回初始位置",
                value=True,
                info="执行完成后将机器人移至归位姿态",
                interactive=True,
            )
            play_sounds = gr.Checkbox(
                label="播放声音提示",
                value=True,
                info="音频反馈（回放模式）",
                interactive=True,
            )

        # ====================================================================
        # 启动控制
        # ====================================================================

        with gr.Row():
            launch_btn = gr.Button("🚀 启动", variant="primary", size="lg", scale=2)
            stop_btn = gr.Button("⏹️ 停止", variant="stop", size="lg", scale=1, interactive=False)

        status_text = gr.Textbox(
            label="状态",
            value="就绪",
            interactive=False,
            max_lines=3,
        )

        # ====================================================================
        # 进程日志
        # ====================================================================

        with gr.Accordion("进程日志", open=False):
            log_output = gr.Textbox(
                label="",
                lines=20,
                max_lines=50,
                interactive=False,
            )

        # ====================================================================
        # 事件处理器
        # ====================================================================

        def on_mode_change(mode_value):
            """根据模式显示/隐藏面板"""
            is_deploy = mode_value == "部署"
            is_replay = mode_value == "回放"
            is_camera = mode_value == "相机预览"

            return {
                # policy_panel 必须也在 camera_preview 模式下可见：show_cameras.py
                # 强制要求 policy.path（推导相机列表或 fallback），用户必须能填。
                policy_panel: gr.update(visible=is_deploy or is_camera),
                inference_panel: gr.update(visible=is_deploy),
                deploy_dataset_panel: gr.update(visible=is_deploy),
                replay_dataset_panel: gr.update(visible=is_replay),
                camera_panel: gr.update(visible=is_camera),
            }

        def on_inference_type_change(inf_type):
            """仅当类型为 rtc 时显示 RTC 参数"""
            return gr.update(visible=(inf_type == "rtc"))

        def on_strategy_change(strat):
            """当策略需要录制时显示数据集面板"""
            needs_dataset = strat != "base"
            return gr.update(visible=needs_dataset)

        def on_load_preset(preset_name):
            """加载预设配置并更新 UI"""
            if not preset_name:
                return {}

            try:
                config = load_preset(preset_name)
                updates = {}

                # 策略设置
                if config.policy:
                    updates[policy_path] = gr.update(value=config.policy.path)
                    updates[policy_device] = gr.update(value=config.policy.device)

                # 机器人设置
                updates[http_url] = gr.update(value=config.robot.http_base_url)
                updates[robot_id] = gr.update(value=config.robot.id)

                # 推理设置
                if config.inference:
                    updates[fps] = gr.update(value=config.inference.fps)
                    updates[strategy] = gr.update(value=config.inference.strategy)
                    updates[inference_type] = gr.update(value=config.inference.type)

                    if config.inference.rtc and config.inference.rtc.execution_horizon:
                        updates[execution_horizon] = gr.update(value=config.inference.rtc.execution_horizon)

                    if config.inference.rename_map:
                        updates[rename_map_json] = gr.update(value=json.dumps(config.inference.rename_map, indent=2))

                # 数据集设置
                if config.dataset:
                    if config.dataset.repo_id:
                        updates[repo_id_deploy] = gr.update(value=config.dataset.repo_id)
                        updates[repo_id_replay] = gr.update(value=config.dataset.repo_id)
                    if config.dataset.root:
                        updates[dataset_root_deploy] = gr.update(value=config.dataset.root)
                        updates[dataset_root_replay] = gr.update(value=config.dataset.root)
                    if config.dataset.single_task:
                        updates[single_task] = gr.update(value=config.dataset.single_task)
                    if config.dataset.episode is not None:
                        updates[episode] = gr.update(value=config.dataset.episode)

                # 运行时设置
                updates[return_to_home] = gr.update(value=config.runtime.return_to_initial_position)
                updates[play_sounds] = gr.update(value=config.runtime.play_sounds)

                # 相机设置
                if config.runtime.camera_list:
                    updates[camera_list] = gr.update(value=config.runtime.camera_list)
                updates[camera_fps] = gr.update(value=config.runtime.camera_fps)
                updates[show_quad] = gr.update(value=config.runtime.show_quad)

                # 状态消息
                updates[status_text] = gr.update(value=f"✅ 已加载预设: {preset_name}")

                return updates

            except Exception as e:
                return {status_text: gr.update(value=f"❌ 加载预设失败: {e}")}

        def on_save_preset(
            preset_name,
            mode, policy_path, policy_device, http_url, robot_id, safety_stats_path,
            fps, strategy, inference_type, execution_horizon, max_guidance_weight,
            duration, interpolation_multiplier, use_torch_compile, show_cameras_inf,
            rename_map_json, repo_id_deploy, dataset_root_deploy, episode, single_task,
            dataset_fps, repo_id_replay, dataset_root_replay,
            camera_list, camera_fps, show_quad, window_width, window_height,
            return_to_home, play_sounds
        ):
            """保存当前配置为预设"""
            if not preset_name or not preset_name.strip():
                return gr.update(value="❌ 请输入预设名称")

            try:
                # 合并 repo_id
                repo_id = repo_id_deploy if mode == "部署" else repo_id_replay
                dataset_root = dataset_root_deploy if mode == "部署" else dataset_root_replay

                # 从 UI 构建配置
                config = build_config_from_ui(
                    mode, policy_path, policy_device, http_url, robot_id, safety_stats_path,
                    fps, strategy, inference_type, execution_horizon, max_guidance_weight,
                    duration, interpolation_multiplier, use_torch_compile, show_cameras_inf,
                    rename_map_json, repo_id, dataset_root, episode, single_task, dataset_fps,
                    camera_list, camera_fps, show_quad, window_width, window_height,
                    return_to_home, play_sounds
                )

                # 保存到文件
                preset_dir = Path(__file__).parent / "presets"
                preset_dir.mkdir(exist_ok=True)
                preset_file = preset_dir / f"{preset_name}.yaml"

                save_yaml(config, str(preset_file))

                return gr.update(value=f"✅ 预设已保存: {preset_file}")

            except Exception as e:
                return gr.update(value=f"❌ 保存失败: {e}")

        def on_export_yaml(
            mode, policy_path, policy_device, http_url, robot_id, safety_stats_path,
            fps, strategy, inference_type, execution_horizon, max_guidance_weight,
            duration, interpolation_multiplier, use_torch_compile, show_cameras_inf,
            rename_map_json, repo_id_deploy, dataset_root_deploy, episode, single_task,
            dataset_fps, repo_id_replay, dataset_root_replay,
            camera_list, camera_fps, show_quad, window_width, window_height,
            return_to_home, play_sounds
        ):
            """导出当前配置为 YAML 文件"""
            try:
                # 合并 repo_id
                repo_id = repo_id_deploy if mode == "部署" else repo_id_replay
                dataset_root = dataset_root_deploy if mode == "部署" else dataset_root_replay

                # 从 UI 构建配置
                config = build_config_from_ui(
                    mode, policy_path, policy_device, http_url, robot_id, safety_stats_path,
                    fps, strategy, inference_type, execution_horizon, max_guidance_weight,
                    duration, interpolation_multiplier, use_torch_compile, show_cameras_inf,
                    rename_map_json, repo_id, dataset_root, episode, single_task, dataset_fps,
                    camera_list, camera_fps, show_quad, window_width, window_height,
                    return_to_home, play_sounds
                )

                # 生成临时文件
                import tempfile
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                temp_file = tempfile.gettempdir() + f"/robot_config_{timestamp}.yaml"

                save_yaml(config, temp_file)

                # 返回文件内容供下载
                with open(temp_file, 'r') as f:
                    yaml_content = f.read()

                return yaml_content, gr.update(value=f"✅ 配置已导出到: {temp_file}")

            except Exception as e:
                return "", gr.update(value=f"❌ 导出失败: {e}")

        def on_launch_click(
            mode, policy_path, policy_device, http_url, robot_id, safety_stats_path,
            fps, strategy, inference_type, execution_horizon, max_guidance_weight,
            duration, interpolation_multiplier, use_torch_compile, show_cameras_inf,
            rename_map_json, repo_id_deploy, dataset_root_deploy, episode, single_task,
            dataset_fps, repo_id_replay, dataset_root_replay,
            camera_list, camera_fps, show_quad, window_width, window_height,
            return_to_home, play_sounds
        ):
            """验证配置并启动进程"""
            try:
                # 合并 repo_id
                repo_id = repo_id_deploy if mode == "部署" else repo_id_replay
                dataset_root = dataset_root_deploy if mode == "部署" else dataset_root_replay

                # 从 UI 构建配置
                config = build_config_from_ui(
                    mode, policy_path, policy_device, http_url, robot_id, safety_stats_path,
                    fps, strategy, inference_type, execution_horizon, max_guidance_weight,
                    duration, interpolation_multiplier, use_torch_compile, show_cameras_inf,
                    rename_map_json, repo_id, dataset_root, episode, single_task, dataset_fps,
                    camera_list, camera_fps, show_quad, window_width, window_height,
                    return_to_home, play_sounds
                )

                # 验证
                errors = validate(config)
                if errors:
                    error_msg = "❌ 配置错误：\n" + "\n".join(f"  • {e}" for e in errors)
                    return {
                        status_text: gr.update(value=error_msg),
                        launch_btn: gr.update(interactive=True),
                        stop_btn: gr.update(interactive=False),
                    }

                # 启动相应进程
                if mode == "部署":
                    success, msg = pm.launch_deploy(config)
                elif mode == "回放":
                    success, msg = pm.launch_replay(config)
                elif mode == "相机预览":
                    success, msg = pm.launch_camera_preview(config)
                else:
                    success, msg = False, f"未知模式: {mode}"

                return {
                    status_text: gr.update(value=msg),
                    launch_btn: gr.update(interactive=not success),
                    stop_btn: gr.update(interactive=success),
                }

            except Exception as e:
                return {
                    status_text: gr.update(value=f"❌ 错误: {e}"),
                    launch_btn: gr.update(interactive=True),
                    stop_btn: gr.update(interactive=False),
                }

        def on_stop_click():
            """停止所有运行中的进程"""
            pm.stop_all()
            return {
                status_text: gr.update(value="⏹️ 已停止"),
                launch_btn: gr.update(interactive=True),
                stop_btn: gr.update(interactive=False),
            }

        def refresh_logs():
            """刷新日志输出"""
            return pm.get_logs(last_n_lines=100)

        # 连接事件
        mode.change(
            on_mode_change,
            inputs=[mode],
            outputs=[
                policy_panel,
                inference_panel,
                deploy_dataset_panel,
                replay_dataset_panel,
                camera_panel,
            ],
        )

        inference_type.change(
            on_inference_type_change,
            inputs=[inference_type],
            outputs=[rtc_group],
        )

        strategy.change(
            on_strategy_change,
            inputs=[strategy],
            outputs=[deploy_dataset_panel],
        )

        # 预设加载
        preset_dropdown.change(
            on_load_preset,
            inputs=[preset_dropdown],
            outputs=[
                policy_path,
                policy_device,
                http_url,
                robot_id,
                fps,
                strategy,
                inference_type,
                execution_horizon,
                rename_map_json,
                repo_id_deploy,
                repo_id_replay,
                dataset_root_deploy,
                dataset_root_replay,
                single_task,
                episode,
                return_to_home,
                play_sounds,
                camera_list,
                camera_fps,
                show_quad,
                status_text,
            ],
        )

        # 收集所有输入用于启动
        launch_inputs = [
            mode,
            policy_path,
            policy_device,
            http_url,
            robot_id,
            safety_stats_path,
            fps,
            strategy,
            inference_type,
            execution_horizon,
            max_guidance_weight,
            duration,
            interpolation_multiplier,
            use_torch_compile,
            show_cameras_inf,
            rename_map_json,
            repo_id_deploy,
            dataset_root_deploy,
            episode,
            single_task,
            dataset_fps,
            repo_id_replay,
            dataset_root_replay,
            camera_list,
            camera_fps,
            show_quad,
            window_width,
            window_height,
            return_to_home,
            play_sounds,
        ]

        launch_btn.click(
            on_launch_click,
            inputs=launch_inputs,
            outputs=[status_text, launch_btn, stop_btn],
        )

        stop_btn.click(
            on_stop_click,
            outputs=[status_text, launch_btn, stop_btn],
        )

        # 保存预设按钮
        save_preset_btn.click(
            on_save_preset,
            inputs=[save_preset_name] + launch_inputs,
            outputs=[status_text],
        )

        # 导出 YAML 按钮
        export_btn.click(
            on_export_yaml,
            inputs=launch_inputs,
            outputs=[exported_yaml, status_text],
        ).then(
            lambda: gr.update(visible=True),
            outputs=[exported_yaml],
        )

        # 每 2 秒自动刷新日志
        timer = gr.Timer(value=2, active=True)
        timer.tick(
            fn=refresh_logs,
            outputs=[log_output],
        )

    return app


def main_zh():
    """中文版主入口"""
    app = create_app_zh()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="orange",
            secondary_hue="stone",
        ),
    )


if __name__ == "__main__":
    main_zh()
