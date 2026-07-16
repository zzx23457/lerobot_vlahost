"""中文 Gradio UI 主应用（yaml-centric 版）

UI 是 deploy_config*.yaml / replay_config.yaml 的友好编辑器 + 启动器：
- YAML 是单一真相源
- 表单只是 YAML 的可选结构化视图
- 启动 = 写 yaml 到 tempfile → 脚本 `--config <tmp>`
"""

import json
import tempfile
import time
from pathlib import Path

import gradio as gr

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
    list_templates,
    load_template,
)
from .process_manager import get_process_manager

# 全局进程管理器
pm = get_process_manager()


# ============================================================================
# 辅助函数（文件 / 数据集扫描）
# ============================================================================

def _list_train_dirs():
    """列出 outputs/train/ 下的所有训练目录"""
    base_path = Path("outputs/train")
    if not base_path.exists():
        return []
    return sorted(
        (item.name for item in base_path.iterdir() if item.is_dir()),
        reverse=True,
    )


def _list_checkpoints(train_dir_name: str):
    """列出指定训练目录下的所有 checkpoints"""
    if not train_dir_name:
        return []
    train_path = Path("outputs/train") / train_dir_name
    checkpoints_dir = train_path / "checkpoints"
    if not checkpoints_dir.exists():
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

    def sort_key(ckpt):
        if "last" in ckpt:
            return (0, 0)
        try:
            num = int(ckpt.split("/")[1])
            return (1, -num)
        except (ValueError, IndexError):
            return (2, ckpt)

    return sorted(checkpoints, key=sort_key)


def _list_datasets():
    """列出 datasets/ 下的所有数据集"""
    base_path = Path("datasets")
    if not base_path.exists():
        return []
    return sorted(str(item) for item in base_path.iterdir() if item.is_dir())


def _parse_yaml_text(yaml_text: str) -> UnifiedRobotConfig:
    """解析 yaml 文本 → UnifiedRobotConfig（用 tempfile 走 load_yaml）"""
    ts = int(time.time() * 1000)
    tmp = Path(tempfile.gettempdir()) / f"_yaml_parse_{ts}.yaml"
    tmp.write_text(yaml_text, encoding="utf-8")
    try:
        return load_yaml(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def _config_to_form_updates(config: UnifiedRobotConfig) -> dict:
    """把 config 映射成 {gr.Component: gr.update(value=...)} 字典。
    用于 on_template_change / on_yaml_to_form 两个 handler 共用。"""
    updates: dict = {}

    # Mode
    mode_value = {
        "deploy": "部署",
        "replay": "回放",
        "camera_preview": "相机预览",
    }.get(config.mode, "部署")
    updates["mode_ref"] = gr.update(value=mode_value)

    # Policy
    updates["policy_path"] = gr.update(value=config.policy.path)
    updates["policy_device"] = gr.update(value=config.policy.device)

    # Robot
    updates["http_url"] = gr.update(value=config.robot.http_base_url)
    updates["robot_id"] = gr.update(value=config.robot.robot_id)
    updates["safety_stats_path"] = gr.update(value=config.robot.safety_stats_path or "")

    # Inference
    updates["fps"] = gr.update(value=config.inference.fps)
    updates["strategy"] = gr.update(value=config.inference.strategy)
    updates["inference_type"] = gr.update(value=config.inference.type)
    updates["duration"] = gr.update(value=config.inference.duration)
    updates["interpolation_multiplier"] = gr.update(value=config.inference.interpolation_multiplier)
    updates["use_torch_compile"] = gr.update(value=config.inference.use_torch_compile)
    updates["show_cameras_inf"] = gr.update(value=config.inference.show_cameras)

    if config.inference.rtc:
        updates["execution_horizon"] = gr.update(
            value=config.inference.rtc.execution_horizon or 0
        )
        updates["max_guidance_weight"] = gr.update(value=config.inference.rtc.max_guidance_weight)

    updates["n_action_steps"] = gr.update(value=config.inference.n_action_steps or 20)
    updates["chunk_interval_s"] = gr.update(value=config.inference.chunk_interval_s or 2.0)

    updates["rename_map_json"] = gr.update(
        value=json.dumps(config.inference.rename_map, indent=2) if config.inference.rename_map else ""
    )

    # Dataset
    updates["repo_id_deploy"] = gr.update(value=config.dataset.repo_id)
    updates["dataset_root_deploy"] = gr.update(value=config.dataset.root or "")
    updates["single_task"] = gr.update(value=config.dataset.single_task or "")
    updates["repo_id_replay"] = gr.update(value=config.dataset.repo_id)
    updates["dataset_root_replay"] = gr.update(value=config.dataset.root or "")
    updates["episode"] = gr.update(value=config.dataset.episode if config.dataset.episode is not None else 0)
    updates["dataset_fps"] = gr.update(value=config.dataset.fps)

    # Runtime
    updates["return_to_home"] = gr.update(value=config.runtime.return_to_initial_position)
    updates["play_sounds"] = gr.update(value=config.runtime.play_sounds)

    # Camera
    updates["camera_list"] = gr.update(value=config.runtime.camera_list)
    updates["camera_fps"] = gr.update(value=config.runtime.camera_fps)
    updates["show_quad"] = gr.update(value=config.runtime.show_quad)
    updates["window_width"] = gr.update(value=config.runtime.window_width)
    updates["window_height"] = gr.update(value=config.runtime.window_height)

    return updates


# ============================================================================
# 从表单构建配置（form → UnifiedRobotConfig）
# ============================================================================

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
    n_action_steps: float,
    chunk_interval_s: float,
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

    mode_key = (
        mode.lower()
        .replace(" ", "_")
        .replace("部署", "deploy")
        .replace("回放", "replay")
        .replace("相机预览", "camera_preview")
    )

    rename_map: dict = {}
    if rename_map_json and rename_map_json.strip():
        try:
            rename_map = json.loads(rename_map_json)
        except json.JSONDecodeError:
            pass

    config = UnifiedRobotConfig(
        mode=mode_key,
        robot=RobotConfig(
            http_base_url=http_url,
            robot_id=robot_id,
            safety_stats_path=safety_stats_path if safety_stats_path else None,
        ),
        runtime=RuntimeConfig(
            return_to_initial_position=return_to_home,
            play_sounds=play_sounds,
            camera_list=camera_list if camera_list else [
                "right_eye", "left_eye", "left_wrist", "right_wrist"
            ],
            camera_fps=camera_fps,
            show_quad=show_quad,
            window_width=int(window_width) if window_width else 640,
            window_height=int(window_height) if window_height else 480,
        ),
    )

    if "deploy" in mode_key or mode == "部署":
        config.policy = PolicyConfig(path=policy_path, device=policy_device)
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
            n_action_steps=int(n_action_steps) if n_action_steps else None,
            chunk_interval_s=float(chunk_interval_s) if chunk_interval_s else None,
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

    return config


# ============================================================================
# Gradio 应用
# ============================================================================

def create_app_zh():
    """创建 yaml-centric 中文 Gradio 应用"""

    with gr.Blocks(title="🤖 LeRobot 机器人控制中心") as app:
        gr.Markdown(
            """
            # 🤖 LeRobot 统一控制界面（YAML-Centric）

            一站式机器人推理工作流：部署策略、回放数据集、预览相机

            **使用方式**：
            1. 顶部"加载模板"选择一个 yaml（deploy_config_chunk / replay_config 等）
            2. 在 Tab "YAML" 直接编辑，或在 Tab "表单" 通过结构化视图编辑并点"刷新 YAML"
            3. 点 🚀 启动 — UI 把当前 YAML 写到 tempfile，调用对应脚本的 `--config`
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
                template_dropdown = gr.Dropdown(
                    label="加载模板配置",
                    choices=list_templates(),
                    value=None,
                    interactive=True,
                    info=(
                        "deploy_config_chunk / deploy_config / deploy_config_hybrid / "
                        "replay_config 等"
                    ),
                )
                save_template_name = gr.Textbox(
                    label="新模板名称",
                    placeholder="my_deploy_config",
                    info="输入后点'保存为模板'会写到 workflows/robot_interaction/",
                )
                with gr.Row():
                    save_template_btn = gr.Button("💾 保存为模板", size="sm")
                    refresh_templates_btn = gr.Button("🔄", size="sm", min_width=50)

        # ====================================================================
        # 中部 Tabs：表单 / YAML
        # ====================================================================

        with gr.Tabs():
            # ----------------------------
            # Tab "表单"
            # ----------------------------
            with gr.Tab("📝 表单"):
                # ----- 策略设置 -----
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
                        label="最终模型路径",
                        placeholder="outputs/train/act_v2_20260701_181934/checkpoints/190000/pretrained_model",
                        interactive=True,
                    )

                    policy_device = gr.Radio(
                        choices=["cuda", "cpu"],
                        value="cuda",
                        label="推理设备",
                        interactive=True,
                    )

                    def on_train_dir_change(train_dir):
                        if not train_dir:
                            return gr.update(choices=[], visible=False), ""
                        checkpoints = _list_checkpoints(train_dir)
                        if not checkpoints:
                            return gr.update(choices=[], visible=False), f"outputs/train/{train_dir}"
                        return (
                            gr.update(choices=checkpoints, visible=True, value=checkpoints[0]),
                            "",
                        )

                    train_dir_dropdown.change(
                        fn=on_train_dir_change,
                        inputs=[train_dir_dropdown],
                        outputs=[checkpoint_dropdown, policy_path],
                    )

                    def on_checkpoint_change(train_dir, checkpoint):
                        if not train_dir or not checkpoint:
                            return ""
                        return f"outputs/train/{train_dir}/{checkpoint}"

                    checkpoint_dropdown.change(
                        fn=on_checkpoint_change,
                        inputs=[train_dir_dropdown, checkpoint_dropdown],
                        outputs=[policy_path],
                    )

                    refresh_train_dirs_btn.click(
                        fn=_list_train_dirs,
                        outputs=[train_dir_dropdown],
                    )

                # ----- 机器人设置 -----
                with gr.Accordion("机器人设置", open=False) as robot_panel:
                    http_url = gr.Textbox(
                        label="HTTP 基础 URL",
                        value="http://192.168.10.123:8010",
                        placeholder="http://192.168.10.123:8010",
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
                        interactive=True,
                    )

                # ----- 推理设置 -----
                with gr.Accordion("推理设置", open=False, visible=True) as inference_panel:
                    fps = gr.Slider(1, 60, 30, step=1, label="控制频率 (Hz)")
                    strategy = gr.Dropdown(
                        choices=["base", "sentry", "highlight", "dagger", "episodic"],
                        value="base",
                        label="录制策略",
                        info="base: 仅推理 | sentry: 持续录制 | highlight: 按键保存 | dagger: 人工干预 | episodic: 分段录制",
                    )
                    inference_type = gr.Radio(
                        choices=["sync", "rtc", "chunk"],
                        value="sync",
                        label="推理类型",
                        info="sync: 完整 chunk 执行 | rtc: 实时分块（更快响应） | chunk: 开环分段下发（最稳）",
                    )

                    with gr.Group(visible=False) as rtc_group:
                        gr.Markdown("### Real-Time Chunking 设置")
                        execution_horizon = gr.Slider(1, 100, 4, step=1, label="执行步长")
                        max_guidance_weight = gr.Slider(0, 20, 10.0, step=0.5, label="最大引导权重")

                    with gr.Group(visible=False) as chunk_group:
                        gr.Markdown("### 开环 Chunk 模式设置")
                        gr.Markdown(
                            "每 `chunk_interval_s` 推理一次，把前 `n_action_steps` 个动作"
                            "一次性 POST 给机器人。等价于 deploy_config_chunk.yaml。"
                        )
                        n_action_steps = gr.Slider(1, 100, 20, step=1, label="单次下发动作数")
                        chunk_interval_s = gr.Slider(0.1, 10.0, 2.0, step=0.1, label="推理间隔（秒）")

                    duration = gr.Slider(0, 600, 0, step=10, label="运行时长（秒）", info="0 = 无限")
                    interpolation_multiplier = gr.Slider(1, 10, 1, step=1, label="插值倍数")
                    use_torch_compile = gr.Checkbox(label="使用 Torch 编译", value=False)
                    show_cameras_inf = gr.Checkbox(label="显示相机窗口", value=False)

                    with gr.Accordion("相机重命名映射（高级）", open=False):
                        rename_map_json = gr.Textbox(
                            label="重命名映射 (JSON)",
                            placeholder='{"right_eye": "image", "left_wrist": "wrist_image"}',
                            lines=3,
                            info="留空如果不需要",
                        )

                # ----- 数据集设置（部署） -----
                with gr.Accordion("数据集设置（VLA 和录制需要）", open=False, visible=False) as deploy_dataset_panel:
                    repo_id_deploy = gr.Textbox(
                        label="HuggingFace Repo ID",
                        placeholder="username/dataset_name",
                        info="VLA 模型和非 base 策略需要",
                    )
                    with gr.Row():
                        dataset_dropdown_deploy = gr.Dropdown(
                            label="快速选择本地数据集",
                            choices=_list_datasets(),
                            allow_custom_value=True,
                            scale=2,
                        )
                        refresh_datasets_deploy_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)
                    dataset_root_deploy = gr.Textbox(
                        label="本地数据集根目录（或手动输入）",
                        placeholder="datasets/my_dataset",
                    )
                    single_task = gr.Textbox(
                        label="任务描述（仅 VLA）",
                        placeholder="拿起红色方块放到蓝色碗里",
                        lines=2,
                    )

                    dataset_dropdown_deploy.change(
                        fn=lambda x: x if x else "",
                        inputs=[dataset_dropdown_deploy],
                        outputs=[dataset_root_deploy],
                    )
                    refresh_datasets_deploy_btn.click(
                        fn=_list_datasets,
                        outputs=[dataset_dropdown_deploy],
                    )

                # ----- 数据集设置（回放） -----
                with gr.Accordion("数据集设置", open=True, visible=False) as replay_dataset_panel:
                    repo_id_replay = gr.Textbox(
                        label="HuggingFace Repo ID",
                        placeholder="username/dataset_name",
                        info="必需：要回放的数据集",
                    )
                    episode = gr.Number(
                        label="轨迹索引", value=0, precision=0, minimum=0,
                    )
                    with gr.Row():
                        dataset_dropdown_replay = gr.Dropdown(
                            label="快速选择本地数据集",
                            choices=_list_datasets(),
                            allow_custom_value=True,
                            scale=2,
                        )
                        refresh_datasets_replay_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)
                    dataset_root_replay = gr.Textbox(
                        label="本地数据集根目录（或手动输入）",
                        placeholder="datasets/my_dataset",
                    )
                    dataset_dropdown_replay.change(
                        fn=lambda x: x if x else "",
                        inputs=[dataset_dropdown_replay],
                        outputs=[dataset_root_replay],
                    )
                    refresh_datasets_replay_btn.click(
                        fn=_list_datasets,
                        outputs=[dataset_dropdown_replay],
                    )
                    dataset_fps = gr.Slider(1, 60, 30, step=1, label="回放频率 (Hz)")

                # ----- 相机设置 -----
                with gr.Accordion("相机设置", open=True, visible=False) as camera_panel:
                    camera_list = gr.CheckboxGroup(
                        choices=["right_eye", "left_eye", "left_wrist", "right_wrist"],
                        value=["right_eye", "left_eye", "left_wrist", "right_wrist"],
                        label="要显示的相机",
                    )
                    camera_fps = gr.Slider(1, 60, 30, step=1, label="显示频率 (Hz)")
                    show_quad = gr.Checkbox(label="四宫格布局", value=True)
                    with gr.Row():
                        window_width = gr.Number(label="窗口宽度", value=640, precision=0, minimum=320)
                        window_height = gr.Number(label="窗口高度", value=480, precision=0, minimum=240)

                # ----- 运行时设置 -----
                with gr.Accordion("运行时设置", open=False) as runtime_panel:
                    return_to_home = gr.Checkbox(
                        label="执行后返回初始位置",
                        value=True,
                        info="回放模式正常生效；部署模式因 deploy.py argparse bug 当前无效",
                    )
                    play_sounds = gr.Checkbox(
                        label="播放声音提示（仅回放）",
                        value=True,
                    )

            # ----------------------------
            # Tab "YAML"
            # ----------------------------
            with gr.Tab("📄 YAML"):
                yaml_editor = gr.Code(
                    label="YAML 配置（真相源）",
                    language="yaml",
                    lines=30,
                    interactive=True,
                )
                with gr.Row():
                    refresh_yaml_btn = gr.Button("🔄 从表单刷新 → YAML", size="sm")
                    apply_yaml_btn = gr.Button("📋 YAML → 应用到表单", size="sm")

                gr.Markdown(
                    """
                    **直接编辑**这个 YAML 即可。点启动时，UI 会把这段 YAML 写到
                    `tempfile/robot_config_*.yaml`，然后以 `--config <tmp>` 调用
                    对应的脚本。

                    - 部署模式脚本：[`deploy.py`](../../robot_interaction/deploy.py)
                    - 回放模式脚本：[`replay.py`](../../robot_interaction/replay.py)
                    - 相机预览脚本：[`show_cameras.py`](../../robot_interaction/show_cameras.py)

                    YAML schema 与这三个脚本直接吃的格式一致（除相机运行参数外），
                    所以你也能从外部编辑器改 `deploy_config_*.yaml` 然后回到 UI 点启动。
                    """
                )

        # ====================================================================
        # 底部：启动 + 状态 + 日志
        # ====================================================================

        with gr.Row():
            launch_btn = gr.Button("🚀 启动", variant="primary", size="lg", scale=2)
            stop_btn = gr.Button("⏹️ 停止", variant="stop", size="lg", scale=1, interactive=False)

        status_text = gr.Textbox(label="状态", value="就绪", interactive=False, max_lines=4)

        with gr.Accordion("进程日志", open=False):
            log_output = gr.Textbox(label="", lines=20, max_lines=50, interactive=False)

        # ====================================================================
        # 事件处理器
        # ====================================================================

        # 收集所有表单字段作为 inputs（form ↔ yaml 同步用）
        form_inputs = [
            mode, policy_path, policy_device, http_url, robot_id, safety_stats_path,
            fps, strategy, inference_type, execution_horizon, max_guidance_weight,
            n_action_steps, chunk_interval_s,
            duration, interpolation_multiplier, use_torch_compile, show_cameras_inf,
            rename_map_json,
            repo_id_deploy, dataset_root_deploy, episode, single_task, dataset_fps,
            repo_id_replay, dataset_root_replay,
            camera_list, camera_fps, show_quad, window_width, window_height,
            return_to_home, play_sounds,
        ]

        form_outputs_for_yaml_to_form = [
            mode, policy_path, policy_device, http_url, robot_id, safety_stats_path,
            fps, strategy, inference_type, execution_horizon, max_guidance_weight,
            n_action_steps, chunk_interval_s,
            duration, interpolation_multiplier, use_torch_compile, show_cameras_inf,
            rename_map_json,
            repo_id_deploy, dataset_root_deploy, episode, single_task, dataset_fps,
            repo_id_replay, dataset_root_replay,
            camera_list, camera_fps, show_quad, window_width, window_height,
            return_to_home, play_sounds,
            status_text,
        ]

        def on_template_change(name):
            """加载模板 yaml 到 yaml_editor，并同步填充表单字段。"""
            if not name:
                return {}
            try:
                path = Path(__file__).parent.parent / f"{name}.yaml"
                yaml_text = path.read_text(encoding="utf-8")
                config = load_template(name)
                updates = _config_to_form_updates(config)
                updates[yaml_editor] = gr.update(value=yaml_text)
                updates[status_text] = gr.update(value=f"✅ 已加载模板: {name}")
                return updates
            except Exception as e:
                return {status_text: gr.update(value=f"❌ 加载模板失败: {e}")}

        def on_form_to_yaml(*form_values):
            """表单 → UnifiedRobotConfig → save_yaml → 灌到 yaml_editor。"""
            try:
                config = build_config_from_ui(*form_values)
                ts = int(time.time() * 1000)
                tmp = Path(tempfile.gettempdir()) / f"_form_to_yaml_{ts}.yaml"
                save_yaml(config, tmp)
                yaml_text = tmp.read_text(encoding="utf-8")
                tmp.unlink(missing_ok=True)
                return gr.update(value=yaml_text), gr.update(value="✅ 已从表单刷新 YAML")
            except Exception as e:
                return (
                    gr.update(value=f"# ❌ 刷新失败: {e}"),
                    gr.update(value=f"❌ 刷新 YAML 失败: {e}"),
                )

        def on_yaml_to_form(yaml_text):
            """YAML → UnifiedRobotConfig → 逐字段灌到表单。"""
            try:
                config = _parse_yaml_text(yaml_text or "")
                updates = _config_to_form_updates(config)
                updates[status_text] = gr.update(value="✅ 已应用 YAML 到表单")
                return updates
            except Exception as e:
                return {status_text: gr.update(value=f"❌ YAML 解析失败: {e}")}

        def on_save_template(yaml_text, name):
            """把当前 yaml_editor 文本写到 workflows/robot_interaction/<name>.yaml。"""
            if not name or not name.strip():
                return gr.update(value="❌ 请输入模板名称")
            if not yaml_text or not yaml_text.strip():
                return gr.update(value="❌ yaml_editor 为空")
            try:
                # 先 parse 一下，确保 yaml 合法
                _parse_yaml_text(yaml_text)
                target = Path(__file__).parent.parent / f"{name}.yaml"
                if target.exists():
                    return gr.update(value=f"❌ 文件已存在: {target}")
                target.write_text(yaml_text, encoding="utf-8")
                return gr.update(value=f"✅ 模板已保存: {target}")
            except Exception as e:
                return gr.update(value=f"❌ 保存失败: {e}")

        def on_launch_click(yaml_text):
            """从 yaml_editor 解析 → validate → 启动对应进程。"""
            try:
                config = _parse_yaml_text(yaml_text or "")
            except Exception as e:
                return {
                    status_text: gr.update(value=f"❌ YAML 解析失败: {e}"),
                    launch_btn: gr.update(interactive=True),
                    stop_btn: gr.update(interactive=False),
                }

            errors = validate(config)
            if errors:
                error_msg = "❌ 配置错误：\n" + "\n".join(f"  • {e}" for e in errors)
                return {
                    status_text: gr.update(value=error_msg),
                    launch_btn: gr.update(interactive=True),
                    stop_btn: gr.update(interactive=False),
                }

            mode_key = config.mode
            if mode_key == "deploy":
                success, msg = pm.launch_deploy(config)
            elif mode_key == "replay":
                success, msg = pm.launch_replay(config)
            elif mode_key == "camera_preview":
                success, msg = pm.launch_camera_preview(config)
            else:
                success, msg = False, f"未知 mode: {mode_key}"

            return {
                status_text: gr.update(value=msg),
                launch_btn: gr.update(interactive=not success),
                stop_btn: gr.update(interactive=success),
            }

        def on_stop_click():
            pm.stop_all()
            return {
                status_text: gr.update(value="⏹️ 已停止"),
                launch_btn: gr.update(interactive=True),
                stop_btn: gr.update(interactive=False),
            }

        def refresh_logs():
            return pm.get_logs(last_n_lines=100)

        def on_mode_change(mode_value):
            is_deploy = mode_value == "部署"
            is_replay = mode_value == "回放"
            is_camera = mode_value == "相机预览"
            return {
                policy_panel: gr.update(visible=is_deploy),
                inference_panel: gr.update(visible=is_deploy),
                deploy_dataset_panel: gr.update(visible=is_deploy),
                replay_dataset_panel: gr.update(visible=is_replay),
                camera_panel: gr.update(visible=is_camera),
            }

        def on_inference_type_change(inf_type):
            return (
                gr.update(visible=(inf_type == "rtc")),
                gr.update(visible=(inf_type == "chunk")),
            )

        def on_strategy_change(strat):
            return gr.update(visible=(strat != "base"))

        def on_refresh_templates():
            return gr.update(choices=list_templates())

        # ====================================================================
        # 事件绑定
        # ====================================================================

        mode.change(
            on_mode_change,
            inputs=[mode],
            outputs=[policy_panel, inference_panel, deploy_dataset_panel, replay_dataset_panel, camera_panel],
        )
        inference_type.change(
            on_inference_type_change,
            inputs=[inference_type],
            outputs=[rtc_group, chunk_group],
        )
        strategy.change(
            on_strategy_change,
            inputs=[strategy],
            outputs=[deploy_dataset_panel],
        )

        template_dropdown.change(
            on_template_change,
            inputs=[template_dropdown],
            outputs=[yaml_editor] + form_outputs_for_yaml_to_form,
        )

        refresh_templates_btn.click(
            on_refresh_templates,
            outputs=[template_dropdown],
        )

        refresh_yaml_btn.click(
            on_form_to_yaml,
            inputs=form_inputs,
            outputs=[yaml_editor, status_text],
        )

        apply_yaml_btn.click(
            on_yaml_to_form,
            inputs=[yaml_editor],
            outputs=form_outputs_for_yaml_to_form,
        )

        save_template_btn.click(
            on_save_template,
            inputs=[yaml_editor, save_template_name],
            outputs=[status_text],
        )

        launch_btn.click(
            on_launch_click,
            inputs=[yaml_editor],
            outputs=[status_text, launch_btn, stop_btn],
        )

        stop_btn.click(
            on_stop_click,
            outputs=[status_text, launch_btn, stop_btn],
        )

        timer = gr.Timer(value=2, active=True)
        timer.tick(fn=refresh_logs, outputs=[log_output])

    return app


def main_zh():
    """中文版主入口"""
    app = create_app_zh()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="stone"),
    )