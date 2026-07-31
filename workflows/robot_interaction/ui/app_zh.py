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
from dataclasses import asdict
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
    DataProcessingConfig,
    SanityCheckConfig,
    CleanDatasetConfig,
    TimestampCheckConfig,
    MergeDatasetsConfig,
    V2ConvertConfig,
    ModelTrainingConfig,
    TrainingOptimizationConfig,
    TrainingTrackingConfig,
    SmolVLAConfig,
    FineTuneConfig,
    MODE_TO_KEY,
    KEY_TO_MODE,
    validate,
    save_yaml,
    load_yaml,
    list_presets,        # deprecated alias, 默认列 robot 模板
    load_preset,         # deprecated alias, 默认加载 robot 模板
    list_user_presets,   # 新: 按 kind 列用户预设
    load_user_preset,    # 新: 按 kind 加载用户预设
    save_user_preset,    # 新: 按 kind 保存用户预设
    list_all_robot_choices,  # 新: robot 模板 + robot 用户预设合并
    preset_kind_for_mode,    # 新: 中文/英文 mode → preset kind
    PRESET_SUBDIRS,      # 新: 子目录路径字典
)
from .process_manager import get_process_manager
from .components.data_processing_panel import (
    create_data_processing_panel,
    DataProcessingPanel,
)
from .components.model_training_panel import (
    create_model_training_panel,
    ModelTrainingPanel,
)

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
    """列出 datasets/ 下的所有数据集 (返回完整路径)"""
    return _list_datasets_full()


def _list_datasets_full() -> list[str]:
    """列出 datasets/ 下的所有目录,返回完整路径字符串 (按名字排序)

    自动跳过隐藏目录 (以 `.` 开头) 和 LeRobotHub 缓存 (.cache/huggingface)。
    """
    base_path = Path("datasets")
    if not base_path.exists():
        return []

    datasets = []
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
        # 跳过隐藏目录和缓存
        if item.name.startswith("."):
            continue
        datasets.append(str(item))

    return sorted(datasets)


def build_data_processing_config(
    operation: str,
    dataset_path: str,
    output_path: str,
    n_samples: int,
    dry_run: bool,
    report_only: bool,
    zero_threshold: int,
    source_roots_text: str,
    merge_repo_id: str,
    merge_video_size_mb: float,
    video_key: str,
    tolerance_ms: float,
    ts_report_output: str,
    ts_output_format: str,
    v2_variant: str,
    v2_output_root: str,
    v2_suffix: str,
    cam_left_eye: bool,
    cam_right_eye: bool,
    cam_left_wrist: bool,
    cam_right_wrist: bool,
    v2_dry_run: bool,
) -> DataProcessingConfig:
    """Build DataProcessingConfig from UI values."""
    variant = v2_variant if v2_variant in ("standard", "next_joint") else "standard"
    return DataProcessingConfig(
        operation=operation,
        dataset_path=dataset_path,
        output_path=output_path,
        sanity=SanityCheckConfig(n_samples=int(n_samples)),
        clean=CleanDatasetConfig(
            dry_run=bool(dry_run),
            report_only=bool(report_only),
            zero_threshold=int(zero_threshold),
        ),
        timestamp=TimestampCheckConfig(
            video_key=video_key,
            tolerance_ms=float(tolerance_ms) if tolerance_ms is not None else 1.0,
            report_output=ts_report_output,
            output_format=ts_output_format if ts_output_format in ("text", "json") else "text",
        ),
        merge=MergeDatasetsConfig(
            source_roots_text=source_roots_text,
            repo_id=merge_repo_id,
            video_files_size_mb=float(merge_video_size_mb) if merge_video_size_mb else 0.001,
        ),
        v2_convert=V2ConvertConfig(
            variant=variant,
            output_root=v2_output_root,
            v2_suffix=v2_suffix or "_v2",
            camera_enabled={
                "left_eye": bool(cam_left_eye),
                "right_eye": bool(cam_right_eye),
                "left_wrist": bool(cam_left_wrist),
                "right_wrist": bool(cam_right_wrist),
            },
            dry_run=bool(v2_dry_run),
        ),
    )


def build_model_training_config(
    script: str,
    phase: str,
    dataset_root: str,
    output_root: str,
    batch_size: int,
    steps: int,
    eval_freq: int,
    save_freq: int,
    log_freq: int,
    wandb_project: str,
    wandb_enable: bool,
    push_to_hub: bool,
    policy_chunk_size: int,
    policy_n_action_steps: int,
    policy_lr: float,
    policy_path: str,
    load_vlm_weights: bool,
    freeze_vision_encoder: bool,
    train_expert_only: bool,
    hf_endpoint: str,
    rename_map_json: str,
    pretrained_ckpt: str,
    new_dataset: str,
) -> ModelTrainingConfig:
    """Build ModelTrainingConfig from UI values."""
    # parse rename_map JSON
    rename_map: dict[str, str] = {}
    if rename_map_json and rename_map_json.strip():
        try:
            parsed = json.loads(rename_map_json)
            if isinstance(parsed, dict):
                rename_map = {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass  # validate() 阶段会捕获

    return ModelTrainingConfig(
        script=script,
        phase=phase,
        dataset_root=dataset_root,
        output_root=output_root,
        optimization=TrainingOptimizationConfig(
            batch_size=int(batch_size) if batch_size else 8,
            steps=int(steps) if steps else 400000,
            eval_freq=int(eval_freq) if eval_freq else 20000,
            save_freq=int(save_freq) if save_freq else 20000,
            log_freq=int(log_freq) if log_freq else 50,
        ),
        tracking=TrainingTrackingConfig(
            wandb_project=wandb_project,
            wandb_enable=bool(wandb_enable),
            push_to_hub=bool(push_to_hub),
        ),
        smolvla=SmolVLAConfig(
            policy_chunk_size=int(policy_chunk_size) if policy_chunk_size else 50,
            policy_n_action_steps=int(policy_n_action_steps) if policy_n_action_steps else 50,
            policy_lr=float(policy_lr) if policy_lr else 1e-4,
            policy_path=policy_path,
            load_vlm_weights=bool(load_vlm_weights),
            freeze_vision_encoder=bool(freeze_vision_encoder),
            train_expert_only=bool(train_expert_only),
            hf_endpoint=hf_endpoint,
            rename_map=rename_map,
        ),
        finetune=FineTuneConfig(
            pretrained_ckpt=pretrained_ckpt,
            new_dataset=new_dataset,
        ),
    )


def build_config_from_ui(
    mode: str,
    # Policy
    policy_path: str,
    policy_device: str,
    # Robot — basic
    http_url: str,
    robot_id: str,
    robot_type: str,
    robot_timeout: float,
    safety_stats_path: str,
    warn_on_observation_out_of_range: bool,
    action_clip_margin_deg: float,
    max_relative_target_deg: float,
    # Robot — hybrid (only used when robot_type == "marvain_m6_hybrid")
    robot_ip: str,
    robot_control_mode: str,
    robot_vel_ratio: int,
    robot_acc_ratio: int,
    robot_disable_torque_on_disconnect: bool,
    # Robot — joint names (JSON list string from textarea)
    joint_names_json: str,
    # Robot — cameras (JSON dict string from textarea)
    cameras_json: str,
    # Inference — basic
    fps: float,
    strategy: str,
    inference_type: str,
    duration: float,
    interpolation_multiplier: float,
    max_steps: int,
    use_torch_compile: bool,
    torch_compile_backend: str,
    torch_compile_mode: str,
    compile_warmup_inferences: int,
    show_cameras_inf: bool,
    rename_map_json: str,
    # Inference — RTC
    execution_horizon: float,
    max_guidance_weight: float,
    # Inference — chunk
    n_action_steps: int,
    chunk_interval_s: float,
    # Dataset
    repo_id: str,
    dataset_root: str,
    episode: float,
    single_task: str,
    dataset_fps: float,
    # Camera (preview only)
    camera_list: list,
    camera_fps: float,
    show_quad: bool,
    window_width: float,
    window_height: float,
    # Runtime
    return_to_home: bool,
    play_sounds: bool,
    # Data processing
    dp_operation: str = "sanity",
    dp_dataset_path: str = "",
    dp_output_path: str = "",
    dp_n_samples: int = 5,
    dp_dry_run: bool = True,
    dp_report_only: bool = False,
    dp_zero_threshold: int = 7,
    dp_source_roots_text: str = "",
    dp_merge_repo_id: str = "",
    dp_merge_video_size_mb: float = 0.001,
    dp_video_key: str = "",
    dp_tolerance_ms: float = 1.0,
    dp_ts_report_output: str = "",
    dp_ts_output_format: str = "text",
    dp_v2_variant: str = "standard",
    dp_v2_output_root: str = "",
    dp_v2_suffix: str = "_v2",
    dp_cam_left_eye: bool = True,
    dp_cam_right_eye: bool = True,
    dp_cam_left_wrist: bool = True,
    dp_cam_right_wrist: bool = True,
    dp_v2_dry_run: bool = True,
    # Model training
    mt_script: str = "act",
    mt_phase: str = "smoke",
    mt_dataset_root: str = "",
    mt_output_root: str = "",
    mt_batch_size: int = 8,
    mt_steps: int = 400000,
    mt_eval_freq: int = 20000,
    mt_save_freq: int = 20000,
    mt_log_freq: int = 50,
    mt_wandb_project: str = "",
    mt_wandb_enable: bool = False,
    mt_push_to_hub: bool = False,
    mt_policy_chunk_size: int = 50,
    mt_policy_n_action_steps: int = 50,
    mt_policy_lr: float = 1e-4,
    mt_policy_path: str = "",
    mt_load_vlm_weights: bool = False,
    mt_freeze_vision_encoder: bool = True,
    mt_train_expert_only: bool = True,
    mt_hf_endpoint: str = "",
    mt_rename_map_json: str = "",
    mt_pretrained_ckpt: str = "",
    mt_new_dataset: str = "",
) -> UnifiedRobotConfig:
    """从 UI 组件值构建 UnifiedRobotConfig

    这是 UI 表单 → dataclass 的唯一通道。所有 RobotConfig/InferenceConfig/
    DatasetConfig 字段都必须在这里透传，否则会丢字段（见 Phase 1 审计）。
    """

    mode_key = MODE_TO_KEY.get(mode, mode.lower().replace(" ", "_"))

    # 解析 rename_map JSON
    rename_map = {}
    if rename_map_json and rename_map_json.strip():
        try:
            rename_map = json.loads(rename_map_json)
        except json.JSONDecodeError:
            pass  # 将在验证中捕获

    # 解析 joint_names JSON（list[str]）
    joint_names: list[str] | None = None
    if joint_names_json and joint_names_json.strip():
        try:
            parsed = json.loads(joint_names_json)
            if isinstance(parsed, list):
                joint_names = [str(j) for j in parsed]
        except json.JSONDecodeError:
            pass  # 验证阶段报错

    # 解析 cameras JSON（dict[name → {fps,width,height,color_mode}]）
    cameras_dict: dict[str, dict] | None = None
    if cameras_json and cameras_json.strip():
        try:
            parsed = json.loads(cameras_json)
            if isinstance(parsed, dict):
                cameras_dict = parsed
        except json.JSONDecodeError:
            pass  # 验证阶段报错

    # 构建 RobotConfig：所有 dataclass 字段都从 UI 来，不再依赖默认。
    robot_kwargs: dict = dict(
        http_base_url=http_url,
        id=robot_id,  # RobotConfig field is `id`（matches YAML/deploy.py contract）
        type=robot_type if robot_type else "marvain_m6_http",
        timeout=float(robot_timeout) if robot_timeout else 5.0,
        safety_stats_path=safety_stats_path if safety_stats_path else None,
        warn_on_observation_out_of_range=bool(warn_on_observation_out_of_range),
        action_clip_margin_deg=float(action_clip_margin_deg) if action_clip_margin_deg is not None else 5.0,
        max_relative_target_deg=float(max_relative_target_deg) if max_relative_target_deg is not None else 10.0,
        ip=robot_ip if robot_ip else "192.168.10.190",
        control_mode=robot_control_mode if robot_control_mode else "impedance",
        vel_ratio=int(robot_vel_ratio) if robot_vel_ratio is not None else 20,
        acc_ratio=int(robot_acc_ratio) if robot_acc_ratio is not None else 20,
        disable_torque_on_disconnect=bool(robot_disable_torque_on_disconnect),
    )
    if joint_names is not None:
        robot_kwargs["joint_names"] = joint_names
    if cameras_dict is not None:
        # cameras_dict 是 dict[name → dict]，转成 dict[name → CameraConfig]
        from .config_manager import CameraConfig
        robot_kwargs["cameras"] = {
            name: CameraConfig(
                **{k: v for k, v in (cam or {}).items() if k in {"type", "fps", "width", "height", "color_mode"}}
            )
            for name, cam in cameras_dict.items()
        }

    config = UnifiedRobotConfig(
        mode=mode_key,
        robot=RobotConfig(**robot_kwargs),
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
        # InferenceConfig：所有 dataclass 字段都从 UI 来。
        inference_kwargs: dict = dict(
            type=inference_type,
            strategy=strategy,
            fps=fps,
            duration=duration,
            interpolation_multiplier=int(interpolation_multiplier),
            max_steps=int(max_steps) if max_steps is not None else 10000,
            use_torch_compile=use_torch_compile,
            torch_compile_backend=torch_compile_backend if torch_compile_backend else "inductor",
            torch_compile_mode=torch_compile_mode if torch_compile_mode else "default",
            compile_warmup_inferences=int(compile_warmup_inferences) if compile_warmup_inferences is not None else 2,
            show_cameras=show_cameras_inf,
            rename_map=rename_map,
            rtc=RTCConfig(
                execution_horizon=int(execution_horizon) if execution_horizon else None,
                max_guidance_weight=max_guidance_weight,
            ),
            # chunk 模式字段：build_config_from_ui 必须把 None 也传过去（而不是丢），
            # 这样 deploy.py 才不会因为 KeyError 炸掉。
            n_action_steps=int(n_action_steps) if n_action_steps is not None else None,
            chunk_interval_s=float(chunk_interval_s) if chunk_interval_s is not None else None,
        )
        config.inference = InferenceConfig(**inference_kwargs)
        config.dataset = DatasetConfig(
            repo_id=repo_id if repo_id else "",
            root=dataset_root if dataset_root else None,
            single_task=single_task if single_task else None,
            fps=float(dataset_fps) if dataset_fps is not None else 30.0,
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

    elif mode_key == "data_processing":
        config.data_processing = build_data_processing_config(
            dp_operation, dp_dataset_path, dp_output_path,
            dp_n_samples, dp_dry_run, dp_report_only, dp_zero_threshold,
            dp_source_roots_text, dp_merge_repo_id, dp_merge_video_size_mb,
            dp_video_key, dp_tolerance_ms, dp_ts_report_output, dp_ts_output_format,
            dp_v2_variant, dp_v2_output_root, dp_v2_suffix,
            dp_cam_left_eye, dp_cam_right_eye, dp_cam_left_wrist, dp_cam_right_wrist,
            dp_v2_dry_run,
        )

    elif mode_key == "model_training":
        config.model_training = build_model_training_config(
            mt_script, mt_phase, mt_dataset_root, mt_output_root,
            mt_batch_size, mt_steps, mt_eval_freq, mt_save_freq, mt_log_freq,
            mt_wandb_project, mt_wandb_enable, mt_push_to_hub,
            mt_policy_chunk_size, mt_policy_n_action_steps, mt_policy_lr, mt_policy_path,
            mt_load_vlm_weights, mt_freeze_vision_encoder, mt_train_expert_only,
            mt_hf_endpoint, mt_rename_map_json,
            mt_pretrained_ckpt, mt_new_dataset,
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
                choices=["部署", "回放", "相机预览", "数据处理", "模型训练"],
                value="部署",
                label="操作模式",
                scale=2,
            )

            with gr.Column(scale=1):
                preset_dropdown = gr.Dropdown(
                    label="加载预设配置 (按当前 mode 过滤)",
                    choices=list_all_robot_choices(),  # 初始: 部署模式
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

            with gr.Row():
                robot_type = gr.Dropdown(
                    choices=["marvain_m6_http", "marvain_m6_hybrid"],
                    value="marvain_m6_http",
                    label="机器人后端",
                    info="HTTP: 全部走 HTTP API | Hybrid: HTTP 取观测 + SDK 下发动作",
                    interactive=True,
                    scale=1,
                )
                robot_timeout = gr.Slider(
                    minimum=0.5,
                    maximum=30.0,
                    value=5.0,
                    step=0.5,
                    label="HTTP 超时（秒）",
                    interactive=True,
                    scale=1,
                )

            with gr.Row():
                warn_on_observation_out_of_range = gr.Checkbox(
                    label="观测超界时警告",
                    value=True,
                    info="观测值超出训练集分布时打 warning",
                    interactive=True,
                )

            with gr.Row():
                action_clip_margin_deg = gr.Slider(
                    minimum=0.0,
                    maximum=30.0,
                    value=5.0,
                    step=0.5,
                    label="动作裁剪余量（度）",
                    interactive=True,
                )
                max_relative_target_deg = gr.Slider(
                    minimum=0.0,
                    maximum=90.0,
                    value=10.0,
                    step=1.0,
                    label="单步最大相对位移（度）",
                    interactive=True,
                )

            # Hybrid 字段：仅 robot_type == "marvain_m6_hybrid" 时展开
            with gr.Group(visible=False) as hybrid_group:
                gr.Markdown("### Hybrid 后端专属（仅 marvain_m6_hybrid 时生效）")
                robot_ip = gr.Textbox(
                    label="机器人 IP",
                    value="192.168.10.190",
                    info="SDK 直连的机械臂 IP",
                    interactive=True,
                )
                robot_control_mode = gr.Dropdown(
                    choices=["impedance", "position"],
                    value="impedance",
                    label="控制模式",
                    interactive=True,
                )
                with gr.Row():
                    robot_vel_ratio = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=20,
                        step=1,
                        label="速度比例 (%)",
                        interactive=True,
                    )
                    robot_acc_ratio = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=20,
                        step=1,
                        label="加速度比例 (%)",
                        interactive=True,
                    )
                robot_disable_torque_on_disconnect = gr.Checkbox(
                    label="断连时关闭扭矩",
                    value=False,
                    info="True: disconnect 时调用 release_robot；False: 保留使能",
                    interactive=True,
                )

            # 关节名 / 相机元数据：sub-accordion，编辑频率低
            with gr.Accordion("高级：关节名与相机元数据（JSON）", open=False):
                joint_names_json = gr.Textbox(
                    label="关节名列表（JSON 数组）",
                    placeholder='["left_arm_joint_1", ..., "right_gripper"]',
                    lines=3,
                    info="留空使用默认 16 关节；JSON 必须能解析为 list[str]",
                    interactive=True,
                )
                cameras_json = gr.Textbox(
                    label="相机元数据（JSON dict）",
                    placeholder='{"right_eye": {"type": "http", "fps": 30, "width": 640, "height": 480, "color_mode": "rgb"}, ...}',
                    lines=5,
                    info="留空使用默认 4 摄像头；JSON 必须能解析为 dict[name → 配置]",
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

            # Chunk 模式设置（仅 inference_type == "chunk" 时显示）
            with gr.Group(visible=False) as chunk_group:
                gr.Markdown("### Chunk 模式设置（开环）")
                with gr.Row():
                    n_action_steps = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=50,
                        step=1,
                        label="每次发送的动作步数",
                        info="每个 chunk 推理后连续执行的动作数",
                        interactive=True,
                    )
                    chunk_interval_s = gr.Slider(
                        minimum=0.1,
                        maximum=30.0,
                        value=5.0,
                        step=0.1,
                        label="chunk 间隔（秒）",
                        info="两次推理的间隔；上一次发出去的动作仍继续执行",
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

            max_steps = gr.Number(
                label="最大推理步数",
                value=10000,
                precision=0,
                minimum=1,
                info="达到后自动停止（0=无限）",
                interactive=True,
            )

            use_torch_compile = gr.Checkbox(
                label="使用 Torch 编译",
                value=False,
                info="PyTorch 2.0+ 编译加速",
                interactive=True,
            )

            # Torch compile 子设置（仅启用时显示）
            with gr.Group(visible=False) as torch_compile_group:
                gr.Markdown("### Torch Compile 子设置")
                with gr.Row():
                    torch_compile_backend = gr.Dropdown(
                        choices=["inductor", "cudagraphs"],
                        value="inductor",
                        label="编译后端",
                        interactive=True,
                    )
                    torch_compile_mode = gr.Dropdown(
                        choices=["default", "reduce-overhead", "max-autotune"],
                        value="default",
                        label="编译模式",
                        interactive=True,
                    )
                compile_warmup_inferences = gr.Number(
                    label="预热推理次数",
                    value=2,
                    precision=0,
                    minimum=0,
                    info="正式运行前先跑几次让 torch.compile 完成 trace",
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

            dataset_fps_deploy = gr.Slider(
                minimum=1,
                maximum=60,
                value=30,
                step=1,
                label="数据集 fps（仅 sentry/highlight/dagger/episodic 时生效）",
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
        # 数据处理 / 模型训练 panel (默认隐藏, 通过 mode 切换)
        # ====================================================================

        dp_panel = create_data_processing_panel()
        mt_panel = create_model_training_panel()

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
            is_data = mode_value == "数据处理"
            is_train = mode_value == "模型训练"

            # 按 mode 切换 preset 下拉的 options
            kind = preset_kind_for_mode(mode_value)
            if kind == "robot":
                preset_choices = list_all_robot_choices()
            else:
                preset_choices = list_user_presets(kind)

            return {
                # policy_panel 必须也在 camera_preview 模式下可见：show_cameras.py
                # 强制要求 policy.path（推导相机列表或 fallback），用户必须能填。
                policy_panel: gr.update(visible=is_deploy or is_camera),
                inference_panel: gr.update(visible=is_deploy),
                deploy_dataset_panel: gr.update(visible=is_deploy),
                replay_dataset_panel: gr.update(visible=is_replay),
                camera_panel: gr.update(visible=is_camera),
                dp_panel.panel: gr.update(visible=is_data),
                mt_panel.panel: gr.update(visible=is_train),
                # mode 切换 → 刷新预设下拉
                preset_dropdown: gr.update(choices=preset_choices, value=None),
                # 提示当前预设保存位置
                save_preset_name: gr.update(info=f"将保存到 ui/presets/{kind}/<name>.yaml"),
            }

        def on_inference_type_change(inf_type):
            """根据推理类型显示对应子面板（rtc / chunk）"""
            return (
                gr.update(visible=(inf_type == "rtc")),    # rtc_group
                gr.update(visible=(inf_type == "chunk")),  # chunk_group
            )

        def on_data_operation_change(op):
            """根据 operation 显示对应子面板 (sanity / clean / merge / ts_check / v2_convert)"""
            return (
                gr.update(visible=(op == "sanity")),
                gr.update(visible=(op == "clean")),
                gr.update(visible=(op == "merge")),
                gr.update(visible=(op == "ts_check")),
                gr.update(visible=(op == "v2_convert")),
            )

        def on_training_script_change(script):
            """根据 script 显示对应子面板 (act / smolvla / finetune)"""
            # 三个脚本都支持 BATCH_SIZE/STEPS/EVAL_FREQ/SAVE_FREQ/LOG_FREQ 和 WANDB_* env
            # (finetune_act.sh 第 45-53 行确认), 所以 opt_group + trk_group 三个都显示
            has_opt_trk = script in ("act", "smolvla", "finetune")
            return (
                gr.update(visible=has_opt_trk),   # opt_group
                gr.update(visible=has_opt_trk),   # trk_group
                gr.update(visible=(script == "smolvla")),  # smolvla_group
                gr.update(visible=(script == "finetune")), # finetune_group
            )

        def on_robot_type_change(rt):
            """robot.type == marvain_m6_hybrid 时显示 hybrid_group"""
            return gr.update(visible=(rt == "marvain_m6_hybrid"))

        def on_use_torch_compile_change(use_tc):
            """启用 torch.compile 时显示子设置"""
            return gr.update(visible=bool(use_tc))

        def on_strategy_change(strat):
            """当策略需要录制时显示数据集面板"""
            needs_dataset = strat != "base"
            return gr.update(visible=needs_dataset)

        # on_load_preset 期望返回的 47 个 output components (顺序必须与
        # preset_dropdown.change 绑定的 outputs 一一对应)
        _ALL_PRESET_OUTPUTS = (
            policy_path, policy_device,
            http_url, robot_id, robot_type, robot_timeout, safety_stats_path,
            warn_on_observation_out_of_range, action_clip_margin_deg, max_relative_target_deg,
            robot_ip, robot_control_mode, robot_vel_ratio, robot_acc_ratio, robot_disable_torque_on_disconnect,
            joint_names_json, cameras_json,
            fps, strategy, inference_type,
            duration, interpolation_multiplier, max_steps,
            use_torch_compile, torch_compile_backend, torch_compile_mode, compile_warmup_inferences,
            show_cameras_inf, rename_map_json,
            execution_horizon, max_guidance_weight,
            n_action_steps, chunk_interval_s,
            repo_id_deploy, repo_id_replay,
            dataset_root_deploy, dataset_root_replay,
            single_task, episode,
            dataset_fps_deploy, dataset_fps,
            return_to_home, play_sounds,
            camera_list, camera_fps, show_quad,
            status_text,
        )
        assert len(_ALL_PRESET_OUTPUTS) == 47, f"期望 47 个 output, 实际 {len(_ALL_PRESET_OUTPUTS)}"

        def _no_preset_update() -> dict:
            """不修改任何 output 的统一占位 dict (47 个 gr.update() 无变化)"""
            return {c: gr.update() for c in _ALL_PRESET_OUTPUTS}

        def on_load_preset(preset_name, current_mode):
            """加载预设配置并更新 UI (按当前 mode 过滤).

            只更新与当前 mode 相关的字段, 其它 mode 的字段保持 UI 现状不动,
            避免在 data_processing / model_training 模式下写入无意义的 robot 字段。
            """
            if not preset_name:
                return _no_preset_update()

            try:
                kind = preset_kind_for_mode(current_mode)
                # 模板 (kind=robot) 与用户预设统一入口
                if kind == "robot" and " (预设)" not in preset_name:
                    config = load_preset(preset_name)  # 从 workflows/robot_interaction/ 加载模板
                else:
                    config = load_user_preset(kind, preset_name)
                updates = {}

                # —— Policy ——
                if config.policy:
                    updates[policy_path] = gr.update(value=config.policy.path)
                    updates[policy_device] = gr.update(value=config.policy.device)

                # —— Robot ——
                r = config.robot
                updates[http_url] = gr.update(value=r.http_base_url)
                updates[robot_id] = gr.update(value=r.id)
                updates[robot_type] = gr.update(value=r.type)
                updates[robot_timeout] = gr.update(value=r.timeout)
                updates[warn_on_observation_out_of_range] = gr.update(value=r.warn_on_observation_out_of_range)
                updates[action_clip_margin_deg] = gr.update(value=r.action_clip_margin_deg)
                updates[max_relative_target_deg] = gr.update(value=r.max_relative_target_deg)
                # hybrid 字段（仅在 hybrid 时用户能看到，但仍写回 UI 状态）
                updates[robot_ip] = gr.update(value=r.ip)
                updates[robot_control_mode] = gr.update(value=r.control_mode)
                updates[robot_vel_ratio] = gr.update(value=r.vel_ratio)
                updates[robot_acc_ratio] = gr.update(value=r.acc_ratio)
                updates[robot_disable_torque_on_disconnect] = gr.update(value=r.disable_torque_on_disconnect)
                # joint_names / cameras：JSON 化以便编辑
                updates[joint_names_json] = gr.update(value=json.dumps(r.joint_names, indent=2) if r.joint_names else "")
                updates[cameras_json] = gr.update(
                    value=json.dumps(
                        {k: asdict(v) for k, v in r.cameras.items()},
                        indent=2,
                    )
                    if r.cameras else ""
                )
                # safety_stats_path 可空
                updates[safety_stats_path] = gr.update(value=r.safety_stats_path or "")

                # —— Inference ——
                if config.inference:
                    inf = config.inference
                    updates[fps] = gr.update(value=inf.fps)
                    updates[strategy] = gr.update(value=inf.strategy)
                    updates[inference_type] = gr.update(value=inf.type)
                    updates[duration] = gr.update(value=inf.duration)
                    updates[interpolation_multiplier] = gr.update(value=inf.interpolation_multiplier)
                    updates[max_steps] = gr.update(value=inf.max_steps)
                    updates[use_torch_compile] = gr.update(value=inf.use_torch_compile)
                    updates[torch_compile_backend] = gr.update(value=inf.torch_compile_backend)
                    updates[torch_compile_mode] = gr.update(value=inf.torch_compile_mode)
                    updates[compile_warmup_inferences] = gr.update(value=inf.compile_warmup_inferences)
                    updates[show_cameras_inf] = gr.update(value=inf.show_cameras)

                    if inf.rtc and inf.rtc.execution_horizon is not None:
                        updates[execution_horizon] = gr.update(value=inf.rtc.execution_horizon)
                    if inf.rtc and inf.rtc.max_guidance_weight is not None:
                        updates[max_guidance_weight] = gr.update(value=inf.rtc.max_guidance_weight)

                    if inf.rename_map:
                        updates[rename_map_json] = gr.update(value=json.dumps(inf.rename_map, indent=2))

                    # chunk 模式字段
                    if inf.n_action_steps is not None:
                        updates[n_action_steps] = gr.update(value=inf.n_action_steps)
                    if inf.chunk_interval_s is not None:
                        updates[chunk_interval_s] = gr.update(value=inf.chunk_interval_s)

                # —— Dataset ——
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
                    # dataset.fps 同时写回 deploy 和 replay 两个 Slider
                    if config.dataset.fps is not None:
                        updates[dataset_fps_deploy] = gr.update(value=config.dataset.fps)
                        updates[dataset_fps] = gr.update(value=config.dataset.fps)

                # —— Runtime ——
                updates[return_to_home] = gr.update(value=config.runtime.return_to_initial_position)
                updates[play_sounds] = gr.update(value=config.runtime.play_sounds)
                if config.runtime.camera_list:
                    updates[camera_list] = gr.update(value=config.runtime.camera_list)
                updates[camera_fps] = gr.update(value=config.runtime.camera_fps)
                updates[show_quad] = gr.update(value=config.runtime.show_quad)

                # —— Data processing ——
                if config.data_processing:
                    dp = config.data_processing
                    updates[dp_panel.operation] = gr.update(value=dp.operation)
                    updates[dp_panel.dataset_path] = gr.update(value=dp.dataset_path)
                    updates[dp_panel.output_path] = gr.update(value=dp.output_path)
                    updates[dp_panel.n_samples] = gr.update(value=dp.sanity.n_samples)
                    updates[dp_panel.dry_run] = gr.update(value=dp.clean.dry_run)
                    updates[dp_panel.report_only] = gr.update(value=dp.clean.report_only)
                    updates[dp_panel.zero_threshold] = gr.update(value=dp.clean.zero_threshold)
                    updates[dp_panel.source_roots_text] = gr.update(value=dp.merge.source_roots_text)
                    updates[dp_panel.merge_repo_id] = gr.update(value=dp.merge.repo_id)
                    updates[dp_panel.merge_video_size_mb] = gr.update(value=dp.merge.video_files_size_mb)
                    updates[dp_panel.video_key] = gr.update(value=dp.timestamp.video_key)
                    updates[dp_panel.tolerance_ms] = gr.update(value=dp.timestamp.tolerance_ms)
                    updates[dp_panel.ts_report_output] = gr.update(value=dp.timestamp.report_output)
                    updates[dp_panel.ts_output_format] = gr.update(value=dp.timestamp.output_format)
                    updates[dp_panel.v2_variant] = gr.update(value=dp.v2_convert.variant)
                    updates[dp_panel.v2_output_root] = gr.update(value=dp.v2_convert.output_root)
                    updates[dp_panel.v2_suffix] = gr.update(value=dp.v2_convert.v2_suffix)
                    cam_enabled = dp.v2_convert.camera_enabled
                    updates[dp_panel.cam_left_eye] = gr.update(value=bool(cam_enabled.get("left_eye", True)))
                    updates[dp_panel.cam_right_eye] = gr.update(value=bool(cam_enabled.get("right_eye", True)))
                    updates[dp_panel.cam_left_wrist] = gr.update(value=bool(cam_enabled.get("left_wrist", True)))
                    updates[dp_panel.cam_right_wrist] = gr.update(value=bool(cam_enabled.get("right_wrist", True)))
                    updates[dp_panel.v2_dry_run] = gr.update(value=dp.v2_convert.dry_run)

                # —— Model training ——
                if config.model_training:
                    mt = config.model_training
                    updates[mt_panel.script] = gr.update(value=mt.script)
                    updates[mt_panel.phase] = gr.update(value=mt.phase)
                    updates[mt_panel.dataset_root] = gr.update(value=mt.dataset_root)
                    updates[mt_panel.output_root] = gr.update(value=mt.output_root)
                    updates[mt_panel.batch_size] = gr.update(value=mt.optimization.batch_size)
                    updates[mt_panel.steps] = gr.update(value=mt.optimization.steps)
                    updates[mt_panel.eval_freq] = gr.update(value=mt.optimization.eval_freq)
                    updates[mt_panel.save_freq] = gr.update(value=mt.optimization.save_freq)
                    updates[mt_panel.log_freq] = gr.update(value=mt.optimization.log_freq)
                    updates[mt_panel.wandb_project] = gr.update(value=mt.tracking.wandb_project)
                    updates[mt_panel.wandb_enable] = gr.update(value=mt.tracking.wandb_enable)
                    updates[mt_panel.push_to_hub] = gr.update(value=mt.tracking.push_to_hub)
                    updates[mt_panel.policy_chunk_size] = gr.update(value=mt.smolvla.policy_chunk_size)
                    updates[mt_panel.policy_n_action_steps] = gr.update(value=mt.smolvla.policy_n_action_steps)
                    updates[mt_panel.policy_lr] = gr.update(value=mt.smolvla.policy_lr)
                    updates[mt_panel.policy_path] = gr.update(value=mt.smolvla.policy_path)
                    updates[mt_panel.load_vlm_weights] = gr.update(value=mt.smolvla.load_vlm_weights)
                    updates[mt_panel.freeze_vision_encoder] = gr.update(value=mt.smolvla.freeze_vision_encoder)
                    updates[mt_panel.train_expert_only] = gr.update(value=mt.smolvla.train_expert_only)
                    updates[mt_panel.hf_endpoint] = gr.update(value=mt.smolvla.hf_endpoint)
                    if mt.smolvla.rename_map:
                        updates[mt_panel.rename_map_json] = gr.update(value=json.dumps(mt.smolvla.rename_map))
                    updates[mt_panel.pretrained_ckpt] = gr.update(value=mt.finetune.pretrained_ckpt)
                    updates[mt_panel.new_dataset] = gr.update(value=mt.finetune.new_dataset)

                # 状态消息
                updates[status_text] = gr.update(value=f"✅ 已加载预设: {preset_name}")

                # 过滤 updates: 只能包含 _ALL_PRESET_OUTPUTS 里的 key
                # (避免 Gradio 报 "Returned component not specified as output")
                # dp_panel / mt_panel 字段虽然会写入 updates 字典, 但因为它们
                # 不在 preset_dropdown.change 的 outputs 列表里, 必须过滤掉
                _valid_keys = set(_ALL_PRESET_OUTPUTS)
                full_updates = _no_preset_update()
                for k, v in updates.items():
                    if k in _valid_keys:
                        full_updates[k] = v
                # 也保证 status_text 出现
                full_updates[status_text] = gr.update(value=f"✅ 已加载预设: {preset_name}")
                return full_updates

            except Exception as e:
                err_update = _no_preset_update()
                err_update[status_text] = gr.update(value=f"❌ 加载预设失败: {e}")
                return err_update

        def on_save_preset(
            preset_name,
            mode, policy_path, policy_device,
            http_url, robot_id, robot_type, robot_timeout, safety_stats_path,
            warn_on_observation_out_of_range, action_clip_margin_deg, max_relative_target_deg,
            robot_ip, robot_control_mode, robot_vel_ratio, robot_acc_ratio, robot_disable_torque_on_disconnect,
            joint_names_json, cameras_json,
            fps, strategy, inference_type,
            duration, interpolation_multiplier, max_steps,
            use_torch_compile, torch_compile_backend, torch_compile_mode, compile_warmup_inferences,
            show_cameras_inf, rename_map_json,
            execution_horizon, max_guidance_weight,
            n_action_steps, chunk_interval_s,
            repo_id_deploy, dataset_root_deploy, episode, single_task, dataset_fps_deploy,
            repo_id_replay, dataset_root_replay, dataset_fps,
            camera_list, camera_fps, show_quad, window_width, window_height,
            return_to_home, play_sounds,
            dp_operation, dp_dataset_path, dp_output_path,
            dp_n_samples, dp_dry_run, dp_report_only, dp_zero_threshold,
            dp_source_roots_text, dp_merge_repo_id, dp_merge_video_size_mb,
            dp_video_key, dp_tolerance_ms, dp_ts_report_output, dp_ts_output_format,
            dp_v2_variant, dp_v2_output_root, dp_v2_suffix,
            dp_cam_left_eye, dp_cam_right_eye, dp_cam_left_wrist, dp_cam_right_wrist, dp_v2_dry_run,
            mt_script, mt_phase, mt_dataset_root, mt_output_root,
            mt_batch_size, mt_steps, mt_eval_freq, mt_save_freq, mt_log_freq,
            mt_wandb_project, mt_wandb_enable, mt_push_to_hub,
            mt_policy_chunk_size, mt_policy_n_action_steps, mt_policy_lr, mt_policy_path,
            mt_load_vlm_weights, mt_freeze_vision_encoder, mt_train_expert_only,
            mt_hf_endpoint, mt_rename_map_json,
            mt_pretrained_ckpt, mt_new_dataset,
        ):
            """保存当前配置为预设"""
            if not preset_name or not preset_name.strip():
                return gr.update(value="❌ 请输入预设名称")

            try:
                # 合并 repo_id / dataset_fps（deploy 和 replay 两个面板各一份）
                repo_id = repo_id_deploy if mode == "部署" else repo_id_replay
                dataset_root = dataset_root_deploy if mode == "部署" else dataset_root_replay
                fps_value = dataset_fps_deploy if mode == "部署" else dataset_fps

                config = build_config_from_ui(
                    mode, policy_path, policy_device,
                    http_url, robot_id, robot_type, robot_timeout, safety_stats_path,
                    warn_on_observation_out_of_range, action_clip_margin_deg, max_relative_target_deg,
                    robot_ip, robot_control_mode, robot_vel_ratio, robot_acc_ratio, robot_disable_torque_on_disconnect,
                    joint_names_json, cameras_json,
                    fps, strategy, inference_type,
                    duration, interpolation_multiplier, max_steps,
                    use_torch_compile, torch_compile_backend, torch_compile_mode, compile_warmup_inferences,
                    show_cameras_inf, rename_map_json,
                    execution_horizon, max_guidance_weight,
                    n_action_steps, chunk_interval_s,
                    repo_id, dataset_root, episode, single_task, fps_value,
                    camera_list, camera_fps, show_quad, window_width, window_height,
                    return_to_home, play_sounds,
                    dp_operation, dp_dataset_path, dp_output_path,
                    dp_n_samples, dp_dry_run, dp_report_only, dp_zero_threshold,
                    dp_source_roots_text, dp_merge_repo_id, dp_merge_video_size_mb,
                    dp_video_key, dp_tolerance_ms, dp_ts_report_output, dp_ts_output_format,
                    dp_v2_variant, dp_v2_output_root, dp_v2_suffix,
                    dp_cam_left_eye, dp_cam_right_eye, dp_cam_left_wrist, dp_cam_right_wrist, dp_v2_dry_run,
                    mt_script, mt_phase, mt_dataset_root, mt_output_root,
                    mt_batch_size, mt_steps, mt_eval_freq, mt_save_freq, mt_log_freq,
                    mt_wandb_project, mt_wandb_enable, mt_push_to_hub,
                    mt_policy_chunk_size, mt_policy_n_action_steps, mt_policy_lr, mt_policy_path,
                    mt_load_vlm_weights, mt_freeze_vision_encoder, mt_train_expert_only,
                    mt_hf_endpoint, mt_rename_map_json,
                    mt_pretrained_ckpt, mt_new_dataset,
                )

                preset_kind = preset_kind_for_mode(mode)
                preset_file = save_user_preset(preset_kind, preset_name, config)

                return gr.update(value=f"✅ 预设已保存到 {preset_kind}/{preset_name}.yaml: {preset_file}")

            except Exception as e:
                return gr.update(value=f"❌ 保存失败: {e}")

        def on_export_yaml(
            mode, policy_path, policy_device,
            http_url, robot_id, robot_type, robot_timeout, safety_stats_path,
            warn_on_observation_out_of_range, action_clip_margin_deg, max_relative_target_deg,
            robot_ip, robot_control_mode, robot_vel_ratio, robot_acc_ratio, robot_disable_torque_on_disconnect,
            joint_names_json, cameras_json,
            fps, strategy, inference_type,
            duration, interpolation_multiplier, max_steps,
            use_torch_compile, torch_compile_backend, torch_compile_mode, compile_warmup_inferences,
            show_cameras_inf, rename_map_json,
            execution_horizon, max_guidance_weight,
            n_action_steps, chunk_interval_s,
            repo_id_deploy, dataset_root_deploy, episode, single_task, dataset_fps_deploy,
            repo_id_replay, dataset_root_replay, dataset_fps,
            camera_list, camera_fps, show_quad, window_width, window_height,
            return_to_home, play_sounds,
            dp_operation, dp_dataset_path, dp_output_path,
            dp_n_samples, dp_dry_run, dp_report_only, dp_zero_threshold,
            dp_source_roots_text, dp_merge_repo_id, dp_merge_video_size_mb,
            dp_video_key, dp_tolerance_ms, dp_ts_report_output, dp_ts_output_format,
            dp_v2_variant, dp_v2_output_root, dp_v2_suffix,
            dp_cam_left_eye, dp_cam_right_eye, dp_cam_left_wrist, dp_cam_right_wrist, dp_v2_dry_run,
            mt_script, mt_phase, mt_dataset_root, mt_output_root,
            mt_batch_size, mt_steps, mt_eval_freq, mt_save_freq, mt_log_freq,
            mt_wandb_project, mt_wandb_enable, mt_push_to_hub,
            mt_policy_chunk_size, mt_policy_n_action_steps, mt_policy_lr, mt_policy_path,
            mt_load_vlm_weights, mt_freeze_vision_encoder, mt_train_expert_only,
            mt_hf_endpoint, mt_rename_map_json,
            mt_pretrained_ckpt, mt_new_dataset,
        ):
            """导出当前配置为 YAML 文件"""
            try:
                repo_id = repo_id_deploy if mode == "部署" else repo_id_replay
                dataset_root = dataset_root_deploy if mode == "部署" else dataset_root_replay
                fps_value = dataset_fps_deploy if mode == "部署" else dataset_fps

                config = build_config_from_ui(
                    mode, policy_path, policy_device,
                    http_url, robot_id, robot_type, robot_timeout, safety_stats_path,
                    warn_on_observation_out_of_range, action_clip_margin_deg, max_relative_target_deg,
                    robot_ip, robot_control_mode, robot_vel_ratio, robot_acc_ratio, robot_disable_torque_on_disconnect,
                    joint_names_json, cameras_json,
                    fps, strategy, inference_type,
                    duration, interpolation_multiplier, max_steps,
                    use_torch_compile, torch_compile_backend, torch_compile_mode, compile_warmup_inferences,
                    show_cameras_inf, rename_map_json,
                    execution_horizon, max_guidance_weight,
                    n_action_steps, chunk_interval_s,
                    repo_id, dataset_root, episode, single_task, fps_value,
                    camera_list, camera_fps, show_quad, window_width, window_height,
                    return_to_home, play_sounds,
                    dp_operation, dp_dataset_path, dp_output_path,
                    dp_n_samples, dp_dry_run, dp_report_only, dp_zero_threshold,
                    dp_source_roots_text, dp_merge_repo_id, dp_merge_video_size_mb,
                    dp_video_key, dp_tolerance_ms, dp_ts_report_output, dp_ts_output_format,
                    dp_v2_variant, dp_v2_output_root, dp_v2_suffix,
                    dp_cam_left_eye, dp_cam_right_eye, dp_cam_left_wrist, dp_cam_right_wrist, dp_v2_dry_run,
                    mt_script, mt_phase, mt_dataset_root, mt_output_root,
                    mt_batch_size, mt_steps, mt_eval_freq, mt_save_freq, mt_log_freq,
                    mt_wandb_project, mt_wandb_enable, mt_push_to_hub,
                    mt_policy_chunk_size, mt_policy_n_action_steps, mt_policy_lr, mt_policy_path,
                    mt_load_vlm_weights, mt_freeze_vision_encoder, mt_train_expert_only,
                    mt_hf_endpoint, mt_rename_map_json,
                    mt_pretrained_ckpt, mt_new_dataset,
                )

                import tempfile
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                temp_file = tempfile.gettempdir() + f"/robot_config_{timestamp}.yaml"
                save_yaml(config, temp_file)

                with open(temp_file, 'r') as f:
                    yaml_content = f.read()

                return yaml_content, gr.update(value=f"✅ 配置已导出到: {temp_file}")

            except Exception as e:
                return "", gr.update(value=f"❌ 导出失败: {e}")

        def on_launch_click(
            mode, policy_path, policy_device,
            http_url, robot_id, robot_type, robot_timeout, safety_stats_path,
            warn_on_observation_out_of_range, action_clip_margin_deg, max_relative_target_deg,
            robot_ip, robot_control_mode, robot_vel_ratio, robot_acc_ratio, robot_disable_torque_on_disconnect,
            joint_names_json, cameras_json,
            fps, strategy, inference_type,
            duration, interpolation_multiplier, max_steps,
            use_torch_compile, torch_compile_backend, torch_compile_mode, compile_warmup_inferences,
            show_cameras_inf, rename_map_json,
            execution_horizon, max_guidance_weight,
            n_action_steps, chunk_interval_s,
            repo_id_deploy, dataset_root_deploy, episode, single_task, dataset_fps_deploy,
            repo_id_replay, dataset_root_replay, dataset_fps,
            camera_list, camera_fps, show_quad, window_width, window_height,
            return_to_home, play_sounds,
            dp_operation, dp_dataset_path, dp_output_path,
            dp_n_samples, dp_dry_run, dp_report_only, dp_zero_threshold,
            dp_source_roots_text, dp_merge_repo_id, dp_merge_video_size_mb,
            dp_video_key, dp_tolerance_ms, dp_ts_report_output, dp_ts_output_format,
            dp_v2_variant, dp_v2_output_root, dp_v2_suffix,
            dp_cam_left_eye, dp_cam_right_eye, dp_cam_left_wrist, dp_cam_right_wrist, dp_v2_dry_run,
            mt_script, mt_phase, mt_dataset_root, mt_output_root,
            mt_batch_size, mt_steps, mt_eval_freq, mt_save_freq, mt_log_freq,
            mt_wandb_project, mt_wandb_enable, mt_push_to_hub,
            mt_policy_chunk_size, mt_policy_n_action_steps, mt_policy_lr, mt_policy_path,
            mt_load_vlm_weights, mt_freeze_vision_encoder, mt_train_expert_only,
            mt_hf_endpoint, mt_rename_map_json,
            mt_pretrained_ckpt, mt_new_dataset,
        ):
            """验证配置并启动进程"""
            try:
                repo_id = repo_id_deploy if mode == "部署" else repo_id_replay
                dataset_root = dataset_root_deploy if mode == "部署" else dataset_root_replay
                fps_value = dataset_fps_deploy if mode == "部署" else dataset_fps

                config = build_config_from_ui(
                    mode, policy_path, policy_device,
                    http_url, robot_id, robot_type, robot_timeout, safety_stats_path,
                    warn_on_observation_out_of_range, action_clip_margin_deg, max_relative_target_deg,
                    robot_ip, robot_control_mode, robot_vel_ratio, robot_acc_ratio, robot_disable_torque_on_disconnect,
                    joint_names_json, cameras_json,
                    fps, strategy, inference_type,
                    duration, interpolation_multiplier, max_steps,
                    use_torch_compile, torch_compile_backend, torch_compile_mode, compile_warmup_inferences,
                    show_cameras_inf, rename_map_json,
                    execution_horizon, max_guidance_weight,
                    n_action_steps, chunk_interval_s,
                    repo_id, dataset_root, episode, single_task, fps_value,
                    camera_list, camera_fps, show_quad, window_width, window_height,
                    return_to_home, play_sounds,
                    dp_operation, dp_dataset_path, dp_output_path,
                    dp_n_samples, dp_dry_run, dp_report_only, dp_zero_threshold,
                    dp_source_roots_text, dp_merge_repo_id, dp_merge_video_size_mb,
                    dp_video_key, dp_tolerance_ms, dp_ts_report_output, dp_ts_output_format,
                    dp_v2_variant, dp_v2_output_root, dp_v2_suffix,
                    dp_cam_left_eye, dp_cam_right_eye, dp_cam_left_wrist, dp_cam_right_wrist, dp_v2_dry_run,
                    mt_script, mt_phase, mt_dataset_root, mt_output_root,
                    mt_batch_size, mt_steps, mt_eval_freq, mt_save_freq, mt_log_freq,
                    mt_wandb_project, mt_wandb_enable, mt_push_to_hub,
                    mt_policy_chunk_size, mt_policy_n_action_steps, mt_policy_lr, mt_policy_path,
                    mt_load_vlm_weights, mt_freeze_vision_encoder, mt_train_expert_only,
                    mt_hf_endpoint, mt_rename_map_json,
                    mt_pretrained_ckpt, mt_new_dataset,
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
                elif mode == "数据处理":
                    success, msg = pm.launch_data_processing(config)
                elif mode == "模型训练":
                    success, msg = pm.launch_model_training(config)
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

        def refresh_status():
            """检测进程完成事件,更新状态文本与启动/停止键可用性.

            每 0.5s 由 timer.tick 调用:
              - 读取 pm.get_changes() 获取 newly-finished notices 与当前 running 状态
              - 如果有新的完成事件,把 notice 列表追加到 status_text (以 \\n 分隔)
              - 根据是否有进程在跑, 切换 launch_btn / stop_btn 可用性
            """
            changes = pm.get_changes()
            notices = changes["notices"]
            any_running = changes["any_running"]
            statuses = changes["statuses"]

            status_updates = []
            if notices:
                # 已有 status 末尾追加; 没有就初始化
                status_updates.append("\n".join(notices))
            elif statuses:
                # 当前没有新增事件,但显示活跃进程的状态
                running_lines = [
                    f"[{name}] {st}" for name, st in statuses.items() if "运行中" in st
                ]
                status_updates.append("\n".join(running_lines) if running_lines else "就绪")

            return {
                status_text: gr.update(value="\n".join(status_updates)) if status_updates else gr.update(),
                launch_btn: gr.update(interactive=not any_running),
                stop_btn: gr.update(interactive=any_running),
            }

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
                dp_panel.panel,
                mt_panel.panel,
                preset_dropdown,    # mode 切换 → 重新列预设
                save_preset_name,   # mode 切换 → 更新保存路径提示
            ],
        )

        inference_type.change(
            on_inference_type_change,
            inputs=[inference_type],
            outputs=[rtc_group, chunk_group],
        )

        robot_type.change(
            on_robot_type_change,
            inputs=[robot_type],
            outputs=[hybrid_group],
        )

        use_torch_compile.change(
            on_use_torch_compile_change,
            inputs=[use_torch_compile],
            outputs=[torch_compile_group],
        )

        strategy.change(
            on_strategy_change,
            inputs=[strategy],
            outputs=[deploy_dataset_panel],
        )

        # 新模式动态子面板切换
        dp_panel.operation.change(
            on_data_operation_change,
            inputs=[dp_panel.operation],
            outputs=[
                dp_panel.sanity_group,
                dp_panel.clean_group,
                dp_panel.merge_group,
                dp_panel.ts_check_group,
                dp_panel.v2_convert_group,
            ],
        )

        # 数据集下拉刷新按钮 — 重新扫描 datasets/
        dp_panel.dataset_path_refresh_btn.click(
            lambda: gr.update(choices=_list_datasets_full()),
            inputs=[],
            outputs=[dp_panel.dataset_path],
        )

        mt_panel.script.change(
            on_training_script_change,
            inputs=[mt_panel.script],
            outputs=[
                mt_panel.opt_group,
                mt_panel.trk_group,
                mt_panel.smolvla_group,
                mt_panel.finetune_group,
            ],
        )

        # 预设加载 (按当前 mode 决定预设子目录)
        preset_dropdown.change(
            on_load_preset,
            inputs=[preset_dropdown, mode],
            outputs=[
                policy_path, policy_device,
                http_url, robot_id, robot_type, robot_timeout, safety_stats_path,
                warn_on_observation_out_of_range, action_clip_margin_deg, max_relative_target_deg,
                robot_ip, robot_control_mode, robot_vel_ratio, robot_acc_ratio, robot_disable_torque_on_disconnect,
                joint_names_json, cameras_json,
                fps, strategy, inference_type,
                duration, interpolation_multiplier, max_steps,
                use_torch_compile, torch_compile_backend, torch_compile_mode, compile_warmup_inferences,
                show_cameras_inf, rename_map_json,
                execution_horizon, max_guidance_weight,
                n_action_steps, chunk_interval_s,
                repo_id_deploy, repo_id_replay,
                dataset_root_deploy, dataset_root_replay,
                single_task, episode,
                dataset_fps_deploy, dataset_fps,
                return_to_home, play_sounds,
                camera_list, camera_fps, show_quad,
                status_text,
            ],
        )

        # 收集所有输入用于启动（顺序必须和 build_config_from_ui 签名一一对应）
        launch_inputs = [
            mode,
            policy_path, policy_device,
            http_url, robot_id, robot_type, robot_timeout, safety_stats_path,
            warn_on_observation_out_of_range, action_clip_margin_deg, max_relative_target_deg,
            robot_ip, robot_control_mode, robot_vel_ratio, robot_acc_ratio, robot_disable_torque_on_disconnect,
            joint_names_json, cameras_json,
            fps, strategy, inference_type,
            duration, interpolation_multiplier, max_steps,
            use_torch_compile, torch_compile_backend, torch_compile_mode, compile_warmup_inferences,
            show_cameras_inf, rename_map_json,
            execution_horizon, max_guidance_weight,
            n_action_steps, chunk_interval_s,
            repo_id_deploy, dataset_root_deploy, episode, single_task, dataset_fps_deploy,
            repo_id_replay, dataset_root_replay, dataset_fps,
            camera_list, camera_fps, show_quad, window_width, window_height,
            return_to_home, play_sounds,
            # Data processing (顺序必须与 build_config_from_ui 的 dp_* 参数一致)
            dp_panel.operation, dp_panel.dataset_path, dp_panel.output_path,
            dp_panel.n_samples, dp_panel.dry_run, dp_panel.report_only, dp_panel.zero_threshold,
            dp_panel.source_roots_text, dp_panel.merge_repo_id, dp_panel.merge_video_size_mb,
            dp_panel.video_key, dp_panel.tolerance_ms, dp_panel.ts_report_output, dp_panel.ts_output_format,
            dp_panel.v2_variant, dp_panel.v2_output_root, dp_panel.v2_suffix,
            dp_panel.cam_left_eye, dp_panel.cam_right_eye, dp_panel.cam_left_wrist, dp_panel.cam_right_wrist, dp_panel.v2_dry_run,
            # Model training (顺序必须与 build_config_from_ui 的 mt_* 参数一致)
            mt_panel.script, mt_panel.phase, mt_panel.dataset_root, mt_panel.output_root,
            mt_panel.batch_size, mt_panel.steps, mt_panel.eval_freq, mt_panel.save_freq, mt_panel.log_freq,
            mt_panel.wandb_project, mt_panel.wandb_enable, mt_panel.push_to_hub,
            mt_panel.policy_chunk_size, mt_panel.policy_n_action_steps, mt_panel.policy_lr, mt_panel.policy_path,
            mt_panel.load_vlm_weights, mt_panel.freeze_vision_encoder, mt_panel.train_expert_only,
            mt_panel.hf_endpoint, mt_panel.rename_map_json,
            mt_panel.pretrained_ckpt, mt_panel.new_dataset,
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

        # 每 0.5 秒自动刷新日志 + 检测完成事件
        # (1) refresh_logs: 刷新日志文本 (tail)
        # (2) refresh_status: 检测进程完成/失败, 追加到 status_text 并切换 launch/stop 按钮可用性
        timer = gr.Timer(value=0.5, active=True)
        timer.tick(
            fn=refresh_logs,
            outputs=[log_output],
        ).then(
            fn=refresh_status,
            outputs=[status_text, launch_btn, stop_btn],
        )

        # 训练数据集下拉刷新按钮 — 重新扫描 datasets/
        mt_panel.dataset_root_refresh_btn.click(
            lambda: gr.update(choices=_list_datasets_full()),
            inputs=[],
            outputs=[mt_panel.dataset_root],
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
