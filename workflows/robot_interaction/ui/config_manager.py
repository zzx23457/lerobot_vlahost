"""Unified configuration manager for robot interaction workflows.

This module provides a mode-discriminated config schema that unifies
deploy, replay, camera preview, data processing, and model training workflows.
"""

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Literal, Any
import yaml


# ============================================================================
# Mode constants (internal values are stable; UI displays Chinese labels)
# ============================================================================
# UI 显示 → 内部值。避免对中文做 substring 匹配导致 schema 漂移。
MODE_TO_KEY: dict[str, str] = {
    "部署": "deploy",
    "回放": "replay",
    "相机预览": "camera_preview",
    "数据处理": "data_processing",
    "模型训练": "model_training",
}
KEY_TO_MODE: dict[str, str] = {v: k for k, v in MODE_TO_KEY.items()}

# 涉及机器人硬件的模式 — 这些模式必须保留 robot / runtime / policy / inference。
ROBOT_MODES = ("deploy", "replay", "camera_preview")
DATA_PROCESSING_OPERATIONS = ("sanity", "clean", "merge", "ts_check", "v2_convert")
TRAINING_SCRIPTS = ("act", "smolvla", "finetune")
TRAINING_PHASES = ("all", "env", "check", "smoke", "train", "eval")


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class PolicyConfig:
    """Policy configuration (deploy mode only)"""
    path: str = ""
    device: Literal["cuda", "cpu"] = "cuda"


@dataclass
class CameraConfig:
    """Single camera configuration"""
    type: str = "http"
    fps: int = 30
    width: int = 640
    height: int = 480
    color_mode: str = "rgb"


@dataclass
class RobotConfig:
    """Robot hardware configuration (all modes)"""
    http_base_url: str = "http://192.168.10.123:8010"
    # Field name MUST be `id` to match deploy.py / replay.py / show_cameras.py contract
    # (they read config['robot']['id']). Renaming from `robot_id` fixes a KeyError.
    id: str = "marvain_m6_01"
    type: str = "marvain_m6_http"  # "marvain_m6_http" or "marvain_m6_hybrid"
    timeout: float = 5.0
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {
        "right_eye": CameraConfig(),
        "left_eye": CameraConfig(),
        "left_wrist": CameraConfig(),
        "right_wrist": CameraConfig(),
    })
    safety_stats_path: str | None = None
    action_clip_margin_deg: float = 5.0
    max_relative_target_deg: float = 10.0
    # Informational: rollout may warn when observation stats exceed training range.
    # deploy.py 不消费，存着是为了 YAML round-trip 不丢字段。
    warn_on_observation_out_of_range: bool = True
    joint_names: list[str] = field(default_factory=lambda: [
        "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3", "left_arm_joint_4",
        "left_arm_joint_5", "left_arm_joint_6", "left_arm_joint_7",
        "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3", "right_arm_joint_4",
        "right_arm_joint_5", "right_arm_joint_6", "right_arm_joint_7",
        "left_gripper", "right_gripper",
    ])
    # —— Hybrid 机器人（marvain_m6_hybrid）专属 ——
    # deploy.py 在 robot.type == "marvain_m6_hybrid" 时读这些字段（deploy.py:347-355），
    # 缺一个会直接 KeyError。HTTP 模式下全部忽略。
    ip: str = "192.168.10.190"
    control_mode: str = "impedance"  # "impedance" | "position"
    vel_ratio: int = 20
    acc_ratio: int = 20
    disable_torque_on_disconnect: bool = False


@dataclass
class RTCConfig:
    """Real-Time Chunking configuration"""
    execution_horizon: int | None = None
    max_guidance_weight: float = 10.0


@dataclass
class InferenceConfig:
    """Inference configuration (deploy mode only)"""
    type: Literal["sync", "rtc", "chunk"] = "sync"
    strategy: Literal["base", "sentry", "highlight", "dagger", "episodic"] = "base"
    fps: float = 30.0
    duration: float = 0.0  # 0 = infinite
    max_steps: int = 10000
    interpolation_multiplier: int = 1
    use_torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "default"
    compile_warmup_inferences: int = 2
    show_cameras: bool = False
    rtc: RTCConfig = field(default_factory=RTCConfig)
    # chunk 模式专属（open-loop：每 chunk_interval_s 推理一次，把前
    # n_action_steps 个 action 一次性发给机器人）。deploy.py:411-417
    # 把这两个拼成 --inference.n_action_steps / --inference.chunk_interval_s。
    n_action_steps: int | None = None
    chunk_interval_s: float | None = None
    rename_map: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Dataset configuration (conditional)"""
    repo_id: str = ""
    root: str | None = None
    episode: int | None = None
    single_task: str | None = None
    fps: float = 30.0


@dataclass
class RuntimeConfig:
    """Runtime behavior configuration (all modes)"""
    return_to_initial_position: bool = True
    play_sounds: bool = True
    # Camera preview specific (for camera_preview mode)
    camera_list: list[str] = field(default_factory=lambda: ["right_eye", "left_eye", "left_wrist", "right_wrist"])
    camera_fps: float = 30.0
    show_quad: bool = True
    window_width: int = 640
    window_height: int = 480


# ----------------------------------------------------------------------------
# Data processing configs (data_processing mode)
# ----------------------------------------------------------------------------

@dataclass
class SanityCheckConfig:
    """Args for sanity_check.py"""
    n_samples: int = 5


@dataclass
class CleanDatasetConfig:
    """Args for clean_dirty_episodes.py"""
    dry_run: bool = True  # UI 默认安全
    report_only: bool = False
    zero_threshold: int = 7


@dataclass
class TimestampCheckConfig:
    """Args for check_timestamp_alignment.py"""
    video_key: str = ""
    tolerance_ms: float = 1.0
    report_output: str = ""
    output_format: Literal["text", "json"] = "text"


@dataclass
class MergeDatasetsConfig:
    """Args for merge_two_datasets.py"""
    source_roots_text: str = ""  # 多行文本，每行一个 path
    repo_id: str = ""
    video_files_size_mb: float = 0.001


@dataclass
class V2ConvertConfig:
    """Args for v2_convert.py"""
    variant: Literal["standard", "next_joint"] = "standard"
    output_root: str = ""  # 为空时由 v2_suffix 自动推导
    v2_suffix: str = "_v2"
    # 4 个相机开关；key 与 RobotConfig.cameras 一致
    camera_enabled: dict[str, bool] = field(default_factory=lambda: {
        "left_eye": True,
        "right_eye": True,
        "left_wrist": True,
        "right_wrist": True,
    })
    dry_run: bool = True


@dataclass
class DataProcessingConfig:
    """Top-level config for data_processing mode"""
    operation: Literal["sanity", "clean", "merge", "ts_check", "v2_convert"] = "sanity"
    dataset_path: str = ""
    output_path: str = ""  # clean / merge / v2_convert 有效；sanity 不需要
    sanity: SanityCheckConfig = field(default_factory=SanityCheckConfig)
    clean: CleanDatasetConfig = field(default_factory=CleanDatasetConfig)
    timestamp: TimestampCheckConfig = field(default_factory=TimestampCheckConfig)
    merge: MergeDatasetsConfig = field(default_factory=MergeDatasetsConfig)
    v2_convert: V2ConvertConfig = field(default_factory=V2ConvertConfig)


# ----------------------------------------------------------------------------
# Model training configs (model_training mode)
# ----------------------------------------------------------------------------

@dataclass
class TrainingOptimizationConfig:
    """Common optimization params for ACT / SmolVLA training"""
    batch_size: int = 8
    steps: int = 400000
    eval_freq: int = 20000
    save_freq: int = 20000
    log_freq: int = 50


@dataclass
class TrainingTrackingConfig:
    """Wandb / Hub tracking"""
    wandb_project: str = ""
    wandb_enable: bool = False
    push_to_hub: bool = False


@dataclass
class SmolVLAConfig:
    """SmolVLA-specific env vars"""
    policy_chunk_size: int = 50
    policy_n_action_steps: int = 50
    policy_lr: float = 1e-4
    policy_path: str = ""  # HF repo or local path
    load_vlm_weights: bool = False
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    hf_endpoint: str = ""  # e.g. https://hf-mirror.com
    # JSON 字符串文本，存储为 dict[str, str]；空表示不传该 env
    rename_map: dict[str, str] = field(default_factory=dict)


@dataclass
class FineTuneConfig:
    """ACT fine-tune specific env vars"""
    pretrained_ckpt: str = ""  # PRETRAINED_CKPT
    new_dataset: str = ""  # NEW_DATASET


@dataclass
class ModelTrainingConfig:
    """Top-level config for model_training mode"""
    script: Literal["act", "smolvla", "finetune"] = "act"
    phase: Literal["all", "env", "check", "smoke", "train", "eval"] = "smoke"
    dataset_root: str = ""
    output_root: str = ""
    optimization: TrainingOptimizationConfig = field(default_factory=TrainingOptimizationConfig)
    tracking: TrainingTrackingConfig = field(default_factory=TrainingTrackingConfig)
    smolvla: SmolVLAConfig = field(default_factory=SmolVLAConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)


@dataclass
class UnifiedRobotConfig:
    """Unified configuration for all modes"""
    mode: Literal[
        "deploy", "replay", "camera_preview", "data_processing", "model_training"
    ] = "deploy"
    robot: RobotConfig = field(default_factory=RobotConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    policy: PolicyConfig | None = field(default_factory=PolicyConfig)
    inference: InferenceConfig | None = field(default_factory=InferenceConfig)
    dataset: DatasetConfig | None = field(default_factory=DatasetConfig)
    data_processing: DataProcessingConfig | None = field(default_factory=DataProcessingConfig)
    model_training: ModelTrainingConfig | None = field(default_factory=ModelTrainingConfig)


# ============================================================================
# Validation Functions
# ============================================================================

def _validate_http_url(url: str) -> bool:
    """Validate HTTP URL format"""
    if not url:
        return False
    return url.startswith("http://") or url.startswith("https://")


def _is_vla_model(policy_path: str) -> bool:
    """Detect if a model is a VLA model by checking its config"""
    if not policy_path:
        return False

    config_path = Path(policy_path) / "config.json"
    if not config_path.exists():
        return False

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Check for VLA-specific fields
        model_type = config.get("model_type", "").lower()
        return "vla" in model_type or "smolvla" in model_type
    except Exception:
        return False


def _validate_path_exists(path: str, kind: str) -> str | None:
    """Return error message if path is empty / missing / not a directory."""
    if not path:
        return f"{kind} 路径不能为空"
    p = Path(path)
    if not p.exists():
        return f"{kind} 路径不存在: {path}"
    if not p.is_dir():
        return f"{kind} 不是目录: {path}"
    return None


def _validate_data_processing(config: DataProcessingConfig) -> list[str]:
    """Validate data_processing config."""
    errors: list[str] = []
    op = config.operation

    if op == "sanity":
        if err := _validate_path_exists(config.dataset_path, "数据集"):
            errors.append(err)
        if config.sanity.n_samples <= 0:
            errors.append("n_samples 必须为正整数")

    elif op == "clean":
        if err := _validate_path_exists(config.dataset_path, "数据集"):
            errors.append(err)
        # report_only / dry_run 都允许; 但实际写盘要求 output_path 非空且 != input
        if not config.clean.report_only and not config.clean.dry_run:
            if not config.output_path:
                errors.append("实际清洗模式必须指定输出路径")
            elif Path(config.output_path).resolve() == Path(config.dataset_path).resolve():
                errors.append("clean 的输入和输出不能是同一路径")
        if config.clean.zero_threshold < 0:
            errors.append("zero_threshold 不能为负数")

    elif op == "merge":
        # 解析多行 sources
        sources = [
            line.strip()
            for line in (config.merge.source_roots_text or "").splitlines()
            if line.strip()
        ]
        if len(sources) < 2:
            errors.append("merge 至少需要 2 个 source 数据集路径")
        else:
            for i, s in enumerate(sources):
                if err := _validate_path_exists(s, f"source[{i}]"):
                    errors.append(err)
        if not config.output_path:
            errors.append("merge 必须指定输出路径")
        else:
            out_p = Path(config.output_path).resolve()
            for s in sources:
                if out_p == Path(s).resolve():
                    errors.append(f"merge 输出不能与 source 相同: {s}")
            if out_p.exists():
                errors.append(f"merge 输出目录已存在，拒绝覆盖: {out_p}")
        if not config.merge.repo_id:
            errors.append("merge 必须指定 repo_id")
        if config.merge.video_files_size_mb <= 0:
            errors.append("video_files_size_mb 必须为正数")

    elif op == "ts_check":
        if err := _validate_path_exists(config.dataset_path, "数据集"):
            errors.append(err)
        if config.timestamp.tolerance_ms < 0:
            errors.append("tolerance_ms 不能为负数")
        if config.timestamp.video_key:
            # 不强校验文件存在，只在执行时报错；这里仅做格式提示
            pass
        if config.timestamp.output_format not in ("text", "json"):
            errors.append(f"output_format 必须是 text 或 json, 得到 {config.timestamp.output_format!r}")

    elif op == "v2_convert":
        if err := _validate_path_exists(config.dataset_path, "v1 数据集"):
            errors.append(err)
        # output_root 可选（未提供则用 v2_suffix 推导）；若提供则必须 != input
        if config.v2_convert.output_root:
            if Path(config.v2_convert.output_root).resolve() == Path(config.dataset_path).resolve():
                errors.append("v2_convert 的输入和输出不能是同一路径")
            if Path(config.v2_convert.output_root).exists():
                errors.append(f"v2_convert 输出目录已存在，拒绝覆盖: {config.v2_convert.output_root}")
        cam_enabled = config.v2_convert.camera_enabled
        # 至少 1 个相机启用
        if not any(cam_enabled.values()):
            errors.append("v2_convert 至少要保留 1 个相机")
        # key 必须落在已知相机集合
        for k in cam_enabled:
            if k not in ("left_eye", "right_eye", "left_wrist", "right_wrist"):
                errors.append(f"v2_convert.camera_enabled 包含未知相机 key: {k}")

    else:
        errors.append(f"未知 operation: {op}")

    return errors


def _validate_model_training(config: ModelTrainingConfig) -> list[str]:
    """Validate model_training config."""
    errors: list[str] = []
    script = config.script
    phase = config.phase

    # dataset_root: env / check 阶段可放宽；其它阶段要求存在
    if phase not in ("env", "eval") and script != "finetune":
        if err := _validate_path_exists(config.dataset_root, "数据集"):
            errors.append(err)

    # output_root: 真正训练的 phase 要求非空
    if phase in ("all", "train", "smoke"):
        if not config.output_root:
            errors.append("output_root 不能为空（训练会写入 checkpoint）")

    # 数值字段校验
    opt = config.optimization
    for label, value in [
        ("batch_size", opt.batch_size),
        ("steps", opt.steps),
        ("eval_freq", opt.eval_freq),
        ("save_freq", opt.save_freq),
        ("log_freq", opt.log_freq),
    ]:
        if value is None or value <= 0:
            errors.append(f"{label} 必须为正整数")

    if script == "smolvla":
        sm = config.smolvla
        if sm.policy_chunk_size <= 0:
            errors.append("policy_chunk_size 必须为正整数")
        if sm.policy_n_action_steps <= 0:
            errors.append("policy_n_action_steps 必须为正整数")
        if sm.policy_lr <= 0:
            errors.append("policy_lr 必须为正数")
        # rename_map 必须是 dict[str, str]
        for k, v in (sm.rename_map or {}).items():
            if not isinstance(k, str) or not isinstance(v, str):
                errors.append("smolvla.rename_map 必须是 {str: str}")
                break

    elif script == "finetune":
        ft = config.finetune
        if not ft.pretrained_ckpt:
            errors.append("finetune 必须指定 pretrained_ckpt (PRETRAINED_CKPT)")
        elif not Path(ft.pretrained_ckpt).exists():
            errors.append(f"pretrained_ckpt 不存在: {ft.pretrained_ckpt}")
        elif not (Path(ft.pretrained_ckpt) / "config.json").exists():
            errors.append(f"pretrained_ckpt 缺少 config.json: {ft.pretrained_ckpt}")
        if not ft.new_dataset:
            errors.append("finetune 必须指定 new_dataset (NEW_DATASET)")
        elif not Path(ft.new_dataset).exists():
            errors.append(f"new_dataset 不存在: {ft.new_dataset}")
        # fine-tune 阶段不需要 optimization，但要求 pretrained 维度匹配 (脚本内会校验)

    return errors


def validate(config: UnifiedRobotConfig) -> list[str]:
    """Validate configuration and return list of error messages."""
    errors: list[str] = []

    # Mode-specific required fields
    if config.mode == "deploy":
        if not config.policy or not config.policy.path:
            errors.append("Policy configuration with path required for deploy mode")
        if not config.inference:
            errors.append("Inference configuration required for deploy mode")

        # VLA model detection
        if config.policy and config.policy.path and _is_vla_model(config.policy.path):
            if not config.dataset or not config.dataset.single_task:
                errors.append("VLA models require dataset.single_task description")

        # Check policy path exists
        if config.policy and config.policy.path:
            policy_path = Path(config.policy.path)
            if not policy_path.exists():
                errors.append(f"Policy path does not exist: {config.policy.path}")
            elif not (policy_path / "config.json").exists():
                # 检查是否是训练输出目录（包含 checkpoints 子目录）
                if (policy_path / "checkpoints").exists():
                    errors.append(f"Policy path seems to be a training directory. Please select a checkpoint, e.g., {config.policy.path}/checkpoints/XXXXX/pretrained_model")
                else:
                    errors.append(f"No config.json found in policy path: {config.policy.path}")

    elif config.mode == "replay":
        if not config.dataset or not config.dataset.repo_id:
            errors.append("Dataset repo_id required for replay mode")
        if not config.dataset or config.dataset.episode is None:
            errors.append("Dataset episode index required for replay mode")

    # 只有机器人模式才校验 robot / inference / RTC / strategy / rename_map
    if config.mode in ROBOT_MODES:
        # RTC-specific validation
        if config.inference and config.inference.type == "rtc":
            if not config.inference.rtc or config.inference.rtc.execution_horizon is None:
                errors.append("RTC mode requires execution_horizon parameter")

        # Strategy-specific validation
        if config.inference and config.inference.strategy != "base":
            if not config.dataset or not config.dataset.repo_id:
                errors.append(f"Strategy '{config.inference.strategy}' requires dataset.repo_id for recording")

        # Robot connectivity
        if not _validate_http_url(config.robot.http_base_url):
            errors.append(f"Invalid HTTP URL: {config.robot.http_base_url}")

        # Camera mapping validation
        if config.inference and config.inference.rename_map:
            physical_cameras = set(config.robot.cameras.keys())
            for physical, model in config.inference.rename_map.items():
                if physical not in physical_cameras:
                    errors.append(f"rename_map key '{physical}' not in robot.cameras")

    # 数据处理模式
    if config.mode == "data_processing":
        if not config.data_processing:
            errors.append("data_processing 配置缺失")
        else:
            errors.extend(_validate_data_processing(config.data_processing))

    # 模型训练模式
    if config.mode == "model_training":
        if not config.model_training:
            errors.append("model_training 配置缺失")
        else:
            errors.extend(_validate_model_training(config.model_training))

    return errors


# ============================================================================
# YAML Serialization
# ============================================================================

def _dataclass_to_dict(obj):
    """Convert dataclass to dict, handling nested dataclasses and None values"""
    if obj is None:
        return None

    result = {}
    for k, v in asdict(obj).items():
        if isinstance(v, dict):
            # Handle nested CameraConfig dict
            if k == "cameras":
                result[k] = {name: dict(cam) for name, cam in v.items()}
            else:
                result[k] = v
        else:
            result[k] = v

    return result


def _filter_dataclass_fields(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields declared on the dataclass.

    之前是静默丢弃未知字段，导致 hybrid yaml / 加新字段后 UI 加载会丢数据。
    现在保留同样的过滤行为（dataclass 构造不接受未知 kwarg），但对被丢弃的
    字段打 warning，方便发现 schema 漂移。
    """
    if not data:
        return {}
    valid = {f.name for f in fields(cls)}
    unknown = [k for k in data if k not in valid]
    for k in unknown:
        import warnings
        warnings.warn(
            f"[{cls.__name__}] YAML 字段 '{k}' 不在 dataclass 里，已被忽略。"
            f" 有效字段: {sorted(valid)}",
            stacklevel=2,
        )
    return {k: v for k, v in data.items() if k in valid}


def _build_data_processing_dict(config: DataProcessingConfig) -> dict[str, Any]:
    """Serialize a DataProcessingConfig to a flat dict for YAML."""
    out: dict[str, Any] = {
        "operation": config.operation,
        "dataset_path": config.dataset_path,
        "output_path": config.output_path,
    }
    if config.operation == "sanity":
        out["sanity"] = _dataclass_to_dict(config.sanity)
    elif config.operation == "clean":
        out["clean"] = _dataclass_to_dict(config.clean)
    elif config.operation == "ts_check":
        out["timestamp"] = _dataclass_to_dict(config.timestamp)
    elif config.operation == "merge":
        out["merge"] = _dataclass_to_dict(config.merge)
    elif config.operation == "v2_convert":
        out["v2_convert"] = _dataclass_to_dict(config.v2_convert)
    return out


def _build_model_training_dict(config: ModelTrainingConfig) -> dict[str, Any]:
    """Serialize a ModelTrainingConfig to a flat dict for YAML."""
    return {
        "script": config.script,
        "phase": config.phase,
        "dataset_root": config.dataset_root,
        "output_root": config.output_root,
        "optimization": _dataclass_to_dict(config.optimization),
        "tracking": _dataclass_to_dict(config.tracking),
        "smolvla": _dataclass_to_dict(config.smolvla),
        "finetune": _dataclass_to_dict(config.finetune),
    }


def _load_data_processing_dict(data: dict[str, Any]) -> DataProcessingConfig:
    """Build a DataProcessingConfig from a YAML dict, restoring nested dataclasses."""
    base = _filter_dataclass_fields(DataProcessingConfig, {
        k: v for k, v in data.items() if k in {"operation", "dataset_path", "output_path"}
    })
    cfg = DataProcessingConfig(**base)

    sanity_data = _filter_dataclass_fields(SanityCheckConfig, data.get("sanity", {}) or {})
    if sanity_data:
        cfg.sanity = SanityCheckConfig(**sanity_data)

    clean_data = _filter_dataclass_fields(CleanDatasetConfig, data.get("clean", {}) or {})
    if clean_data:
        cfg.clean = CleanDatasetConfig(**clean_data)

    ts_data = _filter_dataclass_fields(TimestampCheckConfig, data.get("timestamp", {}) or {})
    if ts_data:
        cfg.timestamp = TimestampCheckConfig(**ts_data)

    merge_data = _filter_dataclass_fields(MergeDatasetsConfig, data.get("merge", {}) or {})
    if merge_data:
        cfg.merge = MergeDatasetsConfig(**merge_data)

    v2_data = _filter_dataclass_fields(V2ConvertConfig, data.get("v2_convert", {}) or {})
    if v2_data:
        cfg.v2_convert = V2ConvertConfig(**v2_data)

    return cfg


def _load_model_training_dict(data: dict[str, Any]) -> ModelTrainingConfig:
    """Build a ModelTrainingConfig from a YAML dict."""
    base = _filter_dataclass_fields(ModelTrainingConfig, {
        k: v for k, v in data.items()
        if k in {"script", "phase", "dataset_root", "output_root"}
    })
    cfg = ModelTrainingConfig(**base)

    opt_data = _filter_dataclass_fields(TrainingOptimizationConfig, data.get("optimization", {}) or {})
    if opt_data:
        cfg.optimization = TrainingOptimizationConfig(**opt_data)

    track_data = _filter_dataclass_fields(TrainingTrackingConfig, data.get("tracking", {}) or {})
    if track_data:
        cfg.tracking = TrainingTrackingConfig(**track_data)

    sm_data = _filter_dataclass_fields(SmolVLAConfig, data.get("smolvla", {}) or {})
    if sm_data:
        cfg.smolvla = SmolVLAConfig(**sm_data)

    ft_data = _filter_dataclass_fields(FineTuneConfig, data.get("finetune", {}) or {})
    if ft_data:
        cfg.finetune = FineTuneConfig(**ft_data)

    return cfg


def save_yaml(config: UnifiedRobotConfig, filepath: Path | str) -> None:
    """Save config to YAML in the schema expected by deploy.py / replay.py / etc.

    Mode-aware:
      - deploy: writes `inference.return_to_initial_position` (from RuntimeConfig)
      - replay: writes root-level `play_sounds` and `return_to_initial_position`
      - camera_preview: only writes robot + policy (camera runtime params go via CLI)
      - data_processing: writes `data_processing` block only
      - model_training: writes `model_training` block only
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    config_dict: dict[str, Any] = {"mode": config.mode}

    if config.mode == "deploy":
        config_dict["robot"] = _dataclass_to_dict(config.robot)
        if config.policy:
            config_dict["policy"] = _dataclass_to_dict(config.policy)
        if config.inference:
            inference_dict = _dataclass_to_dict(config.inference)
            # deploy.py reads inference.return_to_initial_position (deploy.py:392)
            inference_dict["return_to_initial_position"] = (
                config.runtime.return_to_initial_position
            )
            config_dict["inference"] = inference_dict
        if config.dataset:
            config_dict["dataset"] = _dataclass_to_dict(config.dataset)

    elif config.mode == "replay":
        config_dict["robot"] = _dataclass_to_dict(config.robot)
        if config.dataset:
            config_dict["dataset"] = _dataclass_to_dict(config.dataset)
        # replay.py expects root-level play_sounds / return_to_initial_position
        config_dict["play_sounds"] = config.runtime.play_sounds
        config_dict["return_to_initial_position"] = config.runtime.return_to_initial_position

    elif config.mode == "camera_preview":
        config_dict["robot"] = _dataclass_to_dict(config.robot)
        # show_cameras.py reads http_base_url from yaml.robot and policy.path.
        # Always emit the policy block (even with empty path) to keep the schema
        # consistent; show_cameras.py will still validate and error if path is "".
        config_dict["policy"] = {
            "path": config.policy.path if config.policy else "",
            "device": config.policy.device if config.policy else "cuda",
        }

    elif config.mode == "data_processing":
        if config.data_processing is None:
            raise ValueError("data_processing config missing for data_processing mode")
        config_dict["data_processing"] = _build_data_processing_dict(config.data_processing)

    elif config.mode == "model_training":
        if config.model_training is None:
            raise ValueError("model_training config missing for model_training mode")
        config_dict["model_training"] = _build_model_training_dict(config.model_training)

    with open(filepath, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)


def load_yaml(filepath: Path | str) -> UnifiedRobotConfig:
    """Load config from YAML. Mode-aware + tolerant of unknown fields.

    Returns a UnifiedRobotConfig whose runtime / inference / policy fields are
    populated according to the file's `mode` key (default: deploy).
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, "r") as f:
        data = yaml.safe_load(f) or {}

    mode = data.get("mode", "deploy")

    # Robot (tolerant of hybrid-only fields like robot.ip / control_mode)
    robot_data = _filter_dataclass_fields(RobotConfig, data.get("robot", {}) or {})
    cameras_data = robot_data.get("cameras", {}) or {}
    cameras = {
        name: CameraConfig(**_filter_dataclass_fields(CameraConfig, cam or {}))
        for name, cam in cameras_data.items()
    }
    robot_data["cameras"] = cameras
    robot = RobotConfig(**robot_data)

    # Policy
    policy_data = _filter_dataclass_fields(PolicyConfig, data.get("policy") or {})
    policy = PolicyConfig(**policy_data) if policy_data else PolicyConfig()

    # Inference
    inference_data = _filter_dataclass_fields(InferenceConfig, data.get("inference") or {})
    return_to_initial_from_inference: bool | None = None
    if "return_to_initial_position" in (data.get("inference") or {}):
        return_to_initial_from_inference = data["inference"]["return_to_initial_position"]
    if inference_data:
        rtc_raw = inference_data.pop("rtc", None) or {}
        rtc_data = _filter_dataclass_fields(RTCConfig, rtc_raw)
        inference_data["rtc"] = RTCConfig(**rtc_data)
        inference = InferenceConfig(**inference_data)
    else:
        inference = InferenceConfig()

    # Dataset
    dataset_data = _filter_dataclass_fields(DatasetConfig, data.get("dataset") or {})
    dataset = DatasetConfig(**dataset_data) if dataset_data else DatasetConfig()

    # Runtime — populated differently per mode
    if mode == "deploy":
        return_to_initial = (
            return_to_initial_from_inference
            if return_to_initial_from_inference is not None
            else True
        )
        runtime = RuntimeConfig(
            return_to_initial_position=return_to_initial,
            play_sounds=True,
        )
    elif mode == "replay":
        runtime = RuntimeConfig(
            return_to_initial_position=data.get("return_to_initial_position", True),
            play_sounds=data.get("play_sounds", True),
        )
    elif mode == "data_processing":
        runtime = RuntimeConfig()
    elif mode == "model_training":
        runtime = RuntimeConfig()
    else:  # camera_preview or unknown
        runtime = RuntimeConfig()

    # Data processing block (only meaningful for data_processing mode)
    dp_data = data.get("data_processing", {}) or {}
    if dp_data:
        data_processing = _load_data_processing_dict(dp_data)
    else:
        data_processing = DataProcessingConfig()

    # Model training block
    mt_data = data.get("model_training", {}) or {}
    if mt_data:
        model_training = _load_model_training_dict(mt_data)
    else:
        model_training = ModelTrainingConfig()

    return UnifiedRobotConfig(
        mode=mode,
        robot=robot,
        policy=policy,
        inference=inference,
        dataset=dataset,
        runtime=runtime,
        data_processing=data_processing,
        model_training=model_training,
    )


def dump_to_tempfile(config: UnifiedRobotConfig) -> Path:
    """Serialize config to a temp yaml file. Returns the path."""
    import tempfile
    import time

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tmp = Path(tempfile.gettempdir()) / f"robot_config_{timestamp}.yaml"
    save_yaml(config, tmp)
    return tmp


# ============================================================================
# Template Management
# ============================================================================
# Per-mode preset directories (UI 用, 与 robot templates 分离)
# ============================================================================

# UI preset 根目录: <repo>/workflows/robot_interaction/ui/presets/
_PRESETS_BASE = Path(__file__).parent / "presets"

# 每个 kind 单独一个子目录, 互不污染
PRESET_SUBDIRS: dict[str, Path] = {
    "robot":          _PRESETS_BASE / "robot",
    "data_processing": _PRESETS_BASE / "data_processing",
    "model_training":  _PRESETS_BASE / "model_training",
}

# 中文显示名 → preset kind 映射 (与 MODE_TO_KEY 平行)
_MODE_KIND: dict[str, str] = {
    "deploy": "robot",
    "replay": "robot",
    "camera_preview": "robot",
    "data_processing": "data_processing",
    "model_training": "model_training",
}


def preset_kind_for_mode(mode: str) -> str:
    """Return preset subdir kind for a given mode (Chinese or English)."""
    # 先尝试英文 key, 再尝试中文, 最后 fallback 'robot'
    if mode in PRESET_SUBDIRS:
        return mode
    if mode in _MODE_KIND.values():
        return mode
    # 中文 → 英文
    en_mode = MODE_TO_KEY.get(mode, mode)
    return _MODE_KIND.get(en_mode, "robot")


def list_user_presets(kind: str) -> list[str]:
    """List user-saved preset names for a given kind.

    Returns file stems (without .yaml), sorted alphabetically.
    Empty list if subdir doesn't exist.
    """
    subdir = PRESET_SUBDIRS.get(kind)
    if subdir is None or not subdir.exists():
        return []
    return sorted(p.stem for p in subdir.glob("*.yaml"))


def list_all_robot_choices() -> list[str]:
    """Combined list: built-in robot templates + user-saved robot presets.

    The dropdown shows them all together; user presets get a "(预设)" suffix to
    distinguish from built-ins.
    """
    templates = list_templates()  # built-in
    user = list_user_presets("robot")
    return templates + [f"{n} (预设)" for n in user]


def save_user_preset(kind: str, name: str, config: UnifiedRobotConfig) -> Path:
    """Save a user preset to ``ui/presets/<kind>/<name>.yaml``.

    Returns the saved path. Subdir is created on demand.
    """
    if kind not in PRESET_SUBDIRS:
        raise ValueError(f"Unknown preset kind: {kind!r}")
    if not name or not name.strip():
        raise ValueError("preset name must be non-empty")
    subdir = PRESET_SUBDIRS[kind]
    subdir.mkdir(parents=True, exist_ok=True)
    # 防止覆盖模板 (built-in 模板 stem 与 preset 名字相同时)
    safe_name = name.strip()
    if kind == "robot" and safe_name in list_templates():
        # 用户预设可以同名, 实际写到子目录, 加载时优先用户
        pass
    path = subdir / f"{safe_name}.yaml"
    save_yaml(config, str(path))
    return path


def load_user_preset(kind: str, name: str) -> UnifiedRobotConfig:
    """Load a user preset from ``ui/presets/<kind>/<name>.yaml``.

    ``name`` may include the "(预设)" suffix added by list_all_robot_choices.
    """
    if kind not in PRESET_SUBDIRS:
        raise ValueError(f"Unknown preset kind: {kind!r}")
    subdir = PRESET_SUBDIRS[kind]
    # 去掉 "(预设)" 后缀 (robot 模板兼容)
    safe_name = name.replace(" (预设)", "").strip()
    path = subdir / f"{safe_name}.yaml"
    return load_yaml(path)


# ============================================================================
# Built-in robot templates (位于 workflows/robot_interaction/*.yaml)
# ============================================================================

# 默认模板扫描路径: workflows/robot_interaction/ 下的 yaml 文件
_DEFAULT_TEMPLATES_DIR = Path(__file__).parent.parent


def list_templates(templates_dir: Path | str | None = None) -> list[str]:
    """List available config templates from workflows/robot_interaction/*.yaml.

    Returns file stems (without .yaml) sorted alphabetically.
    """
    if templates_dir is None:
        templates_dir = _DEFAULT_TEMPLATES_DIR
    else:
        templates_dir = Path(templates_dir)

    if not templates_dir.exists():
        return []

    templates = []
    for file in sorted(templates_dir.glob("*.yaml")):
        # 跳过 ui/ 自己的 yaml (避免递归列出预设/模板 yaml)
        if "ui/" in str(file) or "presets/" in str(file):
            continue
        templates.append(file.stem)

    return templates


def load_template(template_name: str, templates_dir: Path | str | None = None) -> UnifiedRobotConfig:
    """Load a config template by file stem from workflows/robot_interaction/."""
    if templates_dir is None:
        templates_dir = _DEFAULT_TEMPLATES_DIR
    else:
        templates_dir = Path(templates_dir)

    template_path = templates_dir / f"{template_name}.yaml"
    return load_yaml(template_path)


# 保留旧 API (向后兼容) — 现在会按 kind="robot" 读取
def list_presets(presets_dir: Path | str | None = None) -> list[str]:
    """Deprecated: use list_templates() for built-in robot templates, or
    list_user_presets(kind=...) for user-saved presets."""
    if presets_dir is None:
        return list_templates()
    return list_templates(presets_dir)


def load_preset(preset_name: str, presets_dir: Path | str | None = None) -> UnifiedRobotConfig:
    """Deprecated: use load_template() for built-in, load_user_preset(kind, name) for user presets."""
    if presets_dir is None:
        return load_template(preset_name)
    return load_template(preset_name, presets_dir)


# 保留旧 API（向后兼容）作为模板的别名
def list_presets(presets_dir: Path | str | None = None) -> list[str]:
    """Deprecated: use list_templates(). Kept for backward compat."""
    return list_templates(presets_dir)


def load_preset(preset_name: str, presets_dir: Path | str | None = None) -> UnifiedRobotConfig:
    """Deprecated: use load_template(). Kept for backward compat."""
    return load_template(preset_name, presets_dir)