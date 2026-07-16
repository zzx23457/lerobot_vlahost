"""Unified configuration manager for robot inference workflows.

This module provides a mode-discriminated config schema that unifies
deploy, replay, and camera preview workflows.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal
import yaml


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
    robot_id: str = "marvain_m6_01"
    type: str = "marvain_m6_http"
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
    joint_names: list[str] = field(default_factory=lambda: [
        "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
        "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
        "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4",
        "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
        "left_gripper", "right_gripper"
    ])


@dataclass
class RTCConfig:
    """Real-Time Chunking configuration"""
    execution_horizon: int | None = None
    max_guidance_weight: float = 10.0


@dataclass
class InferenceConfig:
    """Inference configuration (deploy mode only)"""
    type: Literal["sync", "rtc"] = "sync"
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


@dataclass
class UnifiedRobotConfig:
    """Unified configuration for all robot interaction modes"""
    mode: Literal["deploy", "replay", "camera_preview"] = "deploy"
    robot: RobotConfig = field(default_factory=RobotConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    policy: PolicyConfig | None = field(default_factory=PolicyConfig)
    inference: InferenceConfig | None = field(default_factory=InferenceConfig)
    dataset: DatasetConfig | None = field(default_factory=DatasetConfig)


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


def validate(config: UnifiedRobotConfig) -> list[str]:
    """Validate configuration and return list of error messages"""
    errors = []

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

    return errors


# ============================================================================
# CLI Argument Conversion
# ============================================================================

def to_cli_args(config: UnifiedRobotConfig) -> list[str]:
    """Convert unified config to deploy.py / replay.py CLI arguments"""
    args = []

    if config.mode == "deploy":
        if config.policy and config.policy.path:
            args.extend(["--policy-path", config.policy.path])
        if config.policy and config.policy.device:
            args.extend(["--device", config.policy.device])

        if config.inference:
            args.extend(["--fps", str(config.inference.fps)])
            args.extend(["--strategy", config.inference.strategy])
            args.extend(["--inference-type", config.inference.type])

            if config.inference.type == "rtc" and config.inference.rtc and config.inference.rtc.execution_horizon:
                args.extend(["--execution-horizon", str(config.inference.rtc.execution_horizon)])

            if config.inference.type == "rtc" and config.inference.rtc and config.inference.rtc.max_guidance_weight is not None:
                args.extend(["--max-guidance-weight", str(config.inference.rtc.max_guidance_weight)])

            if config.inference.duration > 0:
                args.extend(["--duration", str(config.inference.duration)])

            if config.inference.interpolation_multiplier != 1:
                args.extend(["--interpolation-multiplier", str(config.inference.interpolation_multiplier)])

            if config.inference.use_torch_compile:
                args.append("--use-torch-compile")

            if config.inference.rename_map:
                args.extend(["--rename-map", json.dumps(config.inference.rename_map)])

            if config.inference.show_cameras:
                args.append("--show-cameras")

        if config.dataset and config.dataset.single_task:
            args.extend(["--single-task", config.dataset.single_task])

        if config.dataset and config.dataset.repo_id:
            args.extend(["--repo-id", config.dataset.repo_id])

        if config.dataset and config.dataset.root:
            args.extend(["--dataset-root", config.dataset.root])

    elif config.mode == "replay":
        if config.dataset:
            if config.dataset.repo_id:
                args.extend(["--repo-id", config.dataset.repo_id])
            if config.dataset.episode is not None:
                args.extend(["--episode", str(config.dataset.episode)])
            args.extend(["--fps", str(config.dataset.fps)])

            if config.dataset.root:
                args.extend(["--dataset-root", config.dataset.root])

    # Common args
    args.extend(["--http-base-url", config.robot.http_base_url])
    args.extend(["--robot-id", config.robot.robot_id])

    # Robot safety settings
    if config.robot.safety_stats_path:
        args.extend(["--safety-stats-path", config.robot.safety_stats_path])

    # 始终传递 return_to_initial 参数
    args.append("--return-to-initial")
    args.append(str(config.runtime.return_to_initial_position))

    if config.mode == "replay":
        if config.runtime.play_sounds:
            args.append("--play-sounds")
        else:
            args.append("--no-sounds")

    return args


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


def save_yaml(config: UnifiedRobotConfig, filepath: Path | str) -> None:
    """Save config to YAML file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    config_dict = {
        "mode": config.mode,
        "robot": _dataclass_to_dict(config.robot),
        "runtime": _dataclass_to_dict(config.runtime),
    }

    if config.policy:
        config_dict["policy"] = _dataclass_to_dict(config.policy)

    if config.inference:
        config_dict["inference"] = _dataclass_to_dict(config.inference)

    if config.dataset:
        config_dict["dataset"] = _dataclass_to_dict(config.dataset)

    with open(filepath, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)


def load_yaml(filepath: Path | str) -> UnifiedRobotConfig:
    """Load config from YAML file"""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    # Parse nested configs
    robot_data = data.get("robot", {})
    cameras_data = robot_data.get("cameras", {})
    cameras = {name: CameraConfig(**cam) for name, cam in cameras_data.items()}
    robot_data["cameras"] = cameras
    robot = RobotConfig(**robot_data)

    # 加载 runtime，过滤掉已删除的字段（如 show_cameras）
    runtime_data = data.get("runtime", {})
    # 移除 show_cameras 字段（如果存在）- 这是旧配置文件可能有的字段
    runtime_data.pop("show_cameras", None)
    runtime = RuntimeConfig(**runtime_data)

    policy_data = data.get("policy")
    policy = PolicyConfig(**policy_data) if policy_data else None

    inference_data = data.get("inference")
    if inference_data:
        rtc_data = inference_data.get("rtc", {})
        inference_data["rtc"] = RTCConfig(**rtc_data)
        inference = InferenceConfig(**inference_data)
    else:
        inference = None

    dataset_data = data.get("dataset")
    dataset = DatasetConfig(**dataset_data) if dataset_data else None

    return UnifiedRobotConfig(
        mode=data.get("mode", "deploy"),
        robot=robot,
        runtime=runtime,
        policy=policy,
        inference=inference,
        dataset=dataset,
    )


# ============================================================================
# Preset Management
# ============================================================================

def list_presets(presets_dir: Path | str | None = None) -> list[str]:
    """List available config presets"""
    if presets_dir is None:
        presets_dir = Path(__file__).parent / "presets"
    else:
        presets_dir = Path(presets_dir)

    if not presets_dir.exists():
        return []

    presets = []
    for file in presets_dir.glob("*.yaml"):
        presets.append(file.stem)

    return sorted(presets)


def load_preset(preset_name: str, presets_dir: Path | str | None = None) -> UnifiedRobotConfig:
    """Load a config preset by name"""
    if presets_dir is None:
        presets_dir = Path(__file__).parent / "presets"
    else:
        presets_dir = Path(presets_dir)

    preset_path = presets_dir / f"{preset_name}.yaml"
    return load_yaml(preset_path)
