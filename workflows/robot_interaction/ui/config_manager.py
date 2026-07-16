"""Unified configuration manager for robot inference workflows.

This module provides a mode-discriminated config schema that unifies
deploy, replay, and camera preview workflows.
"""

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Literal, Any
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
        "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3", "left_arm_joint_4",
        "left_arm_joint_5", "left_arm_joint_6", "left_arm_joint_7",
        "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3", "right_arm_joint_4",
        "right_arm_joint_5", "right_arm_joint_6", "right_arm_joint_7",
        "left_gripper", "right_gripper",
    ])


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
    """Filter out fields not declared on the dataclass (for forward compat with
    hybrid yaml and unknown schema fields)."""
    if not data:
        return {}
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


def save_yaml(config: UnifiedRobotConfig, filepath: Path | str) -> None:
    """Save config to YAML in the schema expected by deploy.py / replay.py.

    Mode-aware:
      - deploy: writes `inference.return_to_initial_position` (from RuntimeConfig)
      - replay: writes root-level `play_sounds` and `return_to_initial_position`
      - camera_preview: only writes robot + policy (camera runtime params go via CLI)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    config_dict: dict[str, Any] = {
        "mode": config.mode,
        "robot": _dataclass_to_dict(config.robot),
    }

    if config.mode == "deploy":
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
        if config.dataset:
            config_dict["dataset"] = _dataclass_to_dict(config.dataset)
        # replay.py expects root-level play_sounds / return_to_initial_position
        config_dict["play_sounds"] = config.runtime.play_sounds
        config_dict["return_to_initial_position"] = config.runtime.return_to_initial_position

    elif config.mode == "camera_preview":
        # show_cameras.py reads http_base_url from yaml.robot and policy.path
        if config.policy and config.policy.path:
            config_dict["policy"] = {"path": config.policy.path}

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
    else:  # camera_preview or unknown
        runtime = RuntimeConfig()

    return UnifiedRobotConfig(
        mode=mode,
        robot=robot,
        policy=policy,
        inference=inference,
        dataset=dataset,
        runtime=runtime,
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

# 默认模板扫描路径：workflows/robot_interaction/ 下的 yaml 文件
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
        # 跳过 ui/ 自己的 yaml（避免递归列出预设/模板 yaml）
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


# 保留旧 API（向后兼容）作为模板的别名
def list_presets(presets_dir: Path | str | None = None) -> list[str]:
    """Deprecated: use list_templates(). Kept for backward compat."""
    return list_templates(presets_dir)


def load_preset(preset_name: str, presets_dir: Path | str | None = None) -> UnifiedRobotConfig:
    """Deprecated: use load_template(). Kept for backward compat."""
    return load_template(preset_name, presets_dir)
