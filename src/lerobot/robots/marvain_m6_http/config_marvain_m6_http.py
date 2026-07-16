#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for MarvainM6Http robot (HTTP interface variant)."""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras.configs import CameraConfig, ColorMode

from ..config import RobotConfig


@CameraConfig.register_subclass("http")
@dataclass
class HttpCameraConfig(CameraConfig):
    """Metadata-only camera config for cameras whose frames come from another source.

    Use this for cameras whose images are delivered out-of-band (e.g. embedded
    in a robot's HTTP state response) rather than opened directly by a camera
    backend. The `width`/`height`/`fps` fields are required — they are used by
    ``RobotConfig.__post_init__`` to declare ``observation_features`` shapes —
    but no `index_or_path`/device handle is needed because nothing is opened
    here. The actual frame data is decoded by the robot driver.
    """

    color_mode: ColorMode = ColorMode.RGB

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)


@dataclass(kw_only=True)
class MarvainM6HttpRobotConfig(RobotConfig):
    """Configuration for Marvain M6 robot accessed via HTTP API.

    This robot communicates with a remote HTTP server that wraps the native
    SDK. The server returns observations (joints + images) in radians and
    accepts actions (joints) in radians. This config class defines the
    HTTP endpoint and conversion parameters.

    HTTP API Structure:
    - GET /state returns:
      - joint_states.positions: [14 arm joints in radians]
      - gripper_left: [gripper position in radians]
      - gripper_right: [gripper position in radians]
      - quad_image: {format: "jpeg", data: "base64..."}
    - POST /action accepts:
      - joints: [14 arm joints in radians]
      - gripper_left: [gripper position in radians]
      - gripper_right: [gripper position in radians]

    Attributes:
        http_base_url: Base URL of the HTTP API server (e.g., "http://192.168.10.123:8010")
        timeout: Request timeout in seconds for HTTP calls
        cameras: Camera configurations (for feature declaration; images come from HTTP)
        joint_names: Semantic names for 16 joints (7+7 arms + 2 grippers)
        default_gripper_pos: Default position for gripper joints (degrees) when HTTP doesn't provide
        safety_stats_path: Optional path to dataset root for loading safety bounds
        action_clip_margin_deg: Safety margin in degrees added to min/max bounds
        max_relative_target_deg: Maximum per-tick joint motion in degrees
        warn_on_observation_out_of_range: Log warnings when observations exceed training bounds
    """

    # HTTP connection parameters
    http_base_url: str = "http://192.168.10.123:8010"
    timeout: float = 5.0

    # Endpoint path for whole-chunk action dispatch (chunk inference mode).
    # POSTed body: {"actions": [{jointcmd_left, jointcmd_right, gripper_left, gripper_right}, ...]}
    action_chunk_path: str = "/action_chunk"

    # Robot configuration
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    joint_names: list[str] = field(
        default_factory=lambda: [
            "left_arm_joint_1",
            "left_arm_joint_2",
            "left_arm_joint_3",
            "left_arm_joint_4",
            "left_arm_joint_5",
            "left_arm_joint_6",
            "left_arm_joint_7",
            "right_arm_joint_1",
            "right_arm_joint_2",
            "right_arm_joint_3",
            "right_arm_joint_4",
            "right_arm_joint_5",
            "right_arm_joint_6",
            "right_arm_joint_7",
            "left_gripper",
            "right_gripper",
        ]
    )

    # Gripper configuration
    # HTTP API doesn't return gripper values, so we pad with default values
    default_gripper_pos: float = 0.0  # Default gripper position in degrees

    # Data-driven safety (optional)
    safety_stats_path: Path | None = None
    action_clip_margin_deg: float = 5.0
    max_relative_target_deg: float = 10.0
    warn_on_observation_out_of_range: bool = True

    def __post_init__(self):
        super().__post_init__()
        if len(self.joint_names) != 16:
            raise ValueError(
                f"MarvainM6Http requires exactly 16 joint names, got {len(self.joint_names)}"
            )


# Register with the robot choice registry
RobotConfig.register_subclass("marvain_m6_http", MarvainM6HttpRobotConfig)
