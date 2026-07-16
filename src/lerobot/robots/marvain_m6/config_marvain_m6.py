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

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class MarvainM6Config:
    """Base configuration for the Marvain M6 bimanual follower arm.

    The Marvain M6 is a 7-DOF left arm + 7-DOF right arm + 1-DOF left
    gripper + 1-DOF right gripper = 16 joint system, controlled over
    TCP/IP via the Marvin native SDK wrapper. Cameras are flexible and
    configured via the `cameras` field.
    """

    # --- Network / control ---
    robot_ip: str = "192.168.15.190"
    control_mode: str = "impedance"  # "position" or "impedance"
    # Wrapper-level velocity / acceleration limit (1-100 %).
    vel_ratio: int = 20
    acc_ratio: int = 20

    # --- Impedance-mode tuning (only used when control_mode="impedance") ---
    # Per-joint stiffness array, length 14 (7 per arm: A臂 + B臂).
    # If None, the wrapper's hardcoded default
    # ([2.0, 2.0, 2.0, 1.6, 0.5, 1.0, 1.0] applied to both arms) is used.
    impedance_k: list[float] | None = None
    # Per-joint damping array, length 14. If None, the wrapper's
    # hardcoded default ([0.6, 0.6, 0.6, 0.4, 0.2, 0.2, 0.2] applied
    # to both arms) is used.
    impedance_d: list[float] | None = None

    # --- Disconnect behaviour (Q5: lock = keep torque on) ---
    # If True, the wrapper's full disconnect path is used (down-servo,
    # disable grippers, release SDK). If False, only the SDK connection
    # is released and the arm / grippers stay in their last commanded
    # state (motor torque on, holding position).
    disable_torque_on_disconnect: bool = False

    # --- Joint naming (Q1) ---
    # Order: [left arm 7, right arm 7, left gripper, right gripper].
    # These names MUST match the keys the policy was trained with.
    joint_names: list[str] = field(default_factory=lambda: [
        "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3",
        "left_arm_joint_4", "left_arm_joint_5", "left_arm_joint_6", "left_arm_joint_7",
        "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3",
        "right_arm_joint_4", "right_arm_joint_5", "right_arm_joint_6", "right_arm_joint_7",
        "left_gripper", "right_gripper",
    ])

    # --- Cameras ---
    # Camera configuration is fully flexible. Specify via:
    # - CLI: --robot.cameras='{"cam_name": {"type": "...", ...}}'
    # - Code: cameras={"cam_name": CameraConfig(...)}
    # Camera keys must match the dataset's info.json for training/inference.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # --- Data-driven safety (all derived from training dataset stats) ---
    # Path to the dataset root whose `meta/stats.json` defines the safe
    # operating range. When set, the robot will (a) clamp every sent
    # action to the per-joint [min, max] range observed at training time
    # (+`action_clip_margin_deg`), and (b) emit a warning if a read
    # observation falls outside that range. Set to None to disable both
    # safety checks (NOT recommended for real-machine deployment).
    safety_stats_path: Path | None = None
    # Extra slack (in degrees) added to the training range when clipping
    # actions. The policy's outputs are clipped to [min - margin, max +
    # margin] before being sent to the robot. 5° is a sane default.
    action_clip_margin_deg: float = 5.0
    # If non-None, per-tick joint motion is capped to this many degrees
    # relative to the previously commanded position. Prevents the arm
    # from making sudden large jumps if the policy emits a wild action.
    # (e.g. 10.0 = no joint may move more than 10° between two ticks.)
    max_relative_target_deg: float | None = 10.0
    # If True, the robot logs a WARNING the first time an observation
    # value is outside the training range. The values are still passed
    # through to the policy — this is a "we are now in unknown territory"
    # alarm, not a hard stop.
    warn_on_observation_out_of_range: bool = True

    def __post_init__(self) -> None:
        # Validate control_mode
        if self.control_mode not in ("position", "impedance"):
            raise ValueError(
                f"control_mode must be 'position' or 'impedance', got '{self.control_mode}'"
            )
        super().__post_init__()


@RobotConfig.register_subclass("marvain_m6")
@dataclass
class MarvainM6RobotConfig(RobotConfig, MarvainM6Config):
    pass
