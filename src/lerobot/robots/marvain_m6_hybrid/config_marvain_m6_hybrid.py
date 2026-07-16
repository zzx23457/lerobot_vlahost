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

"""Configuration for the Marvain M6 hybrid robot (SDK joint obs + HTTP camera obs + SDK actions).

The hybrid robot exposes two backends on the same physical arm:

- Marvin SDK (used for joint/gripper observations and actions): joint
  positions and gripper positions come from
  ``MarvinRobotWrapper.get_joint_positions()`` (a single 16-float
  vector in degrees). Actions are dispatched via
  ``MarvinRobotWrapper.set_joint_positions``, bypassing the HTTP
  ``POST /action`` endpoint.
- HTTP backend (used only for camera observations): the configured
  HTTP server's ``GET /state`` endpoint returns a stitched quad image
  (1280x960) which is split into the four named camera views.

The 16-joint dict layout and the 4-camera shape are identical to the
HTTP-only and SDK-only variants so the same trained policy can run
across all three backends unchanged.
"""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras.configs import CameraConfig

# Re-used for camera shape declaration (no actual device is opened;
# frames are decoded from the HTTP quad_image).
from lerobot.robots.marvain_m6_http.config_marvain_m6_http import HttpCameraConfig

from ..config import RobotConfig


@dataclass(kw_only=True)
class MarvainM6HybridRobotConfig(RobotConfig):
    """Configuration for the Marvain M6 hybrid robot.

    Observations come from HTTP; actions go through the Marvin SDK.

    HTTP API structure (GET /state):
      - ``joint_states.positions``: [14 arm joints in radians]
      - ``gripper_left``: [gripper position in radians]
      - ``gripper_right``: [gripper position in radians]
      - ``quad_image``: {format: "jpeg", data: "base64..."}

    SDK action interface (``MarvinRobotWrapper.set_joint_positions``):
      - 16 floats in DEGREES, in joint_names order
      - Internally converts grippers deg→rad and dispatches via TCP/IP

    Attributes:
        # SDK observation interface (``MarvinRobotWrapper.get_joint_positions``):
        #   - 16 floats in DEGREES, in ``joint_names`` order.
        #   - Internally reads ``fb_joint_pos`` for both arms and converts
        #     gripper rad→deg; see ``marvin_robot_wrapper.get_joint_positions``.
        #
        # SDK action interface (``MarvinRobotWrapper.set_joint_positions``):
        #   - 16 floats in DEGREES, in joint_names order
        #   - Internally converts grippers deg→rad and dispatches via TCP/IP

        # HTTP camera interface (GET /state):
        #   - ``quad_image``: {format: "jpeg", data: "base64..."} or
        #                     {format: "mjpeg", stream_url: "/stream/quad.mjpg"}

        # HTTP backend
        http_base_url: HTTP server base URL (e.g. ``http://192.168.10.123:8010``).
            Still required for the camera fetch path.
        timeout: HTTP request timeout in seconds. Still required for the
            camera fetch path.
        default_gripper_pos: **Deprecated/no-op in the hybrid driver.** The
            gripper position now comes from the SDK, not from HTTP. This
            field is kept so existing YAML configs do not break; safe to
            remove from user YAML.

        # SDK backend
        robot_ip: Robot controller IP reachable on TCP/IP.
        control_mode: ``"position"`` (rigid) or ``"impedance"`` (compliant).
        vel_ratio: Velocity percent (1-100) forwarded to wrapper.
        acc_ratio: Acceleration percent (1-100) forwarded to wrapper.
        disable_torque_on_disconnect: If True, run the wrapper's full
            down-servo path on disconnect. If False, release the SDK
            connection but keep motor torque on (arm locks in place).

        # Shared / policy contract
        cameras: Camera configurations. Use ``HttpCameraConfig`` entries
            (frames come from HTTP quad_image; metadata-only stubs).
        joint_names: 16 joint names in the canonical order. Must match
            the keys the policy was trained with.

        # Data-driven safety
        safety_stats_path: Optional dataset root whose ``meta/stats.json``
            defines the safe operating range for action clipping and
            observation warnings.
        action_clip_margin_deg: Extra slack (degrees) added to the
            per-joint [min, max] range when clipping actions.
        max_relative_target_deg: Per-tick joint motion cap (degrees).
            None disables the cap.
        warn_on_observation_out_of_range: If True, log a one-shot
            warning when a read joint value falls outside training range.
    """

    # --- HTTP backend (camera fetch only) ---
    http_base_url: str = "http://192.168.10.123:8010"
    timeout: float = 5.0
    # Kept for backward compatibility with existing YAML configs; not
    # consumed by get_observation() anymore (SDK supplies grippers).
    default_gripper_pos: float = 0.0

    # --- SDK backend (actions) ---
    robot_ip: str = "192.168.10.190"
    control_mode: str = "impedance"
    vel_ratio: int = 20
    acc_ratio: int = 20
    disable_torque_on_disconnect: bool = False

    # --- Shared / policy contract ---
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

    # --- Data-driven safety ---
    safety_stats_path: Path | None = None
    action_clip_margin_deg: float = 5.0
    max_relative_target_deg: float | None = 10.0
    warn_on_observation_out_of_range: bool = True

    def __post_init__(self):
        if len(self.joint_names) != 16:
            raise ValueError(
                f"MarvainM6Hybrid requires exactly 16 joint names, got {len(self.joint_names)}"
            )
        if self.control_mode not in ("position", "impedance"):
            raise ValueError(
                f"control_mode must be 'position' or 'impedance', got '{self.control_mode}'"
            )
        super().__post_init__()


# Register with the robot choice registry. draccus picks this up at
# import time; ``lerobot_rollout.py`` imports the package for that
# side-effect.
RobotConfig.register_subclass("marvain_m6_hybrid", MarvainM6HybridRobotConfig)