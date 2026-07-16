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

"""HTTP-based driver for Marvain M6 bimanual robot (16 joints).

>>> OLD 备份：本会话开始时仓库里的版本。
>>> 与 ../marvain_m6_http.py 当前版本的差异：
>>>   1. connect() 强校验 joint_states.positions 必须存在
>>>   2. get_observation() 直接 data["joint_states"]["positions"]（不防御 None）
>>>   3. get_observation() gripper 用 list[0]（不兼容 dict 格式）
>>>   4. get_observation() quad_image 只处理 base64 data 字段
>>>   5. send_action() payload 字段名 joint_left/right（被服务端静默丢弃）
>>>   6. 没有 _extract_gripper_pos helper
>>>   7. 没有 _grab_mjpeg_frame helper
>>>
>>> 回滚：cp old/marvain_m6_http.py ../marvain_m6_http.py
>>> ⚠ 这个版本跟当前服务端不兼容，仅作历史快照参考。
"""

import base64
import json
import logging
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
import requests

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_marvain_m6_http import MarvainM6HttpRobotConfig

logger = logging.getLogger(__name__)


class MarvainM6HttpRobot(Robot):
    """HTTP-based driver for Marvain M6 bimanual follower arm (14 joints + 2 grippers = 16 total)."""

    config_class = MarvainM6HttpRobotConfig
    name = "marvain_m6_http"

    def __init__(self, config: MarvainM6HttpRobotConfig):
        super().__init__(config)
        self.config = config
        self._session = requests.Session()
        self._connected = False
        self._calibrated = True  # HTTP interface doesn't require calibration

        # Camera names discovered from first observation
        self._camera_names: list[str] = []
        self._has_quad_image: bool = False

        # Data-driven safety
        self._safety_bounds: np.ndarray | None = None  # shape (16, 2) = [min, max] in degrees
        self._safety_warned: set[int] = set()
        self._last_sent_pos: list[float] | None = None  # in degrees

    # ------------------------------------------------------------------
    # Unit conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rad_to_deg(joints_rad: list[float]) -> np.ndarray:
        """Convert joint positions from radians to degrees."""
        return np.degrees(np.array(joints_rad, dtype=np.float64))

    @staticmethod
    def _deg_to_rad(joints_deg: list[float] | np.ndarray) -> list[float]:
        """Convert joint positions from degrees to radians."""
        return np.radians(np.asarray(joints_deg, dtype=np.float64)).tolist()

    @staticmethod
    def _decode_image(base64_str: str) -> np.ndarray:
        """Decode base64-encoded image string to numpy array (H, W, 3) RGB uint8."""
        img_bytes = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image from base64 string")
        # Convert BGR to RGB (OpenCV uses BGR, LeRobot expects RGB)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Quad cell layout: which physical camera sits in which corner of the 2x2 grid.
    # Corners: tl=top-left, tr=top-right, bl=bottom-left, br=bottom-right.
    # ** Edit this mapping when the server-side camera placement changes. **
    _QUAD_CELL_OF_CAMERA: dict[str, str] = {
        "left_eye": "tl",
        "left_wrist": "bl",
        "right_wrist": "br",
        "right_eye": "tr",
    }

    @staticmethod
    def _cell_slice(corner: str, h_cell: int, w_cell: int) -> tuple[slice, slice]:
        """Return (row_slice, col_slice) for one corner of a 2x2 grid."""
        if corner == "tl":
            return slice(0, h_cell), slice(0, w_cell)
        if corner == "tr":
            return slice(0, h_cell), slice(w_cell, None)
        if corner == "bl":
            return slice(h_cell, None), slice(0, w_cell)
        if corner == "br":
            return slice(h_cell, None), slice(w_cell, None)
        raise ValueError(f"Unknown quad cell {corner!r}; expected one of tl/tr/bl/br")

    @staticmethod
    def _split_quad_image(quad_image: np.ndarray) -> dict[str, np.ndarray]:
        """Split a 2x2 quad_image into named camera views at any resolution.

        Per-camera placement is driven by ``_QUAD_CELL_OF_CAMERA`` above —
        edit that mapping to change which corner of the grid each camera
        comes from.

        Cell size is inferred from the quad dimensions (H/2 × W/2), so this
        works for any per-camera resolution as long as H and W are even.

        If the right_eye cell is blank (hardware may expose only one front
        camera), it falls back to a copy of left_eye so the model never
        receives an all-black image.

        Args:
            quad_image: Full quad image from HTTP server (H, W, 3)

        Returns:
            Dict mapping each camera name listed in ``_QUAD_CELL_OF_CAMERA``
            to an image array of shape (H/2, W/2, 3).
        """
        h, w = quad_image.shape[:2]

        if h % 2 or w % 2:
            raise ValueError(
                f"Unexpected quad_image dimensions: {h}x{w}. "
                f"Expected even H and W (2x2 grid)."
            )

        h_cell, w_cell = h // 2, w // 2

        cells: dict[str, np.ndarray] = {}
        for cam_name, corner in MarvainM6HttpRobot._QUAD_CELL_OF_CAMERA.items():
            rs, cs = MarvainM6HttpRobot._cell_slice(corner, h_cell, w_cell)
            cells[cam_name] = quad_image[rs, cs].copy()

        # right_eye fallback: if its cell is blank, reuse left_eye so the
        # model doesn't see all zeros.
        if "right_eye" in cells and cells["right_eye"].size and cells["right_eye"].mean() < 10:
            cells["right_eye"] = cells["left_eye"].copy()

        return cells

    # ------------------------------------------------------------------
    # Safety bounds loading (lazy)
    # ------------------------------------------------------------------
    def _load_safety_bounds(self) -> np.ndarray:
        """Load safety bounds from dataset stats.json (in degrees)."""
        if self._safety_bounds is not None:
            return self._safety_bounds

        stats_path = self.config.safety_stats_path
        if stats_path is None:
            self._safety_bounds = np.empty((0, 2))
            return self._safety_bounds

        stats_file = Path(stats_path) / "meta" / "stats.json"
        if not stats_file.is_file():
            logger.warning(
                f"{self} safety_stats_path={stats_path} has no meta/stats.json; "
                "data-driven safety checks disabled."
            )
            self._safety_bounds = np.empty((0, 2))
            return self._safety_bounds

        with open(stats_file) as f:
            stats = json.load(f)

        # Prefer 'action' range, fall back to 'observation.state'
        key = "action" if "action" in stats else "observation.state"
        lo = np.asarray(stats[key]["min"], dtype=np.float64)
        hi = np.asarray(stats[key]["max"], dtype=np.float64)

        if lo.shape != (16,) or hi.shape != (16,):
            logger.warning(
                f"{self} stats.json['{key}'] has shape {lo.shape}/{hi.shape}, "
                "expected (16,). Safety checks disabled."
            )
            self._safety_bounds = np.empty((0, 2))
            return self._safety_bounds

        margin = self.config.action_clip_margin_deg
        self._safety_bounds = np.stack([lo - margin, hi + margin], axis=1)  # (16, 2)
        logger.info(
            f"{self} loaded safety bounds from {stats_file} "
            f"(per-joint range with ±{margin}° margin, source='{key}')."
        )
        return self._safety_bounds

    # ------------------------------------------------------------------
    # Feature declarations
    # ------------------------------------------------------------------
    @cached_property
    def observation_features(self) -> dict:
        """Observation structure: 16 joint positions + N camera images."""
        features = {f"{j}.pos": float for j in self.config.joint_names}
        # Camera features (shape inferred from config)
        for cam_name, cam_cfg in self.config.cameras.items():
            features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict:
        """Action structure: 16 joint target positions."""
        return {f"{j}.pos": float for j in self.config.joint_names}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Establish HTTP connection by testing GET /state."""
        url = f"{self.config.http_base_url}/state"
        try:
            response = self._session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()

            # Validate response structure
            if "joint_states" not in data or "positions" not in data["joint_states"]:
                raise ValueError(
                    f"HTTP response missing required keys. Expected 'joint_states.positions', "
                    f"got: {list(data.keys())}"
                )

            joints = data["joint_states"]["positions"]
            if len(joints) != 14:
                raise ValueError(f"Expected 14 joints, got {len(joints)}")

            # Check for quad_image
            self._has_quad_image = "quad_image" in data and data["quad_image"] is not None
            if self._has_quad_image:
                logger.info(f"{self} detected quad_image from HTTP")
                # Try to decode and split to discover actual cameras
                try:
                    quad_img_data = data["quad_image"]
                    if "data" in quad_img_data:
                        quad_image = self._decode_image(quad_img_data["data"])
                        camera_images = self._split_quad_image(quad_image)
                        self._camera_names = list(camera_images.keys())
                        logger.info(f"{self} split quad_image into cameras: {self._camera_names}")
                except Exception as e:
                    logger.warning(f"{self} failed to split quad_image during connect: {e}")
                    self._camera_names = []
            else:
                logger.warning(f"{self} no quad_image found in HTTP response")
                self._camera_names = []

            self._connected = True
            logger.info(f"{self} connected to {self.config.http_base_url}")

        except requests.RequestException as e:
            raise DeviceNotConnectedError(
                f"Failed to connect to HTTP server at {self.config.http_base_url}: {e}"
            ) from e

    def calibrate(self) -> None:
        """No-op: HTTP interface doesn't require calibration."""
        pass

    def configure(self) -> None:
        """No-op: HTTP interface configuration is server-side."""
        pass

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Get current observation from HTTP endpoint.

        Returns:
            Dictionary with joint positions (in degrees) and camera images (numpy arrays).
        """
        url = f"{self.config.http_base_url}/state"
        try:
            response = self._session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()

            # Convert joints from radians to degrees
            joints_rad = data["joint_states"]["positions"]
            if len(joints_rad) != 14:
                raise ValueError(f"Expected 14 arm joints from HTTP, got {len(joints_rad)}")

            joints_deg = self._rad_to_deg(joints_rad)

            # Safety warning for out-of-range observations
            if self.config.warn_on_observation_out_of_range:
                self._check_observation_bounds(joints_deg)

            # Build observation dict with 16 joints (14 arms + 2 grippers)
            obs: RobotObservation = {}
            for i in range(14):
                obs[f"{self.config.joint_names[i]}.pos"] = float(joints_deg[i])

            # Get gripper positions from HTTP (if available)
            # HTTP returns gripper_left and gripper_right arrays, we take the first element
            try:
                if "gripper_left" in data and data["gripper_left"] is not None:
                    gripper_left_rad = data["gripper_left"][0] if isinstance(data["gripper_left"], list) else data["gripper_left"]
                    obs[f"{self.config.joint_names[14]}.pos"] = float(np.degrees(gripper_left_rad))
                else:
                    obs[f"{self.config.joint_names[14]}.pos"] = float(self.config.default_gripper_pos)

                if "gripper_right" in data and data["gripper_right"] is not None:
                    gripper_right_rad = data["gripper_right"][0] if isinstance(data["gripper_right"], list) else data["gripper_right"]
                    obs[f"{self.config.joint_names[15]}.pos"] = float(np.degrees(gripper_right_rad))
                else:
                    obs[f"{self.config.joint_names[15]}.pos"] = float(self.config.default_gripper_pos)
            except (IndexError, KeyError, TypeError) as e:
                logger.debug(f"{self} gripper data not available or invalid: {e}, using defaults")
                obs[f"{self.config.joint_names[14]}.pos"] = float(self.config.default_gripper_pos)
                obs[f"{self.config.joint_names[15]}.pos"] = float(self.config.default_gripper_pos)

            # Decode quad_image if present and split into individual cameras
            if self._has_quad_image and "quad_image" in data and data["quad_image"] is not None:
                quad_img_data = data["quad_image"]
                if "data" in quad_img_data:
                    try:
                        # Decode the full quad image
                        quad_image = self._decode_image(quad_img_data["data"])

                        # Split into individual camera views
                        camera_images = self._split_quad_image(quad_image)

                        # Add each camera to observation
                        for cam_name, cam_img in camera_images.items():
                            obs[cam_name] = cam_img

                        # Update camera names list
                        self._camera_names = list(camera_images.keys())

                    except Exception as e:
                        logger.warning(f"{self} failed to decode/split quad_image: {e}")

            return obs

        except requests.RequestException as e:
            raise DeviceNotConnectedError(
                f"Failed to get observation from {url}: {e}"
            ) from e

    def _check_observation_bounds(self, joints_deg: np.ndarray) -> None:
        """Warn once per joint if observation is out of training range (only for arm joints)."""
        bounds = self._load_safety_bounds()
        if bounds.size == 0:
            return

        # Only check first 14 joints (arms), not grippers
        for i in range(14):
            if i in self._safety_warned:
                continue
            val = joints_deg[i]
            lo, hi = bounds[i]
            if val < lo or val > hi:
                logger.warning(
                    f"{self} joint {i} ({self.config.joint_names[i]}) observation "
                    f"{val:.2f}° is outside training range [{lo:.2f}, {hi:.2f}]°"
                )
                self._safety_warned.add(i)

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Send action command to HTTP endpoint.

        Args:
            action: Dictionary with joint target positions (in degrees) for all 16 joints

        Returns:
            The action actually sent (potentially clipped) in degrees for all 16 joints
        """
        # Extract arm joint positions in degrees (first 14)
        arm_joints_deg = np.array(
            [action[f"{j}.pos"] for j in self.config.joint_names[:14]],
            dtype=np.float64
        )

        # Apply safety clipping (only for arm joints)
        arm_joints_deg = self._apply_safety_clipping(arm_joints_deg)

        # Apply max relative target constraint (only for arm joints)
        arm_joints_deg = self._apply_max_relative_target(arm_joints_deg)

        # Convert to radians for HTTP
        arm_joints_rad = self._deg_to_rad(arm_joints_deg)

        # Extract gripper positions in degrees and convert to radians
        left_gripper_deg = action[f"{self.config.joint_names[14]}.pos"]
        right_gripper_deg = action[f"{self.config.joint_names[15]}.pos"]
        left_gripper_rad = np.radians(left_gripper_deg)
        right_gripper_rad = np.radians(right_gripper_deg)

        # Send to HTTP endpoint with arm joints and grippers
        url = f"{self.config.http_base_url}/action"

        # 分离左臂和右臂（前7个是左臂，后7个是右臂）
        joint_left_rad = arm_joints_rad[:7].tolist() if hasattr(arm_joints_rad, 'tolist') else list(arm_joints_rad[:7])
        joint_right_rad = arm_joints_rad[7:14].tolist() if hasattr(arm_joints_rad, 'tolist') else list(arm_joints_rad[7:14])

        payload = {
            "joint_left": joint_left_rad,    # 左臂7个关节
            "joint_right": joint_right_rad,  # 右臂7个关节
            "gripper_left": float(left_gripper_rad),  # 单个浮点数，不是数组
            "gripper_right": float(right_gripper_rad),
        }
        try:
            response = self._session.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            # 添加更多调试信息
            error_msg = f"Failed to send action to {url}: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f"\nServer response: {error_detail}"
                except:
                    error_msg += f"\nServer response: {e.response.text[:200]}"
            raise DeviceNotConnectedError(error_msg) from e

        # Update last sent position (only arm joints for relative target limiting)
        self._last_sent_pos = arm_joints_deg.tolist()

        # Return the clipped action for all 16 joints
        result = {}
        for i in range(14):
            result[f"{self.config.joint_names[i]}.pos"] = float(arm_joints_deg[i])
        result[f"{self.config.joint_names[14]}.pos"] = float(left_gripper_deg)
        result[f"{self.config.joint_names[15]}.pos"] = float(right_gripper_deg)

        return result

    def _apply_safety_clipping(self, joints_deg: np.ndarray) -> np.ndarray:
        """Clip joint positions to safety bounds (only for arm joints, not grippers)."""
        bounds = self._load_safety_bounds()
        if bounds.size == 0:
            return joints_deg

        # Only clip first 14 joints (arms), bounds for grippers are ignored
        clipped = np.clip(joints_deg, bounds[:14, 0], bounds[:14, 1])

        # Log clipping events
        if not np.allclose(joints_deg, clipped, atol=1e-6):
            for i in range(14):
                if abs(joints_deg[i] - clipped[i]) > 1e-6:
                    logger.warning(
                        f"{self} action clipped: joint {i} ({self.config.joint_names[i]}) "
                        f"{joints_deg[i]:.2f}° → {clipped[i]:.2f}° "
                        f"(bounds: [{bounds[i, 0]:.2f}, {bounds[i, 1]:.2f}]°)"
                    )

        return clipped

    def _apply_max_relative_target(self, joints_deg: np.ndarray) -> np.ndarray:
        """Limit per-tick joint motion."""
        if self._last_sent_pos is None:
            return joints_deg

        max_delta = self.config.max_relative_target_deg
        if max_delta <= 0:
            return joints_deg

        last_pos = np.array(self._last_sent_pos, dtype=np.float64)
        delta = joints_deg - last_pos
        clamped_delta = np.clip(delta, -max_delta, max_delta)
        clamped_joints = last_pos + clamped_delta

        if not np.allclose(delta, clamped_delta, atol=1e-6):
            logger.debug(
                f"{self} max_relative_target applied: max delta was {np.abs(delta).max():.2f}°"
            )

        return clamped_joints

    # ------------------------------------------------------------------
    # Disconnect
    # ------------------------------------------------------------------
    def disconnect(self) -> None:
        """Close HTTP session."""
        if self._connected:
            self._session.close()
            self._connected = False
            logger.info(f"{self} disconnected")