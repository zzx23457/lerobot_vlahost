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

"""HTTP-based driver for Marvain M6 bimanual robot (16 joints)."""

import base64
import json
import logging
import time
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
    """HTTP-based driver for Marvain M6 bimanual follower arm (14 joints + 2 grippers = 16 total).

    Communicates with a remote HTTP server at the configured base URL. The
    server provides endpoints for:
      - GET /state: Returns {
          "joint_states": {"positions": [14 floats in radians], ...},
          "gripper_left": [float in radians],
          "gripper_right": [float in radians],
          "quad_image": {"format": "jpeg", "data": "base64..."}
        }
      - POST /action: Accepts {
          "joints": [14 floats in radians],
          "gripper_left": [float in radians],
          "gripper_right": [float in radians]
        }

    Unit Conversion:
      - HTTP API uses radians for all joint positions
      - Internal LeRobot convention uses degrees
      - This class converts radians ↔ degrees at the HTTP boundary

    Safety:
      - Optional action clipping based on dataset stats (if safety_stats_path is set)
      - Per-tick motion limiting (max_relative_target_deg) for arm joints only
      - Out-of-range observation warnings

    Camera images: The server returns a single "quad_image" (base64-encoded JPEG),
    which is a 2x2 grid (1280x960) split into 4 cameras named by body part,
    ordered left→right: left_eye, right_eye, left_wrist, right_wrist.

    Quad layout (read top→bottom, left→right):
        - left_eye    = quad top-left      (0:480,   0:640)
        - left_wrist  = quad top-right     (0:480,   640:1280)
        - right_wrist = quad bottom-left   (480:960, 0:640)
        - right_eye   = quad bottom-right  (480:960, 640:1280)

    Mapping from physical camera keys → policy input slots is configurable
    per deployment via deploy_config.yaml's `rename_map` field (forwarded
    as `--rename_map` to lerobot-rollout, picked up by
    RenameObservationsProcessorStep in the preprocessor pipeline).
    """

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

    @staticmethod
    def _extract_gripper_pos(value) -> float | None:
        """Extract gripper position (radians) from a /state gripper field.

        服务端返回的 gripper 字段历史上变化过：
        - 早期是 ``[number]``（list 单元素数组）
        - 当前是 ``{"position": number, "velocity": ..., ...}``（dict）

        Returns:
            夹爪位置（弧度），格式不识别时返回 None。
        """
        if isinstance(value, list):
            return float(value[0]) if value else None
        if isinstance(value, dict):
            pos = value.get("position")
            return float(pos) if pos is not None else None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _grab_mjpeg_frame(self, stream_url: str, timeout: float = 2.0) -> np.ndarray:
        """从 MJPEG 流拉取单帧 JPEG，解码为 RGB numpy 数组 (H, W, 3)。

        服务端 /state 当前返回的 quad_image 用 ``stream_url``（multipart MJPEG 流）
        而不是 base64 内嵌帧。MJPEG 流格式是 multipart/x-mixed-replace，每帧 JPEG
        前面带 boundary header。本方法流式读取，找到第一个 JPEG SOI (0xFFD8) 开始
        累积字节，读到 JPEG EOI (0xFFD9) 即截断，作为完整一帧。
        """
        if not stream_url.startswith("http"):
            # stream_url 是相对路径（如 /stream/quad.mjpg），拼上 base_url
            base = self.config.http_base_url.rstrip("/")
            stream_url = f"{base}/{stream_url.lstrip('/')}"

        response = self._session.get(stream_url, stream=True, timeout=timeout)
        response.raise_for_status()

        buf = bytearray()
        found_soi = False
        try:
            for chunk in response.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                if not found_soi:
                    idx = chunk.find(b"\xff\xd8")
                    if idx >= 0:
                        buf.extend(chunk[idx:])
                        found_soi = True
                else:
                    buf.extend(chunk)
                    idx = buf.rfind(b"\xff\xd9")
                    if idx >= 0:
                        buf = buf[: idx + 2]
                        break
        finally:
            response.close()

        if not buf or len(buf) < 4:
            raise ValueError("no complete JPEG frame found in MJPEG stream")
        img_array = np.frombuffer(bytes(buf), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("failed to decode JPEG frame from MJPEG stream")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Quad cell layout: which physical camera sits in which corner of the 2x2 grid.
    # Corners: tl=top-left, tr=top-right, bl=bottom-left, br=bottom-right.
    # ** Edit this mapping when the server-side camera placement changes. **
    # _QUAD_CELL_OF_CAMERA: dict[str, str] = {
    #     # 鱼眼
    #     "left_eye": "tl",
    #     "left_wrist": "bl",
    #     "right_wrist": "br",
    #     "right_eye": "tr",
    # }
    _QUAD_CELL_OF_CAMERA: dict[str, str] = {
        # realsense 双边架
        "left_eye": "tl",
        "left_wrist": "tr",
        "right_wrist": "bl",
        "right_eye": "br",
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

            # 服务端 vlahost 简化接口：/state 不一定含 joint_states.positions。
            # 连接本身只验证服务端可达 + 返回 dict；关节位置缺失留给 get_observation()
            # 按接口契约如实报错。仅保留 isinstance 兜底，避免 None / 非 dict 触发 TypeError。
            if not isinstance(data, dict):
                raise ValueError(
                    f"HTTP /state returned non-dict body (got {type(data).__name__}): {data!r}"
                )

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

            if not isinstance(data, dict):
                raise ValueError(
                    f"HTTP /state returned non-dict body (got {type(data).__name__}): {data!r}"
                )

            # 关节位置按接口契约从 data["joint_states"]["positions"] 读取。
            # 服务端 vlahost 当前不返回 joint_states，缺字段时如实报错（不引入假值），
            # 让调用方知道"必须解决关节状态来源"才能继续。
            joint_states = data.get("joint_states")
            if not isinstance(joint_states, dict) or "positions" not in joint_states:
                raise RuntimeError(
                    f"HTTP /state did not return joint_states.positions "
                    f"(got keys: {list(data.keys())}). The robot server must expose "
                    f"joint positions for observation; cannot proceed without it."
                )
            joints_rad = joint_states["positions"]
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
            # 服务端返回 gripper 字段的格式历史上变化过：早期是 list[number]（取 [0]），
            # 当前是 dict（取 ["position"]）。两种格式都兼容，按实际 payload 自适应。
            try:
                if "gripper_left" in data and data["gripper_left"] is not None:
                    gripper_left_rad = self._extract_gripper_pos(data["gripper_left"])
                    if gripper_left_rad is None:
                        obs[f"{self.config.joint_names[14]}.pos"] = float(self.config.default_gripper_pos)
                    else:
                        obs[f"{self.config.joint_names[14]}.pos"] = float(np.degrees(gripper_left_rad))
                else:
                    obs[f"{self.config.joint_names[14]}.pos"] = float(self.config.default_gripper_pos)

                if "gripper_right" in data and data["gripper_right"] is not None:
                    gripper_right_rad = self._extract_gripper_pos(data["gripper_right"])
                    if gripper_right_rad is None:
                        obs[f"{self.config.joint_names[15]}.pos"] = float(self.config.default_gripper_pos)
                    else:
                        obs[f"{self.config.joint_names[15]}.pos"] = float(np.degrees(gripper_right_rad))
                else:
                    obs[f"{self.config.joint_names[15]}.pos"] = float(self.config.default_gripper_pos)
            except (IndexError, KeyError, TypeError) as e:
                logger.debug(f"{self} gripper data not available or invalid: {e}, using defaults")
                obs[f"{self.config.joint_names[14]}.pos"] = float(self.config.default_gripper_pos)
                obs[f"{self.config.joint_names[15]}.pos"] = float(self.config.default_gripper_pos)

            # Decode quad_image if present and split into individual cameras.
            # quad_image 字段历史上变化过：
            # - 旧格式：``{"data": "base64..."}``（base64 内嵌单帧 JPEG）
            # - 新格式：``{"stream_url": "/stream/quad.mjpg"}``（MJPEG 流，从流拉单帧）
            if self._has_quad_image and "quad_image" in data and data["quad_image"] is not None:
                quad_img_data = data["quad_image"]
                try:
                    if "data" in quad_img_data:
                        # 旧格式：base64 内嵌帧
                        quad_image = self._decode_image(quad_img_data["data"])
                    elif "stream_url" in quad_img_data:
                        # 新格式：从 MJPEG 流拉单帧
                        quad_image = self._grab_mjpeg_frame(quad_img_data["stream_url"])
                    else:
                        raise ValueError("quad_image has neither 'data' nor 'stream_url'")

                    # Split into individual camera views
                    camera_images = self._split_quad_image(quad_image)

                    # Add each camera to observation
                    for cam_name, cam_img in camera_images.items():
                        obs[cam_name] = cam_img

                    # Update camera names list
                    self._camera_names = list(camera_images.keys())

                except Exception as e:
                    logger.warning(f"{self} failed to decode/split quad_image: {e}")

            # Pass through need_new_chunk field if present (for chunk-based inference)
            if "need_new_chunk" in data:
                obs["need_new_chunk"] = data["need_new_chunk"]

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
        payload, result = self._prepare_action(action)

        # Send to HTTP endpoint with arm joints and grippers
        url = f"{self.config.http_base_url}/action"
        try:
            response = self._session.post(url, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise DeviceNotConnectedError(self._action_error_msg(url, e)) from e

        return result

    @check_if_not_connected
    def send_action_chunk(self, actions: list[RobotAction]) -> list[RobotAction]:
        """Send a whole action chunk to the HTTP endpoint in a single request.

        Each element of ``actions`` is a 16-joint target dict (in degrees), same
        format as :meth:`send_action`.  Every action is put through the same
        safety clipping / max-relative-target / deg→rad conversion (applied
        sequentially so per-step motion limiting chains through the chunk), then
        all per-action payloads are POSTed together to ``config.action_chunk_path``.

        The server must accept a body of the form::

            {"actions": [
                {"jointcmd_left": [7 rad], "jointcmd_right": [7 rad],
                 "gripper_left": rad, "gripper_right": rad},
                ...
            ]}

        Args:
            actions: Ordered list of 16-joint target dicts (in degrees).

        Returns:
            The list of actions actually sent (potentially clipped), in degrees.
        """
        if not actions:
            return []

        chunk_payload: list[dict] = []
        results: list[RobotAction] = []
        for action in actions:
            payload, result = self._prepare_action(action)
            chunk_payload.append(payload)
            results.append(result)

        url = f"{self.config.http_base_url}{self.config.action_chunk_path}"
        # send_http_base_url: str = "http://0.0.0.0:8010"

        # url = f"{send_http_base_url}{self.config.action_chunk_path}"
        body = {"actions": chunk_payload}
        try:
            response = self._session.post(url, json=body, timeout=self.config.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise DeviceNotConnectedError(self._action_error_msg(url, e)) from e

        return results

    def _prepare_action(self, action: RobotAction) -> tuple[dict, RobotAction]:
        """Build the HTTP payload (radians) and the returned result dict (degrees) for one action.

        Applies safety clipping and max-relative-target limiting to the 14 arm
        joints (grippers are passed through), updates ``_last_sent_pos`` so
        successive calls chain (important when sending a chunk), and splits the
        arm joints into the server's left/right command fields.
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

        # 分离左臂和右臂（前7个是左臂，后7个是右臂）
        joint_left_rad = arm_joints_rad[:7].tolist() if hasattr(arm_joints_rad, 'tolist') else list(arm_joints_rad[:7])
        joint_right_rad = arm_joints_rad[7:14].tolist() if hasattr(arm_joints_rad, 'tolist') else list(arm_joints_rad[7:14])

        # 服务端 /action 合法字段是 jointcmd_left/right（参见调试面板 JS 和 OpenAPI schema），
        # 用错字段名（joint_left/right）会被服务端静默丢弃，机器人不动。
        payload = {
            "jointcmd_left": joint_left_rad,    # 左臂7个关节
            "jointcmd_right": joint_right_rad,  # 右臂7个关节
            "gripper_left": float(left_gripper_rad),  # 单个浮点数，不是数组
            "gripper_right": float(right_gripper_rad),
        }

        # Update last sent position (only arm joints for relative target limiting)
        self._last_sent_pos = arm_joints_deg.tolist()

        # Build the clipped result for all 16 joints (in degrees)
        result: RobotAction = {}
        for i in range(14):
            result[f"{self.config.joint_names[i]}.pos"] = float(arm_joints_deg[i])
        result[f"{self.config.joint_names[14]}.pos"] = float(left_gripper_deg)
        result[f"{self.config.joint_names[15]}.pos"] = float(right_gripper_deg)

        return payload, result

    @staticmethod
    def _action_error_msg(url: str, e: Exception) -> str:
        """Build a verbose error message including the server response body if present."""
        error_msg = f"Failed to send action to {url}: {e}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                error_msg += f"\nServer response: {error_detail}"
            except Exception:
                error_msg += f"\nServer response: {e.response.text[:200]}"
        return error_msg

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
    # Go Home
    # ------------------------------------------------------------------
    @check_if_not_connected
    def go_home(self, wait_before: float = 1.0) -> bool:
        """Call the vlahost server's /go_home endpoint to return robot to home position.

        This uses the server's ROS 2 service call rather than sending joint commands,
        which is more reliable and handles the motion planning server-side.

        Args:
            wait_before: Seconds to wait before sending the go_home command, allowing
                the robot to stabilize after any previous motion. Default is 1.0 seconds.

        Returns:
            bool: True if the go_home command succeeded, False otherwise.
        """
        # Wait for robot to stabilize before sending go_home command
        if wait_before > 0:
            logger.info(f"{self} waiting {wait_before:.1f}s before go_home...")
            time.sleep(wait_before)

        url = f"{self.config.http_base_url}/go_home"
        try:
            response = self._session.post(url, timeout=self.config.timeout)
            response.raise_for_status()
            result = response.json()

            if result.get("success"):
                logger.info(f"{self} go_home succeeded: {result.get('message', '')}")
                return True
            else:
                logger.error(f"{self} go_home failed: {result.get('error', 'unknown error')}")
                return False

        except requests.RequestException as e:
            error_msg = f"Failed to call go_home at {url}: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg += f"\nServer response: {error_detail}"
                except Exception:
                    error_msg += f"\nServer response: {e.response.text[:200]}"
            logger.error(error_msg)
            return False

    # ------------------------------------------------------------------
    # Disconnect
    # ------------------------------------------------------------------
    def disconnect(self) -> None:
        """Close HTTP session."""
        if self._connected:
            self._session.close()
            self._connected = False
            logger.info(f"{self} disconnected")
