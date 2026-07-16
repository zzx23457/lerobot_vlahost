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

"""Hybrid driver for the Marvain M6 bimanual follower arm.

Combines two transports on the same physical robot:

- **Joint + gripper observations** come from the Marvin SDK
  (``MarvinRobotWrapper.get_joint_positions()``), which returns a
  16-element list in degrees ``[A-arm 7, B-arm 7, left_gripper, right_gripper]``.

- **Camera observations** come from the configured HTTP server
  (GET /state), specifically the stitched 2x2 quad image which is
  split into four named camera views.

- **Actions** are dispatched directly through the Marvin TCP/IP SDK
  (``MarvinRobotWrapper.set_joint_positions``), bypassing the HTTP
  POST /action endpoint entirely.

The 16-joint dict layout and 4-camera shape match the HTTP-only and
SDK-only variants so a policy trained against one backend can run
against any of the three unchanged.
"""

import base64
import json
import logging
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
import requests

from lerobot.Marvin_sdk_pro.marvin_robot_wrapper import MarvinRobotWrapper
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_marvain_m6_hybrid import MarvainM6HybridRobotConfig

logger = logging.getLogger(__name__)


class MarvainM6HybridRobot(Robot):
    """Marvain M6 hybrid driver (SDK joint obs + HTTP camera obs + SDK actions).

    Communication model:
      - ``get_observation()`` → SDK ``wrapper.get_joint_positions()``
        (returns 16 floats in degrees: arm joints + grippers).
        Cameras come from HTTP GET {http_base_url}/state (quad_image).
      - ``send_action()``     → SDK ``wrapper.set_joint_positions(degrees)``
        (the SDK already converts arm joint degrees straight through and
        grippers deg→rad internally; see
        ``src/lerobot/Marvin_sdk_pro/UNIT_CONVERSION.md``)

    Quad image layout (read top→bottom, left→right):
        - left_eye    = quad top-left      (0:480,   0:640)
        - left_wrist  = quad top-right     (0:480,   640:1280)
        - right_wrist = quad bottom-left   (480:960, 0:640)
        - right_eye   = quad bottom-right  (480:960, 640:1280)

    Mapping from physical camera keys → policy input slots is
    configurable per deployment via deploy_config.yaml's ``rename_map``
    field (forwarded as ``--rename_map`` to ``lerobot-rollout``, picked
    up by ``RenameObservationsProcessorStep`` in the preprocessor).
    """

    config_class = MarvainM6HybridRobotConfig
    name = "marvain_m6_hybrid"

    def __init__(self, config: MarvainM6HybridRobotConfig):
        super().__init__(config)
        self.config = config

        # Two backends, two states. ``is_connected`` ANDs them.
        self._wrapper = MarvinRobotWrapper(
            robot_ip=config.robot_ip,
            control_mode=config.control_mode,
        )
        self._session = requests.Session()

        # Connection state for each backend
        self._http_connected = False
        self._sdk_connected = False

        # HTTP observation bookkeeping
        self._has_quad_image: bool = False
        self._camera_names: list[str] = []

        # Data-driven safety (lazy-loaded on first use so the class
        # can be instantiated in environments without the dataset on
        # disk).
        self._safety_bounds: np.ndarray | None = None  # (16, 2)
        self._safety_warned: set[int] = set()
        self._last_sent_pos: list[float] | None = None  # for max_relative_target check

    # ------------------------------------------------------------------
    # Static helpers (copied from MarvainM6HttpRobot)
    # ------------------------------------------------------------------
    @staticmethod
    def _decode_image(base64_str: str) -> np.ndarray:
        """Decode base64-encoded image string to numpy array (H, W, 3) RGB uint8."""
        img_bytes = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image from base64 string")
        # OpenCV returns BGR; LeRobot expects RGB.
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

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
        for cam_name, corner in MarvainM6HybridRobot._QUAD_CELL_OF_CAMERA.items():
            rs, cs = MarvainM6HybridRobot._cell_slice(corner, h_cell, w_cell)
            cells[cam_name] = quad_image[rs, cs].copy()

        # right_eye fallback: if its cell is blank, reuse left_eye so the
        # model doesn't see all zeros.
        if "right_eye" in cells and cells["right_eye"].size and cells["right_eye"].mean() < 10:
            cells["right_eye"] = cells["left_eye"].copy()

        return cells

    # ------------------------------------------------------------------
    # Feature declarations
    # ------------------------------------------------------------------
    @cached_property
    def observation_features(self) -> dict:
        """Observation structure: 16 joint positions + N camera images."""
        features = {f"{j}.pos": float for j in self.config.joint_names}
        for cam_name, cam_cfg in self.config.cameras.items():
            features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict:
        """Action structure: 16 joint target positions."""
        return {f"{j}.pos": float for j in self.config.joint_names}

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return (
            self._http_connected
            and self._sdk_connected
            and self._wrapper.is_connected()
        )

    @property
    def is_calibrated(self) -> bool:
        # SDK handles homing internally; HTTP doesn't require calibration.
        return True

    def calibrate(self) -> None:
        """No-op: SDK handles homing internally; HTTP doesn't calibrate."""
        logger.debug(f"{self} calibrate() is a no-op.")

    def configure(self) -> None:
        """No-op: wrapper has no per-session configuration step."""
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect to both backends. HTTP first (cheap), then SDK (slow).

        Failure modes:
          - HTTP fails: SDK is never touched; ``DeviceNotConnectedError`` raised.
          - SDK fails: HTTP session is closed, ``_http_connected`` reset,
            ``DeviceNotConnectedError`` raised.
          - Both succeed: ``is_connected`` returns True.
        """
        # Pre-load safety bounds so path errors surface at connect time.
        self._load_safety_bounds()

        # 1. HTTP first. Cheap, fast to fail.
        try:
            url = f"{self.config.http_base_url}/state"
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

            self._has_quad_image = "quad_image" in data and data["quad_image"] is not None
            if self._has_quad_image:
                logger.info(f"{self} detected quad_image from HTTP")
                # Try to decode and split to discover actual cameras
                try:
                    quad_img_data = data["quad_image"]
                    if "data" in quad_img_data:
                        quad_image = self._decode_image(quad_img_data["data"])
                    elif "stream_url" in quad_img_data:
                        quad_image = self._grab_mjpeg_frame(quad_img_data["stream_url"])
                    else:
                        raise ValueError("quad_image has neither 'data' nor 'stream_url'")
                    camera_images = self._split_quad_image(quad_image)
                    self._camera_names = list(camera_images.keys())
                    logger.info(f"{self} split quad_image into cameras: {self._camera_names}")
                except Exception as e:
                    logger.warning(f"{self} failed to split quad_image during connect: {e}")
                    self._camera_names = []
            else:
                logger.warning(f"{self} no quad_image found in HTTP response")
                self._camera_names = []

            self._http_connected = True
            logger.info(f"{self} connected to {self.config.http_base_url}")

        except requests.RequestException as e:
            raise DeviceNotConnectedError(
                f"HTTP backend failed to connect at {self.config.http_base_url}: {e}"
            ) from e

        # 2. SDK second. Slow (~3-4 s). If this fails, roll back HTTP.
        try:
            self._wrapper.connect()
            self._sdk_connected = True
            logger.info(
                f"{self} SDK connected to {self.config.robot_ip} "
                f"(mode={self.config.control_mode})"
            )
        except Exception as e:
            try:
                self._session.close()
            except Exception:
                pass
            self._http_connected = False
            raise DeviceNotConnectedError(
                f"SDK backend failed to connect at {self.config.robot_ip}: {e}"
            ) from e

        self._last_sent_pos = None
        self.configure()
        logger.info(f"{self} connected (hybrid: HTTP obs + SDK act)")

    def disconnect(self) -> None:
        """Release SDK first (arm safety), then close HTTP session.

        SDK failure during release is logged but does not stop us from
        closing the HTTP session — the operator gets a clear log line
        (``sdk_release_ok=False``) and the next ``connect()`` will
        retry from a clean slate.
        """
        sdk_ok = False

        # SDK first — release the arm so it doesn't stay torqued mid-action.
        if self._sdk_connected and self._wrapper.is_connected():
            try:
                if self.config.disable_torque_on_disconnect:
                    self._wrapper.disconnect()
                else:
                    # Lock mode: bypass wrapper's down-servo path; keep motor torque on.
                    self._wrapper.robot.release_robot()
                    self._wrapper._connected = False
                    self._wrapper._gripper_connected = False
                sdk_ok = True
                logger.info(
                    f"{self} SDK released (lock={not self.config.disable_torque_on_disconnect})"
                )
            except Exception as e:
                logger.error(f"{self} SDK disconnect failed: {type(e).__name__}: {e}")
                # Best-effort: force the wrapper into a "released" state from our POV.
                try:
                    self._wrapper._connected = False
                    self._wrapper._gripper_connected = False
                except Exception:
                    pass

        # HTTP second — closing the session cannot fail meaningfully.
        if self._http_connected:
            try:
                self._session.close()
            except Exception as e:
                logger.warning(f"{self} HTTP session close raised: {e}")
            self._http_connected = False

        self._sdk_connected = False
        logger.info(f"{self} disconnected (sdk_release_ok={sdk_ok})")

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
    # Observation (joints from SDK, cameras from HTTP)
    # ------------------------------------------------------------------
    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Get current observation: joints from SDK, cameras from HTTP.

        Returns:
            Dict with 16 joint positions (degrees) + 4 camera images.
        """
        if not self._sdk_connected:
            raise DeviceNotConnectedError(
                f"{self} SDK backend is down; joint observations unavailable."
            )
        if not self._http_connected:
            raise DeviceNotConnectedError(
                f"{self} HTTP backend is down; camera observations unavailable."
            )

        # --- Joints from the Marvin SDK ---
        # Wrapper returns 16 floats in DEGREES, in the order
        # [left_arm_1..7, right_arm_1..7, left_gripper, right_gripper].
        # See marvin_robot_wrapper.get_joint_positions() and
        # Marvin_sdk_pro/UNIT_CONVERSION.md.
        try:
            pos = list(self._wrapper.get_joint_positions())
        except Exception as e:
            raise DeviceNotConnectedError(
                f"SDK get_joint_positions failed: {type(e).__name__}: {e}"
            ) from e

        if len(pos) != len(self.config.joint_names):
            raise RuntimeError(
                f"SDK returned {len(pos)} joints but config expects "
                f"{len(self.config.joint_names)}. Check joint_names order."
            )

        joints_deg = np.asarray(pos, dtype=np.float64)

        # Safety warning for out-of-range observations (all 16 joints).
        if self.config.warn_on_observation_out_of_range:
            self._check_observation_bounds(joints_deg)

        obs: RobotObservation = {
            f"{name}.pos": float(val)
            for name, val in zip(self.config.joint_names, pos)
        }

        # --- Cameras from HTTP /state (quad_image only) ---
        url = f"{self.config.http_base_url}/state"
        try:
            response = self._session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise DeviceNotConnectedError(
                f"Failed to get camera observation from {url}: {e}"
            ) from e

        if not isinstance(data, dict):
            raise ValueError(
                f"HTTP /state returned non-dict body (got {type(data).__name__}): {data!r}"
            )

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

        return obs

    def _check_observation_bounds(self, joints_deg: np.ndarray) -> None:
        """Warn once per joint if observation is out of training range (all 16 joints)."""
        bounds = self._load_safety_bounds()
        if bounds.size == 0:
            return

        for i in range(len(self.config.joint_names)):
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
    # Action (SDK)
    # ------------------------------------------------------------------
    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Send action command directly through the Marvin SDK.

        Args:
            action: Dict with joint target positions (degrees) for all 16 joints.

        Returns:
            The action actually sent (potentially clipped) in degrees for all 16 joints.
        """
        if not self._sdk_connected:
            raise DeviceNotConnectedError(
                f"{self} SDK backend is down; cannot send action."
            )

        # Repack dict into the wrapper's positional layout
        try:
            pos = [action[f"{name}.pos"] for name in self.config.joint_names]
        except KeyError as e:
            raise KeyError(
                f"Action missing key {e}. Expected 16 keys of the form "
                f"'<joint_name>.pos' matching config.joint_names."
            ) from e

        arr = np.asarray(pos, dtype=np.float64)

        # --- Safety 1: clip to training range ---
        bounds = self._safety_bounds
        if bounds is None:
            bounds = self._load_safety_bounds()
        if bounds.shape[0] == 16:
            clipped = np.clip(arr, bounds[:, 0], bounds[:, 1])
            if not np.allclose(clipped, arr):
                n_clipped = int(np.sum(clipped != arr))
                logger.warning(
                    f"{self} clipped {n_clipped}/16 joint commands to training range "
                    f"[{bounds[:, 0].min():.1f}, {bounds[:, 1].max():.1f}] "
                    "(policy asked for out-of-distribution values)."
                )
                arr = clipped

        # --- Safety 2: cap per-tick relative motion ---
        if (
            self.config.max_relative_target_deg is not None
            and self._last_sent_pos is not None
        ):
            prev = np.asarray(self._last_sent_pos, dtype=np.float64)
            delta = arr - prev
            delta_capped = np.clip(
                delta,
                -self.config.max_relative_target_deg,
                self.config.max_relative_target_deg,
            )
            if not np.allclose(delta_capped, delta):
                n_capped = int(np.sum(delta_capped != delta))
                max_actual = float(np.max(np.abs(delta)))
                logger.warning(
                    f"{self} capped {n_capped}/16 joints' per-tick motion to "
                    f"±{self.config.max_relative_target_deg}° "
                    f"(largest requested delta was {max_actual:.1f}°)."
                )
                arr = prev + delta_capped

        pos_deg = arr.tolist()

        # SDK direct dispatch — bypass HTTP POST /action.
        # Wrapper expects degrees for all 16 joints (incl. grippers) and
        # converts grippers deg→rad internally (see UNIT_CONVERSION.md).
        self._wrapper.set_joint_positions(
            pos_deg,
            vel_ratio=self.config.vel_ratio,
            acc_ratio=self.config.acc_ratio,
        )
        self._last_sent_pos = pos_deg

        # Return the *actually-sent* action in degrees (so the caller and
        # any downstream logging see the policy-frame value).
        return {
            f"{name}.pos": float(v)
            for name, v in zip(self.config.joint_names, pos_deg)
        }