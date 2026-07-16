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

import json
import logging
import time
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.Marvin_sdk_pro.marvin_robot_wrapper import MarvinRobotWrapper
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_marvain_m6 import MarvainM6RobotConfig

logger = logging.getLogger(__name__)


class MarvainM6(Robot):
    """Driver for the Marvain M6 bimanual follower arm (16 joints).

    Talks to the controller over TCP/IP via the Marvin SDK wrapper
    (MarvinRobotWrapper) and exposes a LeRobot-compatible interface:
    16 `.pos` proprioceptive keys + N camera image keys (N is
    config-driven, currently 3 for the 188-ep deployment dataset).

    Joint semantic names and feature layout are defined in
    MarvainM6RobotConfig and MUST match the keys used to train the
    downstream policy.

    Optional data-driven safety (enabled by setting
    `config.safety_stats_path` to a dataset root):
      * `send_action` clips each commanded joint to the [min - margin,
        max + margin] range observed in `meta/stats.json` of the
        training dataset, preventing the policy from asking for
        out-of-distribution positions.
      * `get_observation` emits a one-shot WARNING the first time a
        read joint value falls outside that range, so a sensor
        misconfiguration (e.g. radians-vs-degrees bug) is flagged
        immediately at deployment.
      * If `config.max_relative_target_deg` is set, per-tick joint
        motion is capped to that many degrees relative to the last
        commanded position.
    """

    config_class = MarvainM6RobotConfig
    name = "marvain_m6"

    # Unit convention: degrees, end-to-end. The SDK wrapper's
    # `get_joint_positions` returns 16 angles in DEGREES (gripper values
    # are converted from rad internally) and `set_joint_positions` expects
    # 16 angles in DEGREES (gripper values are converted to rad internally
    # before being sent to the motor's controlMIT). This is the contract
    # documented in `Marvin_sdk_pro/UNIT_CONVERSION.md` — the wrapper
    # handles all rad↔deg conversion at its boundary, callers pass degrees.
    # The training dataset and the policy operate in degrees, so this class
    # passes values straight through with no rad remap.

    def __init__(self, config: MarvainM6RobotConfig):
        super().__init__(config)
        self.config = config
        # SDK wrapper handles TCP/IP connect, state machine, and dual
        # gripper (KM_CAN DM4310) management. We only wrap it.
        self._wrapper = MarvinRobotWrapper(
            robot_ip=config.robot_ip,
            control_mode=config.control_mode,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

        # Data-driven safety bounds (lazy-loaded on first use so that
        # the class can be instantiated in environments without the
        # dataset on disk, e.g. unit tests).
        self._safety_bounds: np.ndarray | None = None  # shape (16, 2) = [min, max]
        self._safety_warned: set[int] = set()  # joint indices already warned about
        self._last_sent_pos: list[float] | None = None  # for max_relative_target check

    # ------------------------------------------------------------------
    # Safety stats loading (lazy)
    # ------------------------------------------------------------------
    def _load_safety_bounds(self) -> np.ndarray:
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
        # Prefer `action` range (what the policy is allowed to command),
        # fall back to `observation.state` if action stats are missing.
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
        m = self.config.action_clip_margin_deg
        self._safety_bounds = np.stack([lo - m, hi + m], axis=1)  # (16, 2)
        logger.info(
            f"{self} loaded safety bounds from {stats_file} "
            f"(per-joint range with ±{m}° margin, source='{key}')."
        )
        return self._safety_bounds

    # ------------------------------------------------------------------
    # Feature declarations (consumed by record / eval / inference)
    # ------------------------------------------------------------------
    @cached_property
    def observation_features(self) -> dict:
        return {
            **{f"{j}.pos": float for j in self.config.joint_names},
            **{
                cam: (self.config.cameras[cam].height,
                      self.config.cameras[cam].width, 3)
                for cam in self.cameras
            },
        }

    @cached_property
    def action_features(self) -> dict:
        return {f"{j}.pos": float for j in self.config.joint_names}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return (
            self._wrapper.is_connected()
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        # Pre-load safety stats so any path-related error surfaces at
        # connect() time, not on the first action.
        self._load_safety_bounds()
        # Wrapper establishes TCP/IP, verifies data flow, and switches
        # the arm into the requested control mode (position / impedance).
        self._wrapper.connect()
        # If user provided custom impedance K/D, override the wrapper's
        # hardcoded defaults. This is cheaper than a full mode switch
        # (we just re-write K/D, no IDLE->TORQ dance).
        if (
            self.config.control_mode == "impedance"
            and (self.config.impedance_k is not None
                 or self.config.impedance_d is not None)
        ):
            k = self.config.impedance_k or ([2.0, 2.0, 2.0, 1.6, 0.5, 1.0, 1.0] * 2)
            d = self.config.impedance_d or ([0.6, 0.6, 0.6, 0.4, 0.2, 0.2, 0.2] * 2)
            self._apply_impedance_kd(k, d)
            logger.info(f"{self} applied custom impedance K/D from config.")
        for cam in self.cameras.values():
            cam.connect()
        # Apply one-time runtime configuration (PID, limits, etc.) —
        # mirrors the `self.configure()` call in so_follower.connect().
        # No-op for Marvin today; kept in the call chain so adding
        # future SDK config (e.g. impedance K/D) requires no upstream
        # change.
        self.configure()
        self._last_sent_pos = None  # reset on each (re)connect
        logger.info(f"{self} connected.")

    @check_if_not_connected
    def disconnect(self) -> None:
        # Disconnect cameras first.
        for cam in self.cameras.values():
            cam.disconnect()
        # Shutdown the robot:
        #   disable_torque_on_disconnect=False (Q5 "lock"): keep motor
        #     torque on, just release the SDK TCP/IP resources so the
        #     arm physically holds its last commanded position.
        #   disable_torque_on_disconnect=True: full release path
        #     (down-servo, disable grippers, release SDK).
        if not self._wrapper.is_connected():
            return
        if self.config.disable_torque_on_disconnect:
            self._wrapper.disconnect()
        else:
            # Soft shutdown: bypass the wrapper's set_state(0) call.
            self._wrapper.robot.release_robot()
            self._wrapper._connected = False
            self._wrapper._gripper_connected = False
        logger.info(f"{self} disconnected (lock={not self.config.disable_torque_on_disconnect}).")

    # ------------------------------------------------------------------
    # Runtime control-mode switching
    # ------------------------------------------------------------------
    def set_control_mode(
        self,
        mode: Literal["position", "impedance"],
        k: list[float] | None = None,
        d: list[float] | None = None,
    ) -> None:
        """Switch the arm's motion control mode at runtime.

        Args:
            mode: ``"position"`` (rigid) or ``"impedance"`` (compliant).
            k: per-joint stiffness, length 14 (7 per arm: A臂 + B臂).
                Used only when ``mode="impedance"``. If ``None``, the
                wrapper's default
                ``[2.0, 2.0, 2.0, 1.6, 0.5, 1.0, 1.0]`` is applied to
                both arms.
            d: per-joint damping, length 14. If ``None``, the wrapper's
                default ``[0.6, 0.6, 0.6, 0.4, 0.2, 0.2, 0.2]`` is
                applied to both arms.

        Notes:
            * The switch is a blocking 1-2 second operation; do not
              call it inside the high-rate inference loop.
            * During the switch the arm briefly enters the IDLE
              state. Any in-flight action from the policy will be
              dropped.
            * The current mode is reflected in ``self.config.control_mode``
              after the call returns.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if mode not in ("position", "impedance"):
            raise ValueError(
                f"Unknown mode {mode!r}; expected 'position' or 'impedance'."
            )

        if mode == "position":
            self._switch_to_position_mode()
        else:  # impedance
            k_default = [2.0, 2.0, 2.0, 1.6, 0.5, 1.0, 1.0] * 2  # 14 values
            d_default = [0.6, 0.6, 0.6, 0.4, 0.2, 0.2, 0.2] * 2
            k_use = list(k) if k is not None else k_default
            d_use = list(d) if d is not None else d_default
            if len(k_use) != 14 or len(d_use) != 14:
                raise ValueError(
                    f"K and D must be length 14 (7 per arm); got "
                    f"len(k)={len(k_use)}, len(d)={len(d_use)}."
                )
            self._switch_to_impedance_mode(k_use, d_use)

        self.config.control_mode = mode
        logger.info(f"{self} control mode -> {mode}")

    def _switch_to_position_mode(self) -> None:
        """Switch both arms to position (rigid) control mode (state=1)."""
        robot = self._wrapper.robot
        for arm in ("A", "B"):
            robot.clear_set()
            robot.set_state(arm=arm, state=1)
            robot.send_cmd()
            time.sleep(0.1)
        time.sleep(0.3)  # settle
        logger.info(f"{self} position mode set on both arms.")

    def _switch_to_impedance_mode(self, k: list[float], d: list[float]) -> None:
        """Switch both arms to joint-impedance mode.

        Args:
            k, d: length 14 (left arm 7, then right arm 7).

        Mirrors the wrapper's `_enable_impedance_mode` sequence but
        accepts per-arm K, D instead of the hardcoded defaults.
        """
        k_a, k_b = k[:7], k[7:]
        d_a, d_b = d[:7], d[7:]
        robot = self._wrapper.robot

        # 1. IDLE first (down-servo) — required before changing state.
        for arm in ("A", "B"):
            robot.clear_set()
            robot.set_state(arm=arm, state=0)
            robot.send_cmd()
        time.sleep(0.5)

        # 2. Switch to torque mode (state=3).
        for arm in ("A", "B"):
            robot.clear_set()
            robot.set_state(arm=arm, state=3)
            robot.send_cmd()
        time.sleep(0.5)

        # 3. Set joint-impedance type (type=1).
        for arm in ("A", "B"):
            robot.clear_set()
            robot.set_impedance_type(arm=arm, type=1)
            robot.send_cmd()
        time.sleep(0.3)

        # 4. Set per-arm K, D.
        for arm, k_arm, d_arm in (("A", k_a, d_a), ("B", k_b, d_b)):
            robot.clear_set()
            robot.set_joint_kd_params(arm=arm, K=k_arm, D=d_arm)
            robot.send_cmd()
        time.sleep(0.3)

        # 5. Apply vel / acc limits.
        for arm in ("A", "B"):
            robot.clear_set()
            robot.set_vel_acc(
                arm=arm,
                velRatio=self.config.vel_ratio,
                AccRatio=self.config.acc_ratio,
            )
            robot.send_cmd()
        time.sleep(0.3)
        logger.info(f"{self} impedance mode set on both arms (custom K/D applied).")

    def _apply_impedance_kd(self, k: list[float], d: list[float]) -> None:
        """Overwrite K, D without doing a full mode switch.

        Cheaper than ``_switch_to_impedance_mode`` (skips the
        IDLE -> TORQ dance). Use this when the arm is already in
        impedance mode and you just want to retune the gains.
        """
        if len(k) != 14 or len(d) != 14:
            raise ValueError(
                f"K and D must be length 14 (7 per arm); got "
                f"len(k)={len(k)}, len(d)={len(d)}."
            )
        robot = self._wrapper.robot
        for arm, k_arm, d_arm in (("A", k[:7], d[:7]), ("B", k[7:], d[7:])):
            robot.clear_set()
            robot.set_joint_kd_params(arm=arm, K=k_arm, D=d_arm)
            robot.send_cmd()
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Calibration (Q2 decision: no-op; SDK handles homing internally)
    # ------------------------------------------------------------------
    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.warning(
            f"{self} calibrate() is a no-op: SDK handles homing internally. "
            "If the arm reports invalid positions at startup, contact the hardware team."
        )

    # ------------------------------------------------------------------
    # Configure: wrapper does not expose PID/torque-limit configuration
    # at the Python level, so this is a no-op for now.
    # ------------------------------------------------------------------
    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Observation / action IO
    # ------------------------------------------------------------------
    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Wrapper returns a 16-element list in the order:
        #   [left_arm_1..7, right_arm_1..7, left_gripper, right_gripper]
        # where the last 2 (grippers) are in RADIANS. Convert to degrees
        # so the rest of the pipeline is unit-consistent.
        raw = self._wrapper.get_joint_positions()
        if len(raw) != len(self.config.joint_names):
            raise RuntimeError(
                f"Wrapper returned {len(raw)} joints but config expects "
                f"{len(self.config.joint_names)}. Check joint_names order."
            )
        # Wrapper already returns degrees for all 16 joints (incl. grippers).
        # Pass through unchanged — see class-level unit-convention note and
        # Marvin_sdk_pro/UNIT_CONVERSION.md.
        pos = list(raw)

        # --- Data-driven sanity check: flag values outside training range.
        bounds = self._safety_bounds
        if (
            self.config.warn_on_observation_out_of_range
            and bounds.shape[0] == 16
        ):
            arr = np.asarray(pos, dtype=np.float64)
            for i, val in enumerate(arr):
                if i in self._safety_warned:
                    continue
                lo, hi = bounds[i]
                if not (lo <= val <= hi):
                    logger.warning(
                        f"{self} observation {self.config.joint_names[i]}.pos = "
                        f"{val:.3f} is outside training range "
                        f"[{lo:.2f}, {hi:.2f}] (margin {self.config.action_clip_margin_deg}°). "
                        "Possible causes: (1) wrong units (rad vs deg), "
                        "(2) wrong joint_names order, (3) sensor not homed. "
                        "This warning fires once per joint per session."
                    )
                    self._safety_warned.add(i)

        obs: RobotObservation = {
            f"{name}.pos": float(val)
            for name, val in zip(self.config.joint_names, pos)
        }
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        # Repack the dict back into the wrapper's positional layout.
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
            delta_clipped = np.clip(
                delta,
                -self.config.max_relative_target_deg,
                self.config.max_relative_target_deg,
            )
            if not np.allclose(delta_clipped, delta):
                n_capped = int(np.sum(delta_clipped != delta))
                max_actual = float(np.max(np.abs(delta)))
                logger.warning(
                    f"{self} capped {n_capped}/16 joints' per-tick motion to "
                    f"±{self.config.max_relative_target_deg}° "
                    f"(largest requested delta was {max_actual:.1f}°)."
                )
                arr = prev + delta_clipped

        pos_clamped_deg = arr.tolist()
        # Wrapper expects degrees for all 16 joints (incl. grippers) and
        # handles the rad conversion against the SDK internally. Pass
        # through unchanged — see class-level unit-convention note and
        # Marvin_sdk_pro/UNIT_CONVERSION.md.
        self._wrapper.set_joint_positions(
            pos_clamped_deg,
            vel_ratio=self.config.vel_ratio,
            acc_ratio=self.config.acc_ratio,
        )
        # Track the *clipped* value in degrees (the policy / safety view).
        self._last_sent_pos = pos_clamped_deg

        # Return the *actually-sent* action in DEGREES (so the caller
        # and any downstream logging see the policy-frame value, not
        # the raw radian value the SDK consumed).
        return {f"{name}.pos": float(v) for name, v in zip(self.config.joint_names, pos_clamped_deg)}
