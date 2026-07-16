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

"""Tests for MarvainM6HybridRobot (SDK joint obs + HTTP camera obs + SDK actions).

All tests mock both backends:
  - ``MarvinRobotWrapper`` → SDK TCP/IP path (used by ``get_observation``
    for joint + gripper readings, AND by ``send_action``).
  - ``requests.Session``    → HTTP path (used by ``get_observation``
    for camera observations only).

This lets us assert routing — that joints come from SDK, cameras come
from HTTP, actions go through SDK — without any real hardware.
"""

import base64
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from lerobot.robots.marvain_m6_hybrid import (
    MarvainM6HybridRobot,
    MarvainM6HybridRobotConfig,
)
from lerobot.utils.errors import DeviceNotConnectedError

# Module path where MarvinRobotWrapper / requests.Session are imported
# inside the hybrid robot. We patch them at this location.
WRAPPER_MODULE = "lerobot.robots.marvain_m6_hybrid.marvain_m6_hybrid"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_state_response(
    *,
    joint_positions_rad: list[float] | None = None,
    gripper_left: float = 0.0,
    gripper_right: float = 0.0,
    quad_image_b64: str | None = None,
):
    """Build a mock HTTP /state response."""
    if joint_positions_rad is None:
        joint_positions_rad = [0.0] * 14
    resp = MagicMock(name="HTTPStateResponse")
    resp.raise_for_status = MagicMock()
    body = {
        "joint_states": {"positions": list(joint_positions_rad)},
        "gripper_left": [gripper_left],
        "gripper_right": [gripper_right],
    }
    if quad_image_b64 is not None:
        body["quad_image"] = {"format": "jpeg", "data": quad_image_b64}
    else:
        body["quad_image"] = None
    resp.json.return_value = body
    return resp


def _make_wrapper_mock(*, joint_positions: list[float] | None = None):
    """Build a mock MarvinRobotWrapper with sane defaults.

    Args:
        joint_positions: Value returned by ``get_joint_positions``. Defaults
            to 16 zeros. Tests that need specific values pass an explicit
            16-element list in degrees.
    """
    wrapper = MagicMock(name="MarvinRobotWrapperMock")
    wrapper.is_connected.return_value = True
    wrapper.connect.return_value = True
    wrapper.get_joint_positions.return_value = (
        list(joint_positions) if joint_positions is not None else [0.0] * 16
    )
    wrapper.set_joint_positions.return_value = None
    wrapper.disconnect.return_value = None
    wrapper.robot = MagicMock()
    wrapper.robot.release_robot.return_value = None
    return wrapper


def _encode_quad_image(image: np.ndarray) -> str:
    """Encode an RGB image to base64 JPEG (round-trip via OpenCV)."""
    import cv2

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


@pytest.fixture
def hybrid_robot():
    """Default fixture: SDK and HTTP both up; no quad image in state."""
    wrapper_mock = _make_wrapper_mock()
    state_resp = _make_state_response()
    session_mock = MagicMock(name="HTTPSession")
    session_mock.get.return_value = state_resp

    with (
        patch(f"{WRAPPER_MODULE}.MarvinRobotWrapper", return_value=wrapper_mock),
        patch(f"{WRAPPER_MODULE}.requests.Session", return_value=session_mock),
    ):
        cfg = MarvainM6HybridRobotConfig(
            id="test_hybrid",
            http_base_url="http://127.0.0.1:8010",
            robot_ip="127.0.0.1",
            control_mode="position",
        )
        robot = MarvainM6HybridRobot(cfg)
        yield robot, wrapper_mock, session_mock
        if robot.is_connected:
            robot.disconnect()


@pytest.fixture
def hybrid_robot_with_quad():
    """Variant: HTTP state response includes a 1280x960 white quad_image."""
    wrapper_mock = _make_wrapper_mock()
    quad = np.full((960, 1280, 3), 255, dtype=np.uint8)
    # Make a 100x100 black square in the bottom-right to verify the
    # right_eye-blank fallback is NOT triggered when the region has data.
    quad[860:960, 1180:1280] = 0
    state_resp = _make_state_response(quad_image_b64=_encode_quad_image(quad))
    session_mock = MagicMock(name="HTTPSession")
    session_mock.get.return_value = state_resp

    with (
        patch(f"{WRAPPER_MODULE}.MarvinRobotWrapper", return_value=wrapper_mock),
        patch(f"{WRAPPER_MODULE}.requests.Session", return_value=session_mock),
    ):
        cfg = MarvainM6HybridRobotConfig(
            id="test_hybrid_q",
            http_base_url="http://127.0.0.1:8010",
            robot_ip="127.0.0.1",
            control_mode="position",
        )
        robot = MarvainM6HybridRobot(cfg)
        yield robot, wrapper_mock, session_mock
        if robot.is_connected:
            robot.disconnect()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_features_match_joints(hybrid_robot):
    """Observation/action feature dicts contain 16 joint keys."""
    robot, _, _ = hybrid_robot
    joint_keys = {f"{n}.pos" for n in robot.config.joint_names}
    assert set(robot.observation_features.keys()) == joint_keys
    assert set(robot.action_features.keys()) == joint_keys


def test_connect_calls_both_backends(hybrid_robot):
    """connect() probes HTTP and brings up SDK."""
    robot, wrapper_mock, session_mock = hybrid_robot
    assert not robot.is_connected

    robot.connect()

    # HTTP /state was probed at least once (the connect-time probe).
    assert session_mock.get.called
    # SDK connect was called exactly once.
    wrapper_mock.connect.assert_called_once()
    assert robot.is_connected


def test_is_connected_requires_both_backends(hybrid_robot):
    """is_connected is False if either backend drops."""
    robot, wrapper_mock, _ = hybrid_robot
    robot.connect()
    assert robot.is_connected

    # Simulate SDK drop.
    wrapper_mock.is_connected.return_value = False
    assert not robot.is_connected


def test_get_observation_uses_sdk_for_joints_http_for_cameras(hybrid_robot):
    """get_observation() pulls joints from SDK and cameras from HTTP."""
    robot, wrapper_mock, session_mock = hybrid_robot
    robot.connect()

    # Reset call counts after connect-time probe.
    wrapper_mock.get_joint_positions.reset_mock()
    session_mock.get.reset_mock()

    obs = robot.get_observation()

    # SDK observation path WAS used for joints.
    wrapper_mock.get_joint_positions.assert_called_once()
    # HTTP /state was hit for camera fetch.
    session_mock.get.assert_called()
    assert session_mock.get.call_args.args[0].endswith("/state")

    # All 16 joint keys present.
    joint_keys = {f"{n}.pos" for n in robot.config.joint_names}
    assert joint_keys.issubset(set(obs.keys()))


def test_get_observation_with_quad_image(hybrid_robot_with_quad):
    """quad_image in HTTP response is split into 4 named cameras."""
    robot, _, _ = hybrid_robot_with_quad
    robot.connect()
    obs = robot.get_observation()

    for cam in ("left_eye", "right_eye", "left_wrist", "right_wrist"):
        assert cam in obs, f"missing camera key {cam!r}"
        assert obs[cam].shape == (480, 640, 3)


def test_send_action_routes_to_sdk_not_http(hybrid_robot):
    """send_action() goes through the SDK wrapper, never HTTP POST."""
    robot, wrapper_mock, session_mock = hybrid_robot
    robot.connect()
    wrapper_mock.set_joint_positions.reset_mock()
    session_mock.post.reset_mock()

    action = {f"{n}.pos": 5.0 for n in robot.config.joint_names}
    returned = robot.send_action(action)

    # SDK wrapper received the action.
    wrapper_mock.set_joint_positions.assert_called_once()
    sent_values = wrapper_mock.set_joint_positions.call_args.args[0]
    assert len(sent_values) == 16
    assert all(v == 5.0 for v in sent_values)
    # HTTP POST was NOT issued.
    session_mock.post.assert_not_called()
    # Returned dict matches input (no clipping, no scaling).
    assert returned == action


def test_send_action_clamps_to_safety_bounds(tmp_path):
    """send_action() clips commands to dataset stats range + margin."""
    wrapper_mock = _make_wrapper_mock()
    session_mock = MagicMock()
    session_mock.get.return_value = _make_state_response()

    # Tiny stats.json with narrow action bounds: [0, 10] for all 16.
    meta = tmp_path / "meta"
    meta.mkdir()
    stats = {
        "action": {
            "min": [0.0] * 16,
            "max": [10.0] * 16,
        }
    }
    (meta / "stats.json").write_text(json.dumps(stats))

    with (
        patch(f"{WRAPPER_MODULE}.MarvinRobotWrapper", return_value=wrapper_mock),
        patch(f"{WRAPPER_MODULE}.requests.Session", return_value=session_mock),
    ):
        cfg = MarvainM6HybridRobotConfig(
            id="test_clamp",
            http_base_url="http://127.0.0.1:8010",
            robot_ip="127.0.0.1",
            control_mode="position",
            safety_stats_path=tmp_path,
            action_clip_margin_deg=5.0,
        )
        robot = MarvainM6HybridRobot(cfg)
        robot.connect()
        wrapper_mock.set_joint_positions.reset_mock()

        # Try to command 50° on every joint; should clip to 10 + 5 = 15°.
        action = {f"{n}.pos": 50.0 for n in robot.config.joint_names}
        robot.send_action(action)

        sent_values = wrapper_mock.set_joint_positions.call_args.args[0]
        # Each joint command must be ≤ 10 + 5 = 15°.
        assert all(v <= 15.0 + 1e-6 for v in sent_values)
        # Sanity: clipping actually changed the values (not a no-op).
        assert any(abs(v - 50.0) > 1e-6 for v in sent_values)


def test_disconnect_releases_sdk_first_then_http(hybrid_robot):
    """disconnect() releases SDK then closes HTTP session."""
    robot, wrapper_mock, session_mock = hybrid_robot
    robot.connect()

    # disable_torque_on_disconnect=True → wrapper.disconnect() path.
    robot.config.disable_torque_on_disconnect = True
    robot.disconnect()

    # Both backends released.
    wrapper_mock.disconnect.assert_called_once()
    session_mock.close.assert_called_once()
    assert not robot.is_connected


def test_disconnect_lock_mode_skips_full_wrapper_disconnect(hybrid_robot):
    """disable_torque_on_disconnect=False → lock mode (no down-servo)."""
    robot, wrapper_mock, _ = hybrid_robot
    robot.config.disable_torque_on_disconnect = False
    robot.connect()
    robot.disconnect()

    # Lock mode: wrapper.disconnect NOT called; release_robot IS called.
    wrapper_mock.disconnect.assert_not_called()
    wrapper_mock.robot.release_robot.assert_called_once()
    # Wrapper forced into "released" state.
    assert wrapper_mock._connected is False
    assert wrapper_mock._gripper_connected is False
    assert not robot.is_connected


def test_connect_failure_rolls_back_http_when_sdk_fails():
    """If SDK connect raises, the HTTP session is closed and an error is raised."""
    wrapper_mock = _make_wrapper_mock()
    wrapper_mock.connect.side_effect = RuntimeError("sdk down")
    session_mock = MagicMock()
    session_mock.get.return_value = _make_state_response()

    with (
        patch(f"{WRAPPER_MODULE}.MarvinRobotWrapper", return_value=wrapper_mock),
        patch(f"{WRAPPER_MODULE}.requests.Session", return_value=session_mock),
    ):
        cfg = MarvainM6HybridRobotConfig(
            id="test_sdk_fail",
            http_base_url="http://127.0.0.1:8010",
            robot_ip="127.0.0.1",
            control_mode="position",
        )
        robot = MarvainM6HybridRobot(cfg)

        with pytest.raises(DeviceNotConnectedError, match="SDK backend"):
            robot.connect()

        # HTTP session was rolled back.
        session_mock.close.assert_called_once()
        # SDK was attempted (and failed) but no extra connect happened.
        wrapper_mock.connect.assert_called_once()
        assert not robot.is_connected


def test_connect_failure_when_http_fails():
    """If HTTP connect raises, SDK is never touched."""
    wrapper_mock = _make_wrapper_mock()
    session_mock = MagicMock()
    session_mock.get.side_effect = requests.RequestException("http down")

    with (
        patch(f"{WRAPPER_MODULE}.MarvinRobotWrapper", return_value=wrapper_mock),
        patch(f"{WRAPPER_MODULE}.requests.Session", return_value=session_mock),
    ):
        cfg = MarvainM6HybridRobotConfig(
            id="test_http_fail",
            http_base_url="http://127.0.0.1:8010",
            robot_ip="127.0.0.1",
            control_mode="position",
        )
        robot = MarvainM6HybridRobot(cfg)

        with pytest.raises(DeviceNotConnectedError, match="HTTP backend"):
            robot.connect()

        # SDK was never even attempted.
        wrapper_mock.connect.assert_not_called()
        assert not robot.is_connected


def test_get_observation_handles_missing_quad_image(hybrid_robot):
    """If HTTP returns no quad_image, obs still has all 16 joint keys but no camera keys."""
    robot, _, _ = hybrid_robot
    robot.connect()
    obs = robot.get_observation()

    joint_keys = {f"{n}.pos" for n in robot.config.joint_names}
    assert joint_keys.issubset(set(obs.keys()))
    # No camera keys when quad_image is absent.
    for cam in ("left_eye", "right_eye", "left_wrist", "right_wrist"):
        assert cam not in obs


# ---------------------------------------------------------------------------
# Tests synced from HTTP-side updates (MJPEG, lenient connect)
# ---------------------------------------------------------------------------
def test_split_quad_image_works_at_arbitrary_even_resolution():
    """_split_quad_image splits any (H, W) as long as H and W are even."""
    # 1280×960 (the legacy fixed size) — sanity check.
    quad = np.random.randint(0, 256, size=(960, 1280, 3), dtype=np.uint8)
    cells = MarvainM6HybridRobot._split_quad_image(quad)
    assert set(cells.keys()) == {"left_eye", "right_eye", "left_wrist", "right_wrist"}
    for img in cells.values():
        assert img.shape == (480, 640, 3)

    # Different resolution — still works.
    quad2 = np.random.randint(0, 256, size=(600, 800, 3), dtype=np.uint8)
    cells2 = MarvainM6HybridRobot._split_quad_image(quad2)
    for img in cells2.values():
        assert img.shape == (300, 400, 3)


def test_split_quad_image_right_eye_blank_falls_back_to_left_eye():
    """When right_eye cell is blank, falls back to a copy of left_eye."""
    quad = np.full((960, 1280, 3), 255, dtype=np.uint8)
    # Make right_eye (bottom-right) completely black.
    quad[480:960, 640:1280] = 0
    cells = MarvainM6HybridRobot._split_quad_image(quad)
    # right_eye was blank; should equal left_eye (also white).
    assert np.array_equal(cells["right_eye"], cells["left_eye"])
    assert cells["right_eye"].mean() > 10  # not all-black


def test_connect_does_not_require_joint_states():
    """connect() only verifies HTTP reachability, not joint_states presence."""
    wrapper_mock = _make_wrapper_mock()
    session_mock = MagicMock()
    # /state returns a dict without joint_states — connect should still succeed.
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"quad_image": None}  # no joint_states at all
    session_mock.get.return_value = resp

    with (
        patch(f"{WRAPPER_MODULE}.MarvinRobotWrapper", return_value=wrapper_mock),
        patch(f"{WRAPPER_MODULE}.requests.Session", return_value=session_mock),
    ):
        cfg = MarvainM6HybridRobotConfig(
            id="test_lenient_connect",
            http_base_url="http://127.0.0.1:8010",
            robot_ip="127.0.0.1",
            control_mode="position",
        )
        robot = MarvainM6HybridRobot(cfg)
        robot.connect()  # should NOT raise
        assert robot.is_connected


def test_get_observation_sdk_supplies_gripper_values():
    """get_observation takes gripper positions from the SDK, not HTTP.

    HTTP body has NO gripper_* fields; SDK returns specific degrees.
    The obs dict should reflect the SDK values verbatim.
    """
    sdk_positions = [0.0] * 14 + [12.5, -7.25]
    wrapper_mock = _make_wrapper_mock(joint_positions=sdk_positions)
    session_mock = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    # Intentionally omit gripper_* to prove they don't come from HTTP.
    resp.json.return_value = {"quad_image": None}
    session_mock.get.return_value = resp

    with (
        patch(f"{WRAPPER_MODULE}.MarvinRobotWrapper", return_value=wrapper_mock),
        patch(f"{WRAPPER_MODULE}.requests.Session", return_value=session_mock),
    ):
        cfg = MarvainM6HybridRobotConfig(
            id="test_sdk_gripper",
            http_base_url="http://127.0.0.1:8010",
            robot_ip="127.0.0.1",
            control_mode="position",
        )
        robot = MarvainM6HybridRobot(cfg)
        robot.connect()
        obs = robot.get_observation()

    # Grippers come straight from the SDK (no rad→deg conversion).
    assert obs[f"{robot.config.joint_names[14]}.pos"] == pytest.approx(12.5)
    assert obs[f"{robot.config.joint_names[15]}.pos"] == pytest.approx(-7.25)


def test_get_observation_mjpeg_stream_url():
    """When quad_image uses stream_url, _grab_mjpeg_frame is called."""
    wrapper_mock = _make_wrapper_mock()
    session_mock = MagicMock()
    # /state response uses stream_url (no base64 data).
    state_resp = MagicMock()
    state_resp.raise_for_status = MagicMock()
    state_resp.json.return_value = {
        "joint_states": {"positions": [0.0] * 14},
        "gripper_left": [0.0],
        "gripper_right": [0.0],
        "quad_image": {"stream_url": "/stream/quad.mjpg"},
    }
    session_mock.get.return_value = state_resp

    # Mock the MJPEG stream response (iter_content yields one complete JPEG).
    quad = np.full((960, 1280, 3), 255, dtype=np.uint8)
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(quad, cv2.COLOR_RGB2BGR))
    assert ok
    jpeg_bytes = buf.tobytes()
    # Wrap in MJPEG-style chunks with SOI/EOI markers.
    mjpeg_resp = MagicMock()
    mjpeg_resp.raise_for_status = MagicMock()
    mjpeg_resp.iter_content.return_value = [
        b"\xff\xd8" + jpeg_bytes[2:-2] + b"\xff\xd9"
    ]
    mjpeg_resp.close = MagicMock()

    # First call returns /state, second call returns MJPEG stream.
    # Side effects cover connect-time (state + mjpeg for camera discovery) AND
    # observation-time (state + mjpeg for frame fetch).
    session_mock.get.side_effect = [
        state_resp, mjpeg_resp,  # connect()
        state_resp, mjpeg_resp,  # get_observation()
    ]

    with (
        patch(f"{WRAPPER_MODULE}.MarvinRobotWrapper", return_value=wrapper_mock),
        patch(f"{WRAPPER_MODULE}.requests.Session", return_value=session_mock),
    ):
        cfg = MarvainM6HybridRobotConfig(
            id="test_mjpeg",
            http_base_url="http://127.0.0.1:8010",
            robot_ip="127.0.0.1",
            control_mode="position",
        )
        robot = MarvainM6HybridRobot(cfg)
        robot.connect()
        obs = robot.get_observation()

    # MJPEG stream was hit at least once during observation.
    assert session_mock.get.call_count >= 4
    # All 4 camera keys are present.
    for cam in ("left_eye", "right_eye", "left_wrist", "right_wrist"):
        assert cam in obs
        assert obs[cam].shape == (480, 640, 3)


# ---------------------------------------------------------------------------
# New tests for SDK-as-joint-source behaviour
# ---------------------------------------------------------------------------
def test_get_observation_pulls_joint_values_from_sdk(hybrid_robot):
    """Joint values in obs come straight from wrapper.get_joint_positions."""
    robot, wrapper_mock, _ = hybrid_robot

    # Distinctive, non-symmetric 16-element vector.
    sdk_positions = [float(i) * 1.25 for i in range(16)]
    wrapper_mock.get_joint_positions.return_value = list(sdk_positions)

    robot.connect()
    wrapper_mock.get_joint_positions.reset_mock()

    obs = robot.get_observation()

    wrapper_mock.get_joint_positions.assert_called_once()
    for i, name in enumerate(robot.config.joint_names):
        assert obs[f"{name}.pos"] == pytest.approx(sdk_positions[i])


def test_get_observation_returns_joints_and_cameras_together(hybrid_robot_with_quad):
    """obs contains BOTH 16 joint keys (from SDK) AND 4 camera images (from HTTP)."""
    robot, _, _ = hybrid_robot_with_quad
    robot.connect()
    obs = robot.get_observation()

    joint_keys = {f"{n}.pos" for n in robot.config.joint_names}
    assert joint_keys.issubset(set(obs.keys()))
    for cam in ("left_eye", "right_eye", "left_wrist", "right_wrist"):
        assert cam in obs
        assert obs[cam].shape == (480, 640, 3)


def test_get_observation_raises_when_sdk_disconnects(hybrid_robot):
    """get_observation raises DeviceNotConnectedError if SDK is down."""
    robot, wrapper_mock, _ = hybrid_robot
    robot.connect()

    # Simulate SDK dropping mid-session.
    wrapper_mock.is_connected.return_value = False

    with pytest.raises(DeviceNotConnectedError):
        robot.get_observation()


def test_get_observation_wraps_sdk_runtime_error(hybrid_robot):
    """If the SDK raises inside get_joint_positions, get_observation surfaces DeviceNotConnectedError."""
    robot, wrapper_mock, _ = hybrid_robot
    wrapper_mock.get_joint_positions.side_effect = RuntimeError("robot not connected")
    robot.connect()

    with pytest.raises(DeviceNotConnectedError):
        robot.get_observation()


def test_get_observation_length_mismatch_raises_runtime_error(hybrid_robot):
    """SDK returning the wrong number of joints raises RuntimeError."""
    robot, wrapper_mock, _ = hybrid_robot
    wrapper_mock.get_joint_positions.return_value = [0.0] * 14  # missing grippers
    robot.connect()

    with pytest.raises(RuntimeError, match="SDK returned 14 joints"):
        robot.get_observation()