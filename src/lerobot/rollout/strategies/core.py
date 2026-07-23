# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Rollout strategy ABC and shared action-dispatch helper."""

from __future__ import annotations
from pathlib import Path

import abc
import logging
import time
from typing import TYPE_CHECKING

import torch

from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import log_rerun_data

from ..inference import InferenceEngine

if TYPE_CHECKING:
    from ..configs import RolloutStrategyConfig
    from ..context import HardwareContext, ProcessorContext, RolloutContext, RuntimeContext

logger = logging.getLogger(__name__)

if_continue: bool = False
reject_count: int = 0  # 连续拒绝计数器
is_first_chunk: bool = True  # 标记是否是第一次发送chunk

class RolloutStrategy(abc.ABC):
    """Abstract base for rollout execution strategies.

    Each concrete strategy implements a self-contained control loop with
    its own recording/interaction semantics.  Strategies are mutually
    exclusive — only one runs per session.
    """

    def __init__(self, config: RolloutStrategyConfig) -> None:
        self.config = config
        self._engine: InferenceEngine | None = None
        self._interpolator: ActionInterpolator | None = None
        self._warmup_flushed: bool = False
        self._cached_obs_processed: dict | None = None
        

    def _init_engine(self, ctx: RolloutContext) -> None:
        """Attach the inference engine and action interpolator, then start the backend.

        Creates an :class:`ActionInterpolator` from the config's
        ``interpolation_multiplier`` and starts the inference engine.
        Call this from ``setup()`` so strategies share identical
        initialisation without duplicating code.
        """
        self._interpolator = ActionInterpolator(multiplier=ctx.runtime.cfg.interpolation_multiplier)
        self._engine = ctx.policy.inference
        logger.info("Starting inference engine...")
        self._engine.reset()
        self._engine.start()
        self._warmup_flushed = False
        self._cached_obs_processed = None
        logger.info("Inference engine started")

    def _process_observation_and_notify(self, processors: ProcessorContext, obs_raw: dict) -> dict:
        """Run the observation processor and notify the engine — throttled to policy ticks.

        Callers are responsible for calling ``robot.get_observation()`` every loop
        iteration so ``obs_raw`` stays fresh for the action post-processor.  This
        helper gates only the comparatively expensive bits — the processor pipeline
        and ``engine.notify_observation`` — to fire when the interpolator signals
        it needs a new action (once per ``interpolation_multiplier`` ticks).  On
        interpolated ticks the cached ``obs_processed`` is reused.

        With ``interpolation_multiplier == 1`` this is equivalent to the unthrottled
        path: ``needs_new_action()`` is True every tick.

        The cache is implicitly invalidated whenever ``interpolator.reset()`` is
        called (warmup completion, DAgger phase transitions back to AUTONOMOUS),
        because reset makes ``needs_new_action()`` return True on the next call.
        """
        if self._cached_obs_processed is None or self._interpolator.needs_new_action():
            obs_processed = processors.robot_observation_processor(obs_raw)
            self._engine.notify_observation(obs_processed)
            self._cached_obs_processed = obs_processed
        return self._cached_obs_processed

    def _handle_warmup(self, use_torch_compile: bool, loop_start: float, control_interval: float) -> bool:
        """Handle torch.compile warmup phase.

        Returns ``True`` if the caller should ``continue`` (still warming
        up).  On the first post-warmup iteration the engine and
        interpolator are reset so stale warmup state is discarded.
        """
        engine = self._engine
        interpolator = self._interpolator
        if not use_torch_compile:
            return False
        if not engine.ready:
            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            return True
        if not self._warmup_flushed:
            logger.info("Warmup complete — flushing stale state and resuming engine")
            engine.reset()
            interpolator.reset()
            self._warmup_flushed = True
            engine.resume()
        return False

    def _teardown_hardware(self, hw: HardwareContext, return_to_initial_position: bool = True) -> None:
        """Stop the inference engine, optionally return robot to initial position, and disconnect hardware."""
        if self._engine is not None:
            logger.info("Stopping inference engine...")
            self._engine.stop()
        robot = hw.robot_wrapper.inner
        if robot.is_connected:
            if return_to_initial_position and hw.initial_position:
                logger.info("Returning robot to initial position before shutdown...")
                self._return_to_initial_position(hw)
            elif not return_to_initial_position:
                logger.info(
                    "Skipping return-to-initial-position (disabled by config); leaving robot in final pose."
                )
            logger.info("Disconnecting robot...")
            robot.disconnect()
        teleop = hw.teleop
        if teleop is not None and teleop.is_connected:
            logger.info("Disconnecting teleoperator...")
            teleop.disconnect()

    @staticmethod
    def _return_to_initial_position(hw: HardwareContext, duration_s: float = 3.0, fps: int = 50) -> None:
        """Smoothly interpolate the robot back to its initial position."""
        robot = hw.robot_wrapper
        target = hw.initial_position

        # Try using go_home() method if available (more reliable for HTTP robots)
        if hasattr(robot.inner, "go_home"):
            try:
                logger.info("Using robot's go_home() method...")
                if robot.inner.go_home():
                    logger.info("go_home() succeeded")
                    return
                else:
                    logger.warning("go_home() returned False, falling back to interpolation")
            except Exception as e:
                logger.warning("go_home() failed: %s, falling back to interpolation", e)

        # Fallback: smooth interpolation from current position to target
        try:
            current_obs = robot.get_observation()
            current_pos = {k: v for k, v in current_obs.items() if k in target}
            steps = max(int(duration_s * fps), 1)
            for step in range(1, steps + 1):
                t = step / steps
                interp = {}
                for k in current_pos:
                    interp[k] = current_pos[k] * (1 - t) + target[k] * t
                robot.send_action(interp)
                precise_sleep(1 / fps)
        except Exception as e:
            logger.warning("Could not return to initial position: %s", e)

    @staticmethod
    def _log_telemetry(
        obs_processed: dict | None,
        action_dict: dict | None,
        runtime_ctx: RuntimeContext,
    ) -> None:
        """Log observation/action telemetry to Rerun if display_data is enabled."""
        cfg = runtime_ctx.cfg
        if not cfg.display_data:
            return
        log_rerun_data(
            observation=obs_processed,
            action=action_dict,
            compress_images=cfg.display_compressed_images,
        )

    @abc.abstractmethod
    def setup(self, ctx: RolloutContext) -> None:
        """Strategy-specific initialisation (keyboard listeners, buffers, etc.)."""

    @abc.abstractmethod
    def run(self, ctx: RolloutContext) -> None:
        """Main rollout loop.  Returns when shutdown is requested or duration expires."""

    @abc.abstractmethod
    def teardown(self, ctx: RolloutContext) -> None:
        """Cleanup: save dataset, stop threads, disconnect hardware."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def safe_push_to_hub(dataset, tags=None, private=False) -> bool:
    """Push dataset to hub, skipping if no episodes have been saved.

    Returns ``True`` if the push was attempted, ``False`` if skipped.
    """
    if dataset.num_episodes == 0:
        logger.warning("No episodes saved — skipping push to hub")
        return False
    dataset.push_to_hub(tags=tags, private=private)
    return True


def estimate_max_episode_seconds(
    dataset_features: dict,
    fps: float,
    target_size_mb: float = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
) -> float:
    """Conservatively estimate how many seconds of video will exceed *target_size_mb*.

    Each camera produces its own video file, so the episode duration is
    driven by the **slowest** camera to fill ``target_size_mb`` — i.e.
    the one with the fewest pixels per frame (lowest bitrate).

    Uses a deliberately **low** bits-per-pixel estimate so the computed
    duration is *longer* than reality.  By the time the timer fires the
    actual video file is guaranteed to have crossed the target size,
    which aligns episode boundaries with the dataset's video-file
    chunking — each ``push_to_hub`` uploads complete files rather than
    re-uploading a still-growing one.

    The estimate ignores codec-specific settings (CRF, preset) on purpose:
    we only need a rough lower bound on bitrate, not a precise prediction.

    Falls back to 300 s (5 min) when no video features are present.
    """
    # 0.1 bits-per-pixel is a *low* estimate for CRF-30 streaming video of
    # robot footage (real-world is typically 0.1 – 0.3 bpp).  Under-
    # estimating the bitrate over-estimates the time → the episode will be
    # *larger* than target_size_mb when we save, which is what we want.
    conservative_bpp = 0.1

    # Collect per-camera pixel counts — each camera has its own video file.
    camera_pixels = []
    for feat in dataset_features.values():
        if feat.get("dtype") == "video":
            shape = feat.get("shape", ())

            # (H, W, C) — bits-per-pixel is a per-spatial-pixel metric,
            # so we exclude the channel dimension from the count.
            if len(shape) == 3:
                pixels = shape[0] * shape[1]
                camera_pixels.append(pixels)
            else:
                raise ValueError(f"Unexpected video feature shape: {shape}")

    if not camera_pixels:
        return 300.0

    # Use the smallest camera: it produces the lowest bitrate and therefore
    # takes the longest to reach the target — the conservative choice.
    min_pixels = min(camera_pixels)
    bits_per_frame = min_pixels * conservative_bpp
    bytes_per_second = (bits_per_frame * fps) / 8

    # Guard against division by zero just in case
    if bytes_per_second <= 0:
        return 300.0

    return (target_size_mb * 1024 * 1024) / bytes_per_second


# ---------------------------------------------------------------------------
# Shared action-dispatch helper
# ---------------------------------------------------------------------------


def send_next_action(
    obs_processed: dict,
    obs_raw: dict,
    ctx: RolloutContext,
    interpolator: ActionInterpolator,
) -> dict | None:
    """Dispatch the next action to the robot.

    Pulls the next action tensor from the inference engine, feeds the
    interpolator, and sends the interpolated action through the
    ``robot_action_processor`` to the robot.  Works identically for
    sync and async backends — the rollout strategy never needs to branch.

    Returns the action dict that was sent, or ``None`` if no action was
    ready (e.g. empty async queue, interpolator not yet primed).
    """
    engine = ctx.policy.inference
    features = ctx.data.dataset_features
    ordered_keys = ctx.data.ordered_action_keys

    if interpolator.needs_new_action():
        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
        action_tensor = engine.get_action(obs_frame)
        # print("action_tensor shape:", action_tensor.shape)
        if action_tensor is not None:
            interpolator.add(action_tensor.cpu())

    interp = interpolator.get()
    if interp is None:
        return None

    if len(interp) != len(ordered_keys):
        raise ValueError(f"Interpolated tensor length ({len(interp)}) != action keys ({len(ordered_keys)})")
    action_dict = {k: interp[i].item() for i, k in enumerate(ordered_keys)}
    processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
    ctx.hardware.robot_wrapper.send_action(processed)
    # print(f"---------------- action size:",
    # print("action_dict 键数量:", len(action_dict))
    return action_dict


def send_next_action_chunk(
    obs_processed: dict,
    obs_raw: dict,
    ctx: RolloutContext,
) -> dict | None:
    """Dispatch a whole action chunk to the robot in a single send.

    Pulls a ``[N, A]`` chunk from the inference engine (``get_action_chunk``),
    runs each row through the ``robot_action_processor``, and hands the full
    list of processed action dicts to ``robot.send_action_chunk`` so the robot
    driver can transmit them in one HTTP request.  The robot executes the chunk
    open-loop until the next chunk is dispatched.

    Returns the last action dict in the chunk (for telemetry), or ``None`` when
    no chunk was ready (e.g. the engine is still within its interval).
    """
    global if_continue, reject_count
    engine = ctx.policy.inference
    features = ctx.data.dataset_features
    ordered_keys = ctx.data.ordered_action_keys

    obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)

    # Pass through need_new_chunk if present in raw observation (先传递原始值)
    if "need_new_chunk" in obs_raw:
        obs_frame["need_new_chunk"] = obs_raw["need_new_chunk"]

    # 如果 if_continue 为 True，强制覆盖为 1（优先级更高）
    if if_continue:
        if_continue = False
        obs_frame["need_new_chunk"] = 1
        print("⚠️ if_continue=True, forcing need_new_chunk=1 to request new chunk")

    chunk = engine.get_action_chunk(obs_frame)

    if engine.failed:
        return None
    if chunk is None or len(chunk) == 0:
        return None

    # 检查 chunk 的第一帧与输入的 obs_frame 的机械臂 14 个关节角度
    if chunk.shape[1] == len(ordered_keys):
        # 提取 obs_frame 中的 14 个关节位置（排除 gripper）
        arm_joint_keys = [k for k in ordered_keys if 'gripper' not in k.lower()][:14]

        # 从 obs_raw 中获取当前关节位置
        current_positions = []
        for joint_key in arm_joint_keys:
            obs_key = f"{joint_key}.pos"  # 转换为 observation 格式
            if obs_key in obs_raw:
                current_positions.append(obs_raw[obs_key])
            else:
                # 如果找不到，尝试直接用 joint_key
                if joint_key in obs_raw:
                    current_positions.append(obs_raw[joint_key])
                else:
                    print(f"⚠️ Warning: joint {joint_key} not found in obs_raw")
                    current_positions.append(0.0)  # 默认值

        # 提取 chunk 第一帧的 14 个关节目标位置
        first_action = chunk[0][:14]  # 假设前 14 个是关节，后 2 个是 gripper

        # 计算每个关节的位置差距（模型输出已经是角度，直接比较角度值）
        max_diff = 0.0
        max_diff_joint = None
        threshold_deg = 1.0  # 5 度阈值

        for i, (current, target) in enumerate(zip(current_positions, first_action)):
            diff = abs(float(target) - float(current))
            if diff > max_diff:
                max_diff = diff
                max_diff_joint = arm_joint_keys[i] if i < len(arm_joint_keys) else f"joint_{i}"

        # 第一次发送chunk，直接执行降级方案（无条件）
        global is_first_chunk
        if is_first_chunk:
            print("🟦 第一次发送chunk，启用线性插值降级方案（无条件）")
            is_first_chunk = False  # 标记已经不是第一次了

            chunk_size = chunk.shape[0]
            K = 10  # 插值位置参数：前K帧用插值替换

            # 保存原始 chunk 的副本
            original_chunk = chunk.clone()

            # 对所有14个关节执行插值（第一次无条件执行）
            for i, joint_key in enumerate(arm_joint_keys[:16]):
                current_val = current_positions[i]

                if chunk_size > K:
                    # 目标：原 chunk 的第 K 帧（索引 K-1）
                    target_val_at_K = float(original_chunk[K-1, i])

                    print(f"   → 关节 {joint_key}: 当前={current_val:.2f}°, chunk第{K}帧={target_val_at_K:.2f}°")

                    # 前K帧：从 current_val 插值到原 chunk 第K帧
                    linear_trajectory = torch.linspace(current_val, target_val_at_K, K,
                                                      dtype=chunk.dtype, device=chunk.device)
                    chunk[:K, i] = linear_trajectory

                    print(f"      前{K}帧插值到第{K}帧，后{chunk_size-K}帧保持原chunk不变")
                else:
                    # chunk 总长度 <= K，全部插值到最后一帧
                    target_val = float(original_chunk[-1, i])
                    linear_trajectory = torch.linspace(current_val, target_val, chunk_size,
                                                      dtype=chunk.dtype, device=chunk.device)
                    chunk[:, i] = linear_trajectory
                    print(f"      整个chunk({chunk_size}帧)全部插值")

            print(f"✓ 完成第一次chunk降级处理，继续发送")

        # 非第一次：检查阈值
        elif max_diff > threshold_deg:
            reject_count += 1
            print(f"⚠️ Chunk rejected ({reject_count}/2): max joint diff = {max_diff:.4f}° at {max_diff_joint}")
            print(f"   Threshold = {threshold_deg:.1f}°")

            # 连续拒绝 2 次，使用线性插值降级方案
            if reject_count >= 2:
                print("🔴 连续拒绝 2 次，启用线性插值降级方案")
                reject_count = 0  # 重置计数器

                chunk_size = chunk.shape[0]
                K = 40  # 插值位置参数：前K帧用插值替换

                # 保存原始 chunk 的副本
                original_chunk = chunk.clone()

                # 逐个检查前 14 个关节，只对超出阈值的关节进行插值替换
                for i, joint_key in enumerate(arm_joint_keys[:14]):
                    current_val = current_positions[i]
                    diff = abs(float(first_action[i]) - float(current_val))

                    if diff > threshold_deg:
                        # 该关节超出阈值，使用新的插值策略
                        if chunk_size > K:
                            # 目标：原 chunk 的第 K 帧（索引 K-1）
                            target_val_at_K = float(original_chunk[K-1, i])

                            print(f"   → 关节 {joint_key}: 当前={current_val:.2f}°, chunk第{K}帧={target_val_at_K:.2f}°, 差距={diff:.2f}°")

                            # 前K帧：从 current_val 插值到原 chunk 第K帧
                            linear_trajectory = torch.linspace(current_val, target_val_at_K, K,
                                                              dtype=chunk.dtype, device=chunk.device)
                            chunk[:K, i] = linear_trajectory

                            # 第K+1帧到第N帧：保持原 chunk 不变（已经是原chunk的值，不需要修改）
                            print(f"      前{K}帧插值到第{K}帧，后{chunk_size-K}帧保持原chunk不变")
                        else:
                            # chunk 总长度 <= K，全部插值到最后一帧
                            target_val = float(original_chunk[-1, i])
                            linear_trajectory = torch.linspace(current_val, target_val, chunk_size,
                                                              dtype=chunk.dtype, device=chunk.device)
                            chunk[:, i] = linear_trajectory
                            print(f"      整个chunk({chunk_size}帧)全部插值")

                print(f"✓ 完成降级处理，继续发送 chunk")
            else:
                # 第一次拒绝，继续尝试
                if_continue = True
                return None
        else:
            # 通过检查，重置计数器
            reject_count = 0
            print(f"✓ Chunk accepted: max joint diff = {max_diff:.4f}°")


    if chunk.shape[1] != len(ordered_keys):
        raise ValueError(
            f"Chunk action width ({chunk.shape[1]}) != action keys ({len(ordered_keys)})"
        )

    # 在发送之前，检查并平滑每个关节的轨迹：如果中间有波峰/波谷偏离起点和终点超过 10 度，
    # 将终点设置为该波峰/波谷的值，然后从起点线性插值到新终点
    chunk_size = chunk.shape[0]
    num_joints = chunk.shape[1]
    wave_threshold_deg = 100.0

    # 只检查前 14 个关节（排除 gripper）
    arm_joint_count = min(14, num_joints)

    for joint_idx in range(arm_joint_count):
        # 提取该关节的完整轨迹
        trajectory = chunk[:, joint_idx]

        # 起点和终点
        start_val = float(trajectory[0])
        end_val = float(trajectory[-1])

        # 找到轨迹中的最大值和最小值
        max_val = float(trajectory.max())
        min_val = float(trajectory.min())

        # 检查波峰和波谷是否偏离起点和终点都超过 10 度
        max_deviation = max(abs(max_val - start_val), abs(max_val - end_val))
        min_deviation = max(abs(min_val - start_val), abs(min_val - end_val))

        # 优先处理偏离更大的那个
        if max_deviation > wave_threshold_deg and max_deviation >= min_deviation:
            print(f"⚠️ Joint {joint_idx} ({ordered_keys[joint_idx] if joint_idx < len(ordered_keys) else joint_idx}): "
                  f"peak {max_val:.2f}° deviates {max_deviation:.2f}° from endpoints")
            print(f"   → 将终点改为 {max_val:.2f}°，线性插值: {start_val:.2f}° → {max_val:.2f}°")

            # 从起点线性插值到波峰值
            linear_trajectory = torch.linspace(start_val, max_val, chunk_size,
                                              dtype=chunk.dtype, device=chunk.device)
            chunk[:, joint_idx] = linear_trajectory

        elif min_deviation > wave_threshold_deg:
            print(f"⚠️ Joint {joint_idx} ({ordered_keys[joint_idx] if joint_idx < len(ordered_keys) else joint_idx}): "
                  f"valley {min_val:.2f}° deviates {min_deviation:.2f}° from endpoints")
            print(f"   → 将终点改为 {min_val:.2f}°，线性插值: {start_val:.2f}° → {min_val:.2f}°")

            # 从起点线性插值到波谷值
            linear_trajectory = torch.linspace(start_val, min_val, chunk_size,
                                              dtype=chunk.dtype, device=chunk.device)
            chunk[:, joint_idx] = linear_trajectory

    processed_actions: list[dict] = []
    for row in chunk:
        action_dict = {k: row[i].item() for i, k in enumerate(ordered_keys)}
        processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
        processed_actions.append(processed)

    robot = ctx.hardware.robot_wrapper
    if not hasattr(robot.inner, "send_action_chunk"):
        raise AttributeError(
            f"Robot '{robot.name}' does not implement send_action_chunk; "
            "chunk inference requires a robot that can dispatch an action chunk."
        )
    # 记录 action chunk 到文件
    chunk_dir = Path("/home/zzx23457/文档/test_chunk")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = chunk_dir / "record_chunk.txt"

    # 获取当前是第几个 chunk（通过读取文件行数）
    if chunk_file.exists():
        with open(chunk_file, "r") as f:
            chunk_count = sum(1 for line in f if line.startswith("=== Chunk"))
    else:
        chunk_count = 0

    chunk_count += 1

    # 追加写入
    with open(chunk_file, "a") as f:
        f.write(f"=== Chunk {chunk_count} ===\n")

        # 收集所有关节位置作为 robot state（按顺序）
        joint_names = [
            'left_arm_joint_1.pos', 'left_arm_joint_2.pos', 'left_arm_joint_3.pos',
            'left_arm_joint_4.pos', 'left_arm_joint_5.pos', 'left_arm_joint_6.pos',
            'left_arm_joint_7.pos', 'right_arm_joint_1.pos', 'right_arm_joint_2.pos',
            'right_arm_joint_3.pos', 'right_arm_joint_4.pos', 'right_arm_joint_5.pos',
            'right_arm_joint_6.pos', 'right_arm_joint_7.pos', 'left_gripper.pos',
            'right_gripper.pos'
        ]

        robot_state = []
        for joint in joint_names:
            if joint in obs_raw:
                robot_state.append(obs_raw[joint])
            elif joint in obs_processed:
                robot_state.append(obs_processed[joint])

        if robot_state:
            f.write(f"Robot State (16 joints): {robot_state}\n")
        else:
            f.write("Robot State: Not found in observation\n")

        f.write(f"Actions ({len(processed_actions)} total):\n")
        for i, action in enumerate(processed_actions):
            # action 是 list/tuple，转成字符串写入
            f.write(f"  Action {i}: {action}\n")
        f.write("\n")

    robot.send_action_chunk(processed_actions)
    print("action_chunk 长度:", len(processed_actions))
    # Return the last action for telemetry/logging.
    return {k: chunk[-1][i].item() for i, k in enumerate(ordered_keys)}
