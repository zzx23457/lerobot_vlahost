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

"""Open-loop action-chunk inference engine.

Every ``chunk_interval_s`` seconds this runs the policy once and hands the
first ``n_action_steps`` actions of the produced chunk to the caller as a
single ``[N, A]`` tensor.  The rollout strategy dispatches the whole chunk in
one shot via ``send_next_action_chunk`` (one HTTP send per chunk) — the robot
executes the chunk open-loop between inferences.

The whole chunk is post-processed in one shot (like the RTC engine), which
sidesteps the per-tick relative-action re-anchoring drift that blocks the sync
engine for relative-action policies.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from copy import copy

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from .base import InferenceEngine

logger = logging.getLogger(__name__)


class ChunkInferenceEngine(InferenceEngine):
    """Open-loop chunk execution: one inference produces one whole-chunk send.

    ``get_action_chunk`` runs the full policy pipeline via
    ``predict_action_chunk`` when at least ``chunk_interval_s`` has elapsed
    since the last chunk, keeps the first ``n_action_steps`` actions
    (post-processed and reordered to the dataset action keys), and returns them
    as a single ``[N, A]`` CPU tensor.  Between intervals it returns ``None`` so
    the control loop simply idles (fresh observations keep flowing but no new
    chunk is emitted).

    ``produces_chunks`` is True so rollout strategies route through
    ``send_next_action_chunk`` (one HTTP send per chunk) rather than the
    per-tick ``send_next_action``.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
        n_action_steps: int,
        chunk_interval_s: float,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type

        if n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")
        if chunk_interval_s < 0:
            raise ValueError(f"chunk_interval_s must be >= 0, got {chunk_interval_s}")
        self._n_action_steps = n_action_steps
        self._chunk_interval_s = chunk_interval_s

        # Episode-scoped state.
        self._last_chunk_t: float | None = None
        self._chunk_len_warned = False

        logger.info(
            "ChunkInferenceEngine initialized (device=%s, n_action_steps=%d, chunk_interval_s=%.3f, "
            "action_keys=%d)",
            self._device,
            self._n_action_steps,
            self._chunk_interval_s,
            len(ordered_action_keys),
        )

    @property
    def produces_chunks(self) -> bool:
        return True

    def start(self) -> None:
        """No background resources to start."""
        logger.info("ChunkInferenceEngine started (inline open-loop chunk mode)")

    def stop(self) -> None:
        """No background resources to stop."""
        logger.info("ChunkInferenceEngine stopped")

    def reset(self) -> None:
        """Reset the policy, processors, and clear the interval timer."""
        logger.info("Resetting chunk inference state (policy + processors)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._last_chunk_t = None

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Not used in chunk mode — the strategy calls ``get_action_chunk``."""
        return None

    def get_action_chunk(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Return a ``[N, A]`` action chunk, or ``None`` when throttled/unavailable."""
        if obs_frame is None:
            return None

        # Check if this is the very first inference (no chunk sent yet)
        is_first_chunk = self._last_chunk_t is None

        # Check if server requests a new chunk via the 'need_new_chunk' flag.
        # Special case: if this is the first chunk, always infer regardless of need_new_chunk value
        # to ensure the robot starts moving immediately.
        need_new_chunk = obs_frame.get("need_new_chunk")
        # print(f"need_new_chunk: {need_new_chunk}")
        if need_new_chunk is not None:
            # Event-driven mode: server explicitly signals when it needs a new chunk
            if need_new_chunk == 0 and not is_first_chunk:
                # Server is still executing the previous chunk, do not infer
                # (but allow first chunk to go through regardless)
                return None
            elif need_new_chunk == 1 or is_first_chunk:
                if is_first_chunk:
                    print(f"🟢 First chunk: inferring regardless of need_new_chunk={need_new_chunk}")
                else:
                    print(f"Inferring new chunk...")
                # Server requests a new chunk, or this is the first chunk - infer immediately

                chunk = self._infer_chunk(obs_frame)
                if chunk is None or chunk.numel() == 0:
                    return None
                self._last_chunk_t = time.perf_counter()
                return chunk
            else:
                logger.warning(
                    "Unexpected need_new_chunk value: %s (expected 0 or 1), falling back to time-based throttling",
                    need_new_chunk
                )

        # Fall back to time-based throttling when need_new_chunk is absent.
        # This maintains backward compatibility with servers that don't send the flag,
        # and also handles the very first inference (before the server can signal readiness).
        now = time.perf_counter()
        if (
            self._last_chunk_t is not None
            and self._chunk_interval_s > 0.0
            and now - self._last_chunk_t < self._chunk_interval_s
        ):
            return None
        print("Using chunk_interval_s")
        chunk = self._infer_chunk(obs_frame)
        if chunk is None or chunk.numel() == 0:
            return None

        self._last_chunk_t = now
        return chunk

    def _infer_chunk(self, obs_frame: dict) -> torch.Tensor | None:
        """Run the full pipeline once and return the first ``n_action_steps`` actions.

        Returns a ``[N, A]`` CPU tensor whose columns follow
        ``ordered_action_keys`` so the caller treats it uniformly.
        """
        # Shallow copy is intentional: the caller builds ``obs_frame`` fresh per
        # tick, so the tensor/array values are not shared with any other reader.
        observation = copy(obs_frame)
        # Remove metadata fields that should not be passed to the inference pipeline
        observation.pop("need_new_chunk", None)

        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)
            # predict_action_chunk returns [1, chunk_size, action_dim]
            chunk = self._policy.predict_action_chunk(observation)
            # Post-process the whole chunk in one shot (matches the RTC engine).
            chunk = self._postprocessor(chunk)

        # [1, T, A] -> [T, A] on CPU.
        chunk = chunk.squeeze(0).cpu()
        if chunk.ndim != 2:
            raise ValueError(f"Expected chunk of shape [T, A] after squeeze, got {tuple(chunk.shape)}")

        chunk_len = chunk.shape[0]
        n = min(self._n_action_steps, chunk_len)
        if self._n_action_steps > chunk_len and not self._chunk_len_warned:
            logger.warning(
                "ChunkInferenceEngine n_action_steps=%d exceeds policy chunk length %d; clamping to %d.",
                self._n_action_steps,
                chunk_len,
                chunk_len,
            )
            self._chunk_len_warned = True

        # Reorder each step to the dataset action-key order, stack to [N, A].
        rows: list[torch.Tensor] = []
        for t in range(n):
            action_dict = make_robot_action(chunk[t], self._dataset_features)
            rows.append(torch.tensor([action_dict[k] for k in self._ordered_action_keys]))
        return torch.stack(rows, dim=0)
