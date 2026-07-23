# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Replays the actions of an episode from a dataset on a robot using chunk-based sending.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Examples:

```shell
lerobot-replay-chunk \
    --robot.type=marvain_m6_http \
    --robot.id=black \
    --robot.http_base_url=http://192.168.10.123:8010 \
    --dataset.repo_id=<USER>/record-test \
    --dataset.episode=0 \
    --chunk_size=100
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    marvain_m6_http,
    marvain_m6_hybrid,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, log_say


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path'). If None, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Chunk size for action chunks
    chunk_size: int = 100
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Poll interval in seconds for checking need_new_chunk
    poll_interval: float = 0.01


@parser.wrap()
def replay_chunk(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    actions = dataset.select_columns(ACTION)
    total_frames = dataset.num_frames

    # 检查机器人是否支持 send_action_chunk
    if not hasattr(robot, "send_action_chunk"):
        raise AttributeError(
            f"Robot '{robot.__class__.__name__}' does not implement send_action_chunk; "
            "chunk replay requires a robot that can dispatch action chunks."
        )

    robot.connect()

    try:
        log_say("Replaying episode with chunk mode", cfg.play_sounds, blocking=True)

        # 分割成多个 chunks
        chunk_count = 0
        for start_idx in range(0, total_frames, cfg.chunk_size):
            end_idx = min(start_idx + cfg.chunk_size, total_frames)
            chunk_actions = []

            logging.info(f"Preparing chunk {chunk_count + 1}: frames {start_idx} to {end_idx - 1}")

            # 准备当前 chunk 的所有 actions
            for idx in range(start_idx, end_idx):
                action_array = actions[idx][ACTION]
                action = {}
                for i, name in enumerate(dataset.features[ACTION]["names"]):
                    action[name] = action_array[i]

                # 获取当前观测（用于 processor）
                robot_obs = robot.get_observation()
                processed_action = robot_action_processor((action, robot_obs))
                chunk_actions.append(processed_action)

            chunk_count += 1
            print(f"\n{'='*60}")
            print(f"Sending chunk {chunk_count}, size={len(chunk_actions)}")
            print(f"Frames: {start_idx} to {end_idx - 1}")
            print(f"{'='*60}")

            # 发送 chunk
            start_time = time.perf_counter()
            robot.send_action_chunk(chunk_actions)
            send_duration = time.perf_counter() - start_time
            print(f"✓ Chunk sent in {send_duration*1000:.2f} ms")

            # 如果不是最后一个 chunk，等待 need_new_chunk
            if end_idx < total_frames:
                print("Waiting for robot to signal need_new_chunk...")
                wait_start = time.perf_counter()

                while True:
                    robot_obs = robot.get_observation()
                    if robot_obs.get("need_new_chunk", False):
                        wait_duration = time.perf_counter() - wait_start
                        print(f"✓ Robot ready for next chunk (waited {wait_duration:.3f}s)")
                        break
                    time.sleep(cfg.poll_interval)
            else:
                print("✓ Last chunk sent, replay complete")

        print(f"\n{'='*60}")
        print(f"Replay finished: {chunk_count} chunks, {total_frames} frames total")
        print(f"{'='*60}\n")

    finally:
        robot.disconnect()


def main():
    register_third_party_plugins()
    replay_chunk()


if __name__ == "__main__":
    main()
