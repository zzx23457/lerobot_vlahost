"""Process manager for robot interaction workflows.

This module manages the lifecycle of:
  - deploy.py / replay.py / show_cameras.py (Python, --config yaml)
  - workflows/data_processing/*.py        (Python, --xxx CLI flags)
  - workflows/model_training/*.sh         (Bash, env-overrides + <phase> arg)

Three launch modes share one process core (``_launch_command``) for log streaming,
stop, and cleanup. We never let the UI inject arbitrary commands or env vars —
each script path is whitelisted and each training env var is whitelisted in
``_build_training_env``.
"""

import atexit
import ctypes
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal, Any

from .config_manager import (
    UnifiedRobotConfig,
    DataProcessingConfig,
    ModelTrainingConfig,
    dump_to_tempfile,
    DATA_PROCESSING_OPERATIONS,
    TRAINING_SCRIPTS,
    TRAINING_PHASES,
)


# ---------------------------------------------------------------------------
# Script path whitelist (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent.parent

DATA_PROCESSING_SCRIPTS: dict[str, Path] = {
    "sanity":     REPO_ROOT / "workflows" / "data_processing" / "sanity_check.py",
    "clean":      REPO_ROOT / "workflows" / "data_processing" / "clean_dirty_episodes.py",
    "merge":      REPO_ROOT / "workflows" / "data_processing" / "merge_two_datasets.py",
    "ts_check":   REPO_ROOT / "workflows" / "data_processing" / "check_timestamp_alignment.py",
    "v2_convert":          REPO_ROOT / "workflows" / "data_processing" / "v2_convert.py",
    "v2_convert_next_joint": REPO_ROOT / "workflows" / "data_processing" / "v2_convert_next_joint.py",
}

TRAINING_BASH_SCRIPTS: dict[str, Path] = {
    "act":      REPO_ROOT / "workflows" / "model_training" / "train_act.sh",
    "smolvla":  REPO_ROOT / "workflows" / "model_training" / "train_smolvla.sh",
    "finetune": REPO_ROOT / "workflows" / "model_training" / "finetune_act.sh",
}


# Mirror deploy.py's _child_preexec so the rollout child sits in its own session
# and gets SIGTERM if the Gradio parent is killed (avoids orphaning an energized
# arm). deploy.py already does this for its grandchild; doing it here too means
# the UI-launched child behaves identically to a directly-launched one.
_PR_SET_PDEATHSIG = 1


def _child_preexec() -> None:
    os.setsid()
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except OSError:
        # Non-Linux: best-effort only.
        pass


# ---------------------------------------------------------------------------
# Pure builders (no I/O, fully unit-testable)
# ---------------------------------------------------------------------------

def _build_data_processing_args(
    config: DataProcessingConfig,
) -> tuple[Path, list[str]]:
    """Construct (script_path, argv) for a data processing operation.

    Boolean flags are only emitted when True; empty values are skipped.
    """
    if config.operation not in DATA_PROCESSING_OPERATIONS:
        raise ValueError(f"Unknown operation: {config.operation!r}")

    op = config.operation
    # v2_convert 包含两种 variant, 需要选具体脚本路径
    if op == "v2_convert":
        variant = getattr(config.v2_convert, "variant", "standard")
        script_key = "v2_convert_next_joint" if variant == "next_joint" else "v2_convert"
        script_path = DATA_PROCESSING_SCRIPTS[script_key]
    else:
        script_path = DATA_PROCESSING_SCRIPTS[op]

    args: list[str] = []

    if op == "sanity":
        args.extend(["--dataset-root", config.dataset_path])
        if config.sanity.n_samples:
            args.extend(["--n-samples", str(int(config.sanity.n_samples))])

    elif op == "clean":
        args.extend(["--dataset-path", config.dataset_path])
        # report_only 模式下 output_path 也可不传
        if config.output_path and not config.clean.report_only:
            args.extend(["--output-path", config.output_path])
        if config.clean.dry_run:
            args.append("--dry-run")
        if config.clean.report_only:
            args.append("--report-only")
        args.extend(["--zero-threshold", str(int(config.clean.zero_threshold))])

    elif op == "merge":
        # 多 source
        sources = [
            line.strip()
            for line in (config.merge.source_roots_text or "").splitlines()
            if line.strip()
        ]
        for src in sources:
            args.extend(["--source-root", src])
        if config.output_path:
            args.extend(["--output-root", config.output_path])
        if config.merge.repo_id:
            args.extend(["--repo-id", config.merge.repo_id])
        if config.merge.video_files_size_mb:
            args.extend(["--video-files-size-mb", str(config.merge.video_files_size_mb)])

    elif op == "ts_check":
        args.extend(["--dataset-root", config.dataset_path])
        if config.timestamp.video_key:
            args.extend(["--video-key", config.timestamp.video_key])
        if config.timestamp.tolerance_ms:
            args.extend(["--tolerance-ms", str(config.timestamp.tolerance_ms)])
        if config.timestamp.report_output:
            args.extend(["--output", config.timestamp.report_output])
        if config.timestamp.output_format:
            args.extend(["--format", config.timestamp.output_format])

    elif op == "v2_convert":
        args.extend(["--dataset-root", config.dataset_path])
        if config.v2_convert.output_root:
            args.extend(["--output-root", config.v2_convert.output_root])
        elif config.v2_convert.v2_suffix:
            args.extend(["--v2-suffix", config.v2_convert.v2_suffix])
        # camera_enabled 序列化为 "1,1,1,1"
        cam_list = [
            "1" if bool(config.v2_convert.camera_enabled.get(k, True)) else "0"
            for k in ("left_eye", "right_eye", "left_wrist", "right_wrist")
        ]
        args.extend(["--camera-enabled", ",".join(cam_list)])
        if config.v2_convert.dry_run:
            args.append("--dry-run")

    return script_path, args


def _build_training_env(config: ModelTrainingConfig) -> dict[str, str]:
    """Construct the env-var overrides for a training launch.

    Only keys with non-empty values are returned (so empty string doesn't
    override the script default). Boolean → ``"true"``/``"false"``.
    """
    env: dict[str, str] = {}
    script = config.script
    phase = config.phase
    opt = config.optimization
    trk = config.tracking

    if script in ("act", "smolvla"):
        if config.dataset_root:
            env["DATASET_ROOT"] = config.dataset_root
    if config.output_root:
        env["OUTPUT_ROOT"] = config.output_root

    # ACT / SmolVLA / Fine-tune 都用 optimization + tracking (脚本都支持这些 env)
    if script in ("act", "smolvla", "finetune"):
        env["BATCH_SIZE"] = str(int(opt.batch_size))
        env["STEPS"] = str(int(opt.steps))
        env["EVAL_FREQ"] = str(int(opt.eval_freq))
        env["SAVE_FREQ"] = str(int(opt.save_freq))
        env["LOG_FREQ"] = str(int(opt.log_freq))
        env["WANDB_ENABLE"] = "true" if trk.wandb_enable else "false"
        env["WANDB_PROJECT"] = trk.wandb_project  # 空字符串会覆盖脚本默认, 但无害
        env["PUSH_TO_HUB"] = "true" if trk.push_to_hub else "false"

    if script == "smolvla":
        sm = config.smolvla
        env["POLICY_CHUNK_SIZE"] = str(int(sm.policy_chunk_size))
        env["POLICY_N_ACTION_STEPS"] = str(int(sm.policy_n_action_steps))
        env["POLICY_LR"] = str(sm.policy_lr)
        if sm.policy_path:
            env["POLICY_PATH"] = sm.policy_path
        env["LOAD_VLM_WEIGHTS"] = "true" if sm.load_vlm_weights else "false"
        env["FREEZE_VISION_ENCODER"] = "true" if sm.freeze_vision_encoder else "false"
        env["TRAIN_EXPERT_ONLY"] = "true" if sm.train_expert_only else "false"
        if sm.hf_endpoint:
            env["HF_ENDPOINT"] = sm.hf_endpoint
        if sm.rename_map:
            # JSON 字符串（紧凑，不带空格 — train_smolvla.sh:95-100 要求）
            env["RENAME_MAP"] = json.dumps(sm.rename_map, separators=(",", ":"))

    elif script == "finetune":
        ft = config.finetune
        if ft.pretrained_ckpt:
            env["PRETRAINED_CKPT"] = ft.pretrained_ckpt
        if ft.new_dataset:
            env["NEW_DATASET"] = ft.new_dataset

    # phase 单独作为 argv 位置参数传给 bash 脚本，不进 env
    _ = phase
    return env


# ---------------------------------------------------------------------------
# Exit-code interpretation helpers
# ---------------------------------------------------------------------------
def interpret_exit_code(operation: str, returncode: int | None) -> tuple[bool, str]:
    """Return (is_success, human_msg) given an operation and its exit code.

    Used by ``launch_data_processing`` to distinguish infrastructure failures
    from business outcomes (e.g. ts_check finding drift).
    """
    if returncode is None:
        return False, "❌ 进程异常结束"
    if returncode == 0:
        return True, f"✅ 完成 (exit 0)"
    if operation == "ts_check" and returncode == 1:
        # ts_check 用 1 表示 "发现 drift" — 业务结果，不是失败
        return True, "⚠️ 完成：发现时间戳偏移 (exit 1)"
    if operation in ("v2_convert", "v2_convert_next_joint", "v2_convert_standard") and returncode in (1, 2, 3):
        labels = {
            1: "配置错误",
            2: "v1 数据集不存在",
            3: "转换失败",
        }
        return False, f"❌ v2_convert 失败: {labels[returncode]} (exit {returncode})"
    return False, f"❌ 失败 (exit {returncode})"


def interpret_training_exit_code(script: str, phase: str, returncode: int | None) -> tuple[bool, str]:
    """Interpret exit code for a training script invocation.

    For now: only ``exit 0`` is success; anything else is a failure.
    Different scripts may have richer semantics later.
    """
    if returncode is None:
        return False, "❌ 训练进程异常结束"
    if returncode == 0:
        return True, f"✅ {script}/{phase} 训练完成 (exit 0)"
    return False, f"❌ {script}/{phase} 训练失败 (exit {returncode})"


# ===========================================================================
# ProcessManager
# ===========================================================================

class ProcessManager:
    """Manages subprocess lifecycle for all workflows.

    Single process core: ``_launch_command``. The ``_launch_script`` and
    ``_launch_bash_script`` wrappers are thin adapters that build the right
    command + env for each kind of target.
    """

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_queues: dict[str, queue.Queue] = {}
        self.log_buffers: dict[str, deque] = {}
        self.monitor_threads: dict[str, threading.Thread] = {}
        # (log_path, file_handle) per process — needed so we can close the log
        # file cleanly and tail it from the reader thread.
        self._log_files: dict[str, tuple[Path, "typing.IO[bytes]"]] = {}
        self._current_process_name: str | None = None
        # Per-process metadata for completion interpretation: name -> dict
        # e.g. {"kind": "data_processing"|"model_training", "operation": "clean", "script": "smolvla"}
        self._process_meta: dict[str, dict] = {}
        # 记录已被通知过完成事件的 process name (避免同一完成被多次弹窗)
        self._notified_completion: set[str] = set()
        # 新完成的 process 累积列表, 每次 get_changes() 调用后清空
        self._pending_notices: list[tuple[str, str]] = []  # (process_name, human_message)

        # Register cleanup on exit
        atexit.register(self._cleanup_all)

    # -- log streaming --------------------------------------------------------

    def _stream_output(self, process_name: str, log_path: Path):
        """Tail the log file and append new lines to the in-memory buffer.

        Method B: stdout/stderr go to a file (no kernel pipe → no backpressure).
        We poll the file at ~50ms intervals; this is fast enough that the UI's
        0.5s refresh timer always sees fresh lines.
        """
        log_queue = self.log_queues[process_name]
        log_buffer = self.log_buffers[process_name]
        pos = 0
        try:
            while True:
                process = self.processes.get(process_name)
                running = process is not None and process.poll() is None
                pos = self._drain_log_file(log_path, pos, log_queue, log_buffer)
                if not running:
                    # Final drain after process exit
                    self._drain_log_file(log_path, pos, log_queue, log_buffer)
                    break
                time.sleep(0.05)
        except Exception as e:
            log_queue.put(f"[ERROR] Log streaming error: {e}")

    def _drain_log_file(
        self,
        log_path: Path,
        pos: int,
        log_queue: queue.Queue,
        log_buffer: deque,
    ) -> int:
        """Read any new bytes from log_path since `pos` and feed lines to the buffer."""
        try:
            if not log_path.exists():
                return pos
            with open(log_path, "rb") as f:
                f.seek(pos)
                new_data = f.read()
                new_pos = f.tell()
            if new_data:
                for line in new_data.splitlines():
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    log_queue.put(decoded)
                    log_buffer.append(decoded)
                    if len(log_buffer) > 1000:
                        log_buffer.popleft()
            return new_pos
        except FileNotFoundError:
            return pos

    # -- core launch ----------------------------------------------------------

    def _launch_command(
        self,
        process_name: str,
        cmd: list[str],
        *,
        env_overrides: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> tuple[bool, str]:
        """Spawn ``cmd`` and wire log streaming + status bookkeeping.

        ``env_overrides`` are merged on top of ``os.environ.copy()`` and only
        affect the child — the Gradio parent process is never mutated.
        """
        if self.is_running(process_name):
            return False, f"❌ {process_name} is already running"

        # Initialize log structures
        self.log_queues[process_name] = queue.Queue()
        self.log_buffers[process_name] = deque(maxlen=1000)

        # Method B: redirect stdout/stderr to a log file. Eliminates the kernel
        # pipe → no 64KB backpressure, no GIL contention with the reader
        # thread. 4 层 parent：ui/ → robot_interaction/ → workflows/ → <REPO_ROOT>。
        repo_root = Path(__file__).parent.parent.parent.parent
        # 使用毫秒级时间戳避免同秒重启同 process_name 的日志文件碰撞
        log_ts = int(time.time() * 1000)
        log_path = (repo_root / "outputs" / "deploy_logs" / f"{process_name}_{log_ts}.log").resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "wb")
        self._log_files[process_name] = (log_path, log_fh)

        try:
            child_env = os.environ.copy()
            if env_overrides:
                child_env.update(env_overrides)
            child_env["PYTHONUNBUFFERED"] = "1"  # 强制 child stdout 行缓冲

            process = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=cwd if cwd is not None else repo_root,
                env=child_env,
                preexec_fn=_child_preexec,
            )

            self.processes[process_name] = process
            self._current_process_name = process_name

            # Start log monitoring thread (file tailer)
            monitor_thread = threading.Thread(
                target=self._stream_output,
                args=(process_name, log_path),
                daemon=True,
            )
            monitor_thread.start()
            self.monitor_threads[process_name] = monitor_thread

            # Wait a moment to check for immediate failures, but DON'T treat
            # quick normal exits as "failed to start" — that's handled by
            # the caller via interpret_exit_code().
            time.sleep(0.5)
            if process.poll() is not None and process.returncode != 0:
                # Process exited with non-zero within 0.5s — likely immediate failure.
                stderr = self.get_logs(process_name, last_n_lines=20)
                return False, f"❌ Process failed to start:\n{stderr}"

            if process.poll() is not None:
                # Quick normal exit (e.g. env check, sanity on tiny data).
                # Don't tear down; let caller decide via interpret_exit_code.
                rc = process.returncode
                return True, f"✅ {process_name} 已完成 (exit {rc}, log: {log_path})"

            return True, f"✅ {process_name} started successfully (PID: {process.pid}, log: {log_path})"

        except Exception as e:
            try:
                log_fh.close()
            except Exception:
                pass
            self._log_files.pop(process_name, None)
            return False, f"❌ Failed to launch {process_name}: {e}"

    # -- script launchers (thin adapters) -------------------------------------

    def _launch_script(
        self,
        process_name: str,
        script_name: str,
        config: UnifiedRobotConfig,
        custom_args: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Launch a Python workflow script with ``--config <tmp_yaml>``."""
        # Resolve script path against workflows/robot_interaction/
        script_path = Path(__file__).parent.parent / script_name
        if not script_path.exists():
            return False, f"❌ Script not found: {script_path}"

        cmd = [sys.executable, str(script_path)]
        if custom_args is None:
            raise RuntimeError(
                f"{process_name}: custom_args is required (yaml-centric 架构要求 --config)"
            )
        cmd.extend(custom_args)
        return self._launch_command(process_name, cmd)

    def _launch_bash_script(
        self,
        process_name: str,
        bash_script: Path,
        phase: str,
        env_overrides: dict[str, str],
    ) -> tuple[bool, str]:
        """Launch a bash training script with whitelisted env overrides."""
        if not bash_script.exists():
            return False, f"❌ Bash script not found: {bash_script}"
        cmd = ["bash", str(bash_script), phase]
        return self._launch_command(process_name, cmd, env_overrides=env_overrides)

    # -- existing robot launchers ---------------------------------------------

    def launch_deploy(self, config: UnifiedRobotConfig) -> tuple[bool, str]:
        """Launch deploy.py with config (serialized to a temp yaml)."""
        tmp = dump_to_tempfile(config)
        return self._launch_script(
            "deploy",
            "deploy.py",
            config,
            custom_args=["--config", str(tmp)],
        )

    def launch_replay(self, config: UnifiedRobotConfig) -> tuple[bool, str]:
        """Launch replay.py with config (serialized to a temp yaml)."""
        tmp = dump_to_tempfile(config)
        return self._launch_script(
            "replay",
            "replay.py",
            config,
            custom_args=["--config", str(tmp)],
        )

    def launch_camera_preview(self, config: UnifiedRobotConfig) -> tuple[bool, str]:
        """Launch show_cameras.py with config (yaml for robot/policy,
        CLI override for camera runtime params not exposed in yaml).
        """
        tmp = dump_to_tempfile(config)
        args = ["--config", str(tmp)]

        # 只有用户填了 policy.path 才传 --policy-path；
        # 纯相机预览（用户已勾选 camera_list）不需要策略模型。
        if config.policy and config.policy.path:
            args.extend(["--policy-path", config.policy.path])

        # show_cameras.py 仍通过 CLI 接 camera_list / camera_fps /
        # show_quad / window_size（其 yaml schema 没暴露这 4 个）。
        if config.runtime.camera_list:
            args.extend(["--cameras", *config.runtime.camera_list])

        if config.runtime.camera_fps:
            args.extend(["--fps", str(int(config.runtime.camera_fps))])

        if config.runtime.show_quad:
            args.append("--show-quad")

        if config.runtime.window_width and config.runtime.window_height:
            args.extend([
                "--window-size",
                str(int(config.runtime.window_width)),
                str(int(config.runtime.window_height)),
            ])

        return self._launch_script(
            "camera_preview",
            "show_cameras.py",
            config,
            custom_args=args,
        )

    # -- new launchers: data_processing / model_training ----------------------

    def launch_data_processing(self, config: UnifiedRobotConfig) -> tuple[bool, str]:
        """Launch a data processing operation via the whitelisted Python script."""
        if config.data_processing is None:
            return False, "❌ data_processing 配置为空"
        dp: DataProcessingConfig = config.data_processing

        try:
            script_path, args = _build_data_processing_args(dp)
        except ValueError as e:
            return False, f"❌ 配置错误: {e}"

        # process_name 包含 variant, 便于并发区分
        suffix = ""
        if dp.operation == "v2_convert":
            suffix = f"_{dp.v2_convert.variant}"
        process_name = f"data_processing_{dp.operation}{suffix}"
        # 注册 metadata, 完成时用于解释退出码
        self._process_meta[process_name] = {
            "kind": "data_processing",
            "operation": dp.operation,
        }
        cmd = [sys.executable, str(script_path), *args]
        # 注意：先调用 _launch_command 检查 running，再启动；否则 interpret_exit_code
        # 会在 quick-success 时被 main_zh 路径用到（caller 负责）。
        return self._launch_command(process_name, cmd)

    def launch_model_training(self, config: UnifiedRobotConfig) -> tuple[bool, str]:
        """Launch a model training script (bash) with whitelisted env vars."""
        if config.model_training is None:
            return False, "❌ model_training 配置为空"
        mt: ModelTrainingConfig = config.model_training

        if mt.script not in TRAINING_SCRIPTS:
            return False, f"❌ 未知 script: {mt.script!r}"
        if mt.phase not in TRAINING_PHASES:
            return False, f"❌ 未知 phase: {mt.phase!r}"

        bash_script = TRAINING_BASH_SCRIPTS[mt.script]
        env_overrides = _build_training_env(mt)
        process_name = f"model_training_{mt.script}_{mt.phase}"
        # 注册 metadata
        self._process_meta[process_name] = {
            "kind": "model_training",
            "script": mt.script,
            "phase": mt.phase,
        }

        return self._launch_bash_script(process_name, bash_script, mt.phase, env_overrides)

    # -- stop / status --------------------------------------------------------

    def stop(self, process_name: str, timeout: float = 10.0) -> bool:
        """Stop a running process gracefully (kills the whole process group)."""
        if process_name not in self.processes:
            return False

        process = self.processes[process_name]

        if process.poll() is not None:
            # Already terminated
            self._cleanup_process(process_name)
            return True

        try:
            # 用 killpg 终止整个进程组 (bash 启动的训练会派生 lerobot-train 子进程,
            # 仅 terminate shell 会留下 orphan)
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()

            # Wait for graceful exit
            try:
                process.wait(timeout=timeout)
                self._cleanup_process(process_name)
                return True
            except subprocess.TimeoutExpired:
                # Force kill the whole group
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                process.wait(timeout=2.0)
                self._cleanup_process(process_name)
                return True

        except Exception:
            return False

    def stop_all(self) -> None:
        """Stop all running processes"""
        process_names = list(self.processes.keys())
        for name in process_names:
            self.stop(name)

    def is_running(self, process_name: str) -> bool:
        """Check if a process is still running"""
        if process_name not in self.processes:
            return False

        process = self.processes[process_name]
        return process.poll() is None

    def get_logs(self, process_name: str | None = None, last_n_lines: int = 100) -> str:
        """Get recent log output for a process"""
        if process_name is None:
            process_name = self._current_process_name

        if process_name is None or process_name not in self.log_buffers:
            return ""

        buffer = self.log_buffers[process_name]
        lines = list(buffer)[-last_n_lines:]
        return "\n".join(lines)

    def get_status(self, process_name: str) -> str:
        """Get current status of a process"""
        if process_name not in self.processes:
            return "Not started"

        if self.is_running(process_name):
            process = self.processes[process_name]
            return f"Running (PID: {process.pid})"
        else:
            process = self.processes[process_name]
            returncode = process.poll()
            return f"Stopped (exit code: {returncode})"

    def _cleanup_process(self, process_name: str):
        """Clean up process resources"""
        # 若进程已结束,生成 completion notice (供 UI 弹窗)
        if process_name in self.processes:
            proc = self.processes[process_name]
            rc = proc.poll()
            if rc is not None and process_name not in self._notified_completion:
                self._notified_completion.add(process_name)
                meta = self._process_meta.get(process_name, {})
                if meta.get("kind") == "data_processing":
                    ok, msg = interpret_exit_code(meta.get("operation", "sanity"), rc)
                elif meta.get("kind") == "model_training":
                    ok, msg = interpret_training_exit_code(
                        meta.get("script", "?"), meta.get("phase", "?"), rc,
                    )
                else:
                    ok, msg = (rc == 0, f"{'✅' if rc == 0 else '❌'} {process_name} 已结束 (exit {rc})")
                self._pending_notices.append((process_name, msg))
            del self.processes[process_name]

        if process_name in self.log_queues:
            del self.log_queues[process_name]

        if process_name in self.monitor_threads:
            del self.monitor_threads[process_name]

        if process_name in self._log_files:
            log_path, log_fh = self._log_files.pop(process_name)
            try:
                log_fh.close()
            except Exception:
                pass

        if self._current_process_name == process_name:
            self._current_process_name = None

    def _cleanup_all(self):
        """Cleanup handler called on exit"""
        self.stop_all()

    def get_changes(self) -> dict[str, Any]:
        """Detect newly-stopped processes and return UI-facing change info.

        Returns a dict:
            - any_running: bool         # True if any process is still running
            - notices: list[str]        # Human-readable completion messages (one per newly-finished)
            - statuses: dict[str, str]  # process_name -> brief status ("Running" / "Done (exit N)")
        """
        any_running = False
        notices: list[str] = []

        # Drain pending notices first (queued by _cleanup_process when a
        # process has finished and been cleaned up).
        while self._pending_notices:
            name, msg = self._pending_notices.pop(0)
            notices.append(f"[{name}] {msg}")
            self._process_meta.pop(name, None)
            self._notified_completion.discard(name)

        # Also detect finished processes that haven't been cleaned up yet
        # (e.g. user is just observing). Their completion is announced via
        # notices but they stay in self.processes until stop() or replace.
        for name in list(self.processes.keys()):
            proc = self.processes[name]
            rc = proc.poll()
            if rc is not None:
                if name not in self._notified_completion:
                    self._notified_completion.add(name)
                    meta = self._process_meta.get(name, {})
                    if meta.get("kind") == "data_processing":
                        ok, msg = interpret_exit_code(meta.get("operation", "sanity"), rc)
                    elif meta.get("kind") == "model_training":
                        ok, msg = interpret_training_exit_code(
                            meta.get("script", "?"), meta.get("phase", "?"), rc,
                        )
                    else:
                        ok, msg = (rc == 0, f"{'✅' if rc == 0 else '❌'} 已结束 (exit {rc})")
                    notices.append(f"[{name}] {msg}")
            else:
                any_running = True

        statuses: dict[str, str] = {}
        for name, proc in self.processes.items():
            rc = proc.poll()
            if rc is None:
                statuses[name] = "运行中"
            else:
                statuses[name] = f"已完成 (exit {rc})"

        return {
            "any_running": any_running,
            "notices": notices,
            "statuses": statuses,
        }


# Global instance
_process_manager: ProcessManager | None = None


def get_process_manager() -> ProcessManager:
    """Get global ProcessManager instance"""
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager