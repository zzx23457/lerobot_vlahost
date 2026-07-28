"""Process manager for robot interaction workflows.

This module manages the lifecycle of deploy.py, replay.py, and show_cameras.py
subprocesses, including launching, monitoring, log streaming, and cleanup.

All three scripts accept a yaml config via ``--config``. The UI serializes
the current ``UnifiedRobotConfig`` to a temp yaml and passes it as
``--config <tmp>``. Camera preview keeps a few CLI-only runtime params
(``--cameras`` / ``--fps`` / ``--show-quad`` / ``--window-size``) because
``show_cameras.py`` does not yet expose them in yaml.
"""

import atexit
import ctypes
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal

from .config_manager import UnifiedRobotConfig, dump_to_tempfile


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


class ProcessManager:
    """Manages subprocess lifecycle for robot interaction workflows"""

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_queues: dict[str, queue.Queue] = {}
        self.log_buffers: dict[str, deque] = {}
        self.monitor_threads: dict[str, threading.Thread] = {}
        # (log_path, file_handle) per process — needed so we can close the log
        # file cleanly and tail it from the reader thread.
        self._log_files: dict[str, tuple[Path, "typing.IO[bytes]"]] = {}
        self._current_process_name: str | None = None

        # Register cleanup on exit
        atexit.register(self._cleanup_all)

    def _get_script_path(self, script_name: str) -> Path:
        """Get absolute path to workflow script"""
        workflows_dir = Path(__file__).parent.parent
        return workflows_dir / script_name

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

    def _launch_script(
        self,
        process_name: str,
        script_name: str,
        config: UnifiedRobotConfig,
        custom_args: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Launch a workflow script as subprocess"""

        # Check if already running
        if self.is_running(process_name):
            return False, f"❌ {process_name} is already running"

        # Get script path
        script_path = self._get_script_path(script_name)
        if not script_path.exists():
            return False, f"❌ Script not found: {script_path}"

        # Build command
        cmd = [sys.executable, str(script_path)]
        # 所有 launch_* 现在都通过 custom_args 传 --config（yaml-centric 架构）。
        if custom_args is None:
            raise RuntimeError(
                f"{process_name}: custom_args is required (yaml-centric 架构要求 --config)"
            )
        cmd.extend(custom_args)

        # Initialize log structures
        self.log_queues[process_name] = queue.Queue()
        self.log_buffers[process_name] = deque(maxlen=1000)

        # Method B: redirect stdout/stderr to a log file. Eliminates the kernel
        # pipe → no 64KB backpressure, no GIL contention with the reader
        # thread, identical behavior to a directly-launched deploy.py whose
        # stdout happens to be a terminal.
        # 4 层 parent：ui/ → robot_interaction/ → workflows/ → <REPO_ROOT>。
        repo_root = Path(__file__).parent.parent.parent.parent
        log_path = (repo_root / "outputs" / "deploy_logs" / f"{process_name}_{int(time.time())}.log").resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "wb")
        self._log_files[process_name] = (log_path, log_fh)

        try:
            # Launch process
            # PYTHONUNBUFFERED=1 强制 rollout 端 stdout 行缓冲（每次 print 立即 flush），
            # 配合 stdout=log_fh 让日志尽快落到磁盘文件里，UI 那边 50ms tail 一次。
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=repo_root,  # Repository root
                env=child_env,
                preexec_fn=_child_preexec,  # 与直接跑 deploy.py 完全一致
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

            # Wait a moment to check for immediate failures
            time.sleep(0.5)
            if process.poll() is not None:
                # Process exited immediately
                stderr = self.get_logs(process_name, last_n_lines=20)
                return False, f"❌ Process failed to start:\n{stderr}"

            return True, f"✅ {process_name} started successfully (PID: {process.pid}, log: {log_path})"

        except Exception as e:
            try:
                log_fh.close()
            except Exception:
                pass
            self._log_files.pop(process_name, None)
            return False, f"❌ Failed to launch {process_name}: {e}"

    def stop(self, process_name: str, timeout: float = 10.0) -> bool:
        """Stop a running process gracefully"""
        if process_name not in self.processes:
            return False

        process = self.processes[process_name]

        if process.poll() is not None:
            # Already terminated
            self._cleanup_process(process_name)
            return True

        try:
            # Send SIGTERM for graceful shutdown
            process.terminate()

            # Wait for graceful exit
            try:
                process.wait(timeout=timeout)
                self._cleanup_process(process_name)
                return True
            except subprocess.TimeoutExpired:
                # Force kill if still running
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
        if process_name in self.processes:
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


# Global instance
_process_manager: ProcessManager | None = None


def get_process_manager() -> ProcessManager:
    """Get global ProcessManager instance"""
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager
