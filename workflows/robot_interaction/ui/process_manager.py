"""Process manager for robot interaction workflows.

This module manages the lifecycle of deploy.py, replay.py, and show_cameras.py
subprocesses, including launching, monitoring, log streaming, and cleanup.
"""

import atexit
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

from .config_manager import UnifiedRobotConfig, to_cli_args


class ProcessManager:
    """Manages subprocess lifecycle for robot interaction workflows"""

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_queues: dict[str, queue.Queue] = {}
        self.log_buffers: dict[str, deque] = {}
        self.monitor_threads: dict[str, threading.Thread] = {}
        self._current_process_name: str | None = None

        # Register cleanup on exit
        atexit.register(self._cleanup_all)

    def _get_script_path(self, script_name: str) -> Path:
        """Get absolute path to workflow script"""
        workflows_dir = Path(__file__).parent.parent
        return workflows_dir / script_name

    def _stream_output(self, process: subprocess.Popen, process_name: str):
        """Stream process output to queue (runs in separate thread)"""
        log_queue = self.log_queues[process_name]
        log_buffer = self.log_buffers[process_name]

        try:
            # Read both stdout and stderr
            for line in process.stdout:
                line = line.decode("utf-8", errors="replace").rstrip()
                log_queue.put(line)
                log_buffer.append(line)

                # Keep buffer at reasonable size (last 1000 lines)
                if len(log_buffer) > 1000:
                    log_buffer.popleft()
        except Exception as e:
            log_queue.put(f"[ERROR] Log streaming error: {e}")

    def launch_deploy(self, config: UnifiedRobotConfig) -> tuple[bool, str]:
        """Launch deploy.py with config"""
        return self._launch_script("deploy", "deploy.py", config)

    def launch_replay(self, config: UnifiedRobotConfig) -> tuple[bool, str]:
        """Launch replay.py with config"""
        return self._launch_script("replay", "replay.py", config)

    def launch_camera_preview(self, config: UnifiedRobotConfig) -> tuple[bool, str]:
        """Launch show_cameras.py with config"""
        # Build args for show_cameras
        args = [
            "--http-base-url", config.robot.http_base_url,
        ]

        if config.policy and config.policy.path:
            args.extend(["--policy-path", config.policy.path])

        if config.inference and config.inference.rename_map:
            import json
            args.extend(["--rename-map", json.dumps(config.inference.rename_map)])

        # 添加相机列表
        if config.runtime.camera_list:
            args.append("--cameras")
            args.extend(config.runtime.camera_list)
        # 如果没有指定，show_cameras.py 会自动使用所有相机

        # 添加 FPS
        if config.runtime.camera_fps:
            args.extend(["--fps", str(int(config.runtime.camera_fps))])

        # 添加四宫格选项
        if config.runtime.show_quad:
            args.append("--show-quad")

        # 添加窗口尺寸 (注意：--window-size 接受两个参数 W H)
        if config.runtime.window_width and config.runtime.window_height:
            args.extend([
                "--window-size",
                str(int(config.runtime.window_width)),
                str(int(config.runtime.window_height))
            ])

        return self._launch_script("camera_preview", "show_cameras.py", config, custom_args=args)

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

        if custom_args:
            cmd.extend(custom_args)
        else:
            cmd.extend(to_cli_args(config))

        # Initialize log structures
        self.log_queues[process_name] = queue.Queue()
        self.log_buffers[process_name] = deque(maxlen=1000)

        try:
            # Launch process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,  # 改为无缓冲，避免警告
                cwd=Path(__file__).parent.parent.parent,  # Repository root
                env=os.environ.copy(),
            )

            self.processes[process_name] = process
            self._current_process_name = process_name

            # Start log monitoring thread
            monitor_thread = threading.Thread(
                target=self._stream_output,
                args=(process, process_name),
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

            return True, f"✅ {process_name} started successfully (PID: {process.pid})"

        except Exception as e:
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
