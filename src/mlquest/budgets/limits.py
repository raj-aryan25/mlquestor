"""Bounded subprocess execution primitives for future experiment runners."""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Sequence


class ExecutionLimitError(RuntimeError):
    """Raised when a command exceeds its timeout or output quota."""


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    timeout_seconds: float
    max_output_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    command: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    output_limited: bool = False


def run_limited_command(
    command: Sequence[str],
    limits: ExecutionLimits,
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> ExecutionResult:
    """Run a command with a process-group timeout and bounded captured output."""

    argv = tuple(str(item) for item in command)
    if not argv:
        raise ValueError("command must not be empty")
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    assert process.stderr is not None
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    output = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + limits.timeout_seconds
    output_limited = False
    timed_out = False
    while selector.get_map():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        events = selector.select(remaining)
        if not events:
            timed_out = True
            break
        for key, _ in events:
            chunk = os.read(key.fd, 65_536)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            output[key.data].extend(chunk)
            if len(output["stdout"]) + len(output["stderr"]) > limits.max_output_bytes:
                output_limited = True
                break
        if timed_out or output_limited:
            break
        if process.poll() is not None and not selector.get_map():
            break
    selector.close()
    if timed_out or output_limited:
        _terminate_process_group(process)
        stdout, stderr = process.communicate()
        output["stdout"].extend(stdout)
        output["stderr"].extend(stderr)
        return ExecutionResult(
            argv,
            process.returncode if process.returncode is not None else -signal.SIGKILL,
            _bounded(bytes(output["stdout"]), limits.max_output_bytes),
            _bounded(bytes(output["stderr"]), limits.max_output_bytes),
            timed_out=timed_out,
            output_limited=output_limited,
        )
    return ExecutionResult(
        argv,
        process.wait(),
        bytes(output["stdout"]),
        bytes(output["stderr"]),
    )


def _bounded(value: bytes, limit: int) -> bytes:
    return value[:limit]


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass


__all__ = ["ExecutionLimitError", "ExecutionLimits", "ExecutionResult", "run_limited_command"]
