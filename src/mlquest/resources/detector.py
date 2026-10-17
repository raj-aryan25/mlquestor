"""Side-effect-free detection of resources available to MLQuest."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Final

from mlquest.domain import ResourceProfile

_PROC_MEMINFO: Final = "/proc/meminfo"
_MEMINFO_KIB: Final = "kB"
_COMMAND_TIMEOUT_SECONDS: Final = 2.0


class ResourceDetector:
    """Detect host resources without requiring optional system packages.

    Detection only reads operating-system metadata and invokes read-only
    capability queries. Failures are represented by ``None`` or conservative
    model-compatible fallbacks rather than escaping to callers.
    """

    def __init__(self, disk_path: str | os.PathLike[str] = ".") -> None:
        self.disk_path = os.fspath(disk_path)

    def detect(self) -> ResourceProfile:
        """Return a snapshot of resources visible to the current process."""

        memory_bytes, memory_available_bytes = _detect_memory()
        disk_bytes, disk_free_bytes = _detect_disk(self.disk_path)
        gpu_count, gpu_model = _detect_gpu()

        return ResourceProfile(
            cpu_count=_detect_cpu_count(),
            memory_bytes=memory_bytes,
            memory_available_bytes=memory_available_bytes,
            disk_bytes=disk_bytes,
            disk_free_bytes=disk_free_bytes,
            gpu_count=gpu_count,
            gpu_model=gpu_model,
            python_version=platform.python_version(),
            docker_available=_detect_docker(),
        )


def detect_resources(disk_path: str | os.PathLike[str] = ".") -> ResourceProfile:
    """Detect resources visible to the current process.

    ``disk_path`` selects the filesystem whose capacity and free space are
    reported. It defaults to the current working directory's filesystem.
    """

    return ResourceDetector(disk_path).detect()


def detect_resource_profile(disk_path: str | os.PathLike[str] = ".") -> ResourceProfile:
    """Compatibility-oriented name for :func:`detect_resources`."""

    return detect_resources(disk_path)


def _detect_cpu_count() -> int:
    try:
        count = os.cpu_count()
    except (OSError, TypeError):
        count = None
    return count if isinstance(count, int) and count > 0 else 1


def _detect_memory() -> tuple[int, int | None]:
    values = _read_proc_meminfo()
    total = values.get("MemTotal")
    available = values.get("MemAvailable")

    if total is None or available is None:
        total_from_sysconf, available_from_sysconf = _read_sysconf_memory()
        if total is None:
            total = total_from_sysconf
        if available is None:
            available = available_from_sysconf

    # ResourceProfile intentionally requires a positive total. A value of one
    # keeps detection safe when both Linux procfs and sysconf are unavailable;
    # callers can distinguish an unknown optional measurement by its None value.
    total = total if total is not None and total > 0 else 1
    if available is not None and available < 0:
        available = None
    return total, available


def _read_proc_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open(_PROC_MEMINFO, encoding="utf-8") as meminfo:
            lines = meminfo.readlines()
    except (OSError, UnicodeError):
        return values

    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = int(parts[1])
        except ValueError:
            continue
        multiplier = 1024 if len(parts) > 2 and parts[2] == _MEMINFO_KIB else 1
        values[parts[0].rstrip(":")] = value * multiplier
    return values


def _read_sysconf_memory() -> tuple[int | None, int | None]:
    page_size = _read_sysconf_value("SC_PAGE_SIZE")
    physical_pages = _read_sysconf_value("SC_PHYS_PAGES")
    available_pages = _read_sysconf_value("SC_AVPHYS_PAGES")

    total = (
        page_size * physical_pages
        if page_size is not None
        and page_size > 0
        and physical_pages is not None
        and physical_pages > 0
        else None
    )
    available = (
        page_size * available_pages
        if page_size is not None
        and page_size > 0
        and available_pages is not None
        and available_pages >= 0
        else None
    )
    return total, available


def _read_sysconf_value(name: str) -> int | None:
    try:
        value = int(os.sysconf(name))
    except (OSError, TypeError, ValueError):
        return None
    return value


def _detect_disk(path: str) -> tuple[int | None, int | None]:
    try:
        usage = shutil.disk_usage(path)
        total = int(usage.total)
        free = int(usage.free)
    except (AttributeError, OSError, TypeError, ValueError):
        return None, None

    if total < 0 or free < 0:
        return None, None
    return total, free


def _detect_gpu() -> tuple[int, str | None]:
    commands = (
        ("nvidia-smi", ("--query-gpu=name", "--format=csv,noheader,nounits")),
        ("rocm-smi", ("--showproductname", "--csv")),
    )
    for executable_name, arguments in commands:
        executable = _which(executable_name)
        if executable is None:
            continue
        output = _run_command((executable, *arguments))
        if output is None:
            continue
        models = tuple(line.strip() for line in output.splitlines() if line.strip())
        if models:
            return len(models), models[0]
    return 0, None


def _detect_docker() -> bool:
    executable = _which("docker")
    if executable is None:
        return False
    return _run_command((executable, "info", "--format", "{{.ServerVersion}}")) is not None


def _run_command(command: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if result.returncode != 0:
        return None
    stdout = getattr(result, "stdout", None)
    return stdout.strip() if isinstance(stdout, str) else None


def _which(name: str) -> str | None:
    try:
        return shutil.which(name)
    except (AttributeError, OSError, TypeError):
        return None
