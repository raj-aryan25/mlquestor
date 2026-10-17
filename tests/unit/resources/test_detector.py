from __future__ import annotations

from types import SimpleNamespace

from mlquest import resources
from mlquest.resources import detector


def test_detect_resources_reads_memory_disk_python_and_capabilities(monkeypatch) -> None:
    proc_meminfo = "MemTotal:       16384 kB\nMemAvailable:    8192 kB\n"
    monkeypatch.setattr(
        detector, "open", lambda *_args, **_kwargs: _MemoryInfo(proc_meminfo), raising=False
    )
    monkeypatch.setattr(detector.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        detector.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=500, free=125),
    )
    monkeypatch.setattr(detector.platform, "python_version", lambda: "3.12.9")

    def which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in {"nvidia-smi", "docker"} else None

    monkeypatch.setattr(detector.shutil, "which", which)

    def run(command, **_kwargs):
        if command[1] == "--query-gpu=name":
            return SimpleNamespace(returncode=0, stdout="NVIDIA A10\nNVIDIA A10\n")
        assert command[1:] == ("info", "--format", "{{.ServerVersion}}")
        return SimpleNamespace(returncode=0, stdout="27.5.1\n")

    monkeypatch.setattr(detector.subprocess, "run", run)

    profile = resources.detect_resources("/workspace")

    assert profile.cpu_count == 8
    assert profile.memory_bytes == 16 * 1024 * 1024
    assert profile.memory_available_bytes == 8 * 1024 * 1024
    assert profile.disk_bytes == 500
    assert profile.disk_free_bytes == 125
    assert profile.gpu_count == 2
    assert profile.gpu_model == "NVIDIA A10"
    assert profile.python_version == "3.12.9"
    assert profile.docker_available is True


def test_detection_degrades_when_optional_sources_are_unavailable(monkeypatch) -> None:
    def missing_open(*_args, **_kwargs):
        raise FileNotFoundError("/proc unavailable")

    monkeypatch.setattr(detector, "open", missing_open, raising=False)
    monkeypatch.setattr(
        detector.os,
        "sysconf",
        lambda _name: (_ for _ in ()).throw(OSError("sysconf unavailable")),
    )
    monkeypatch.setattr(detector.os, "cpu_count", lambda: None)
    monkeypatch.setattr(
        detector.shutil,
        "disk_usage",
        lambda _path: (_ for _ in ()).throw(OSError("filesystem unavailable")),
    )
    monkeypatch.setattr(detector.shutil, "which", lambda _name: None)

    profile = resources.ResourceDetector("/missing").detect()

    assert profile.cpu_count == 1
    assert profile.memory_bytes == 1
    assert profile.memory_available_bytes is None
    assert profile.disk_bytes is None
    assert profile.disk_free_bytes is None
    assert profile.gpu_count == 0
    assert profile.gpu_model is None
    assert profile.docker_available is False


def test_gpu_and_docker_failures_do_not_escape(monkeypatch) -> None:
    monkeypatch.setattr(detector.shutil, "which", lambda name: f"/bin/{name}")

    def run(_command, **_kwargs):
        raise detector.subprocess.TimeoutExpired("tool", 2)

    monkeypatch.setattr(detector.subprocess, "run", run)

    assert detector._detect_gpu() == (0, None)
    assert detector._detect_docker() is False


class _MemoryInfo:
    def __init__(self, content: str) -> None:
        self.content = content

    def __enter__(self) -> _MemoryInfo:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def readlines(self) -> list[str]:
        return self.content.splitlines()
