"""Production-oriented Docker execution for MLQuest experiments.

The runner deliberately accepts host paths only for approved source and
run-specific directories.  The SQLite ledger and artifact manifests are never
mounted into the container; only their resulting records/artifacts are written
after the bounded command has finished.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mlquest.artifacts import FileSystemArtifactStore, StoredArtifact
from mlquest.budgets import BudgetManager, ExecutionLimits, RunRequest
from mlquest.domain import ExperimentSpecification, RunAttempt, RunStatus
from mlquest.domain.common import utc_now

_NUMERIC_USER = re.compile(r"^[1-9][0-9]*:[1-9][0-9]*$")
_TARGET_PART = re.compile(r"^[A-Za-z0-9_.-]+$")


class DockerRunnerError(ValueError):
    """Raised when an experiment cannot be represented as a safe Docker run."""


@dataclass(frozen=True, slots=True)
class DockerMount:
    """One host bind mount exposed at an absolute container target."""

    source: Path | str
    target: str
    read_only: bool = True


@dataclass(frozen=True, slots=True)
class DockerRunResult:
    """Structured output from one bounded Docker experiment attempt."""

    attempt: RunAttempt
    command: tuple[str, ...]
    stdout: bytes
    stderr: bytes
    metadata: dict[str, Any]
    resource_info: dict[str, float]
    artifacts: tuple[StoredArtifact, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.attempt.status is RunStatus.SUCCEEDED


def build_docker_run_command(
    image: str,
    command: Sequence[str],
    *,
    source_mounts: Mapping[Path | str, str] | Sequence[DockerMount] = (),
    work_dir: Path | str,
    output_dir: Path | str,
    log_dir: Path | str,
    cpu_count: int = 1,
    memory_bytes: int = 512 * 1024 * 1024,
    pids_limit: int = 256,
    user: str = "65532:65532",
    container_name: str | None = None,
) -> tuple[str, ...]:
    """Build a deterministic, isolated ``docker run`` argv.

    Source mounts are read-only by construction.  Work, output, and log
    mounts are separate and explicitly writable.  The returned tuple is safe
    to pass directly to ``subprocess`` without a shell.
    """

    _validate_image(image)
    argv_command = _validate_command(command)
    if cpu_count < 1:
        raise DockerRunnerError("cpu_count must be positive")
    if memory_bytes < 1:
        raise DockerRunnerError("memory_bytes must be positive")
    if pids_limit < 1:
        raise DockerRunnerError("pids_limit must be positive")
    _validate_user(user)

    writable = (
        DockerMount(_directory_path(work_dir, "work_dir"), "/work", False),
        DockerMount(_directory_path(output_dir, "output_dir"), "/output", False),
        DockerMount(_directory_path(log_dir, "log_dir"), "/logs", False),
    )
    normalized_sources = tuple(_normalize_source_mounts(source_mounts))
    targets = [mount.target for mount in normalized_sources] + [mount.target for mount in writable]
    if len(set(targets)) != len(targets):
        raise DockerRunnerError("mount targets must be unique")
    if any(
        _paths_overlap(_source_path(mount), _source_path(other))
        for mount in normalized_sources
        for other in writable
    ):
        raise DockerRunnerError("source mounts must not overlap writable run directories")

    argv: list[str] = ["docker", "run", "--read-only", "--network", "none"]
    if container_name is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", container_name):
            raise DockerRunnerError("container_name contains unsafe characters")
        argv.extend(("--name", container_name))
    argv.extend(
        (
            "--user",
            user,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--cpus",
            str(cpu_count),
            "--memory",
            str(memory_bytes),
            "--pids-limit",
            str(pids_limit),
        )
    )
    for mount in (*normalized_sources, *writable):
        mode = "readonly" if mount.read_only else "rw"
        argv.extend(
            (
                "--mount",
                f"type=bind,src={mount.source},dst={mount.target},{mode}",
            )
        )
    argv.extend((image, *argv_command))
    return tuple(argv)


class DockerExperimentRunner:
    """Run one experiment through Docker and persist its terminal evidence."""

    def __init__(
        self,
        budget_manager: BudgetManager,
        image: str,
        *,
        docker_executable: str | Path = "docker",
        user: str = "65532:65532",
        pids_limit: int = 256,
    ) -> None:
        self.budget_manager = budget_manager
        _validate_image(image)
        self.image = image
        self.docker_executable = str(docker_executable)
        self.user = user
        self.pids_limit = pids_limit
        _validate_user(user)
        if pids_limit < 1:
            raise DockerRunnerError("pids_limit must be positive")

    def run(
        self,
        specification: ExperimentSpecification,
        command: Sequence[str],
        *,
        source_mounts: Mapping[Path | str, str] | Sequence[DockerMount] = (),
        work_dir: Path | str,
        output_dir: Path | str,
        log_dir: Path | str,
        limits: ExecutionLimits,
        cpu_count: int | None = None,
        memory_bytes: int | None = None,
        pids_limit: int | None = None,
        ledger: Any | None = None,
        artifact_store: FileSystemArtifactStore | None = None,
        session_id: str | None = None,
        attempt: RunAttempt | None = None,
    ) -> DockerRunResult:
        """Execute and optionally persist one experiment attempt.

        ``source_mounts`` maps host paths to container targets when supplied as
        a mapping.  Passing :class:`DockerMount` objects permits explicit
        read-only values, although source mounts are always normalized to
        read-only for this runner.
        """

        if attempt is not None and attempt.experiment_id != specification.id:
            raise DockerRunnerError("attempt belongs to a different experiment")
        current_attempt = attempt or RunAttempt(experiment_id=specification.id)
        session = session_id or self.budget_manager.session_id
        work_path = _directory_path(work_dir, "work_dir")
        output_path = _directory_path(output_dir, "output_dir")
        log_path = _directory_path(log_dir, "log_dir")
        normalized_sources = _normalize_source_mounts(source_mounts)
        _validate_mount_safety(
            normalized_sources,
            (work_path, output_path, log_path),
            ledger=ledger,
            artifact_store=artifact_store,
        )

        profile = specification.resource_profile
        cpu = cpu_count if cpu_count is not None else (profile.cpu_count if profile else 1)
        memory = memory_bytes if memory_bytes is not None else (
            profile.memory_bytes if profile else 512 * 1024 * 1024
        )
        pids = pids_limit if pids_limit is not None else self.pids_limit
        docker_name = _container_name(current_attempt.id)
        command_argv = build_docker_run_command(
            self.image,
            command,
            source_mounts=normalized_sources,
            work_dir=work_path,
            output_dir=output_path,
            log_dir=log_path,
            cpu_count=cpu,
            memory_bytes=memory,
            pids_limit=pids,
            user=self.user,
            container_name=docker_name,
        )
        executable_command = (self.docker_executable, *command_argv[1:])
        metadata = _metadata(
            self.image,
            command_argv,
            normalized_sources,
            work_path,
            output_path,
            log_path,
            cpu,
            memory,
            pids,
            self.user,
        )
        _persist_attempt(ledger, current_attempt, session)
        started = utc_now()
        running_attempt = current_attempt.model_copy(
            update={"status": RunStatus.RUNNING, "started_at": started}
        )
        _persist_attempt(ledger, running_attempt, session)
        dataset_size = sum(_path_size(_source_path(mount)) for mount in normalized_sources)
        request = RunRequest(
            model_family=specification.model_family,
            dataset_size=dataset_size,
            configuration=specification.model_config_values,
            estimated_memory_bytes=memory,
        )
        started_monotonic = time.monotonic()
        result = None
        try:
            result, _ = self.budget_manager.run_limited_command(
                request,
                list(executable_command),
                limits,
            )
        except Exception as exc:
            finished = utc_now()
            failed_attempt = running_attempt.model_copy(
                update={
                    "status": RunStatus.FAILED,
                    "finished_at": finished,
                    "exit_code": None,
                    "resource_usage": {
                        "runtime_seconds": round(max(0.0, time.monotonic() - started_monotonic), 6),
                        "cpu_limit": float(cpu),
                        "memory_limit_bytes": float(memory),
                        "pids_limit": float(pids),
                    },
                    "failure_reason": f"runner error: {exc}",
                }
            )
            _persist_attempt(ledger, failed_attempt, session)
            raise
        finally:
            _remove_container(self.docker_executable, docker_name)
        assert result is not None

        (log_path / "stdout.log").write_bytes(result.stdout)
        (log_path / "stderr.log").write_bytes(result.stderr)
        captured = _capture_artifacts(
            artifact_store, log_path, output_path, specification.id, running_attempt.id
        )
        status, failure_reason = _terminal_state(
            result.returncode, result.timed_out, result.output_limited
        )
        finished = utc_now()
        runtime = round(max(0.0, time.monotonic() - started_monotonic), 6)
        usage = {
            "runtime_seconds": runtime,
            "cpu_limit": float(cpu),
            "memory_limit_bytes": float(memory),
            "pids_limit": float(pids),
            "stdout_bytes": float(len(result.stdout)),
            "stderr_bytes": float(len(result.stderr)),
        }
        final_attempt = running_attempt.model_copy(
            update={
                "status": status,
                "finished_at": finished,
                "exit_code": result.returncode,
                "artifact_ids": [artifact.manifest.id for artifact in captured],
                "resource_usage": usage,
                "failure_reason": failure_reason,
            }
        )
        _persist_attempt(ledger, final_attempt, session)
        return DockerRunResult(
            attempt=final_attempt,
            command=executable_command,
            stdout=result.stdout,
            stderr=result.stderr,
            metadata=metadata,
            resource_info=usage,
            artifacts=tuple(captured),
        )


def _validate_image(image: str) -> None:
    if (
        not isinstance(image, str)
        or not image
        or image.startswith("-")
        or any(char.isspace() or ord(char) < 32 for char in image)
    ):
        raise DockerRunnerError("image must be a non-empty safe string")


def _validate_command(command: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(item) for item in command)
    if not values or any(not value or "\x00" in value for value in values):
        raise DockerRunnerError("command must contain at least one safe argument")
    return values


def _validate_user(user: str) -> None:
    if not _NUMERIC_USER.fullmatch(user):
        raise DockerRunnerError("user must be a non-root numeric uid:gid")


def _directory_path(value: Path | str, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise DockerRunnerError(f"{name} must be an absolute path")
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise DockerRunnerError(f"{name} must be a directory")
    return path


def _normalize_source_mounts(
    mounts: Mapping[Path | str, str] | Sequence[DockerMount],
) -> tuple[DockerMount, ...]:
    values = (
        tuple(DockerMount(source=source, target=target) for source, target in mounts.items())
        if isinstance(mounts, Mapping)
        else tuple(mounts)
    )
    normalized: list[DockerMount] = []
    for mount in values:
        if not isinstance(mount, DockerMount):
            raise DockerRunnerError("source mounts must be DockerMount values or a path mapping")
        source = Path(mount.source).expanduser()
        if not source.is_absolute() or not source.exists():
            raise DockerRunnerError(f"source mount does not exist as an absolute path: {source}")
        source = source.resolve(strict=True)
        if "," in str(source):
            raise DockerRunnerError("source mount paths must not contain commas")
        target = _validate_target(mount.target)
        normalized.append(DockerMount(source, target, True))
    return tuple(normalized)


def _validate_target(target: str) -> str:
    if not isinstance(target, str) or "\x00" in target or "\\" in target:
        raise DockerRunnerError("mount target contains unsafe characters")
    parts = target.split("/")
    if not target.startswith("/") or target == "/" or any(
        not part or part in {".", ".."} or not _TARGET_PART.fullmatch(part) for part in parts[1:]
    ):
        raise DockerRunnerError("mount target must be a normalized absolute path")
    if target in {"/work", "/output", "/logs"}:
        raise DockerRunnerError("source mount target is reserved for a writable run directory")
    return target


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        try:
            second.relative_to(first)
            return True
        except ValueError:
            return False


def _validate_mount_safety(
    sources: Sequence[DockerMount],
    writable: Sequence[Path],
    *,
    ledger: Any | None,
    artifact_store: FileSystemArtifactStore | None,
) -> None:
    if any(
        _paths_overlap(_source_path(mount), path)
        for mount in sources
        for path in writable
    ):
        raise DockerRunnerError("source mounts must not overlap writable run directories")
    protected: list[Path] = []
    ledger_protected: list[Path] = []
    if artifact_store is not None:
        protected.append(Path(artifact_store.root).resolve())
    ledger_path = getattr(ledger, "path", None)
    if ledger_path is not None and str(ledger_path) != ":memory:":
        ledger_protected.append(Path(ledger_path).expanduser().resolve())
        protected.extend(ledger_protected)
    for path in writable:
        if any(_paths_overlap(path, protected_path) for protected_path in protected):
            raise DockerRunnerError("writable run directories must not overlap protected state")
    for mount in sources:
        if any(
            _paths_overlap(_source_path(mount), protected_path)
            for protected_path in ledger_protected
        ):
            raise DockerRunnerError("source mounts must not expose the session ledger")


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def _source_path(mount: DockerMount) -> Path:
    return Path(mount.source)


def _container_name(attempt_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", attempt_id)
    return f"mlquest-{safe[:115]}"


def _metadata(
    image: str,
    command: Sequence[str],
    sources: Sequence[DockerMount],
    work: Path,
    output: Path,
    logs: Path,
    cpu: int,
    memory: int,
    pids: int,
    user: str,
) -> dict[str, Any]:
    return {
        "runner": "mlquest.runner.docker",
        "image": image,
        "command": list(command),
        "user": user,
        "read_only_root": True,
        "network": "none",
        "cap_drop": ["ALL"],
        "no_new_privileges": True,
        "mounts": [
            {"source": str(mount.source), "target": mount.target, "read_only": True}
            for mount in sources
        ]
        + [
            {"source": str(work), "target": "/work", "read_only": False},
            {"source": str(output), "target": "/output", "read_only": False},
            {"source": str(logs), "target": "/logs", "read_only": False},
        ],
        "limits": {"cpu_count": cpu, "memory_bytes": memory, "pids": pids},
    }


def _terminal_state(
    returncode: int, timed_out: bool, output_limited: bool
) -> tuple[RunStatus, str | None]:
    if timed_out:
        return RunStatus.TIMED_OUT, "execution timed out"
    if output_limited:
        return RunStatus.FAILED, "execution output exceeded the configured limit"
    if returncode != 0:
        return RunStatus.FAILED, f"docker command exited with code {returncode}"
    return RunStatus.SUCCEEDED, None


def _capture_artifacts(
    store: FileSystemArtifactStore | None,
    log_dir: Path,
    output_dir: Path,
    experiment_id: str,
    attempt_id: str,
) -> list[StoredArtifact]:
    if store is None:
        return []
    files = [
        path
        for root in (log_dir, output_dir)
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    captured: list[StoredArtifact] = []
    seen: set[Path] = set()
    for path in sorted(files):
        if path in seen:
            continue
        seen.add(path)
        relative = (
            path.relative_to(log_dir).as_posix()
            if path.is_relative_to(log_dir)
            else path.relative_to(output_dir).as_posix()
        )
        media_type = (
            "text/plain"
            if path.suffix in {".log", ".txt", ".json", ".csv"}
            else "application/octet-stream"
        )
        captured.append(
            store.put_source(
                path,
                role="auxiliary",
                media_type=media_type,
                metadata={
                    "experiment_id": experiment_id,
                    "run_attempt_id": attempt_id,
                    "artifact_kind": "run_log" if path.is_relative_to(log_dir) else "run_output",
                    "relative_path": relative,
                },
            )
        )
    return captured


def _persist_attempt(ledger: Any | None, attempt: RunAttempt, session_id: str) -> None:
    if ledger is not None:
        ledger.put_record("run_attempts", attempt, session_id=session_id)


def _remove_container(docker_executable: str, container_name: str) -> None:
    try:
        subprocess.run(
            [docker_executable, "rm", "--force", container_name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


__all__ = [
    "DockerExperimentRunner",
    "DockerMount",
    "DockerRunResult",
    "DockerRunnerError",
    "build_docker_run_command",
]
