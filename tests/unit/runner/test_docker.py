from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from mlquest.artifacts import FileSystemArtifactStore
from mlquest.budgets import BudgetConfiguration, BudgetManager, ExecutionLimits
from mlquest.domain import (
    ExperimentSpecification,
    MetricDirection,
    RunStatus,
    SplitSpecification,
    SplitStrategy,
    ValidationSpecification,
)
from mlquest.persistence import LocalLedger
from mlquest.runner import (
    DockerExperimentRunner,
    DockerMount,
    DockerRunnerError,
    build_docker_run_command,
)
from mlquest.sessions import SessionService


def _experiment() -> ExperimentSpecification:
    return ExperimentSpecification(
        id="experiment-1",
        name="fixture",
        model_family="fixture-model",
        validation=ValidationSpecification(
            split=SplitSpecification(strategy=SplitStrategy.HOLDOUT, test_size=0.2),
            primary_metric="accuracy",
            metric_direction=MetricDirection.MAXIMIZE,
        ),
    )


def _budget() -> BudgetManager:
    return BudgetManager(
        "session-1",
        BudgetConfiguration(session_seconds=300, model_execution_seconds=300),
    )


@pytest.fixture
def fake_docker(tmp_path: Path) -> Path:
    script = tmp_path / "fake-docker.py"
    script.write_text(
        """#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

log = Path(__file__).with_suffix('.jsonl')
args = sys.argv[1:]
with log.open('a', encoding='utf-8') as stream:
    stream.write(json.dumps(args) + '\\n')
if args and args[0] == 'rm':
    raise SystemExit(0)
if args[0] != 'run':
    raise SystemExit(9)
mode = args[-1]
if mode == 'timeout':
    time.sleep(10)
elif mode == 'failure':
    print('fixture failure', file=sys.stderr)
    raise SystemExit(7)
else:
    print('fixture stdout')
    output_mount = next(item for item in args if 'dst=/output,' in item)
    output = Path(output_mount.split('src=', 1)[1].split(',dst=', 1)[0])
    (output / 'result.txt').write_text('fixture output\\n', encoding='utf-8')
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _run_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    work = tmp_path / "work"
    output = tmp_path / "output"
    logs = tmp_path / "logs"
    return work, output, logs


def test_build_command_isolation_and_explicit_mount_modes(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    command = build_docker_run_command(
        "mlquest-fixture:1",
        ["python", "-c", "print(1)"],
        source_mounts=[DockerMount(source, "/approved/input")],
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        log_dir=tmp_path / "logs",
        cpu_count=2,
        memory_bytes=123456,
        pids_limit=17,
        container_name="mlquest-attempt-1",
    )
    assert command[:4] == ("docker", "run", "--read-only", "--network")
    assert "none" in command
    user_index = command.index("--user")
    assert ("--user", "65532:65532") == (command[user_index], command[user_index + 1])
    assert ("--cap-drop", "ALL") == (
        command[command.index("--cap-drop")],
        command[command.index("--cap-drop") + 1],
    )
    assert "no-new-privileges:true" in command
    assert "--cpus" in command and "2" in command
    assert "--memory" in command and "123456" in command
    assert "--pids-limit" in command and "17" in command
    mount_values = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
    assert any("dst=/approved/input,readonly" in value for value in mount_values)
    assert any("dst=/output,rw" in value for value in mount_values)
    assert any("dst=/work,rw" in value for value in mount_values)
    assert any("dst=/logs,rw" in value for value in mount_values)


def test_runner_success_writes_output_and_persists_attempt_and_artifacts(
    tmp_path: Path, fake_docker: Path
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("feature\n1\n", encoding="utf-8")
    work, output, logs = _run_paths(tmp_path)
    artifact_store = FileSystemArtifactStore(tmp_path / "artifacts")
    with LocalLedger(tmp_path / "ledger.sqlite3") as ledger:
        ledger.create_session("session-1")
        result = DockerExperimentRunner(
            _budget(), "mlquest-fixture:1", docker_executable=fake_docker
        ).run(
            _experiment(),
            ["fixture", "success"],
            source_mounts={source: "/approved/input"},
            work_dir=work,
            output_dir=output,
            log_dir=logs,
            limits=ExecutionLimits(timeout_seconds=2),
            ledger=ledger,
            artifact_store=artifact_store,
        )
        persisted = ledger.get_record("run_attempts", result.attempt.id)

    assert result.attempt.status is RunStatus.SUCCEEDED
    assert result.stdout == b"fixture stdout\n"
    assert (output / "result.txt").read_text(encoding="utf-8") == "fixture output\n"
    assert len(result.artifacts) == 3
    assert persisted is not None
    assert persisted["status"] == "succeeded"
    assert persisted["artifact_ids"] == [artifact.manifest.id for artifact in result.artifacts]

    invocations = [
        json.loads(line)
        for line in fake_docker.with_suffix(".jsonl").read_text().splitlines()
    ]
    run_args, remove_args = invocations
    assert run_args[0] == "run"
    assert remove_args[:2] == ["rm", "--force"]
    assert str(tmp_path / "ledger.sqlite3") not in " ".join(run_args)


def test_runner_maps_failure_and_timeout_to_terminal_status(
    tmp_path: Path, fake_docker: Path
) -> None:
    runner = DockerExperimentRunner(_budget(), "fixture:1", docker_executable=fake_docker)
    for mode, expected_status in (("failure", RunStatus.FAILED), ("timeout", RunStatus.TIMED_OUT)):
        work, output, logs = _run_paths(tmp_path / mode)
        result = runner.run(
            _experiment(),
            ["fixture", mode],
            work_dir=work,
            output_dir=output,
            log_dir=logs,
            limits=ExecutionLimits(timeout_seconds=2 if mode == "failure" else 0.05),
        )
        assert result.attempt.status is expected_status
        assert result.attempt.failure_reason
        assert result.attempt.exit_code != 0


def test_mount_validation_rejects_traversal_and_protected_writable_paths(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")
    with pytest.raises(DockerRunnerError):
        build_docker_run_command(
            "fixture:1",
            ["fixture"],
            source_mounts=[DockerMount(source, "/approved/../escape")],
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "output",
            log_dir=tmp_path / "logs",
        )

    artifact_store = FileSystemArtifactStore(tmp_path / "artifacts")
    with pytest.raises(DockerRunnerError):
        DockerExperimentRunner(_budget(), "fixture:1").run(
            _experiment(),
            ["fixture"],
            work_dir=artifact_store.root / "work",
            output_dir=tmp_path / "output",
            log_dir=tmp_path / "logs",
            limits=ExecutionLimits(timeout_seconds=1),
            artifact_store=artifact_store,
        )


def test_session_service_runs_registered_artifact_and_persists_budget_snapshot(
    tmp_path: Path, fake_docker: Path
) -> None:
    project = tmp_path / "competition"
    service = SessionService()
    service.init_project(project)
    (project / "data" / "train.csv").write_text("target,feature\n0,1\n1,2\n", encoding="utf-8")
    (project / "mlquest.toml").write_text(
        """[task]
name = "fixture"
problem_statement = "predict target"
task_type = "classification"
target_column = "target"

[data]
train = "data/train.csv"

[evaluation]
metric = "accuracy"
direction = "maximize"

[budget]
session_seconds = 300
execution_minutes = 4

[policy]
mode = "autonomous"
""",
        encoding="utf-8",
    )
    session = service.start(project)
    registration = session["dataset_registrations"][0]

    result = service.run_docker_experiment(
        project,
        session["id"],
        _experiment(),
        ["fixture", "success"],
        image="mlquest-fixture:1",
        source_artifacts={registration["artifact_id"]: "/approved/input"},
        timeout_seconds=2,
        docker_executable=fake_docker,
    )

    assert result.succeeded
    assert result.attempt.artifact_ids
    status = service.get_status(project, session["id"])
    assert status["budget_status"]["model_execution_elapsed_seconds"] > 0
    assert (project / ".mlquest" / "sessions" / session["id"] / "budget.json").is_file()
