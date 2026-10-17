from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from mlquest.cli import app
from mlquest.persistence import LocalLedger

runner = CliRunner()


def _project(tmp_path: Path, *, mode: str = "approval") -> Path:
    project = tmp_path / "competition"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0, result.output
    (project / "data" / "train.csv").write_text("target,feature\n0,1\n1,2\n", encoding="utf-8")
    (project / "data" / "test.csv").write_text("feature\n3\n", encoding="utf-8")
    (project / "mlquest.toml").write_text(
        f'''[task]
name = "fixture"
problem_statement = "predict target"
task_type = "classification"
target_column = "target"

[data]
train = "data/train.csv"
test = "data/test.csv"

[evaluation]
metric = "accuracy"
direction = "maximize"

[policy]
mode = "{mode}"
''',
        encoding="utf-8",
    )
    return project


def _start(project: Path) -> str:
    result = runner.invoke(app, ["start", "--project", str(project)])
    assert result.exit_code == 0, result.output
    return next(
        line.split(": ", 1)[1]
        for line in result.output.splitlines()
        if line.startswith("Started session:")
    )


def test_init_creates_template_and_help_is_available(tmp_path: Path) -> None:
    project = tmp_path / "project"
    result = runner.invoke(app, ["init", str(project)])
    assert result.exit_code == 0
    assert (project / "mlquest.toml").is_file()
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "start" in help_result.output
    assert "experiments" in help_result.output


def test_start_status_and_state_survive_a_new_cli_invocation(tmp_path: Path) -> None:
    project = _project(tmp_path)
    session_id = _start(project)
    status = runner.invoke(app, ["status", "--project", str(project), "--session", session_id])
    assert status.exit_code == 0
    assert session_id in status.output
    assert "awaiting_approval" in status.output
    assert "Session" in status.output
    assert "Model execution" in status.output
    ledger_path = project / ".mlquest" / "sessions" / session_id / "ledger.sqlite"
    with LocalLedger(ledger_path) as ledger:
        assert ledger.get_session(session_id) is not None
        assert ledger.list_records("datasets", session_id=session_id)


def test_approval_pause_and_resume_are_persisted(tmp_path: Path) -> None:
    project = _project(tmp_path)
    session_id = _start(project)
    approved = runner.invoke(app, ["approve", "--project", str(project), "--session", session_id])
    assert approved.exit_code == 0, approved.output
    paused = runner.invoke(app, ["pause", "--project", str(project), "--session", session_id])
    assert paused.exit_code == 0
    resumed = runner.invoke(app, ["resume", "--project", str(project), "--session", session_id])
    assert resumed.exit_code == 0
    status = runner.invoke(app, ["status", "--project", str(project), "--session", session_id])
    assert "Status: active" in status.output
    assert "Remaining" in status.output


def test_score_add_requires_valid_external_score_and_persists_it(tmp_path: Path) -> None:
    project = _project(tmp_path, mode="autonomous")
    session_id = _start(project)
    result = runner.invoke(
        app,
        [
            "score",
            "add",
            "--project",
            str(project),
            "--session",
            session_id,
            "--source",
            "manual leaderboard",
            "--metric",
            "accuracy",
            "--score",
            "0.81",
            "--direction",
            "maximize",
            "--reported-by",
            "tester",
            "--timestamp",
            "2026-01-01T00:00:00Z",
        ],
    )
    assert result.exit_code == 0, result.output
    bad = runner.invoke(
        app,
        [
            "score",
            "add",
            "--project",
            str(project),
            "--session",
            session_id,
            "--source",
            "manual",
            "--metric",
            "accuracy",
            "--score",
            "nan",
            "--direction",
            "maximize",
            "--reported-by",
            "tester",
        ],
    )
    assert bad.exit_code != 0
    ledger_path = project / ".mlquest" / "sessions" / session_id / "ledger.sqlite"
    with LocalLedger(ledger_path) as ledger:
        scores = ledger.list_records("external_scores", session_id=session_id)
    assert len(scores) == 1
    assert scores[0]["metric_name"] == "accuracy"
