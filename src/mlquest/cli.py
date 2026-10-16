"""Typer command-line interface for local MLQuest projects."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from mlquest.domain import ApprovalDecisionType, ExternalScoreObservation, MetricDirection
from mlquest.sessions import SessionService, SessionServiceError


app = typer.Typer(
    help="Budget-aware autonomous ML competition research agent.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["--help", "-h"]},
)
experiments_app = typer.Typer(help="Inspect persisted experiments.")
research_app = typer.Typer(help="Inspect persisted research records.")
score_app = typer.Typer(help="Record user-provided score observations.")
app.add_typer(experiments_app, name="experiments")
app.add_typer(research_app, name="research")
app.add_typer(score_app, name="score")

ProjectOption = Annotated[Path, typer.Option(..., "--project", help="MLQuest project directory.")]
SessionOption = Annotated[str, typer.Option(..., "--session", help="Durable session ID.")]


def _service() -> SessionService:
    return SessionService()


def _error(error: Exception) -> NoReturn:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=2)


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Directory for the local MLQuest project.")],
) -> None:
    """Create a project directory, task template, and session workspace."""

    try:
        root = _service().init_project(path)
    except (OSError, ValueError, SessionServiceError) as error:
        _error(error)
    typer.echo(f"Initialized MLQuest project: {root}")
    typer.echo(f"Task file: {root / 'mlquest.toml'}")


@app.command()
def start(
    project: ProjectOption,
    task: Annotated[Path, typer.Option("--task", help="Task TOML, relative to project by default.")] = Path("mlquest.toml"),
) -> None:
    """Create a durable session from a task TOML and local datasets."""

    try:
        session = _service().start(project, task)
    except (OSError, ValueError, SessionServiceError) as error:
        _error(error)
    typer.echo(f"Started session: {session['id']}")
    typer.echo(f"Current phase: {session['current_phase']}")
    if session["policy"].get("mode", "approval") == "approval":
        typer.echo("Status: awaiting approval for intake")
    else:
        typer.echo("Status: active")


@app.command()
def status(project: ProjectOption, session: SessionOption) -> None:
    """Show session metadata, phase state, approvals, and configured budgets."""

    try:
        report = _service().get_status(project, session)
    except (OSError, ValueError, SessionServiceError) as error:
        _error(error)
    record = report["session"]
    console = Console()
    console.print(f"[bold]Session[/bold] {report['metadata']['id']}")
    console.print(f"Status: {report['metadata']['status']}    Current phase: {record.get('current_phase', 'unknown')}")
    console.print(f"Task: {record.get('task', {}).get('name', record.get('task_id', 'unknown'))}")
    console.print("Best candidate: unavailable until experiment services are implemented")

    budget = record.get("budget", {})
    budget_table = Table(title="Budgets")
    budget_table.add_column("Budget")
    budget_table.add_column("Configured")
    budget_table.add_column("Remaining")
    for key, label in (
        ("session_seconds", "Session"),
        ("model_execution_seconds", "Model execution"),
        ("finalisation_reserve_seconds", "Finalisation reserve"),
    ):
        value = budget.get(key, "unavailable")
        budget_table.add_row(label, _seconds(value), "unavailable")
    console.print(budget_table)

    phase_table = Table(title="Phases")
    phase_table.add_column("Phase")
    phase_table.add_column("Status")
    phase_table.add_column("Attempt")
    for phase in report["phases"]:
        phase_table.add_row(phase["phase"], phase["status"], str(phase.get("attempt", 0)))
    console.print(phase_table)
    if report["pending_approvals"]:
        approval_table = Table(title="Pending approvals")
        approval_table.add_column("Request")
        approval_table.add_column("Phase")
        approval_table.add_column("Action")
        for approval in report["pending_approvals"]:
            approval_table.add_row(approval["id"], approval["phase"], approval["requested_action"])
        console.print(approval_table)
    else:
        console.print("Pending approvals: none")
    console.print(
        f"Records: {report['experiments']} experiments, {report['research']} research sources, "
        f"{report['scores']} external scores"
    )


@app.command()
def approve(
    project: ProjectOption,
    session: SessionOption,
    request: Annotated[str | None, typer.Option("--request", "--request-id", help="Approval request ID; defaults to the first pending request.")] = None,
    decided_by: Annotated[str, typer.Option("--by", "--decided-by", help="Person or actor making the decision.")] = "cli-user",
    comment: Annotated[str | None, typer.Option("--comment")] = None,
) -> None:
    """Approve a pending phase request and persist the decision."""

    _resolve_approval(project, session, ApprovalDecisionType.APPROVE, request, decided_by, comment)


@app.command()
def reject(
    project: ProjectOption,
    session: SessionOption,
    request: Annotated[str | None, typer.Option("--request", "--request-id", help="Approval request ID; defaults to the first pending request.")] = None,
    decided_by: Annotated[str, typer.Option("--by", "--decided-by", help="Person or actor making the decision.")] = "cli-user",
    comment: Annotated[str | None, typer.Option("--comment")] = None,
) -> None:
    """Reject a pending phase request and persist the decision."""

    _resolve_approval(project, session, ApprovalDecisionType.REJECT, request, decided_by, comment)


def _resolve_approval(
    project: Path,
    session: str,
    decision: ApprovalDecisionType,
    request: str | None,
    decided_by: str,
    comment: str | None,
) -> None:
    try:
        resolved = _service().resolve_approval(
            project,
            session,
            decision=decision,
            request_id=request,
            decided_by=decided_by,
            comment=comment,
        )
    except (OSError, ValueError, SessionServiceError) as error:
        _error(error)
    typer.echo(f"Approval {decision.value}d: {resolved.id} ({resolved.phase.value})")


@app.command()
def pause(project: ProjectOption, session: SessionOption) -> None:
    """Pause a session and persist its lifecycle state."""

    try:
        _service().pause(project, session)
    except (OSError, ValueError, SessionServiceError) as error:
        _error(error)
    typer.echo(f"Paused session: {session}")


@app.command()
def resume(project: ProjectOption, session: SessionOption) -> None:
    """Resume a paused session from its persisted phase checkpoint."""

    try:
        _service().resume(project, session)
    except (OSError, ValueError, SessionServiceError) as error:
        _error(error)
    typer.echo(f"Resumed session: {session}")


@experiments_app.command("list")
def experiments_list(project: ProjectOption, session: SessionOption) -> None:
    """List persisted experiment specifications."""

    try:
        records = _service().records(project, session, "experiments")
    except (OSError, ValueError, SessionServiceError) as error:
        _error(error)
    if not records:
        typer.echo("No experiments recorded.")
        return
    table = Table(title="Experiments")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Model")
    table.add_column("Status")
    for record in records:
        table.add_row(record.get("id", ""), record.get("name", ""), record.get("model_family", ""), record.get("status", "recorded"))
    Console().print(table)


@experiments_app.command("show")
def experiments_show(
    experiment_id: Annotated[str, typer.Argument(help="Experiment ID.")],
    project: ProjectOption,
    session: SessionOption,
) -> None:
    """Show one persisted experiment specification."""

    try:
        records = _service().records(project, session, "experiments")
    except (OSError, ValueError, SessionServiceError) as error:
        _error(error)
    record = next((item for item in records if item.get("id") == experiment_id), None)
    if record is None:
        _error(SessionServiceError(f"experiment does not exist: {experiment_id}"))
    typer.echo(json.dumps(record, indent=2, sort_keys=True))


@research_app.command("list")
def research_list(project: ProjectOption, session: SessionOption) -> None:
    """List persisted research sources and findings."""

    try:
        records = _service().records(project, session, "research")
    except (OSError, ValueError, SessionServiceError) as error:
        _error(error)
    if not records:
        typer.echo("No research records recorded.")
        return
    table = Table(title="Research")
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("URL")
    for record in records:
        table.add_row(record.get("id", ""), record.get("title", ""), record.get("url", ""))
    Console().print(table)


@score_app.command("add")
def score_add(
    project: ProjectOption,
    session: SessionOption,
    source: Annotated[str, typer.Option(..., help="Where the external score came from.")],
    metric: Annotated[str, typer.Option(..., "--metric", "--metric-name", help="Metric name.")],
    score: Annotated[float, typer.Option(..., help="Observed score value.")],
    direction: Annotated[MetricDirection, typer.Option(..., help="Metric direction: maximize or minimize.")],
    reported_by: Annotated[str, typer.Option(..., "--reported-by", help="Person recording the observation.")],
    observed_at: Annotated[str | None, typer.Option("--timestamp", "--observed-at", help="ISO-8601 timestamp; defaults to now in UTC.")] = None,
    notes: Annotated[str | None, typer.Option("--notes")] = None,
    comparison_context_id: Annotated[str | None, typer.Option("--comparison-context")] = None,
) -> None:
    """Persist a manually reported external score observation."""

    try:
        timestamp = _parse_timestamp(observed_at) if observed_at else datetime.now(timezone.utc)
        observation = ExternalScoreObservation(
            source=source,
            metric_name=metric,
            score=score,
            direction=direction,
            observed_at=timestamp,
            reported_by=reported_by,
            notes=notes,
            comparison_context_id=comparison_context_id,
        )
        identifier = _service().add_score(project, session, observation)
    except (OSError, ValueError, SessionServiceError) as error:
        _error(error)
    typer.echo(f"Recorded external score: {identifier}")


@app.command()
def verify(project: ProjectOption, session: SessionOption) -> None:
    """Show the verification command surface until the verification service exists."""

    _not_implemented("verify", project, session)


@app.command()
def report(project: ProjectOption, session: SessionOption) -> None:
    """Show the report command surface until report generation exists."""

    _not_implemented("report", project, session)


@app.command("package")
def package_command(project: ProjectOption, session: SessionOption) -> None:
    """Show the package command surface until solution packaging exists."""

    _not_implemented("package", project, session)


@app.command()
def ui(project: ProjectOption, session: SessionOption) -> None:
    """Show the UI command surface until the localhost interface exists."""

    _not_implemented("ui", project, session)


def _not_implemented(command: str, project: Path, session: str) -> None:
    del project, session
    typer.echo(f"{command}: not yet implemented; the {command} service is not available in v0.")


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    timestamp = datetime.fromisoformat(normalized)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone, for example 2026-01-01T00:00:00Z")
    return timestamp.astimezone(timezone.utc)


def _seconds(value: object) -> str:
    if not isinstance(value, int):
        return str(value)
    return f"{value // 60} min"


def main() -> None:
    app()
