"""Application services for the local MLQuest project lifecycle.

The CLI is intentionally a thin adapter around this module.  A service call
opens the session ledger it needs and closes it before returning, so a command
never relies on process-local session state.
"""

from __future__ import annotations

import json
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from mlquest.artifacts import FileSystemArtifactStore
from mlquest.datasets import DatasetService
from mlquest.domain import (
    ApprovalDecision,
    ApprovalDecisionType,
    ApprovalRequest,
    ApprovalStatus,
    BudgetConfiguration,
    DatasetRole,
    ExternalScoreObservation,
    MetricDirection,
    PhaseName,
    PhaseState,
    PhaseStatus,
    TaskSpecification,
    TaskType,
)
from mlquest.domain.common import utc_now, validate_identifier
from mlquest.persistence import LocalLedger


PHASE_ORDER = tuple(PhaseName)
DEFAULT_SESSION_SECONDS = 120 * 60
DEFAULT_EXECUTION_SECONDS = 90 * 60


class SessionServiceError(RuntimeError):
    """Raised when a project or durable session cannot be operated safely."""


class SessionService:
    """Create and operate on local MLQuest projects and durable sessions."""

    def init_project(self, project: Path | str) -> Path:
        root = Path(project).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        mlquest_root = root / ".mlquest"
        for directory in (
            root / "data",
            root / "solution",
            mlquest_root / "sessions",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        task_path = root / "mlquest.toml"
        if not task_path.exists():
            task_path.write_text(_TASK_TEMPLATE, encoding="utf-8")
        project_file = mlquest_root / "project.json"
        if not project_file.exists():
            _write_json(project_file, {"project": str(root), "sessions": [], "latest_session": None})
        return root

    def start(self, project: Path | str, task_path: Path | str = "mlquest.toml") -> dict[str, Any]:
        root = Path(project).expanduser().resolve()
        if not root.is_dir():
            raise SessionServiceError(f"project directory does not exist: {root}")
        task_file = Path(task_path)
        if not task_file.is_absolute():
            task_file = root / task_file
        task_file = task_file.resolve()
        if not task_file.is_file():
            raise SessionServiceError(f"task file does not exist: {task_file}")
        try:
            with task_file.open("rb") as source:
                config = tomllib.load(source)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise SessionServiceError(f"could not read task TOML: {exc}") from exc

        session_id = str(uuid4())
        session_root = root / ".mlquest" / "sessions" / session_id
        session_root.mkdir(parents=True, exist_ok=False)
        for directory in ("sources", "derived", "reports", "final"):
            (session_root / directory).mkdir()
        ledger = LocalLedger(session_root / "ledger.sqlite")
        try:
            store = FileSystemArtifactStore(session_root / "artifacts")
            task, budget, policy, registrations = self._load_task(
                config, root=root, task_file=task_file, ledger=ledger, store=store, session_id=session_id
            )
            mode = str(policy.get("mode", "approval")).lower()
            if mode not in {"approval", "autonomous"}:
                raise SessionServiceError("policy.mode must be 'approval' or 'autonomous'")
            policy["mode"] = mode
            first_status = (
                PhaseStatus.AWAITING_APPROVAL if mode == "approval" else PhaseStatus.RUNNING
            )
            for phase_name in PHASE_ORDER:
                phase = PhaseState(phase=phase_name, status=PhaseStatus.PENDING)
                if phase_name is PhaseName.INTAKE:
                    phase.status = first_status
                ledger.save_phase(session_id, phase)

            session_record: dict[str, Any] = {
                "id": session_id,
                "task_id": task.id,
                "task": task.model_dump(mode="json", exclude_none=True),
                "budget": budget.model_dump(mode="json", exclude_none=True),
                "policy": policy,
                "project_path": str(root),
                "task_path": str(task_file),
                "status": "active",
                "current_phase": PhaseName.INTAKE.value,
                "created_at": utc_now().isoformat().replace("+00:00", "Z"),
                "dataset_registrations": registrations,
            }
            ledger.save_session(session_id, session_record, status="active")
            ledger.append_event(
                session_id,
                "session.created",
                {"task_id": task.id, "mode": mode},
                entity_type="session",
                entity_id=session_id,
            )
            if mode == "approval":
                request = ApprovalRequest(
                    phase=PhaseName.INTAKE,
                    requested_action="begin intake phase",
                    rationale="Review the task and registered source datasets before workflow execution.",
                )
                ledger.put_record("approvals", request, session_id=session_id)
                ledger.append_event(
                    session_id,
                    "approval.requested",
                    request.model_dump(mode="json", exclude_none=True),
                    entity_type="approval",
                    entity_id=request.id,
                )
            _register_project_session(root, session_id, task.name)
            manifest = {
                "session_id": session_id,
                "task_id": task.id,
                "task_path": str(task_file),
                "ledger": "ledger.sqlite",
                "artifacts": "artifacts",
                "created_at": session_record["created_at"],
            }
            _write_json(session_root / "manifest.json", manifest)
            return session_record
        finally:
            ledger.close()

    @contextmanager
    def open_session(self, project: Path | str, session_id: str) -> Iterator[tuple[Path, LocalLedger]]:
        root = Path(project).expanduser().resolve()
        try:
            identifier = validate_identifier(session_id, "session ID")
        except ValueError as exc:
            raise SessionServiceError(str(exc)) from exc
        session_root = root / ".mlquest" / "sessions" / identifier
        ledger_path = session_root / "ledger.sqlite"
        if not ledger_path.is_file():
            raise SessionServiceError(f"session does not exist: {identifier}")
        ledger = LocalLedger(ledger_path)
        try:
            if ledger.get_session_metadata(identifier) is None:
                raise SessionServiceError(f"session metadata is missing: {identifier}")
            yield session_root, ledger
        finally:
            ledger.close()

    def get_status(self, project: Path | str, session_id: str) -> dict[str, Any]:
        with self.open_session(project, session_id) as (_, ledger):
            metadata = ledger.get_session_metadata(session_id)
            assert metadata is not None
            phases = [PhaseState.model_validate(item) for item in ledger.list_records("phases", session_id=session_id)]
            phases.sort(key=lambda item: PHASE_ORDER.index(item.phase))
            approvals = [
                ApprovalRequest.model_validate(item)
                for item in ledger.list_records("approvals", session_id=session_id)
                if ApprovalRequest.model_validate(item).status is ApprovalStatus.PENDING
            ]
            return {
                "metadata": metadata,
                "session": metadata["record"],
                "phases": [phase.model_dump(mode="json", exclude_none=True) for phase in phases],
                "pending_approvals": [item.model_dump(mode="json", exclude_none=True) for item in approvals],
                "experiments": len(ledger.list_records("experiments", session_id=session_id)),
                "research": len(ledger.list_records("research", session_id=session_id)),
                "scores": len(ledger.list_records("external_scores", session_id=session_id)),
            }

    def resolve_approval(
        self,
        project: Path | str,
        session_id: str,
        *,
        decision: ApprovalDecisionType,
        request_id: str | None = None,
        decided_by: str = "cli-user",
        comment: str | None = None,
    ) -> ApprovalRequest:
        with self.open_session(project, session_id) as (_, ledger):
            candidates = [
                ApprovalRequest.model_validate(item)
                for item in ledger.list_records("approvals", session_id=session_id)
                if ApprovalRequest.model_validate(item).status is ApprovalStatus.PENDING
            ]
            if request_id is not None:
                candidates = [item for item in candidates if item.id == request_id]
            if not candidates:
                detail = f"approval request not found or already decided: {request_id}" if request_id else "no pending approval requests"
                raise SessionServiceError(detail)
            request = candidates[0]
            approval_decision = ApprovalDecision(
                request_id=request.id,
                decision=decision,
                decided_by=decided_by,
                comment=comment,
            )
            request.resolve(approval_decision)
            ledger.put_record("approvals", request, session_id=session_id)
            phase_data = ledger.get_phase(session_id, request.phase.value)
            if phase_data is not None:
                phase = PhaseState.model_validate(phase_data)
                target = PhaseStatus.RUNNING if decision is ApprovalDecisionType.APPROVE else PhaseStatus.REJECTED
                if phase.can_transition_to(target):
                    phase.transition_to(target)
                    ledger.save_phase(session_id, phase)
            ledger.append_event(
                session_id,
                f"approval.{decision.value}d",
                {"request_id": request.id, "decided_by": decided_by, "comment": comment},
                entity_type="approval",
                entity_id=request.id,
            )
            return request

    def pause(self, project: Path | str, session_id: str) -> dict[str, Any]:
        return self._set_lifecycle(project, session_id, "paused")

    def resume(self, project: Path | str, session_id: str) -> dict[str, Any]:
        return self._set_lifecycle(project, session_id, "active")

    def add_score(self, project: Path | str, session_id: str, observation: ExternalScoreObservation) -> str:
        with self.open_session(project, session_id) as (_, ledger):
            ledger.put_record("external_scores", observation, session_id=session_id)
            ledger.append_event(
                session_id,
                "score.external_recorded",
                observation.model_dump(mode="json", exclude_none=True),
                entity_type="external_score",
                entity_id=observation.id,
            )
            return observation.id

    def records(self, project: Path | str, session_id: str, category: str) -> list[dict[str, Any]]:
        with self.open_session(project, session_id) as (_, ledger):
            return [dict(item) for item in ledger.list_records(category, session_id=session_id)]

    def _set_lifecycle(self, project: Path | str, session_id: str, target: str) -> dict[str, Any]:
        with self.open_session(project, session_id) as (_, ledger):
            metadata = ledger.get_session_metadata(session_id)
            assert metadata is not None
            current = str(metadata["status"])
            if target == "paused" and current != "active":
                raise SessionServiceError(f"cannot pause a session in {current} state")
            if target == "active" and current != "paused":
                raise SessionServiceError(f"cannot resume a session in {current} state")
            record = dict(metadata["record"])
            phase_name = str(record.get("current_phase", PhaseName.INTAKE.value))
            phase_data = ledger.get_phase(session_id, phase_name)
            if phase_data is not None:
                phase = PhaseState.model_validate(phase_data)
                if target == "paused" and phase.status is PhaseStatus.RUNNING:
                    phase.transition_to(PhaseStatus.PAUSED)
                    ledger.save_phase(session_id, phase)
                elif target == "active" and phase.status is PhaseStatus.PAUSED:
                    phase.transition_to(PhaseStatus.RUNNING)
                    ledger.save_phase(session_id, phase)
            record["status"] = target
            ledger.save_session(session_id, record, status=target)
            ledger.append_event(session_id, f"session.{target}", {"phase": phase_name})
            return record

    @staticmethod
    def _load_task(
        config: dict[str, Any],
        *,
        root: Path,
        task_file: Path,
        ledger: LocalLedger,
        store: FileSystemArtifactStore,
        session_id: str,
    ) -> tuple[TaskSpecification, BudgetConfiguration, dict[str, Any], list[dict[str, Any]]]:
        task_config = _mapping(config, "task")
        data_config = _mapping(config, "data")
        evaluation = _mapping(config, "evaluation")
        budget_config = _mapping(config, "budget")
        policy = dict(_mapping(config, "policy"))
        try:
            train = _required_path(data_config, "train", root)
            registrations: list[dict[str, Any]] = []
            role_paths: list[tuple[DatasetRole, Path]] = [(DatasetRole.TRAIN, train)]
            for key, role in (("test", DatasetRole.TEST), ("sample_submission", DatasetRole.SAMPLE_SUBMISSION), ("holdout", DatasetRole.HOLDOUT)):
                value = data_config.get(key)
                if value:
                    role_paths.append((role, _required_path(data_config, key, root)))
            service = DatasetService(store, ledger=ledger)
            roles: dict[DatasetRole, str] = {}
            for role, path in role_paths:
                registration = service.register_source(path, role=role, session_id=session_id)
                roles[role] = registration.artifact_id
                registrations.append({"role": role.value, **registration.model_dump(mode="json")})
            task = TaskSpecification(
                name=str(task_config.get("name", root.name)),
                problem_statement=str(task_config.get("problem_statement", "")),
                task_type=TaskType(str(task_config.get("task_type", "classification"))),
                target_column=str(task_config.get("target_column", "target")),
                primary_metric=str(evaluation.get("metric", evaluation.get("primary_metric", ""))),
                metric_direction=MetricDirection(str(evaluation.get("direction", "maximize"))),
                source_artifact_ids=list(dict.fromkeys(roles.values())),
                dataset_roles=roles,
                secondary_metrics=[str(value) for value in evaluation.get("secondary_metrics", [])],
                target_definition=task_config.get("target_definition"),
                constraints=[str(value) for value in task_config.get("constraints", [])],
            )
            budget = BudgetConfiguration(
                session_seconds=_budget_seconds(
                    budget_config, "session_seconds", "session_minutes", DEFAULT_SESSION_SECONDS
                ),
                model_execution_seconds=_budget_seconds(
                    budget_config, "model_execution_seconds", "execution_minutes", DEFAULT_EXECUTION_SECONDS
                ),
                finalisation_reserve_seconds=_budget_seconds(
                    budget_config, "finalisation_reserve_seconds", "finalisation_reserve_minutes", 0
                ),
                max_experiments=budget_config.get("max_experiments"),
            )
            return task, budget, policy, registrations
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise SessionServiceError(f"invalid task configuration in {task_file}: {exc}") from exc


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{key}] must be a TOML table")
    return value


def _required_path(data: dict[str, Any], key: str, root: Path) -> Path:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"data.{key} is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _budget_seconds(
    config: dict[str, Any], seconds_key: str, minutes_key: str, default_seconds: int
) -> int:
    if seconds_key in config:
        return int(config[seconds_key])
    return int(config.get(minutes_key, default_seconds // 60) * 60)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _register_project_session(project: Path, session_id: str, task_name: str) -> None:
    project_file = project / ".mlquest" / "project.json"
    try:
        payload = json.loads(project_file.read_text(encoding="utf-8")) if project_file.exists() else {}
    except json.JSONDecodeError:
        payload = {}
    sessions = payload.get("sessions", [])
    if not isinstance(sessions, list):
        sessions = []
    sessions.append({"id": session_id, "task": task_name})
    payload.update({"project": str(project), "sessions": sessions, "latest_session": session_id})
    _write_json(project_file, payload)


_TASK_TEMPLATE = """[task]
name = \"example-tabular-task\"
problem_statement = \"Predict the target for each row in the test set.\"
task_type = \"classification\"
target_column = \"target\"

[data]
train = \"data/train.csv\"
test = \"data/test.csv\"
sample_submission = \"data/sample_submission.csv\"

[evaluation]
metric = \"roc_auc\"
direction = \"maximize\"

[budget]
session_minutes = 120
execution_minutes = 90

[policy]
mode = \"approval\"
automatic_submission = false
"""
