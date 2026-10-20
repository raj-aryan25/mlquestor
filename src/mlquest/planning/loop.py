"""Serial, approval-aware orchestration for bounded experiment proposals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any, Protocol

from pydantic import Field, field_validator

from mlquest.budgets import BudgetManager, RunRequest
from mlquest.domain import (
    ApprovalDecision,
    ApprovalDecisionType,
    ApprovalRequest,
    ApprovalStatus,
    ExperimentSpecification,
    ExperimentSpecificationProposal,
    PhaseName,
    PhaseState,
    PhaseStatus,
    TaskSpecification,
    ValidationSpecification,
)
from mlquest.domain.common import DomainModel, validate_non_empty
from mlquest.persistence import LocalLedger

from .contracts import (
    CandidateProposal,
    DiagnosticSeverity,
    ExperimentResult,
    ExperimentStatus,
    PlanningDiagnostic,
    assess_evidence_strength,
)
from .planner import AllowlistedExperimentPlanner
from .scheduler import SerialExperimentScheduler


class LoopMode(StrEnum):
    """Whether phase-boundary approvals are required."""

    APPROVAL = "approval"
    AUTONOMOUS = "autonomous"


class LoopStatus(StrEnum):
    """Terminal state of one serial orchestration pass."""

    COMPLETED = "completed"
    AWAITING_APPROVAL = "awaiting_approval"
    STOPPED = "stopped"


class ExperimentExecutor(Protocol):
    """Execution seam for a Docker-backed experiment runner.

    The planner never supplies a command.  Production implementations are
    expected to translate an approved specification into a Docker invocation;
    tests can use a deterministic fake with the same boundary.
    """

    def execute(self, specification: ExperimentSpecification) -> ExperimentResult: ...


class LoopResult(DomainModel):
    """Durable, serializable summary of one planner/scheduler pass."""

    status: LoopStatus
    phase_status: PhaseStatus
    results: list[ExperimentResult] = Field(default_factory=list)
    diagnostics: list[PlanningDiagnostic] = Field(default_factory=list)
    approval_request_id: str | None = None
    stop_reason: str | None = None

    @field_validator("stop_reason")
    @classmethod
    def valid_stop_reason(cls, value: str | None) -> str | None:
        return None if value is None else validate_non_empty(value, "stop reason")


class ExperimentLoop:
    """Run validated proposals one at a time behind budget and phase gates."""

    def __init__(
        self,
        *,
        ledger: LocalLedger,
        session_id: str,
        task: TaskSpecification,
        budget_manager: BudgetManager,
        executor: ExperimentExecutor,
        planner: AllowlistedExperimentPlanner | None = None,
        scheduler: SerialExperimentScheduler | None = None,
        dataset_size: int = 0,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        self.ledger = ledger
        self.session_id = session_id
        self.task = task
        self.budget_manager = budget_manager
        self.executor = executor
        self.planner = planner or AllowlistedExperimentPlanner()
        self.scheduler = scheduler or SerialExperimentScheduler(max_count=1)
        self.dataset_size = dataset_size

    def run(
        self,
        proposals: Iterable[ExperimentSpecificationProposal | Mapping[str, Any] | object],
        *,
        validation: ValidationSpecification | None = None,
        mode: LoopMode | str = LoopMode.AUTONOMOUS,
        approval_granted: bool = False,
        comparison_context_id: str | None = None,
        expected_value: float = 0.5,
        novelty: float = 0.5,
        risk: float = 0.5,
    ) -> LoopResult:
        """Plan, rank, admit, and execute a bounded serial proposal set."""

        resolved_mode = LoopMode(mode)
        diagnostics: list[PlanningDiagnostic] = []
        phase, approval_request = self._prepare_phase(
            resolved_mode, approval_granted=approval_granted
        )
        if approval_request is not None:
            return LoopResult(
                status=LoopStatus.AWAITING_APPROVAL,
                phase_status=phase.status,
                diagnostics=diagnostics,
                approval_request_id=approval_request.id,
                stop_reason="experiment phase approval is pending",
            )

        candidates: list[CandidateProposal] = []
        for proposal in proposals:
            planned = self.planner.plan(
                proposal,
                task=self.task,
                validation=validation,
                comparison_context_id=comparison_context_id,
            )
            diagnostics.extend(planned.diagnostics)
            if planned.specification is None or planned.proposal is None:
                continue
            estimate = self.budget_manager.estimate(
                model_family=planned.specification.model_family,
                dataset_size=self.dataset_size,
                configuration=planned.specification.model_config_values,
            )
            candidates.append(
                CandidateProposal(
                    proposal=planned.proposal,
                    expected_value=expected_value,
                    estimated_cost=estimate.estimated_seconds,
                    novelty=novelty,
                    risk=risk,
                    comparison_context_id=planned.specification.comparison_context_id,
                )
            )

        if not candidates:
            fallback = self.planner.plan(
                self.planner.fallback_proposal(self.task),
                task=self.task,
                validation=validation,
                comparison_context_id=comparison_context_id,
            )
            diagnostics.extend(fallback.diagnostics)
            if fallback.specification is None or fallback.proposal is None:
                return self._stop(phase, diagnostics, "no executable proposal was available")
            estimate = self.budget_manager.estimate(
                model_family=fallback.specification.model_family,
                dataset_size=self.dataset_size,
                configuration=fallback.specification.model_config_values,
            )
            candidates.append(
                CandidateProposal(
                    proposal=fallback.proposal,
                    expected_value=expected_value,
                    estimated_cost=estimate.estimated_seconds,
                    novelty=novelty,
                    risk=risk,
                    comparison_context_id=fallback.specification.comparison_context_id,
                )
            )

        schedule = self.scheduler.schedule(candidates)
        diagnostics.extend(schedule.diagnostics)
        results: list[ExperimentResult] = []
        for ranked in schedule.selected:
            planned = self.planner.plan(
                ranked.candidate.proposal,
                task=self.task,
                validation=validation,
                comparison_context_id=comparison_context_id,
            )
            diagnostics.extend(planned.diagnostics)
            if planned.specification is None:
                continue
            specification = planned.specification
            self.ledger.put_record("experiments", specification, session_id=self.session_id)
            self.ledger.append_event(
                self.session_id,
                "experiment.queued",
                specification.model_dump(mode="json", exclude_none=True),
                entity_type="experiment",
                entity_id=specification.id,
            )
            try:
                admission = self.budget_manager.preflight(
                    RunRequest(
                        model_family=specification.model_family,
                        dataset_size=self.dataset_size,
                        configuration=specification.model_config_values,
                    )
                )
                if not admission.accepted:
                    result = self._failed_result(
                        specification,
                        "budget_rejected",
                        admission.reason or "budget preflight rejected the experiment",
                    )
                else:
                    result = self.executor.execute(specification)
            except Exception as exc:  # executor failures are ledger-visible diagnostics
                result = self._failed_result(specification, "executor_failed", str(exc))
            results.append(result)
            self.ledger.append_event(
                self.session_id,
                "experiment.result_recorded",
                result.model_dump(mode="json", exclude_none=True),
                entity_type="experiment",
                entity_id=specification.id,
            )

        self._complete_phase(phase)
        status = LoopStatus.COMPLETED if results else LoopStatus.STOPPED
        return LoopResult(
            status=status,
            phase_status=phase.status,
            results=results,
            diagnostics=diagnostics,
            stop_reason=None if results else "scheduler selected no candidates",
        )

    def _prepare_phase(
        self, mode: LoopMode, *, approval_granted: bool
    ) -> tuple[PhaseState, ApprovalRequest | None]:
        phase_data = self.ledger.get_phase(self.session_id, PhaseName.EXPERIMENT.value)
        phase = (
            PhaseState.model_validate(phase_data)
            if phase_data is not None
            else PhaseState(phase=PhaseName.EXPERIMENT)
        )
        if phase_data is None:
            self.ledger.save_phase(self.session_id, phase)

        pending = self._pending_approval()
        if mode is LoopMode.APPROVAL and not approval_granted:
            if pending is None:
                phase.transition_to(PhaseStatus.AWAITING_APPROVAL)
                self.ledger.save_phase(self.session_id, phase)
                pending = ApprovalRequest(
                    phase=PhaseName.EXPERIMENT,
                    requested_action="run the next scheduled experiment",
                    rationale=(
                        "Review the validated experiment proposal, estimated cost, and risk "
                        "before Docker execution."
                    ),
                )
                self.ledger.put_record("approvals", pending, session_id=self.session_id)
                self.ledger.append_event(
                    self.session_id,
                    "approval.requested",
                    pending.model_dump(mode="json", exclude_none=True),
                    entity_type="approval",
                    entity_id=pending.id,
                )
            return phase, pending

        if pending is None and phase.status is PhaseStatus.PENDING:
            pending = ApprovalRequest(
                phase=PhaseName.EXPERIMENT,
                requested_action="run the next scheduled experiment",
                rationale="The autonomous policy admitted the validated experiment proposal.",
            )
        if pending is not None:
            decision = ApprovalDecision(
                request_id=pending.id,
                decision=ApprovalDecisionType.APPROVE,
                decided_by="autonomous-policy" if mode is LoopMode.AUTONOMOUS else "loop-user",
            )
            pending.resolve(decision)
            self.ledger.put_record("approvals", pending, session_id=self.session_id)
            self.ledger.append_event(
                self.session_id,
                "approval.auto_approved" if mode is LoopMode.AUTONOMOUS else "approval.approved",
                pending.model_dump(mode="json", exclude_none=True),
                entity_type="approval",
                entity_id=pending.id,
            )
        if phase.status is PhaseStatus.PENDING:
            phase.transition_to(PhaseStatus.RUNNING)
        elif phase.status is PhaseStatus.AWAITING_APPROVAL:
            phase.transition_to(PhaseStatus.RUNNING)
        elif phase.status is PhaseStatus.PAUSED:
            phase.transition_to(PhaseStatus.RUNNING)
        elif phase.status is not PhaseStatus.RUNNING:
            raise ValueError(f"experiment phase cannot run from {phase.status.value}")
        self.ledger.save_phase(self.session_id, phase)
        self.ledger.append_event(
            self.session_id,
            "phase.running",
            {"phase": PhaseName.EXPERIMENT.value, "mode": mode.value},
            entity_type="phase",
            entity_id=phase.id,
        )
        return phase, None

    def _pending_approval(self) -> ApprovalRequest | None:
        requests = self.ledger.list_records("approvals", session_id=self.session_id)
        for payload in requests:
            request = ApprovalRequest.model_validate(payload)
            if request.phase is PhaseName.EXPERIMENT and request.status is ApprovalStatus.PENDING:
                return request
        return None

    def _complete_phase(self, phase: PhaseState) -> None:
        if phase.status is PhaseStatus.RUNNING:
            phase.transition_to(PhaseStatus.COMPLETED)
            self.ledger.save_phase(self.session_id, phase)
            self.ledger.append_event(
                self.session_id,
                "phase.completed",
                {"phase": phase.phase.value},
                entity_type="phase",
                entity_id=phase.id,
            )

    def _stop(
        self,
        phase: PhaseState,
        diagnostics: list[PlanningDiagnostic],
        reason: str,
    ) -> LoopResult:
        if phase.status is PhaseStatus.RUNNING:
            phase.transition_to(PhaseStatus.FAILED)
            self.ledger.save_phase(self.session_id, phase)
        diagnostics.append(
            PlanningDiagnostic(
                code="loop_stopped",
                message=reason,
                severity=DiagnosticSeverity.ERROR,
            )
        )
        return LoopResult(
            status=LoopStatus.STOPPED,
            phase_status=phase.status,
            diagnostics=diagnostics,
            stop_reason=reason,
        )

    @staticmethod
    def _failed_result(
        specification: ExperimentSpecification,
        code: str,
        message: str,
    ) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=specification.id,
            status=ExperimentStatus.FAILED,
            comparison_context_id=specification.comparison_context_id,
            scores=[],
            evidence=assess_evidence_strength(
                [], comparison_context_id=specification.comparison_context_id
            ),
            diagnostics=[
                PlanningDiagnostic(
                    code=code,
                    message=message,
                    severity=DiagnosticSeverity.ERROR,
                )
            ],
        )


__all__ = ["ExperimentExecutor", "ExperimentLoop", "LoopMode", "LoopResult", "LoopStatus"]
