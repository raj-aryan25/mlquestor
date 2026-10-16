"""Workflow phase state and approval contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator

from .common import ApprovalDecisionType, RecordModel, StableId, utc_now, validate_non_empty


class PhaseName(StrEnum):
    INTAKE = "intake"
    DATASET = "dataset"
    BASELINE = "baseline"
    RESEARCH = "research"
    EXPERIMENT = "experiment"
    CONFIRMATION = "confirmation"
    FINALISATION = "finalisation"
    VERIFICATION = "verification"
    REPORTING = "reporting"


class PhaseStatus(StrEnum):
    PENDING = "pending"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class PhaseTransitionError(ValueError):
    pass


VALID_PHASE_TRANSITIONS: dict[PhaseStatus, frozenset[PhaseStatus]] = {
    PhaseStatus.PENDING: frozenset({PhaseStatus.RUNNING, PhaseStatus.AWAITING_APPROVAL, PhaseStatus.REJECTED}),
    PhaseStatus.AWAITING_APPROVAL: frozenset({PhaseStatus.RUNNING, PhaseStatus.REJECTED, PhaseStatus.PAUSED}),
    PhaseStatus.RUNNING: frozenset({PhaseStatus.COMPLETED, PhaseStatus.FAILED, PhaseStatus.PAUSED}),
    PhaseStatus.PAUSED: frozenset({PhaseStatus.RUNNING, PhaseStatus.REJECTED}),
    PhaseStatus.FAILED: frozenset({PhaseStatus.RUNNING, PhaseStatus.REJECTED}),
    PhaseStatus.COMPLETED: frozenset(),
    PhaseStatus.REJECTED: frozenset(),
}


class PhaseState(RecordModel):
    phase: PhaseName
    status: PhaseStatus = PhaseStatus.PENDING
    updated_at: datetime = Field(default_factory=utc_now)
    attempt: int = Field(default=0, ge=0)
    error: str | None = None

    def can_transition_to(self, target: PhaseStatus) -> bool:
        return target in VALID_PHASE_TRANSITIONS[self.status]

    def transition_to(self, target: PhaseStatus) -> "PhaseState":
        if not self.can_transition_to(target):
            raise PhaseTransitionError(f"cannot transition {self.status.value} phase to {target.value}")
        self.status = target
        self.updated_at = utc_now()
        if target == PhaseStatus.RUNNING:
            self.attempt += 1
        return self


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    DECIDED = "decided"


class ApprovalRequest(RecordModel):
    phase: PhaseName
    requested_action: str
    rationale: str
    estimated_cost_seconds: int | None = Field(default=None, ge=0)
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_at: datetime | None = None
    decided_by: str | None = None
    decision: ApprovalDecisionType | None = None
    comment: str | None = None

    @field_validator("requested_action", "rationale")
    @classmethod
    def valid_text(cls, value: str) -> str:
        return validate_non_empty(value, "approval text")

    def resolve(self, decision: "ApprovalDecision") -> "ApprovalRequest":
        if self.status is not ApprovalStatus.PENDING:
            raise ValueError("approval request has already been decided")
        if decision.request_id != self.id:
            raise ValueError("approval decision references a different request")
        self.status = ApprovalStatus.DECIDED
        self.decision = decision.decision
        self.decided_at = decision.decided_at
        self.decided_by = decision.decided_by
        self.comment = decision.comment
        return self


class ApprovalDecision(RecordModel):
    request_id: StableId
    decision: ApprovalDecisionType
    decided_by: str
    decided_at: datetime = Field(default_factory=utc_now)
    comment: str | None = None

    @field_validator("decided_by")
    @classmethod
    def valid_decider(cls, value: str) -> str:
        return validate_non_empty(value, "decided_by")
