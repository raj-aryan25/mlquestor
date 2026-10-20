"""Typed contracts shared by the planner and serial scheduler."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

from pydantic import Field, field_validator, model_validator

from mlquest.domain import (
    EvidenceStrength,
    ExperimentSpecification,
    ExperimentSpecificationProposal,
    StableId,
)
from mlquest.domain.common import DomainModel, validate_non_empty


class DiagnosticSeverity(StrEnum):
    """Severity of a planner or scheduler diagnostic."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class PlanningDiagnostic(DomainModel):
    """A safe, serializable explanation of a planning decision."""

    code: str
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    field: str | None = None

    @field_validator("code", "message")
    @classmethod
    def required_text(cls, value: str) -> str:
        return validate_non_empty(value, "diagnostic field")


class PlanningResult(DomainModel):
    """Result of validating a proposal and optionally applying a fallback."""

    specification: ExperimentSpecification | None = None
    proposal: ExperimentSpecificationProposal | None = None
    diagnostics: list[PlanningDiagnostic] = Field(default_factory=list)
    used_fallback: bool = False
    accepted: bool = False

    @model_validator(mode="after")
    def accepted_requires_specification(self) -> "PlanningResult":
        if self.accepted and self.specification is None:
            raise ValueError("an accepted planning result requires a specification")
        if self.used_fallback and self.proposal is None:
            raise ValueError("a fallback planning result requires a proposal")
        return self


class CandidateProposal(DomainModel):
    """A proposal plus the scheduler's bounded decision signals."""

    proposal: ExperimentSpecificationProposal
    expected_value: float
    estimated_cost: float = Field(ge=0)
    novelty: float = Field(ge=0, le=1)
    risk: float = Field(ge=0, le=1)
    comparison_context_id: StableId | None = None

    @field_validator("estimated_cost")
    @classmethod
    def finite_cost(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("estimated_cost must be finite")
        return value

    @field_validator("expected_value")
    @classmethod
    def finite_expected_value(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("expected_value must be finite")
        return value


class RankedCandidate(DomainModel):
    """A candidate with its deterministic scheduler rank."""

    candidate: CandidateProposal
    ranking_score: float
    rank: int = Field(ge=1)

    @field_validator("ranking_score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("ranking_score must be finite")
        return value


class ScheduleResult(DomainModel):
    """Serial schedule output, including candidates that were not selected."""

    selected: list[RankedCandidate] = Field(default_factory=list)
    ranked: list[RankedCandidate] = Field(default_factory=list)
    diagnostics: list[PlanningDiagnostic] = Field(default_factory=list)


class ExperimentStatus(StrEnum):
    """Terminal or in-flight state understood by the planning loop."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class EvidenceAssessment(DomainModel):
    """Evidence strength derived from repeated comparable observations."""

    strength: EvidenceStrength
    score_count: int = Field(ge=0)
    successful_run_count: int = Field(ge=0)
    comparison_context_id: StableId | None = None
    score_delta: float | None = None

    @field_validator("score_delta")
    @classmethod
    def finite_delta(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("score_delta must be finite")
        return value

    @model_validator(mode="after")
    def strength_needs_observations(self) -> "EvidenceAssessment":
        if self.score_count > self.successful_run_count:
            raise ValueError("score_count cannot exceed successful_run_count")
        if self.strength is EvidenceStrength.CONFIRMED and self.score_count < 2:
            raise ValueError("confirmed evidence requires at least two comparable scores")
        if self.strength is EvidenceStrength.PROVISIONAL and self.score_count < 1:
            raise ValueError("provisional evidence requires a score")
        return self


class ExperimentResult(DomainModel):
    """Typed result retained by the loop after one experiment attempt."""

    experiment_id: StableId
    status: ExperimentStatus
    comparison_context_id: StableId | None = None
    scores: list[float] = Field(default_factory=list)
    baseline_score: float | None = None
    evidence: EvidenceAssessment
    diagnostics: list[PlanningDiagnostic] = Field(default_factory=list)

    @field_validator("scores", mode="after")
    @classmethod
    def finite_scores(cls, values: list[float]) -> list[float]:
        if any(not isfinite(value) for value in values):
            raise ValueError("experiment scores must be finite")
        return values

    @field_validator("baseline_score")
    @classmethod
    def finite_baseline(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("baseline_score must be finite")
        return value

    @model_validator(mode="after")
    def evidence_matches_result(self) -> "ExperimentResult":
        if self.evidence.score_count != len(self.scores):
            raise ValueError("evidence score_count must match scores")
        if self.evidence.comparison_context_id != self.comparison_context_id:
            raise ValueError("result and evidence comparison contexts differ")
        return self

    @property
    def evidence_strength(self) -> EvidenceStrength:
        """Convenience access to the policy-derived strength."""

        return self.evidence.strength


def assess_evidence_strength(
    scores: list[float],
    *,
    successful_run_count: int | None = None,
    comparison_context_id: StableId | None = None,
    score_delta: float | None = None,
) -> EvidenceAssessment:
    """Classify observations without promoting one score to confirmation."""

    count = len(scores)
    successful = count if successful_run_count is None else successful_run_count
    if successful < count:
        raise ValueError("successful_run_count cannot be less than the score count")
    strength = (
        EvidenceStrength.INSUFFICIENT
        if count == 0
        else EvidenceStrength.EXPLORATORY
        if count == 1
        else EvidenceStrength.PROVISIONAL
    )
    return EvidenceAssessment(
        strength=strength,
        score_count=count,
        successful_run_count=successful,
        comparison_context_id=comparison_context_id,
        score_delta=score_delta,
    )


ExperimentCandidate = CandidateProposal
PlannerDiagnostic = PlanningDiagnostic
PlannerOutcome = PlanningResult
Schedule = ScheduleResult


__all__ = [
    "CandidateProposal",
    "DiagnosticSeverity",
    "EvidenceAssessment",
    "ExperimentCandidate",
    "ExperimentResult",
    "ExperimentStatus",
    "PlanningDiagnostic",
    "PlannerDiagnostic",
    "PlannerOutcome",
    "PlanningResult",
    "RankedCandidate",
    "ScheduleResult",
    "Schedule",
    "assess_evidence_strength",
]
