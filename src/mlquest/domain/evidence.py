"""Measured metrics and verification records."""

from __future__ import annotations

from datetime import datetime
from math import isfinite

from pydantic import AliasChoices, Field, field_validator, model_validator

from .common import MetricDirection, RecordModel, StableId, VerificationStatus, validate_non_empty


class MetricResult(RecordModel):
    metric_name: str = Field(validation_alias=AliasChoices("metric_name", "metric"))
    value: float
    direction: MetricDirection
    run_attempt_id: StableId
    comparison_context_id: StableId
    split_name: str | None = None
    is_primary: bool = False

    @field_validator("metric_name")
    @classmethod
    def valid_metric_name(cls, value: str) -> str:
        return validate_non_empty(value, "metric_name")

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("metric value must be finite")
        return value


class VerificationResult(RecordModel):
    run_attempt_id: StableId
    status: VerificationStatus
    checks: dict[str, bool] = Field(default_factory=dict)
    details: str | None = None
    verified_at: datetime | None = None
    artifact_ids: list[StableId] = Field(default_factory=list)

    @model_validator(mode="after")
    def passed_requires_checks(self) -> "VerificationResult":
        if self.status is VerificationStatus.PASSED and self.checks and not all(self.checks.values()):
            raise ValueError("a passed verification cannot contain failed checks")
        return self


class ExternalScoreObservation(RecordModel):
    source: str
    metric_name: str
    score: float
    direction: MetricDirection
    observed_at: datetime
    reported_by: str
    comparison_context_id: StableId | None = None
    notes: str | None = None

    @field_validator("source", "metric_name", "reported_by")
    @classmethod
    def required_text(cls, value: str) -> str:
        return validate_non_empty(value, "external score field")

    @field_validator("score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("external score must be finite")
        return value
