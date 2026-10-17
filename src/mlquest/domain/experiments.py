"""Experiment specifications and execution attempts."""

from __future__ import annotations

from datetime import datetime

from pydantic import AliasChoices, Field, field_validator, model_validator

from .common import RecordModel, RunStatus, StableId, validate_non_empty
from .resources import ResourceProfile
from .validation import ValidationSpecification


class ExperimentSpecification(RecordModel):
    name: str
    model_family: str
    validation: ValidationSpecification
    task_id: StableId | None = None
    dataset_artifact_id: StableId | None = None
    parent_experiment_id: StableId | None = None
    hypothesis_id: StableId | None = None
    derived_artifact_ids: list[StableId] = Field(default_factory=list)
    model_config_values: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("model_config", "model_config_values", "parameters"),
        serialization_alias="model_config",
    )
    random_seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)
    resource_profile: ResourceProfile | None = None
    comparison_context_id: StableId | None = None

    @field_validator("name", "model_family")
    @classmethod
    def valid_text(cls, value: str) -> str:
        return validate_non_empty(value, "experiment field")

    @field_validator("random_seeds")
    @classmethod
    def valid_seeds(cls, values: list[int]) -> list[int]:
        if any(seed < 0 for seed in values):
            raise ValueError("random seeds must be non-negative")
        return values

    @model_validator(mode="after")
    def context_matches_validation(self) -> "ExperimentSpecification":
        if (
            self.comparison_context_id is not None
            and self.validation.comparison_context_id is not None
        ):
            if self.comparison_context_id != self.validation.comparison_context_id:
                raise ValueError("experiment and validation comparison contexts differ")
        return self


class RunAttempt(RecordModel):
    experiment_id: StableId
    attempt_number: int = Field(default=1, ge=1)
    status: RunStatus = RunStatus.QUEUED
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    artifact_ids: list[StableId] = Field(default_factory=list)
    resource_usage: dict[str, float] = Field(default_factory=dict)
    failure_reason: str | None = None

    @model_validator(mode="after")
    def valid_times(self) -> "RunAttempt":
        if self.started_at and self.finished_at and self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        if (
            self.status in {RunStatus.FAILED, RunStatus.TIMED_OUT, RunStatus.CANCELLED}
            and not self.failure_reason
        ):
            raise ValueError("failed, timed-out, and cancelled runs require failure_reason")
        return self
