"""Task and evaluation definitions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import AliasChoices, Field, field_validator, model_validator

from .artifacts import DatasetRole
from .common import MetricDirection, RecordModel, StableId, validate_non_empty


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TaskSpecification(RecordModel):
    name: str
    problem_statement: str
    task_type: TaskType
    target_column: str
    primary_metric: str
    metric_direction: MetricDirection
    source_artifact_ids: list[StableId] = Field(
        default_factory=list,
        validation_alias=AliasChoices("source_artifact_ids", "dataset_artifact_ids"),
    )
    dataset_roles: dict[DatasetRole, StableId] = Field(default_factory=dict)
    secondary_metrics: list[str] = Field(default_factory=list)
    target_definition: str | None = None
    constraints: list[str] = Field(default_factory=list)

    @field_validator("name", "problem_statement", "target_column", "primary_metric")
    @classmethod
    def required_text(cls, value: str) -> str:
        return validate_non_empty(value, "task field")

    @model_validator(mode="after")
    def has_dataset_reference(self) -> "TaskSpecification":
        if not self.source_artifact_ids and not self.dataset_roles:
            raise ValueError("task specification requires at least one source dataset")
        if self.target_column in {"", "none", "None"}:
            raise ValueError("target_column must identify the prediction target")
        return self
