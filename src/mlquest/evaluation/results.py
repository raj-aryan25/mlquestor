"""Typed result and prediction-validation contracts for local evaluation."""

from __future__ import annotations

from math import isfinite

from pydantic import Field, field_validator

from mlquest.domain import MetricDirection, TaskType
from mlquest.domain.common import DomainModel

from .validation import PredictionValidationError, PredictionValidationResult, validate_predictions

__all__ = [
    "AggregateEvaluationResult",
    "BaselineEvaluationResult",
    "FoldEvaluationResult",
    "PredictionValidationError",
    "PredictionValidationResult",
    "PredictionValidationSummary",
    "ResourceMetadata",
    "validate_predictions",
]


class ResourceMetadata(DomainModel):
    """Runtime and process measurements captured for an evaluation."""

    runtime_seconds: float = Field(ge=0)
    cpu_time_seconds: float = Field(ge=0)
    peak_memory_mb: float = Field(ge=0)
    thread_count: int = Field(ge=1)
    cpu_count: int = Field(ge=1)

    @field_validator("runtime_seconds", "cpu_time_seconds", "peak_memory_mb")
    @classmethod
    def finite_measurement(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("resource measurements must be finite")
        return value


class PredictionValidationSummary(DomainModel):
    """Serializable summary retained after the validation API accepts predictions."""

    n_rows: int = Field(ge=1)
    shape: tuple[int, ...]
    prediction_kind: str
    label_domain: list[object] = Field(default_factory=list)


class FoldEvaluationResult(DomainModel):
    """Metrics and execution metadata for one validation fold."""

    fold_index: int = Field(ge=0)
    train_size: int = Field(ge=1)
    validation_size: int = Field(ge=1)
    metrics: dict[str, float]
    primary_metric: str
    primary_value: float
    prediction_validation: PredictionValidationSummary
    resource_metadata: ResourceMetadata

    @field_validator("primary_value")
    @classmethod
    def finite_primary_value(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("primary metric must be finite")
        return value


class AggregateEvaluationResult(DomainModel):
    """Aggregate baseline result, retaining every fold's measured outcome."""

    baseline_name: str
    baseline_tier: str
    task_type: TaskType
    primary_metric: str
    metric_direction: MetricDirection
    fold_results: list[FoldEvaluationResult] = Field(min_length=1)
    aggregate_metrics: dict[str, float]
    metric_stddev: dict[str, float]
    primary_value: float
    ranking_value: float
    resource_metadata: ResourceMetadata
    comparison_context_id: str | None = None
    split_identity: str | None = None

    @property
    def fold_metrics(self) -> list[dict[str, float]]:
        """Return fold metrics without requiring callers to know the contract shape."""

        return [fold.metrics for fold in self.fold_results]

    def is_better_than(self, other: "AggregateEvaluationResult") -> bool:
        """Compare results while respecting the configured metric direction."""

        if self.primary_metric != other.primary_metric:
            raise ValueError("cannot compare different primary metrics")
        if self.metric_direction is not other.metric_direction:
            raise ValueError("cannot compare metrics with different directions")
        return self.ranking_value > other.ranking_value

    @property
    def runtime_seconds(self) -> float:
        """Convenience access to the aggregate wall-clock runtime."""

        return self.resource_metadata.runtime_seconds


BaselineEvaluationResult = AggregateEvaluationResult
