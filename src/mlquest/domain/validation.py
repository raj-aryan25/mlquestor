"""Validation, split, and score-comparison contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from hashlib import sha256
from typing import Any, Self

from pydantic import AliasChoices, Field, field_validator, model_validator

from .common import MetricDirection, RecordModel, StableId, validate_non_empty


class SplitStrategy(StrEnum):
    RANDOM = "random"
    STRATIFIED = "stratified"
    GROUP = "group"
    TIME = "time"
    HOLDOUT = "holdout"
    CROSS_VALIDATION = "cross_validation"


class SplitSpecification(RecordModel):
    strategy: SplitStrategy = Field(validation_alias=AliasChoices("strategy", "method"))
    test_size: float | None = Field(default=None, gt=0, lt=1)
    validation_size: float | None = Field(default=None, gt=0, lt=1)
    n_splits: int | None = Field(default=None, ge=2)
    random_seed: int | None = Field(default=None, ge=0)
    shuffle: bool = True
    group_column: str | None = None
    time_column: str | None = None
    split_id: StableId | None = None

    @model_validator(mode="after")
    def valid_strategy_options(self) -> "SplitSpecification":
        if self.strategy is SplitStrategy.CROSS_VALIDATION and self.n_splits is None:
            raise ValueError("cross-validation requires n_splits")
        if self.strategy is SplitStrategy.GROUP and not self.group_column:
            raise ValueError("group split requires group_column")
        if self.strategy is SplitStrategy.TIME and not self.time_column:
            raise ValueError("time split requires time_column")
        if self.test_size is not None and self.validation_size is not None:
            if self.test_size + self.validation_size >= 1:
                raise ValueError("test and validation fractions must leave training data")
        return self


class ValidationSpecification(RecordModel):
    split: SplitSpecification = Field(validation_alias=AliasChoices("split", "split_specification"))
    primary_metric: str
    metric_direction: MetricDirection
    secondary_metrics: list[str] = Field(default_factory=list)
    comparison_context_id: StableId | None = None

    @field_validator("primary_metric")
    @classmethod
    def valid_metric(cls, value: str) -> str:
        return validate_non_empty(value, "primary_metric")


class ComparisonContext(RecordModel):
    dataset_artifact_ids: list[StableId] = Field(default_factory=list)
    dataset_id: StableId | None = None
    split_identity: StableId | None = None
    split_specification: SplitSpecification | None = None
    metric_name: str
    metric_direction: MetricDirection
    target_column: str
    target_definition: str | None = None
    comparison_context_id: StableId | None = None

    @model_validator(mode="after")
    def establish_identity(self) -> "ComparisonContext":
        if not self.dataset_artifact_ids and self.dataset_id is None:
            raise ValueError("comparison context requires a dataset identity")
        if self.split_identity is None and self.split_specification is None:
            raise ValueError("comparison context requires a split identity or specification")
        expected = self._context_digest()
        if self.comparison_context_id is not None and self.comparison_context_id != expected:
            raise ValueError("comparison_context_id does not match context fields")
        object.__setattr__(self, "comparison_context_id", expected)
        return self

    def _context_digest(self) -> str:
        split = None
        if self.split_specification is not None:
            split = self.split_specification.model_dump(
                mode="json",
                exclude_none=True,
                exclude={"id", "schema_version", "created_at"},
            )
        payload = {
            "dataset_artifact_ids": self.dataset_artifact_ids,
            "dataset_id": self.dataset_id,
            "split_identity": self.split_identity,
            "split_specification": split,
            "metric_name": self.metric_name,
            "metric_direction": self.metric_direction,
            "target_column": self.target_column,
            "target_definition": self.target_definition,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        values = self.model_dump(exclude={"comparison_context_id"})
        if update:
            values.update(update)
        return type(self)(**values)

    @property
    def identity(self) -> StableId:
        return self.comparison_context_id  # type: ignore[return-value]
