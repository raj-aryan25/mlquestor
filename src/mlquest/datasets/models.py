"""Deterministic contracts produced by dataset ingestion and profiling."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field

from mlquest.domain import DatasetRole
from mlquest.domain.common import DomainModel


class WarningSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class SamplingMetadata(DomainModel):
    """How bounded profiling observations were collected."""

    method: Literal["head"] = "head"
    sample_limit: int = Field(ge=1)
    sampled_rows: int = Field(ge=0)
    scanned_rows: int = Field(ge=0)
    scan_batches: int = Field(ge=0)
    chunk_size: int = Field(ge=1)
    sampling_applied: bool
    metrics_based_on_sample: bool = True
    row_count_from_metadata: bool = False


class ColumnProfile(DomainModel):
    name: str
    dtype: str
    nullable: bool
    missing_count: int = Field(ge=0)
    missing_fraction: float = Field(ge=0, le=1)
    cardinality: int = Field(ge=0)
    cardinality_fraction: float = Field(ge=0, le=1)
    is_constant: bool


class TargetDistribution(DomainModel):
    column: str
    counts: dict[str, int]
    proportions: dict[str, float]
    observed_rows: int = Field(ge=0)


class ColumnSuggestions(DomainModel):
    target_columns: list[str] = Field(default_factory=list)
    id_columns: list[str] = Field(default_factory=list)
    group_columns: list[str] = Field(default_factory=list)
    time_columns: list[str] = Field(default_factory=list)


class ProfileWarning(DomainModel):
    code: str
    severity: WarningSeverity
    message: str
    columns: list[str] = Field(default_factory=list)


class DatasetProfile(DomainModel):
    """Machine-readable summary; it intentionally contains no raw rows."""

    source_artifact_id: str
    content_hash: str
    role: DatasetRole
    format: Literal["csv", "parquet"]
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[ColumnProfile] = Field(default_factory=list)
    duplicate_row_count: int = Field(ge=0)
    duplicate_fraction: float = Field(ge=0, le=1)
    constant_columns: list[str] = Field(default_factory=list)
    target_column: str | None = None
    target_distribution: TargetDistribution | None = None
    suggestions: ColumnSuggestions = Field(default_factory=ColumnSuggestions)
    warnings: list[ProfileWarning] = Field(default_factory=list)
    sampling: SamplingMetadata


class SchemaColumnComparison(DomainModel):
    train_dtype: str
    test_dtype: str
    train_missing_fraction: float = Field(ge=0, le=1)
    test_missing_fraction: float = Field(ge=0, le=1)
    train_cardinality: int = Field(ge=0)
    test_cardinality: int = Field(ge=0)


class SchemaComparison(DomainModel):
    train_only_columns: list[str] = Field(default_factory=list)
    test_only_columns: list[str] = Field(default_factory=list)
    common_columns: list[str] = Field(default_factory=list)
    dtype_mismatches: list[str] = Field(default_factory=list)
    columns: dict[str, SchemaColumnComparison] = Field(default_factory=dict)
    duplicate_common_rows: int = Field(ge=0)
    warnings: list[ProfileWarning] = Field(default_factory=list)


class SourceRegistration(DomainModel):
    artifact_id: str
    content_hash: str
    role: DatasetRole
    format: Literal["csv", "parquet"]
    media_type: str
    immutable_location: str
    original_location: str
    size_bytes: int = Field(ge=0)


class DatasetAnalysis(DomainModel):
    train: DatasetProfile
    test: DatasetProfile | None = None
    schema_comparison: SchemaComparison | None = None
    report_markdown: str
