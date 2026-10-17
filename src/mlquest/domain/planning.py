"""Structured, advisory responses returned by an LLM planner."""

from __future__ import annotations

from math import isfinite

from pydantic import Field, field_validator

from .common import DomainModel, MetricDirection, StableId, validate_non_empty
from .task import TaskType
from .validation import SplitSpecification


class TaskInterpretation(DomainModel):
    """A provider's interpretation of the user-supplied task description."""

    task_type: TaskType
    target_column: str
    primary_metric: str
    metric_direction: MetricDirection
    summary: str
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    ambiguities: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("target_column", "primary_metric", "summary")
    @classmethod
    def required_text(cls, value: str) -> str:
        return validate_non_empty(value, "task interpretation field")


class ValidationPlanProposal(DomainModel):
    """A proposed validation plan; it is not an authorization to execute it."""

    split: SplitSpecification
    rationale: str
    risks: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("rationale")
    @classmethod
    def valid_rationale(cls, value: str) -> str:
        return validate_non_empty(value, "validation rationale")


class ResearchQuery(DomainModel):
    query: str
    rationale: str
    max_results: int = Field(default=5, ge=1, le=20)

    @field_validator("query", "rationale")
    @classmethod
    def valid_text(cls, value: str) -> str:
        return validate_non_empty(value, "research query field")


class HypothesisProposal(DomainModel):
    statement: str
    rationale: str
    expected_direction: str | None = Field(default=None, max_length=200)
    priority: int = Field(default=0, ge=0, le=100)

    @field_validator("statement", "rationale")
    @classmethod
    def valid_text(cls, value: str) -> str:
        return validate_non_empty(value, "hypothesis field")


class ExperimentSpecificationProposal(DomainModel):
    """A declarative experiment suggestion with no execution controls."""

    model_family: str
    parameters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    random_seeds: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=20)
    rationale: str
    expected_effect: str | None = Field(default=None, max_length=200)

    @field_validator("model_family", "rationale")
    @classmethod
    def valid_text(cls, value: str) -> str:
        return validate_non_empty(value, "experiment proposal field")

    @field_validator("random_seeds")
    @classmethod
    def valid_seeds(cls, values: list[int]) -> list[int]:
        if any(seed < 0 for seed in values):
            raise ValueError("random seeds must be non-negative")
        return values

    @field_validator("parameters")
    @classmethod
    def safe_parameters(
        cls, values: dict[str, str | int | float | bool | None]
    ) -> dict[str, str | int | float | bool | None]:
        forbidden = {
            "command",
            "shell",
            "docker",
            "image",
            "mount",
            "mounts",
            "environment",
            "env",
            "cwd",
            "path",
            "run_id",
        }
        if any(key.lower() in forbidden for key in values):
            raise ValueError("experiment parameters cannot contain execution controls")
        if any(isinstance(value, float) and not isfinite(value) for value in values.values()):
            raise ValueError("experiment parameters must contain finite numbers")
        return values


class ExperimentConclusion(DomainModel):
    summary: str
    supports_hypothesis: bool | None = None
    recommendation: str
    limitations: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("summary", "recommendation")
    @classmethod
    def valid_text(cls, value: str) -> str:
        return validate_non_empty(value, "experiment conclusion field")


class ReportDraftSection(DomainModel):
    section: str
    body: str
    evidence_record_ids: list[StableId] = Field(default_factory=list, max_length=50)

    @field_validator("section", "body")
    @classmethod
    def valid_text(cls, value: str) -> str:
        return validate_non_empty(value, "report section field")
