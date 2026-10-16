"""Execution resource and budget contracts."""

from __future__ import annotations

from pydantic import AliasChoices, Field, model_validator

from .common import RecordModel


class ResourceProfile(RecordModel):
    cpu_count: int = Field(gt=0)
    memory_bytes: int = Field(gt=0)
    disk_bytes: int | None = Field(default=None, ge=0)
    gpu_count: int = Field(default=0, ge=0)
    gpu_model: str | None = None
    python_version: str | None = None
    docker_available: bool = False


class BudgetConfiguration(RecordModel):
    session_seconds: int = Field(
        gt=0,
        validation_alias=AliasChoices("session_seconds", "total_session_seconds", "total_wall_clock_seconds"),
    )
    model_execution_seconds: int = Field(
        gt=0,
        validation_alias=AliasChoices("model_execution_seconds", "model_budget_seconds"),
    )
    finalisation_reserve_seconds: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("finalisation_reserve_seconds", "finalization_reserve_seconds"),
    )
    max_experiments: int | None = Field(default=None, ge=1)
    max_memory_bytes: int | None = Field(default=None, gt=0)
    max_storage_bytes: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def compatible_windows(self) -> "BudgetConfiguration":
        if self.model_execution_seconds + self.finalisation_reserve_seconds > self.session_seconds:
            raise ValueError("model execution and finalisation reserve exceed session budget")
        return self
