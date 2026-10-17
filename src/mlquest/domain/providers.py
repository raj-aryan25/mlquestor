"""Persisted metadata for advisory provider calls."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from .common import DomainModel, RecordModel, validate_non_empty


class ProviderCallStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ProviderUsage(DomainModel):
    """Token counts returned by a provider, when the provider supplies them."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ProviderCallRecord(RecordModel):
    """Safe audit metadata; request and response bodies are intentionally absent."""

    operation: str
    provider: str
    model: str
    status: ProviderCallStatus
    response_schema: str
    prompt_digest: str
    prompt_chars: int = Field(ge=0)
    response_digest: str | None = None
    response_chars: int | None = Field(default=None, ge=0)
    usage: ProviderUsage | None = None
    attempt_count: int = Field(ge=1)
    http_status: int | None = Field(default=None, ge=100, le=599)
    failure_code: str | None = None
    started_at: datetime
    finished_at: datetime
    fallback_index: int = Field(default=0, ge=0)

    @field_validator("operation", "provider", "model", "response_schema", "prompt_digest")
    @classmethod
    def required_text(cls, value: str) -> str:
        return validate_non_empty(value, "provider call field")

    @model_validator(mode="after")
    def valid_times(self) -> "ProviderCallRecord":
        if self.finished_at < self.started_at:
            raise ValueError("provider call finished_at cannot precede started_at")
        return self
