"""Shared types and invariants used by domain records."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

StableId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
SCHEMA_VERSION: Literal["1.0"] = "1.0"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MetricDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class ApprovalDecisionType(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class EvidenceStrength(StrEnum):
    EXPLORATORY = "exploratory"
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    INSUFFICIENT = "insufficient"


class VerificationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class SourceType(StrEnum):
    PAPER = "paper"
    DOCUMENTATION = "documentation"
    COMPETITION_WRITEUP = "competition_writeup"
    REPOSITORY = "repository"
    BLOG = "blog"
    OTHER = "other"


class DomainModel(BaseModel):
    """Base model with strict field handling and stable JSON serialization."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )

    @field_validator("*", mode="after")
    @classmethod
    def timezone_aware_timestamps(cls, value: object) -> object:
        if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamps must include timezone information")
        return value

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        )

    def content_digest(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()


class RecordModel(DomainModel):
    id: StableId = Field(default_factory=new_id)
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not _ID_PATTERN.fullmatch(value):
            raise ValueError("id must be a non-empty stable identifier")
        return value

    @field_validator("created_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include timezone information")
        return value.astimezone(timezone.utc)


def validate_identifier(value: str, field_name: str = "identifier") -> str:
    if not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a non-empty stable identifier")
    return value


def validate_non_empty(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value
