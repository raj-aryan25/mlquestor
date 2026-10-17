"""Research sources, hypotheses, and evidence claims."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator

from .common import EvidenceStrength, RecordModel, SourceType, StableId, validate_non_empty


class EvidenceClaimType(StrEnum):
    SOURCE = "source"
    MEASURED_RESULT = "measured_result"
    EXTERNAL_OBSERVATION = "external_observation"


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    TESTED = "tested"
    SUPPORTED = "supported"
    REJECTED = "rejected"


class ResearchRecord(RecordModel):
    title: str
    url: str
    requested_url: str | None = None
    final_url: str | None = None
    source_type: SourceType
    accessed_at: datetime
    provider: str | None = None
    media_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    excerpt: str | None = None
    content_hash: str | None = None
    normalized_content_hash: str | None = None
    cache_artifact_id: StableId | None = None
    untrusted_content: bool = True
    safety_flags: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    hypothesis_ids: list[StableId] = Field(default_factory=list)

    @field_validator("title", "url")
    @classmethod
    def required_text(cls, value: str) -> str:
        return validate_non_empty(value, "research field")


class Hypothesis(RecordModel):
    statement: str
    rationale: str
    source_record_ids: list[StableId] = Field(default_factory=list)
    expected_direction: str | None = None
    priority: int = Field(default=0, ge=0)
    status: HypothesisStatus = HypothesisStatus.PROPOSED

    @field_validator("statement", "rationale")
    @classmethod
    def required_text(cls, value: str) -> str:
        return validate_non_empty(value, "hypothesis field")


class EvidenceClaim(RecordModel):
    claim: str
    claim_type: EvidenceClaimType
    strength: EvidenceStrength
    evidence_record_ids: list[StableId] = Field(default_factory=list)
    comparison_context_id: StableId | None = None
    limitations: list[str] = Field(default_factory=list)

    @field_validator("claim")
    @classmethod
    def required_claim(cls, value: str) -> str:
        return validate_non_empty(value, "claim")
