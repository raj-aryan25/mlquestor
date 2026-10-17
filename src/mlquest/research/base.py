"""Provider-neutral contracts for bounded web research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from mlquest.domain import ResearchQuery, SourceType


class ResearchFailureCode(StrEnum):
    CONFIGURATION = "configuration"
    INVALID_URL = "invalid_url"
    SSRF_BLOCKED = "ssrf_blocked"
    TIMEOUT = "timeout"
    RESPONSE_TOO_LARGE = "response_too_large"
    RATE_LIMITED = "rate_limited"
    HTTP_ERROR = "http_error"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    MALFORMED_RESPONSE = "malformed_response"
    CONTENT_SAFETY = "content_safety"
    TRANSPORT_ERROR = "transport_error"


class ResearchError(RuntimeError):
    """A safe research failure without remote bodies or credentials."""

    def __init__(
        self,
        code: ResearchFailureCode,
        message: str,
        *,
        retryable: bool = False,
        attempts: int = 0,
        status_code: int | None = None,
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.attempts = attempts
        self.status_code = status_code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str | ResearchQuery
    max_results: int = 5

    def __post_init__(self) -> None:
        query = self.query.query if isinstance(self.query, ResearchQuery) else self.query
        if not query.strip():
            raise ValueError("research query must not be empty")
        if self.max_results < 1 or self.max_results > 20:
            raise ValueError("max_results must be between one and twenty")

    @property
    def query_text(self) -> str:
        return self.query.query if isinstance(self.query, ResearchQuery) else self.query


@dataclass(frozen=True, slots=True)
class SearchHit:
    title: str
    url: str
    snippet: str = ""
    rank: int = 0
    provider_result_id: str | None = None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    hits: tuple[SearchHit, ...]
    provider: str
    retrieved_at: datetime
    query_digest: str
    partial: bool = False


@dataclass(frozen=True, slots=True)
class FetchRequest:
    url: str

    def __post_init__(self) -> None:
        if not self.url.strip():
            raise ValueError("research URL must not be empty")


@dataclass(frozen=True, slots=True)
class FetchResponse:
    requested_url: str
    final_url: str
    title: str
    media_type: str
    status_code: int
    content: str
    retrieved_at: datetime
    raw_size_bytes: int
    raw_content_hash: str
    normalized_content_hash: str
    untrusted_content: bool = True
    safety_flags: tuple[str, ...] = ()
    cache_hit: bool = False
    source_type: SourceType = SourceType.OTHER


class ResearchProvider(Protocol):
    def search(self, request: SearchRequest) -> SearchResponse: ...

    def fetch(self, request: FetchRequest) -> FetchResponse: ...
