"""Provider-neutral contracts for structured advisory model calls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel

from mlquest.domain.providers import ProviderUsage

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("message content must not be empty")


@dataclass(frozen=True, slots=True)
class StructuredRequest(Generic[T]):
    operation: str
    messages: tuple[ChatMessage, ...]
    response_model: type[T]
    session_id: str | None = None

    def __post_init__(self) -> None:
        if not self.operation.strip():
            raise ValueError("operation must not be empty")
        if not self.messages:
            raise ValueError("at least one message is required")


@dataclass(frozen=True, slots=True)
class ProviderResult(Generic[T]):
    value: T
    provider: str
    model: str
    usage: ProviderUsage | None
    attempt_count: int = 1
    request_id: str | None = None
    fallback_index: int = 0

    def __post_init__(self) -> None:
        if self.attempt_count < 1:
            raise ValueError("attempt_count must be at least one")
        if self.fallback_index < 0:
            raise ValueError("fallback_index must be non-negative")


class ProviderFailureCode(StrEnum):
    AUTHENTICATION = "authentication"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    MALFORMED_RESPONSE = "malformed_response"
    SCHEMA_VALIDATION = "schema_validation"
    TRANSPORT_ERROR = "transport_error"
    CONFIGURATION = "configuration"


class ProviderError(RuntimeError):
    """A safe, typed provider failure with no response-body or header details."""

    def __init__(
        self,
        code: ProviderFailureCode,
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


class StructuredProvider(Protocol):
    def complete(self, request: StructuredRequest[T]) -> ProviderResult[T]: ...
