"""OpenRouter's OpenAI-compatible structured chat-completion adapter."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Callable, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from mlquest.domain.providers import ProviderCallRecord, ProviderCallStatus, ProviderUsage
from mlquest.persistence import LocalLedger

from .base import (
    ChatMessage,
    ProviderError,
    ProviderFailureCode,
    ProviderResult,
    StructuredRequest,
)
from .redaction import prompt_digest, safe_messages

T = TypeVar("T", bound=BaseModel)
_RETRYABLE_STATUSES = {408, 425, 429}


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.initial_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("backoff values must be non-negative")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must not be below initial backoff")


@dataclass(frozen=True, slots=True)
class OpenRouterSettings:
    model: str
    fallback_models: tuple[str, ...] = ()
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 30.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    max_message_chars: int = 20_000
    max_prompt_chars: int = 60_000

    def __post_init__(self) -> None:
        if not self.model.strip() or any(not model.strip() for model in self.fallback_models):
            raise ValueError("model names must not be empty")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between zero and two")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_message_chars < 1 or self.max_prompt_chars < 1:
            raise ValueError("prompt limits must be positive")


class OpenRouterProvider:
    """Call OpenRouter for validated proposals without exposing execution controls."""

    provider_name = "openrouter"

    def __init__(
        self,
        settings: OpenRouterSettings,
        *,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        ledger: LocalLedger | None = None,
        session_id: str | None = None,
    ) -> None:
        configured_key = os.environ.get("OPENROUTER_API_KEY", "") if api_key is None else api_key
        if not configured_key.strip():
            raise ValueError("OpenRouter API key must not be empty")
        self.settings = settings
        self._api_key = configured_key
        self._transport = transport
        self._sleep = sleeper
        self._ledger = ledger
        self._session_id = session_id

    @classmethod
    def from_env(
        cls,
        *,
        settings: OpenRouterSettings | None = None,
        transport: httpx.BaseTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        ledger: LocalLedger | None = None,
        session_id: str | None = None,
        environ: dict[str, str] | None = None,
    ) -> "OpenRouterProvider":
        environment = os.environ if environ is None else environ
        api_key = environment.get("OPENROUTER_API_KEY", "")
        configured = settings or OpenRouterSettings(
            model=environment.get("MLQUEST_LLM_MODEL", "")
        )
        if not api_key.strip():
            raise ValueError("OPENROUTER_API_KEY is required")
        return cls(
            configured,
            api_key=api_key,
            transport=transport,
            sleeper=sleeper,
            ledger=ledger,
            session_id=session_id,
        )

    def complete(self, request: StructuredRequest[T]) -> ProviderResult[T]:
        try:
            messages = safe_messages(
                request.messages,
                secrets=(self._api_key,),
                max_message_chars=self.settings.max_message_chars,
                max_total_chars=self.settings.max_prompt_chars,
            )
        except ValueError as exc:
            raise ProviderError(
                ProviderFailureCode.CONFIGURATION,
                "provider prompt is outside the configured safety limits",
            ) from exc

        digest, prompt_chars = prompt_digest(messages)
        started_at = _utc_now()
        session_id = self._session_id or request.session_id
        models = (self.settings.model, *self.settings.fallback_models)
        last_error: ProviderError | None = None
        for fallback_index, model in enumerate(models):
            try:
                result = self._complete_model(
                    request,
                    messages,
                    model=model,
                    fallback_index=fallback_index,
                )
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or fallback_index == len(models) - 1:
                    self._persist_failure(
                        request,
                        model=model,
                        fallback_index=fallback_index,
                        prompt_digest_value=digest,
                        prompt_chars=prompt_chars,
                        started_at=started_at,
                        error=exc,
                        session_id=session_id,
                    )
                    raise
                continue
            self._persist_success(
                request,
                model=model,
                fallback_index=fallback_index,
                prompt_digest_value=digest,
                prompt_chars=prompt_chars,
                started_at=started_at,
                result=result,
                session_id=session_id,
            )
            return result
        assert last_error is not None
        raise last_error

    def _complete_model(
        self,
        request: StructuredRequest[T],
        messages: tuple[ChatMessage, ...],
        *,
        model: str,
        fallback_index: int,
    ) -> ProviderResult[T]:
        payload = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": _schema_name(request.response_model),
                    "strict": True,
                    "schema": request.response_model.model_json_schema(),
                },
            },
        }
        attempts = 0
        last_status: int | None = None
        for attempt in range(1, self.settings.retry.max_attempts + 1):
            attempts = attempt
            try:
                with httpx.Client(
                    base_url=self.settings.base_url.rstrip("/"),
                    timeout=self.settings.timeout_seconds,
                    transport=self._transport,
                ) as client:
                    response = client.post(
                        "/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
            except httpx.TimeoutException as exc:
                error = ProviderError(
                    ProviderFailureCode.TIMEOUT,
                    "OpenRouter request timed out",
                    retryable=True,
                    attempts=attempts,
                )
                if attempt == self.settings.retry.max_attempts:
                    raise error from exc
                self._backoff(attempt)
                continue
            except httpx.TransportError as exc:
                error = ProviderError(
                    ProviderFailureCode.TRANSPORT_ERROR,
                    "OpenRouter transport failed",
                    retryable=True,
                    attempts=attempts,
                )
                if attempt == self.settings.retry.max_attempts:
                    raise error from exc
                self._backoff(attempt)
                continue

            last_status = response.status_code
            if response.status_code < 200 or response.status_code >= 300:
                error = _http_error(response.status_code)
                if error.retryable and attempt < self.settings.retry.max_attempts:
                    self._backoff(attempt)
                    continue
                error.attempts = attempts
                raise error
            try:
                envelope = response.json()
                content = _extract_content(envelope)
                decoded = json.loads(content)
                value = request.response_model.model_validate(decoded)
            except (ValueError, TypeError, json.JSONDecodeError, ValidationError) as exc:
                code = (
                    ProviderFailureCode.SCHEMA_VALIDATION
                    if isinstance(exc, ValidationError)
                    else ProviderFailureCode.MALFORMED_RESPONSE
                )
                raise ProviderError(
                    code,
                    "OpenRouter returned an invalid structured response",
                    attempts=attempts,
                    status_code=response.status_code,
                ) from exc
            usage = _usage(response)
            return ProviderResult(
                value=value,
                provider=self.provider_name,
                model=model,
                usage=usage,
                attempt_count=attempts,
                request_id=_request_id(envelope),
                fallback_index=fallback_index,
            )
        raise ProviderError(
            ProviderFailureCode.HTTP_ERROR,
            "OpenRouter request failed",
            retryable=last_status in _RETRYABLE_STATUSES
            or (last_status is not None and last_status >= 500),
            attempts=attempts,
            status_code=last_status,
        )

    def _backoff(self, attempt: int) -> None:
        delay = min(
            self.settings.retry.max_backoff_seconds,
            self.settings.retry.initial_backoff_seconds * (2 ** (attempt - 1)),
        )
        if delay:
            self._sleep(delay)

    def _persist_success(
        self,
        request: StructuredRequest[T],
        *,
        model: str,
        fallback_index: int,
        prompt_digest_value: str,
        prompt_chars: int,
        started_at: datetime,
        result: ProviderResult[T],
        session_id: str | None,
    ) -> None:
        self._persist(
            ProviderCallRecord(
                operation=request.operation,
                provider=self.provider_name,
                model=model,
                status=ProviderCallStatus.SUCCEEDED,
                response_schema=_schema_name(request.response_model),
                prompt_digest=prompt_digest_value,
                prompt_chars=prompt_chars,
                response_digest=sha256(
                    result.value.model_dump_json().encode("utf-8")
                ).hexdigest(),
                response_chars=len(result.value.model_dump_json()),
                usage=result.usage,
                attempt_count=result.attempt_count,
                started_at=started_at,
                finished_at=_utc_now(),
                fallback_index=fallback_index,
            ),
            session_id=session_id,
        )

    def _persist_failure(
        self,
        request: StructuredRequest[T],
        *,
        model: str,
        fallback_index: int,
        prompt_digest_value: str,
        prompt_chars: int,
        started_at: datetime,
        error: ProviderError,
        session_id: str | None,
    ) -> None:
        self._persist(
            ProviderCallRecord(
                operation=request.operation,
                provider=self.provider_name,
                model=model,
                status=ProviderCallStatus.FAILED,
                response_schema=_schema_name(request.response_model),
                prompt_digest=prompt_digest_value,
                prompt_chars=prompt_chars,
                usage=None,
                attempt_count=max(1, error.attempts),
                http_status=error.status_code,
                failure_code=error.code.value,
                started_at=started_at,
                finished_at=_utc_now(),
                fallback_index=fallback_index,
            ),
            session_id=session_id,
        )

    def _persist(self, record: ProviderCallRecord, *, session_id: str | None) -> None:
        if self._ledger is not None:
            self._ledger.put_record("provider_calls", record, session_id=session_id)


def _schema_name(model: type[BaseModel]) -> str:
    return f"{model.__module__}.{model.__qualname__}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _http_error(status_code: int) -> ProviderError:
    if status_code in {401, 403}:
        return ProviderError(
            ProviderFailureCode.AUTHENTICATION,
            "OpenRouter authentication failed",
            status_code=status_code,
        )
    if status_code == 429:
        return ProviderError(
            ProviderFailureCode.RATE_LIMITED,
            "OpenRouter rate limit reached",
            retryable=True,
            status_code=status_code,
        )
    retryable = status_code in _RETRYABLE_STATUSES or status_code >= 500
    return ProviderError(
        ProviderFailureCode.HTTP_ERROR,
        "OpenRouter returned an HTTP error",
        retryable=retryable,
        status_code=status_code,
    )


def _extract_content(envelope: Any) -> str:
    if not isinstance(envelope, dict):
        raise ValueError("response envelope must be an object")
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response choices are missing")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict) or message.get("tool_calls"):
        raise ValueError("response contains no direct message content")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        if text_parts and len(text_parts) == len(content):
            return "".join(text_parts)
    raise ValueError("response content is missing")


def _usage(response: httpx.Response) -> ProviderUsage | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    value = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(value, dict):
        return None
    try:
        return ProviderUsage(
            prompt_tokens=value.get("prompt_tokens"),
            completion_tokens=value.get("completion_tokens"),
            total_tokens=value.get("total_tokens"),
        )
    except (TypeError, ValueError, ValidationError):
        return None


def _request_id(envelope: Any) -> str | None:
    value = envelope.get("id") if isinstance(envelope, dict) else None
    return value if isinstance(value, str) else None
