"""Bounded HTTP adapter for a separately deployed fastCRW service."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from html import unescape
from typing import Any, Callable
from urllib.parse import urljoin

import httpx

from .base import (
    FetchRequest,
    FetchResponse,
    ResearchError,
    ResearchFailureCode,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from .safety import canonical_url, validate_fastcrw_base_url, validate_source_url

_TRANSIENT_STATUS = {408, 425, 429}
_TEXT_MEDIA_TYPES = {
    "application/json",
    "application/xhtml+xml",
    "text/html",
    "text/plain",
    "text/markdown",
}
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(?P<tag>script|style)\b[^>]*>.*?</(?P=tag)>", re.I | re.S)
_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_INJECTION_PATTERNS = (
    (re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.I), "instruction_override"),
    (re.compile(r"\b(system|developer)\s+message\b", re.I), "role_impersonation"),
    (re.compile(r"\b(?:tool|function)\s+call\b", re.I), "tool_call_text"),
    (re.compile(r"\b(?:execute|run)\s+(?:a\s+)?(?:shell|command)\b", re.I), "execution_request"),
)


@dataclass(frozen=True, slots=True)
class ResearchRetryPolicy:
    max_attempts: int = 2
    initial_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        if self.initial_backoff_seconds < 0 or (
            self.max_backoff_seconds < self.initial_backoff_seconds
        ):
            raise ValueError("research backoff values are invalid")


@dataclass(frozen=True, slots=True)
class FastCRWSettings:
    base_url: str = "http://127.0.0.1:8080"
    search_path: str = "/search"
    fetch_path: str = "/fetch"
    api_key_env: str = "MLQUEST_FASTCRW_API_KEY"
    timeout_seconds: float = 15.0
    max_response_bytes: int = 2_000_000
    max_content_chars: int = 100_000
    max_results: int = 20
    max_redirects: int = 3
    rate_limit_seconds: float = 0.0
    allow_http_sources: bool = False
    allowed_base_hosts: tuple[str, ...] = ()
    allowed_source_hosts: tuple[str, ...] = ()
    retry: ResearchRetryPolicy = field(default_factory=ResearchRetryPolicy)

    def __post_init__(self) -> None:
        if not self.search_path.startswith("/") or not self.fetch_path.startswith("/"):
            raise ValueError("fastCRW paths must be absolute URL paths")
        if self.timeout_seconds <= 0 or self.max_response_bytes < 1 or self.max_content_chars < 1:
            raise ValueError("fastCRW bounds must be positive")
        if self.max_results < 1 or self.max_results > 100:
            raise ValueError("fastCRW max_results must be between one and one hundred")
        if self.max_redirects < 0 or self.rate_limit_seconds < 0:
            raise ValueError("fastCRW redirect and rate limits must be non-negative")


class FastCRWProvider:
    """Use fastCRW as an untrusted, bounded search and fetch backend."""

    provider_name = "fastcrw"

    def __init__(
        self,
        settings: FastCRWSettings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        api_key: str | None = None,
    ) -> None:
        self.settings = settings or FastCRWSettings()
        self.base_url = validate_fastcrw_base_url(
            self.settings.base_url, allowed_hosts=self.settings.allowed_base_hosts
        )
        self._transport = transport
        self._sleep = sleeper
        self._clock = clock
        self._last_request_at: float | None = None
        self._rate_lock = threading.Lock()
        self._api_key = api_key
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    @classmethod
    def from_env(
        cls,
        *,
        settings: FastCRWSettings | None = None,
        transport: httpx.BaseTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        environ: dict[str, str] | None = None,
    ) -> "FastCRWProvider":
        import os

        environment = os.environ if environ is None else environ
        configured = settings or FastCRWSettings(
            base_url=environment.get("MLQUEST_FASTCRW_URL", "http://127.0.0.1:8080")
        )
        return cls(
            configured,
            transport=transport,
            sleeper=sleeper,
            clock=clock,
            api_key=environment.get(configured.api_key_env),
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        max_results = min(request.max_results, self.settings.max_results)
        payload = {"query": request.query_text, "max_results": max_results}
        envelope, _ = self._post_json(self.settings.search_path, payload)
        hits, partial = self._decode_search(envelope, max_results)
        return SearchResponse(
            hits=tuple(hits),
            provider=self.provider_name,
            retrieved_at=_utc_now(),
            query_digest=sha256(request.query_text.strip().encode("utf-8")).hexdigest(),
            partial=partial,
        )

    def fetch(self, request: FetchRequest) -> FetchResponse:
        requested_url = validate_source_url(
            request.url,
            allow_http=self.settings.allow_http_sources,
            allowed_hosts=self.settings.allowed_source_hosts,
        )
        envelope, headers = self._post_json(self.settings.fetch_path, {"url": requested_url})
        return self._decode_fetch(envelope, headers, requested_url)

    def _post_json(
        self, path: str, payload: dict[str, object]
    ) -> tuple[dict[str, Any], httpx.Headers]:
        endpoint = urljoin(f"{self.base_url}/", path.lstrip("/"))
        last_error: ResearchError | None = None
        for attempt in range(1, self.settings.retry.max_attempts + 1):
            self._wait_for_rate_limit()
            response_headers: httpx.Headers | None = None
            current_endpoint = endpoint
            redirect_count = 0
            try:
                while True:
                    with httpx.Client(
                        timeout=self.settings.timeout_seconds,
                        transport=self._transport,
                        follow_redirects=False,
                    ) as client:
                        response = client.post(
                            current_endpoint, headers=self._headers, json=payload
                        )
                        response_headers = response.headers
                        body = self._read_bounded(response)
                    if response.status_code not in {301, 302, 303, 307, 308}:
                        break
                    location = response.headers.get("location")
                    if not location:
                        raise ResearchError(
                            ResearchFailureCode.MALFORMED_RESPONSE,
                            "fastCRW redirect did not include a location",
                            attempts=attempt,
                            status_code=response.status_code,
                        )
                    redirect_count += 1
                    if redirect_count > self.settings.max_redirects:
                        raise ResearchError(
                            ResearchFailureCode.HTTP_ERROR,
                            "fastCRW redirect limit exceeded",
                            attempts=attempt,
                            status_code=response.status_code,
                        )
                    next_endpoint = urljoin(current_endpoint, location)
                    if current_endpoint.startswith("https://") and next_endpoint.startswith("http://"):
                        raise ResearchError(
                            ResearchFailureCode.SSRF_BLOCKED,
                            "fastCRW redirect downgraded to HTTP",
                            attempts=attempt,
                        )
                    current_endpoint = validate_fastcrw_base_url(
                        next_endpoint, allowed_hosts=self.settings.allowed_base_hosts
                    )
            except ResearchError:
                raise
            except httpx.TimeoutException:
                last_error = ResearchError(
                    ResearchFailureCode.TIMEOUT,
                    "fastCRW request timed out",
                    retryable=True,
                    attempts=attempt,
                )
            except httpx.TransportError:
                last_error = ResearchError(
                    ResearchFailureCode.TRANSPORT_ERROR,
                    "fastCRW transport failed",
                    retryable=True,
                    attempts=attempt,
                )
            else:
                if response.status_code in {401, 403}:
                    raise ResearchError(
                        ResearchFailureCode.HTTP_ERROR,
                        "fastCRW authentication failed",
                        attempts=attempt,
                        status_code=response.status_code,
                    )
                if response.status_code < 200 or response.status_code >= 300:
                    retryable = response.status_code in _TRANSIENT_STATUS or (
                        response.status_code >= 500
                    )
                    last_error = ResearchError(
                        ResearchFailureCode.RATE_LIMITED
                        if response.status_code == 429
                        else ResearchFailureCode.HTTP_ERROR,
                        "fastCRW returned an HTTP error",
                        retryable=retryable,
                        attempts=attempt,
                        status_code=response.status_code,
                    )
                else:
                    try:
                        decoded = json.loads(body)
                    except (TypeError, ValueError) as exc:
                        raise ResearchError(
                            ResearchFailureCode.MALFORMED_RESPONSE,
                            "fastCRW returned invalid JSON",
                            attempts=attempt,
                            status_code=response.status_code,
                        ) from exc
                    if not isinstance(decoded, dict):
                        raise ResearchError(
                            ResearchFailureCode.MALFORMED_RESPONSE,
                            "fastCRW response must be an object",
                            attempts=attempt,
                            status_code=response.status_code,
                        )
                    return decoded, response.headers
            if last_error is not None and (
                not last_error.retryable or attempt == self.settings.retry.max_attempts
            ):
                raise last_error
            self._backoff(attempt, response_headers=response_headers)
        raise last_error or ResearchError(ResearchFailureCode.HTTP_ERROR, "fastCRW request failed")

    def _read_bounded(self, response: httpx.Response) -> bytes:
        declared = response.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.settings.max_response_bytes:
                    raise ResearchError(
                        ResearchFailureCode.RESPONSE_TOO_LARGE,
                        "fastCRW response exceeds the configured size limit",
                    )
            except ValueError as exc:
                raise ResearchError(
                    ResearchFailureCode.MALFORMED_RESPONSE,
                    "fastCRW response length is invalid",
                ) from exc
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > self.settings.max_response_bytes:
                raise ResearchError(
                    ResearchFailureCode.RESPONSE_TOO_LARGE,
                    "fastCRW response exceeds the configured size limit",
                )
            chunks.append(chunk)
        return b"".join(chunks)

    def _decode_search(
        self, envelope: dict[str, Any], max_results: int
    ) -> tuple[list[SearchHit], bool]:
        raw_hits = envelope.get("results", envelope.get("hits"))
        if not isinstance(raw_hits, list):
            raise ResearchError(
                ResearchFailureCode.MALFORMED_RESPONSE,
                "fastCRW search results are missing",
            )
        hits: list[SearchHit] = []
        for index, raw in enumerate(raw_hits[:max_results]):
            if not isinstance(raw, dict) or not isinstance(raw.get("url"), str):
                raise ResearchError(
                    ResearchFailureCode.MALFORMED_RESPONSE,
                    "fastCRW search hit is invalid",
                )
            url = validate_source_url(
                raw["url"],
                allow_http=self.settings.allow_http_sources,
                allowed_hosts=self.settings.allowed_source_hosts,
            )
            title = raw.get("title", "Untitled source")
            snippet = raw.get("snippet", raw.get("content", ""))
            if not isinstance(title, str) or not isinstance(snippet, str):
                raise ResearchError(
                    ResearchFailureCode.MALFORMED_RESPONSE,
                    "fastCRW search text is invalid",
                )
            hits.append(
                SearchHit(
                    title=title[:500],
                    url=url,
                    snippet=snippet[:2_000],
                    rank=index + 1,
                    provider_result_id=raw.get("id") if isinstance(raw.get("id"), str) else None,
                )
            )
        partial = bool(envelope.get("partial", False)) or len(raw_hits) > max_results
        return hits, partial

    def _decode_fetch(
        self, envelope: dict[str, Any], headers: httpx.Headers, requested_url: str
    ) -> FetchResponse:
        content = envelope.get("content", envelope.get("text"))
        if not isinstance(content, str):
            raise ResearchError(
                ResearchFailureCode.MALFORMED_RESPONSE,
                "fastCRW page content is missing",
            )
        media_type_value = envelope.get("media_type", headers.get("content-type", "text/plain"))
        if not isinstance(media_type_value, str):
            raise ResearchError(
                ResearchFailureCode.MALFORMED_RESPONSE,
                "fastCRW media type is invalid",
            )
        media_type = media_type_value.split(";", 1)[0].lower()
        if media_type not in _TEXT_MEDIA_TYPES and not media_type.startswith("text/"):
            raise ResearchError(
                ResearchFailureCode.UNSUPPORTED_MEDIA_TYPE,
                "fastCRW returned unsupported content",
            )
        final_url = validate_source_url(
            envelope.get("final_url", envelope.get("url", requested_url)),
            allow_http=self.settings.allow_http_sources,
            allowed_hosts=self.settings.allowed_source_hosts,
        )
        if canonical_url(requested_url).startswith("https://") and final_url.startswith("http://"):
            raise ResearchError(
                ResearchFailureCode.SSRF_BLOCKED,
                "fastCRW source downgraded to HTTP",
            )
        normalized = _normalize_text(content, media_type)
        if len(normalized) > self.settings.max_content_chars:
            normalized = normalized[: self.settings.max_content_chars]
        raw_bytes = content.encode("utf-8")
        flags = _safety_flags(normalized)
        title = envelope.get("title", final_url)
        if not isinstance(title, str) or not title.strip():
            title = final_url
        status_value = envelope.get("status_code", 200)
        if isinstance(status_value, bool) or not isinstance(status_value, int):
            raise ResearchError(
                ResearchFailureCode.MALFORMED_RESPONSE,
                "fastCRW source status is invalid",
            )
        return FetchResponse(
            requested_url=requested_url,
            final_url=final_url,
            title=title[:500],
            media_type=media_type,
            status_code=status_value,
            content=normalized,
            retrieved_at=_utc_now(),
            raw_size_bytes=len(raw_bytes),
            raw_content_hash=sha256(raw_bytes).hexdigest(),
            normalized_content_hash=sha256(normalized.encode("utf-8")).hexdigest(),
            safety_flags=tuple(flags),
        )

    def _wait_for_rate_limit(self) -> None:
        with self._rate_lock:
            now = self._clock()
            if self._last_request_at is not None:
                wait = self.settings.rate_limit_seconds - (now - self._last_request_at)
                if wait > 0:
                    self._sleep(wait)
            self._last_request_at = self._clock()

    def _backoff(self, attempt: int, response_headers: httpx.Headers | None) -> None:
        delay = min(
            self.settings.retry.max_backoff_seconds,
            self.settings.retry.initial_backoff_seconds * (2 ** (attempt - 1)),
        )
        if response_headers is not None:
            retry_after = response_headers.get("retry-after")
            try:
                delay = min(self.settings.retry.max_backoff_seconds, max(delay, float(retry_after)))
            except (TypeError, ValueError):
                pass
        if delay:
            self._sleep(delay)


def _normalize_text(content: str, media_type: str) -> str:
    normalized = content
    if media_type in {"text/html", "application/xhtml+xml"}:
        normalized = _SCRIPT_RE.sub(" ", normalized)
        normalized = _TAG_RE.sub(" ", normalized)
    normalized = unescape(normalized).replace("\x00", "")
    normalized = "\n".join(_SPACE_RE.sub(" ", line).strip() for line in normalized.splitlines())
    return "\n".join(line for line in normalized.splitlines() if line).strip()


def _safety_flags(content: str) -> list[str]:
    return [flag for pattern, flag in _INJECTION_PATTERNS if pattern.search(content)]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
