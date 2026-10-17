"""Prompt redaction and deterministic size limits for online providers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .base import ChatMessage

_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|secret|"
    r"credential|private[_-]?key)",
    re.I,
)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.I)
_OPENAI_KEY = re.compile(r"\b(?:sk|rk)-[A-Za-z0-9_-]{12,}\b")
_URL_CREDENTIALS = re.compile(r"(https?://)([^/@\s]+):([^/@\s]+)@", re.I)


def redact_text(value: str, *, secrets: Sequence[str] = ()) -> str:
    """Remove common credentials while preserving ordinary task wording."""

    result = value
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        result = result.replace(secret, "[REDACTED]")
    result = _BEARER.sub("Bearer [REDACTED]", result)
    result = _OPENAI_KEY.sub("[REDACTED]", result)
    return _URL_CREDENTIALS.sub(r"\1[REDACTED]@", result)


def redact_mapping(value: Mapping[str, Any], *, secrets: Sequence[str] = ()) -> dict[str, Any]:
    """Recursively redact secret-like mapping keys and text values."""

    result: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if _SECRET_KEY.search(key_text):
            result[key_text] = "[REDACTED]"
        elif isinstance(item, Mapping):
            result[key_text] = redact_mapping(item, secrets=secrets)
        elif isinstance(item, list):
            result[key_text] = [
                redact_mapping(entry, secrets=secrets)
                if isinstance(entry, Mapping)
                else redact_text(entry, secrets=secrets)
                if isinstance(entry, str)
                else entry
                for entry in item
            ]
        elif isinstance(item, str):
            result[key_text] = redact_text(item, secrets=secrets)
        else:
            result[key_text] = item
    return result


def safe_messages(
    messages: Sequence[ChatMessage],
    *,
    secrets: Sequence[str] = (),
    max_message_chars: int = 20_000,
    max_total_chars: int = 60_000,
) -> tuple[ChatMessage, ...]:
    """Redact and bound messages before they cross the provider boundary."""

    if max_message_chars < 1 or max_total_chars < 1:
        raise ValueError("prompt limits must be positive")
    sanitized: list[ChatMessage] = []
    total = 0
    for message in messages:
        content = redact_text(message.content, secrets=secrets)
        if len(content) > max_message_chars:
            raise ValueError("provider message exceeds the configured size limit")
        total += len(content)
        if total > max_total_chars:
            raise ValueError("provider prompt exceeds the configured size limit")
        sanitized.append(ChatMessage(role=message.role, content=content))
    return tuple(sanitized)


def prompt_digest(messages: Sequence[ChatMessage]) -> tuple[str, int]:
    """Return a digest and character count without retaining prompt bodies."""

    from hashlib import sha256

    payload = json.dumps(
        [{"role": item.role, "content": item.content} for item in messages],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest(), sum(len(item.content) for item in messages)
