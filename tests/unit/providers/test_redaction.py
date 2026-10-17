from __future__ import annotations

from mlquest.providers import ChatMessage
from mlquest.providers.redaction import prompt_digest, redact_mapping, redact_text, safe_messages


def test_redaction_removes_credentials_but_preserves_normal_text() -> None:
    text = "Use the target column and Bearer abc.def.ghi with https://user:pass@example.test."
    redacted = redact_text(text)

    assert "target column" in redacted
    assert "abc.def.ghi" not in redacted
    assert "user:pass" not in redacted


def test_nested_secret_mapping_and_prompt_limits() -> None:
    mapping = redact_mapping({"profile": {"api_key": "secret", "rows": ["1,2,3"]}})
    assert mapping["profile"]["api_key"] == "[REDACTED]"
    messages = safe_messages((ChatMessage("user", "secret text"),), secrets=("secret",))

    assert messages[0].content == "[REDACTED] text"
    assert prompt_digest(messages)[1] == len("[REDACTED] text")
