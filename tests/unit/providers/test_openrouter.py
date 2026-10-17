from __future__ import annotations

import json

import httpx
import pytest

from mlquest.domain import (
    ProviderCallRecord,
    ProviderCallStatus,
    TaskInterpretation,
    TaskType,
)
from mlquest.persistence import LocalLedger
from mlquest.providers import (
    ChatMessage,
    OpenRouterProvider,
    OpenRouterSettings,
    ProviderError,
    ProviderFailureCode,
    RetryPolicy,
    StructuredRequest,
)


def _response_body() -> dict[str, object]:
    return {
        "id": "completion-1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "task_type": "classification",
                            "target_column": "target",
                            "primary_metric": "accuracy",
                            "metric_direction": "maximize",
                            "summary": "A bounded tabular classification task.",
                            "assumptions": [],
                            "ambiguities": [],
                        }
                    ),
                }
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }


def _request() -> StructuredRequest[TaskInterpretation]:
    return StructuredRequest(
        operation="interpret_task",
        messages=(
            ChatMessage("system", "Return JSON only."),
            ChatMessage("user", "Classify target."),
        ),
        response_model=TaskInterpretation,
    )


def test_success_validates_response_and_persists_safe_metadata(tmp_path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_response_body(), request=request)

    with LocalLedger(tmp_path / "ledger.sqlite") as ledger:
        provider = OpenRouterProvider(
            OpenRouterSettings(model="test/model"),
            api_key="sk-test-secret-value",
            transport=httpx.MockTransport(handler),
            ledger=ledger,
            session_id="session-1",
        )
        result = provider.complete(_request())
        records = ledger.list_records(
            "provider_calls", session_id="session-1", model=ProviderCallRecord
        )

    assert result.value.task_type is TaskType.CLASSIFICATION
    assert result.attempt_count == 1
    assert len(records) == 1
    assert records[0].status is ProviderCallStatus.SUCCEEDED
    assert records[0].model == "test/model"
    assert records[0].usage is not None
    payload = json.loads(seen[0].content)
    assert "sk-test-secret-value" not in seen[0].content.decode()
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert seen[0].headers["authorization"] == "Bearer sk-test-secret-value"


def test_transient_failures_are_bounded_and_report_attempt_count() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "slow down"}, request=request)
        return httpx.Response(200, json=_response_body(), request=request)

    provider = OpenRouterProvider(
        OpenRouterSettings(model="test/model", retry=RetryPolicy(max_attempts=3)),
        api_key="test-key",
        transport=httpx.MockTransport(handler),
        sleeper=sleeps.append,
    )
    result = provider.complete(_request())

    assert calls == 2
    assert result.attempt_count == 2
    assert sleeps == [0.5]


def test_malformed_response_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": []}, request=request)

    provider = OpenRouterProvider(
        OpenRouterSettings(model="test/model", retry=RetryPolicy(max_attempts=3)),
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderError) as raised:
        provider.complete(_request())

    assert calls == 1
    assert raised.value.code is ProviderFailureCode.MALFORMED_RESPONSE


def test_from_env_reads_secret_without_putting_it_in_settings(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-secret")
    monkeypatch.setenv("MLQUEST_LLM_MODEL", "configured/model")
    provider = OpenRouterProvider.from_env(transport=httpx.MockTransport(lambda request: None))

    assert provider.settings.model == "configured/model"
    assert "environment-secret" not in repr(provider.settings)
