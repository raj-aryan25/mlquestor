from __future__ import annotations

import pytest

from mlquest.research import ResearchError, ResearchFailureCode
from mlquest.research.safety import (
    canonical_url,
    validate_fastcrw_base_url,
    validate_redirect,
    validate_source_url,
)


def test_source_url_policy_rejects_credentials_private_hosts_and_downgrades() -> None:
    with pytest.raises(ResearchError) as credential_error:
        validate_source_url("https://user:password@example.com/page")
    with pytest.raises(ResearchError) as private_error:
        validate_source_url("https://127.0.0.1/admin")
    with pytest.raises(ResearchError) as downgrade_error:
        validate_redirect("https://example.com", "http://example.com/insecure")

    assert credential_error.value.code is ResearchFailureCode.INVALID_URL
    assert private_error.value.code is ResearchFailureCode.SSRF_BLOCKED
    assert downgrade_error.value.code is ResearchFailureCode.SSRF_BLOCKED


def test_base_endpoint_requires_local_or_explicit_allowlist() -> None:
    assert validate_fastcrw_base_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    with pytest.raises(ResearchError):
        validate_fastcrw_base_url("https://research.example")
    assert validate_fastcrw_base_url(
        "https://research.example", allowed_hosts=("research.example",)
    ) == "https://research.example"
    assert canonical_url("https://Example.com:443/a") == "https://example.com/a"
