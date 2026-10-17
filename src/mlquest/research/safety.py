"""URL and bounded-content safety policies for research providers."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import SplitResult, urlsplit, urlunsplit

from .base import ResearchError, ResearchFailureCode

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_BLOCKED_HOSTS = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.google.com",
}


def canonical_url(url: str) -> str:
    parts = _split_url(url)
    hostname = parts.hostname
    assert hostname is not None
    port = parts.port
    netloc = hostname.lower()
    if ":" in netloc and not netloc.startswith("["):
        netloc = f"[{netloc}]"
    default_port = (parts.scheme == "https" and port == 443) or (
        parts.scheme == "http" and port == 80
    )
    if port is not None and not default_port:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))


def validate_fastcrw_base_url(url: str, *, allowed_hosts: tuple[str, ...] = ()) -> str:
    parts = _split_url(url)
    hostname = _hostname(parts)
    allowed = hostname in {host.lower() for host in allowed_hosts}
    if not allowed and not _is_local_host(hostname):
        raise ResearchError(
            ResearchFailureCode.CONFIGURATION,
            "fastCRW endpoints must be local or explicitly allowlisted",
        )
    if _is_private_literal(hostname) and not (_is_local_host(hostname) or allowed):
        raise ResearchError(ResearchFailureCode.SSRF_BLOCKED, "fastCRW host is not allowed")
    return canonical_url(url).rstrip("/")


def validate_source_url(
    url: str,
    *,
    allow_http: bool = False,
    allowed_hosts: tuple[str, ...] = (),
) -> str:
    parts = _split_url(url)
    hostname = _hostname(parts)
    allowlisted = hostname in {host.lower() for host in allowed_hosts}
    if parts.scheme != "https" and not (allow_http and parts.scheme == "http" and allowlisted):
        raise ResearchError(ResearchFailureCode.INVALID_URL, "research sources must use HTTPS")
    if _is_blocked_host(hostname) or (_is_private_literal(hostname) and not allowlisted):
        raise ResearchError(ResearchFailureCode.SSRF_BLOCKED, "research source host is blocked")
    return canonical_url(url)


def validate_redirect(
    source: str,
    target: str,
    *,
    allow_http: bool = False,
    allowed_hosts: tuple[str, ...] = (),
) -> str:
    source_parts = _split_url(source)
    target_parts = _split_url(target)
    if source_parts.scheme == "https" and target_parts.scheme == "http":
        raise ResearchError(
            ResearchFailureCode.SSRF_BLOCKED,
            "HTTPS redirects cannot downgrade to HTTP",
        )
    target_url = validate_source_url(target, allow_http=allow_http, allowed_hosts=allowed_hosts)
    if source_parts.scheme == "https" and urlsplit(target_url).scheme != "https":
        raise ResearchError(
            ResearchFailureCode.SSRF_BLOCKED,
            "HTTPS redirects cannot downgrade to HTTP",
        )
    return target_url


def _split_url(url: str) -> SplitResult:
    if not isinstance(url, str) or len(url) > 2048 or _CONTROL_CHARS.search(url):
        raise ResearchError(ResearchFailureCode.INVALID_URL, "research URL is malformed")
    try:
        parts = urlsplit(url)
        if parts.username is not None or parts.password is not None or parts.fragment:
            raise ResearchError(
                ResearchFailureCode.INVALID_URL,
                "research URL contains unsafe components",
            )
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            raise ResearchError(
                ResearchFailureCode.INVALID_URL,
                "research URL must have an HTTP scheme and host",
            )
        _ = parts.port
    except ValueError as exc:
        raise ResearchError(ResearchFailureCode.INVALID_URL, "research URL is malformed") from exc
    return parts


def _hostname(parts: SplitResult) -> str:
    hostname = parts.hostname
    if hostname is None:
        raise ResearchError(ResearchFailureCode.INVALID_URL, "research URL has no host")
    return hostname.rstrip(".").lower()


def _is_local_host(hostname: str) -> bool:
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _is_blocked_host(hostname: str) -> bool:
    return hostname in _BLOCKED_HOSTS or hostname.endswith((".localhost", ".local", ".internal"))


def _is_private_literal(hostname: str) -> bool:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
        or address.is_multicast
    )
