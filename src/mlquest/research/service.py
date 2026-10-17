"""Application service for durable, provenance-backed research sources."""

from __future__ import annotations

from collections.abc import Iterable

from mlquest.artifacts import FileSystemArtifactStore
from mlquest.domain import ResearchRecord, SourceType
from mlquest.persistence import LocalLedger

from .base import FetchRequest, ResearchError, ResearchProvider, SearchRequest, SearchResponse
from .cache import ResearchCache


class ResearchService:
    """Keep provider results, cache artifacts, and ledger records consistent."""

    def __init__(
        self,
        provider: ResearchProvider,
        *,
        ledger: LocalLedger | None = None,
        artifact_store: FileSystemArtifactStore | None = None,
        session_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.ledger = ledger
        self.artifact_store = artifact_store
        self.session_id = session_id

    def search(self, request: SearchRequest) -> SearchResponse:
        try:
            response = self.provider.search(request)
        except ResearchError as exc:
            self._record_failure("research.query.failed", exc)
            raise
        if self.ledger is not None:
            self.ledger.append_event(
                self.session_id or "unscoped",
                "research.query.completed",
                {
                    "query_digest": response.query_digest,
                    "hit_count": len(response.hits),
                    "partial": response.partial,
                    "provider": response.provider,
                },
            )
        return response

    def fetch_and_record(
        self,
        request: FetchRequest,
        *,
        source_type: SourceType = SourceType.OTHER,
        findings: Iterable[str] = (),
        hypothesis_ids: Iterable[str] = (),
    ) -> ResearchRecord:
        try:
            response = self.provider.fetch(request)
        except ResearchError as exc:
            self._record_failure("research.source.failed", exc)
            raise
        cached = ResearchCache(self.artifact_store).put(response) if self.artifact_store else None
        record = ResearchRecord(
            title=response.title,
            url=response.final_url,
            requested_url=response.requested_url,
            final_url=response.final_url,
            source_type=source_type,
            accessed_at=response.retrieved_at,
            provider=getattr(self.provider, "provider_name", "research-provider"),
            media_type=response.media_type,
            size_bytes=response.raw_size_bytes,
            excerpt=response.content[:2_000] or None,
            content_hash=response.raw_content_hash,
            normalized_content_hash=response.normalized_content_hash,
            cache_artifact_id=cached.artifact.artifact_id if cached else None,
            untrusted_content=response.untrusted_content,
            safety_flags=list(response.safety_flags),
            findings=list(findings),
            hypothesis_ids=list(hypothesis_ids),
        )
        if self.ledger is not None:
            self.ledger.put_record("research", record, session_id=self.session_id)
            self.ledger.append_event(
                self.session_id or "unscoped",
                "research.source.cached" if cached else "research.source.recorded",
                {
                    "record_id": record.id,
                    "url": record.url,
                    "content_hash": record.content_hash,
                    "cache_artifact_id": record.cache_artifact_id,
                    "untrusted_content": record.untrusted_content,
                    "safety_flags": record.safety_flags,
                },
                entity_type="research",
                entity_id=record.id,
            )
        return record

    def _record_failure(self, event_type: str, error: ResearchError) -> None:
        if self.ledger is not None:
            self.ledger.append_event(
                self.session_id or "unscoped",
                event_type,
                {
                    "code": error.code.value,
                    "retryable": error.retryable,
                    "attempts": error.attempts,
                    "status_code": error.status_code,
                },
            )
