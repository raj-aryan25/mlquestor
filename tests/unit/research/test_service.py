from __future__ import annotations

import httpx

from mlquest.artifacts import FileSystemArtifactStore
from mlquest.domain import ResearchRecord, SourceType
from mlquest.persistence import LocalLedger
from mlquest.research import FastCRWProvider, FetchRequest, ResearchService


def test_fetch_and_record_persists_provenance_and_immutable_cache(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "title": "Documentation",
                "final_url": "https://example.com/docs",
                "media_type": "text/plain",
                "content": "A bounded source excerpt.",
            },
            request=request,
        )

    with LocalLedger(tmp_path / "ledger.sqlite") as ledger:
        service = ResearchService(
            FastCRWProvider(transport=httpx.MockTransport(handler)),
            ledger=ledger,
            artifact_store=FileSystemArtifactStore(tmp_path / "artifacts"),
            session_id="session-1",
        )
        record = service.fetch_and_record(
            FetchRequest("https://example.com/docs"),
            source_type=SourceType.DOCUMENTATION,
            findings=["The source describes a bounded validation method."],
        )
        restored = ledger.get_record("research", record.id, ResearchRecord)
        events = ledger.list_events("session-1")

    assert restored == record
    assert record.cache_artifact_id is not None
    assert record.content_hash is not None
    assert record.normalized_content_hash is not None
    assert record.source_type is SourceType.DOCUMENTATION
    assert events[-1].event_type == "research.source.cached"
    assert "A bounded source excerpt." not in events[-1].payload
