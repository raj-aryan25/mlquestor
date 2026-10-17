"""Immutable snapshots for fetched, untrusted research content."""

from __future__ import annotations

import json
from dataclasses import dataclass

from mlquest.artifacts import FileSystemArtifactStore, StoredArtifact

from .base import FetchResponse


@dataclass(frozen=True, slots=True)
class CachedResearch:
    response: FetchResponse
    artifact: StoredArtifact


class ResearchCache:
    """Store a bounded canonical envelope in the content-addressed artifact store."""

    def __init__(self, store: FileSystemArtifactStore) -> None:
        self.store = store

    def put(self, response: FetchResponse) -> CachedResearch:
        envelope = {
            "requested_url": response.requested_url,
            "final_url": response.final_url,
            "title": response.title,
            "media_type": response.media_type,
            "status_code": response.status_code,
            "retrieved_at": response.retrieved_at.isoformat().replace("+00:00", "Z"),
            "raw_size_bytes": response.raw_size_bytes,
            "raw_content_hash": response.raw_content_hash,
            "normalized_content_hash": response.normalized_content_hash,
            "untrusted_content": response.untrusted_content,
            "safety_flags": list(response.safety_flags),
            "content": response.content,
        }
        content = json.dumps(
            envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        artifact = self.store.put_content(
            content,
            media_type="application/json",
            metadata={
                "provider": "fastcrw",
                "source_url": response.final_url,
                "raw_content_hash": response.raw_content_hash,
                "normalized_content_hash": response.normalized_content_hash,
                "untrusted_content": "true",
            },
        )
        return CachedResearch(response=response, artifact=artifact)
