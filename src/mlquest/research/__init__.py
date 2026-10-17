"""Safe, replaceable research-provider integrations."""

from .base import (
    FetchRequest,
    FetchResponse,
    ResearchError,
    ResearchFailureCode,
    ResearchProvider,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from .cache import CachedResearch, ResearchCache
from .fastcrw import FastCRWProvider, FastCRWSettings, ResearchRetryPolicy
from .service import ResearchService

__all__ = [
    "CachedResearch",
    "FetchRequest",
    "FetchResponse",
    "FastCRWProvider",
    "FastCRWSettings",
    "ResearchCache",
    "ResearchError",
    "ResearchFailureCode",
    "ResearchProvider",
    "ResearchRetryPolicy",
    "ResearchService",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
]
