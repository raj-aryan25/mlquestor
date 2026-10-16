"""Immutable source and derived artifact contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator

from .common import DomainModel, RecordModel, StableId, validate_non_empty


class ArtifactKind(StrEnum):
    DATASET = "dataset"
    DERIVED_DATASET = "derived_dataset"
    MODEL = "model"
    PREDICTIONS = "predictions"
    CODE = "code"
    CONFIGURATION = "configuration"
    REPORT = "report"
    LOG = "log"
    OTHER = "other"


class DatasetRole(StrEnum):
    TRAIN = "train"
    TEST = "test"
    HOLDOUT = "holdout"
    SAMPLE_SUBMISSION = "sample_submission"
    METADATA = "metadata"
    AUXILIARY = "auxiliary"


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]


class ArtifactManifest(RecordModel):
    kind: ArtifactKind
    content_hash: Sha256 = Field(validation_alias=AliasChoices("content_hash", "sha256"))
    media_type: str = Field(validation_alias=AliasChoices("media_type", "mime_type"))
    immutable_location: str = Field(
        validation_alias=AliasChoices("immutable_location", "location", "uri")
    )
    size_bytes: int | None = Field(default=None, ge=0)
    parent_artifact_ids: list[StableId] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("media_type", "immutable_location")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        return validate_non_empty(value, "artifact field")

    @property
    def artifact_id(self) -> StableId:
        return self.id


class ArtifactLineage(DomainModel):
    parent_artifact_ids: list[StableId] = Field(min_length=1)
    transformation: str
    fit_scope: str | None = None

    @field_validator("transformation")
    @classmethod
    def valid_transformation(cls, value: str) -> str:
        return validate_non_empty(value, "transformation")


class SourceArtifact(ArtifactManifest):
    kind: ArtifactKind = ArtifactKind.DATASET
    role: DatasetRole
    format: str
    original_location: str | None = None

    @field_validator("format")
    @classmethod
    def valid_format(cls, value: str) -> str:
        return validate_non_empty(value, "format")


class DatasetArtifact(SourceArtifact):
    row_count: int | None = Field(default=None, ge=0)
    column_names: list[str] = Field(default_factory=list)
    target_column: str | None = None


class DerivedArtifact(ArtifactManifest):
    kind: ArtifactKind = ArtifactKind.DERIVED_DATASET
    parent_artifact_ids: list[StableId] = Field(min_length=1)
    transformation: str
    fit_scope: str | None = None
    lineage: ArtifactLineage | None = None

    @field_validator("transformation")
    @classmethod
    def valid_transformation(cls, value: str) -> str:
        return validate_non_empty(value, "transformation")
