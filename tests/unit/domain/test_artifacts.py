import pytest
from pydantic import ValidationError

from mlquest.domain import ArtifactKind, DatasetArtifact, DatasetRole, DerivedArtifact

HASH = "a" * 64


def test_source_and_derived_artifacts_preserve_roles_and_lineage() -> None:
    source = DatasetArtifact(
        role=DatasetRole.TRAIN,
        content_hash=HASH,
        media_type="text/csv",
        immutable_location="artifacts/source/train.csv",
        format="csv",
    )
    derived = DerivedArtifact(
        content_hash="b" * 64,
        media_type="application/parquet",
        immutable_location="artifacts/derived/features.parquet",
        parent_artifact_ids=[source.id],
        transformation="one-hot encode categorical columns",
    )
    assert source.kind is ArtifactKind.DATASET
    assert derived.parent_artifact_ids == [source.id]


def test_artifact_hash_and_lineage_are_validated() -> None:
    with pytest.raises(ValidationError):
        DatasetArtifact(
            role=DatasetRole.TRAIN,
            content_hash="not-a-sha256",
            media_type="text/csv",
            immutable_location="artifacts/train.csv",
            format="csv",
        )
    with pytest.raises(ValidationError):
        DerivedArtifact(
            content_hash="a" * 64,
            media_type="application/parquet",
            immutable_location="artifacts/derived/features.parquet",
            transformation="clean",
            parent_artifact_ids=[],
        )
