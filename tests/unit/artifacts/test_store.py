from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mlquest.artifacts import (
    ArtifactIntegrityError,
    ArtifactPathError,
    FileSystemArtifactStore,
    sha256_file,
)
from mlquest.domain import DatasetRole
from mlquest.persistence import LocalLedger


def test_hashing_matches_sha256(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    source.write_bytes(b"id,value\n1,42\n")

    assert sha256_file(source) == hashlib.sha256(source.read_bytes()).hexdigest()


def test_duplicate_content_resolves_to_one_identity(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(b"same bytes")
    second.write_bytes(first.read_bytes())
    store = FileSystemArtifactStore(tmp_path / "artifacts")

    first_stored = store.put_source(first, role=DatasetRole.TRAIN, media_type="text/csv")
    second_stored = store.put_source(second, role=DatasetRole.TRAIN, media_type="text/csv")

    assert first_stored.artifact_id == second_stored.artifact_id
    assert first_stored.content_path == second_stored.content_path
    assert len(list((tmp_path / "artifacts" / "objects").rglob("*"))) == 2


def test_source_is_copied_and_remains_immutable(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    original = b"feature,target\n1,0\n"
    source.write_bytes(original)
    store = FileSystemArtifactStore(tmp_path / "artifacts")

    stored = store.put_source(source, role=DatasetRole.TRAIN, media_type="text/csv")
    source.write_bytes(b"changed outside the store")

    assert store.read_bytes(stored.artifact_id) == original
    assert stored.manifest.role is DatasetRole.TRAIN  # type: ignore[union-attr]


def test_derived_manifest_records_lineage_deterministically(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    source.write_bytes(b"raw")
    store = FileSystemArtifactStore(tmp_path / "artifacts")
    parent = store.put_source(source, role=DatasetRole.TRAIN, media_type="text/csv")

    derived = store.put_derived(
        b"features",
        parent_artifact_ids=[parent.artifact_id],
        transformation="one-hot encode category",
        fit_scope="training-fold-0",
        media_type="application/parquet",
    )

    manifest = store.get_manifest(derived.artifact_id)
    assert manifest.parent_artifact_ids == [parent.artifact_id]
    assert manifest.transformation == "one-hot encode category"  # type: ignore[union-attr]
    assert manifest.fit_scope == "training-fold-0"  # type: ignore[union-attr]
    assert store.manifest_path(derived.artifact_id).read_text(encoding="utf-8") == store.manifest_json(
        derived
    )


def test_safe_readback_rejects_path_traversal(tmp_path: Path) -> None:
    store = FileSystemArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ArtifactPathError):
        store.manifest_path("../outside")
    with pytest.raises(ArtifactPathError):
        store.resolve_location("objects/../../outside")


def test_existing_object_cannot_be_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable")
    store = FileSystemArtifactStore(tmp_path / "artifacts")
    stored = store.put_source(source)
    stored.content_path.write_bytes(b"tampered")

    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"immutable")
    with pytest.raises(ArtifactIntegrityError):
        store.put_source(replacement)


def test_registration_metadata_has_a_ledger_record_id(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"value\n1\n")
    stored = FileSystemArtifactStore(tmp_path / "artifacts").put_source(source)

    metadata = stored.registration_metadata()

    assert metadata["id"] == stored.artifact_id
    assert metadata["artifact_id"] == stored.artifact_id


def test_registration_metadata_can_be_saved_in_the_ledger(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_bytes(b"value\n1\n")
    stored = FileSystemArtifactStore(tmp_path / "artifacts").put_source(source)

    with LocalLedger(tmp_path / "ledger.sqlite3") as ledger:
        ledger.put_record("artifacts", stored.registration_metadata(), session_id="session-1")
        record = ledger.get_record("artifacts", stored.artifact_id)

    assert record["content_hash"] == stored.content_hash
