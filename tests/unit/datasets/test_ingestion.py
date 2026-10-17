from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlquest.artifacts import FileSystemArtifactStore
from mlquest.datasets import DatasetProfiler, DatasetService
from mlquest.domain import DatasetRole
from mlquest.persistence import LocalLedger


def _service(
    tmp_path: Path, *, sample_limit: int = 10, chunk_size: int = 4
) -> tuple[DatasetService, FileSystemArtifactStore, LocalLedger]:
    store = FileSystemArtifactStore(tmp_path / "artifacts")
    ledger = LocalLedger(tmp_path / "ledger.sqlite3")
    service = DatasetService(
        store,
        ledger=ledger,
        profiler=DatasetProfiler(sample_limit=sample_limit, chunk_size=chunk_size),
    )
    return service, store, ledger


def test_csv_registration_profiles_without_mutating_source_and_records_ledger(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.csv"
    original = b"id,feature,target\n1,10,yes\n2,,no\n1,10,yes\n"
    source.write_bytes(original)
    service, store, ledger = _service(tmp_path)

    registration = service.register_source(source, role=DatasetRole.TRAIN, session_id="session-1")
    profile = service.profile_source(
        store.artifact_path(registration.artifact_id),
        role=DatasetRole.TRAIN,
        target_column="target",
        artifact_id=registration.artifact_id,
        format_name=registration.format,
    )
    source.write_bytes(b"changed outside immutable storage")

    assert store.read_bytes(registration.artifact_id) == original
    assert registration.role is DatasetRole.TRAIN
    assert profile.row_count == 3
    assert profile.column_count == 3
    assert profile.target_distribution is not None
    assert profile.target_distribution.counts == {"no": 1, "yes": 2}
    assert "feature" in [column.name for column in profile.columns if column.missing_count]
    assert "id" in profile.suggestions.id_columns
    assert any(warning.code == "duplicate_rows" for warning in profile.warnings)
    assert ledger.get_record("datasets", registration.artifact_id)["role"] == "train"


def test_parquet_profile_is_deterministic_and_discloses_bounded_sampling(tmp_path: Path) -> None:
    source = tmp_path / "train.parquet"
    pd.DataFrame({"id": range(20), "value": [1] * 20, "target": ["a", "b"] * 10}).to_parquet(
        source, index=False
    )
    profiler = DatasetProfiler(sample_limit=5, chunk_size=4)

    first = profiler.profile(source, role=DatasetRole.TRAIN, target_column="target")
    second = profiler.profile(source, role=DatasetRole.TRAIN, target_column="target")

    assert first.canonical_json() == second.canonical_json()
    assert first.row_count == 20
    assert first.sampling.sampled_rows == 5
    assert first.sampling.sampling_applied is True
    assert first.sampling.row_count_from_metadata is True
    assert first.sampling.scanned_rows == 5
    assert first.sampling.scan_batches == 2
    assert "value" in first.constant_columns
    assert any(warning.code == "sampled_profile" for warning in first.warnings)


def test_target_leakage_and_validation_suggestions_are_reported(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    source.write_text(
        "record_id,customer_group,event_time,leak,target\n1,a,2025-01-01,0,0\n2,b,2025-01-02,1,1\n",
        encoding="utf-8",
    )
    profile = DatasetProfiler().profile(source, target_column="target", role=DatasetRole.TRAIN)

    assert profile.suggestions.id_columns == ["record_id"]
    assert profile.suggestions.group_columns == ["customer_group"]
    assert profile.suggestions.time_columns == ["event_time"]
    assert any(warning.code == "possible_target_leakage" for warning in profile.warnings)
    assert any(warning.code == "group_validation" for warning in profile.warnings)
    assert any(warning.code == "time_validation" for warning in profile.warnings)


def test_train_test_schema_comparison_and_cross_split_duplicates(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("id,feature,target\n1,10,yes\n2,20,no\n", encoding="utf-8")
    test.write_text("id,feature,extra\n1,10,x\n3,30,y\n", encoding="utf-8")
    service, _, ledger = _service(tmp_path)

    analysis = service.ingest(train, test_path=test, target_column="target", session_id="session-1")

    assert analysis.test is not None
    assert analysis.schema_comparison is not None
    assert analysis.schema_comparison.train_only_columns == ["target"]
    assert analysis.schema_comparison.test_only_columns == ["extra"]
    assert analysis.schema_comparison.duplicate_common_rows == 1
    assert any(warning.code == "schema_mismatch" for warning in analysis.schema_comparison.warnings)
    assert any(
        warning.code == "duplicate_rows_across_splits"
        for warning in analysis.schema_comparison.warnings
    )
    assert "raw" not in analysis.report_markdown.lower()
    assert ledger.list_records("datasets", session_id="session-1")


def test_derived_data_helper_records_parent_transformation_and_fit_scope(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    source.write_bytes(b"x\n1\n")
    service, store, ledger = _service(tmp_path)
    parent = service.register_source(source, role=DatasetRole.TRAIN)

    derived = service.store_derived(
        b"transformed",
        parent_artifact_ids=[parent.artifact_id],
        transformation="standardize numeric columns",
        fit_scope="train-fold-0",
        media_type="text/csv",
        session_id="session-1",
    )

    manifest = store.get_manifest(derived.artifact_id)
    assert manifest.parent_artifact_ids == [parent.artifact_id]
    assert manifest.transformation == "standardize numeric columns"  # type: ignore[union-attr]
    assert manifest.fit_scope == "train-fold-0"  # type: ignore[union-attr]
    assert ledger.get_record("artifacts", derived.artifact_id)["parent_artifact_ids"] == [
        parent.artifact_id
    ]
