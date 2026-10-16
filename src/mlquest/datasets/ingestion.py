"""Dataset registration, analysis, reporting, and derived-data lineage."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from mlquest.artifacts import FileSystemArtifactStore, StoredArtifact
from mlquest.domain import DatasetArtifact, DatasetRole
from mlquest.persistence import LocalLedger

from .models import DatasetAnalysis, DatasetProfile, SchemaComparison, SourceRegistration
from .profiling import DatasetFormatError, DatasetProfiler


class DatasetService:
    """Coordinate immutable source registration and local profiling."""

    def __init__(
        self,
        artifact_store: FileSystemArtifactStore,
        *,
        ledger: LocalLedger | None = None,
        profiler: DatasetProfiler | None = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.ledger = ledger
        self.profiler = profiler or DatasetProfiler()

    def register_source(
        self,
        source_path: Path | str,
        *,
        role: DatasetRole | str,
        session_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> SourceRegistration:
        path = Path(source_path)
        format_name = self._format(path)
        media_type = "text/csv" if format_name == "csv" else "application/vnd.apache.parquet"
        stored = self.artifact_store.put_source(
            path,
            role=role,
            format=format_name,
            media_type=media_type,
            original_location=str(path),
            metadata=metadata,
        )
        manifest = stored.manifest
        registration = SourceRegistration(
            artifact_id=stored.artifact_id,
            content_hash=stored.content_hash,
            role=DatasetRole(role),
            format=format_name,  # type: ignore[arg-type]
            media_type=manifest.media_type,
            immutable_location=manifest.immutable_location,
            original_location=str(path),
            size_bytes=manifest.size_bytes or 0,
        )
        if self.ledger is not None:
            dataset = DatasetArtifact(
                id=registration.artifact_id,
                role=registration.role,
                content_hash=registration.content_hash,
                media_type=registration.media_type,
                immutable_location=registration.immutable_location,
                size_bytes=registration.size_bytes,
                metadata=dict(metadata or {}),
                format=registration.format,
                original_location=registration.original_location,
            )
            self.ledger.put_record("artifacts", dataset, session_id=session_id)
            self.ledger.put_record("datasets", dataset, session_id=session_id)
        return registration

    def profile_source(
        self,
        source: Path | str | StoredArtifact,
        *,
        target_column: str | None = None,
        role: DatasetRole | str = DatasetRole.AUXILIARY,
        artifact_id: str | None = None,
        format_name: str | None = None,
    ) -> DatasetProfile:
        return self.profiler.profile(
            source,
            target_column=target_column,
            role=role,
            artifact_id=artifact_id,
            format_name=format_name,
        )

    def ingest(
        self,
        train_path: Path | str,
        *,
        test_path: Path | str | None = None,
        target_column: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> DatasetAnalysis:
        train_registration = self.register_source(
            train_path, role=DatasetRole.TRAIN, session_id=session_id, metadata=metadata
        )
        train_profile = self.profile_source(
            self.artifact_store.artifact_path(train_registration.artifact_id),
            target_column=target_column,
            role=DatasetRole.TRAIN,
            artifact_id=train_registration.artifact_id,
            format_name=train_registration.format,
        )
        test_profile: DatasetProfile | None = None
        comparison: SchemaComparison | None = None
        if test_path is not None:
            test_registration = self.register_source(
                test_path, role=DatasetRole.TEST, session_id=session_id, metadata=metadata
            )
            test_profile = self.profile_source(
                self.artifact_store.artifact_path(test_registration.artifact_id),
                target_column=None,
                role=DatasetRole.TEST,
                artifact_id=test_registration.artifact_id,
                format_name=test_registration.format,
            )
            comparison = self.profiler.compare(
                train_profile,
                test_profile,
                train_sample=self.profiler.sample(
                    self.artifact_store.artifact_path(train_registration.artifact_id),
                    format_name=train_registration.format,
                ),
                test_sample=self.profiler.sample(
                    self.artifact_store.artifact_path(test_registration.artifact_id),
                    format_name=test_registration.format,
                ),
                target_column=target_column,
            )
        report = render_markdown_report(train_profile, test_profile, comparison)
        return DatasetAnalysis(train=train_profile, test=test_profile, schema_comparison=comparison, report_markdown=report)

    def store_derived(
        self,
        content: Path | str | bytes | bytearray | BinaryIO,
        *,
        parent_artifact_ids: list[str] | tuple[str, ...],
        transformation: str,
        fit_scope: str | None = None,
        media_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
        session_id: str | None = None,
    ) -> StoredArtifact:
        stored = self.artifact_store.put_derived(
            content,
            parent_artifact_ids=parent_artifact_ids,
            transformation=transformation,
            fit_scope=fit_scope,
            media_type=media_type,
            metadata=metadata,
        )
        if self.ledger is not None:
            self.ledger.put_record("artifacts", stored.registration_metadata(), session_id=session_id)
        return stored

    register_source_file = register_source
    store_derived_data = store_derived
    register = register_source
    analyze = ingest
    ingest_dataset = ingest

    @staticmethod
    def _format(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return "csv"
        if suffix in {".parquet", ".pq"}:
            return "parquet"
        raise DatasetFormatError(f"unsupported dataset format: {path.suffix or '<none>'}")


DatasetIngestor = DatasetService


def render_markdown_report(
    train: DatasetProfile,
    test: DatasetProfile | None = None,
    comparison: SchemaComparison | None = None,
) -> str:
    """Render a stable report from profiles without including data rows."""

    lines = [
        "# Dataset profile",
        "",
        f"- Artifact: `{train.source_artifact_id}`",
        f"- Role: `{train.role.value}`",
        f"- Format: `{train.format}`",
        f"- Rows: **{train.row_count}**",
        f"- Columns: **{train.column_count}**",
        f"- Duplicate rows observed: **{train.duplicate_row_count}**",
        f"- Constant columns: {', '.join(train.constant_columns) or 'none'}",
        "",
        "## Sampling",
        "",
        f"- Method: `{train.sampling.method}`",
        f"- Sampled rows: **{train.sampling.sampled_rows}** of **{train.row_count}**",
        f"- Chunk size: **{train.sampling.chunk_size}**",
        f"- Metrics based on sample: **{str(train.sampling.metrics_based_on_sample).lower()}**",
        "",
        "## Columns",
        "",
        "| Name | Type | Missing | Cardinality | Constant |",
        "| --- | --- | ---: | ---: | :---: |",
    ]
    lines.extend(
        f"| `{column.name}` | `{column.dtype}` | {column.missing_count} ({column.missing_fraction:.3f}) | {column.cardinality} | {'yes' if column.is_constant else 'no'} |"
        for column in train.columns
    )
    lines.extend(["", "## Suggestions", "", *_suggestion_lines(train)])
    lines.extend(["", "## Warnings", ""])
    lines.extend(_warning_lines(train.warnings))
    if train.target_distribution is not None:
        lines.extend(["", "## Target distribution", "", f"Target: `{train.target_distribution.column}`", ""])
        lines.append("| Value | Count | Proportion |")
        lines.append("| --- | ---: | ---: |")
        lines.extend(
            f"| `{value}` | {train.target_distribution.counts[value]} | {train.target_distribution.proportions[value]:.3f} |"
            for value in train.target_distribution.counts
        )
    if test is not None:
        lines.extend(["", "## Test profile", "", f"- Artifact: `{test.source_artifact_id}`", f"- Rows: **{test.row_count}**", f"- Columns: **{test.column_count}**"])
    if comparison is not None:
        lines.extend(["", "## Train/test schema comparison", ""])
        lines.append(f"- Train-only columns: {', '.join(comparison.train_only_columns) or 'none'}")
        lines.append(f"- Test-only columns: {', '.join(comparison.test_only_columns) or 'none'}")
        lines.append(f"- Dtype mismatches: {', '.join(comparison.dtype_mismatches) or 'none'}")
        lines.append(f"- Duplicate common rows observed: **{comparison.duplicate_common_rows}**")
        lines.extend(["", "### Comparison warnings", ""])
        lines.extend(_warning_lines(comparison.warnings))
    return "\n".join(lines).rstrip() + "\n"


def _suggestion_lines(profile: DatasetProfile) -> list[str]:
    suggestions = profile.suggestions
    return [
        f"- Likely target columns: {', '.join(suggestions.target_columns) or 'none'}",
        f"- Likely ID columns: {', '.join(suggestions.id_columns) or 'none'}",
        f"- Likely group columns: {', '.join(suggestions.group_columns) or 'none'}",
        f"- Likely time columns: {', '.join(suggestions.time_columns) or 'none'}",
    ]


def _warning_lines(warnings: list) -> list[str]:
    if not warnings:
        return ["- None"]
    return [
        f"- **{warning.severity.value}** `{warning.code}`: {warning.message}"
        + (f" ({', '.join(warning.columns)})" if warning.columns else "")
        for warning in warnings
    ]
