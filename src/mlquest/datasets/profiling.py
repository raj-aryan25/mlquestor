"""Bounded, local-only profiling for CSV and Parquet datasets."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as parquet

from mlquest.artifacts import StoredArtifact, sha256_file
from mlquest.domain import DatasetRole
from .models import (
    ColumnProfile,
    ColumnSuggestions,
    DatasetProfile,
    ProfileWarning,
    SamplingMetadata,
    SchemaColumnComparison,
    SchemaComparison,
    TargetDistribution,
    WarningSeverity,
)


class DatasetFormatError(ValueError):
    """Raised when an input is not a supported tabular format."""


def _format_name(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise DatasetFormatError(f"unsupported dataset format: {path.suffix or '<none>'}")


def _canonical_value(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _normalise_dtype(dtype: Any) -> str:
    return str(dtype).lower().replace(" ", "")


def _warning(code: str, message: str, *, columns: Iterable[str] = (), severity: WarningSeverity = WarningSeverity.WARNING) -> ProfileWarning:
    return ProfileWarning(code=code, severity=severity, message=message, columns=sorted(set(columns)))


class DatasetProfiler:
    """Profile a file while retaining at most ``sample_limit`` rows."""

    def __init__(self, *, sample_limit: int = 10_000, chunk_size: int = 4_096) -> None:
        if sample_limit < 1 or chunk_size < 1:
            raise ValueError("sample_limit and chunk_size must be positive")
        self.sample_limit = sample_limit
        self.chunk_size = chunk_size

    def profile(
        self,
        source: Path | str | StoredArtifact,
        *,
        target_column: str | None = None,
        role: DatasetRole | str = DatasetRole.AUXILIARY,
        artifact_id: str | None = None,
        format_name: str | None = None,
    ) -> DatasetProfile:
        path, source_id, source_format = self._resolve_source(source, artifact_id)
        resolved_format = format_name or source_format or _format_name(path)
        if resolved_format == "csv":
            sample, row_count, scanned_rows, batches, names, metadata_count = self._read_csv(path)
        elif resolved_format == "parquet":
            sample, row_count, scanned_rows, batches, names, metadata_count = self._read_parquet(path)
        else:
            raise DatasetFormatError(f"unsupported dataset format: {resolved_format}")

        columns = self._column_profiles(sample, names)
        duplicate_count = int(sample.duplicated(keep="first").sum()) if not sample.empty else 0
        denominator = len(sample)
        duplicate_fraction = duplicate_count / denominator if denominator else 0.0
        resolved_target = target_column if target_column in names else None
        suggestions = self._suggest_columns(sample, names, target_column)
        warnings = self._profile_warnings(
            sample,
            columns,
            duplicate_count,
            resolved_target,
            target_column,
            suggestions,
            sampled=len(sample) < row_count,
        )
        distribution = self._target_distribution(sample, resolved_target)
        return DatasetProfile(
            source_artifact_id=source_id,
            content_hash=source_id,
            role=DatasetRole(role),
            format=resolved_format,  # type: ignore[arg-type]
            row_count=row_count,
            column_count=len(names),
            columns=columns,
            duplicate_row_count=duplicate_count,
            duplicate_fraction=duplicate_fraction,
            constant_columns=[column.name for column in columns if column.is_constant],
            target_column=resolved_target,
            target_distribution=distribution,
            suggestions=suggestions,
            warnings=sorted(warnings, key=lambda item: (item.code, item.message)),
            sampling=SamplingMetadata(
                sample_limit=self.sample_limit,
                sampled_rows=len(sample),
                scanned_rows=scanned_rows,
                scan_batches=batches,
                chunk_size=self.chunk_size,
                sampling_applied=len(sample) < row_count,
                row_count_from_metadata=metadata_count,
            ),
        )

    def sample(self, source: Path | str | StoredArtifact, *, format_name: str | None = None) -> pd.DataFrame:
        """Return only the bounded profiling sample for local comparisons."""

        path, _, source_format = self._resolve_source(source, None)
        resolved_format = format_name or source_format or _format_name(path)
        if resolved_format == "csv":
            frame, *_ = self._read_csv(path)
        elif resolved_format == "parquet":
            frame, *_ = self._read_parquet(path)
        else:
            raise DatasetFormatError(f"unsupported dataset format: {resolved_format}")
        return frame

    profile_dataset = profile

    def compare(
        self,
        train: DatasetProfile,
        test: DatasetProfile,
        *,
        train_sample: pd.DataFrame | None = None,
        test_sample: pd.DataFrame | None = None,
        target_column: str | None = None,
    ) -> SchemaComparison:
        train_names = [column.name for column in train.columns]
        test_names = [column.name for column in test.columns]
        train_set, test_set = set(train_names), set(test_names)
        common = sorted(train_set & test_set)
        train_only = sorted(train_set - test_set)
        test_only = sorted(test_set - train_set)
        train_by_name = {column.name: column for column in train.columns}
        test_by_name = {column.name: column for column in test.columns}
        details: dict[str, SchemaColumnComparison] = {}
        mismatches: list[str] = []
        for name in common:
            train_column, test_column = train_by_name[name], test_by_name[name]
            details[name] = SchemaColumnComparison(
                train_dtype=train_column.dtype,
                test_dtype=test_column.dtype,
                train_missing_fraction=train_column.missing_fraction,
                test_missing_fraction=test_column.missing_fraction,
                train_cardinality=train_column.cardinality,
                test_cardinality=test_column.cardinality,
            )
            if train_column.dtype != test_column.dtype:
                mismatches.append(name)
        comparison_warnings: list[ProfileWarning] = []
        if train_only or test_only:
            comparison_warnings.append(
                _warning(
                    "schema_mismatch",
                    "Train and test contain different columns.",
                    columns=[*train_only, *test_only],
                )
            )
        if mismatches:
            comparison_warnings.append(
                _warning("dtype_mismatch", "Common train/test columns have different inferred types.", columns=mismatches)
            )
        if target_column and target_column in test_set:
            comparison_warnings.append(
                _warning("target_present_in_test", "The supplied target column is present in test data.", columns=[target_column])
            )
        duplicate_common = 0
        if train_sample is not None and test_sample is not None and common:
            train_keys = {_row_key(row) for row in train_sample[common].itertuples(index=False, name=None)}
            duplicate_common = sum(_row_key(row) in train_keys for row in test_sample[common].itertuples(index=False, name=None))
            if duplicate_common:
                comparison_warnings.append(
                    _warning("duplicate_rows_across_splits", "Common feature rows occur in both train and test.", columns=common)
                )
        return SchemaComparison(
            train_only_columns=train_only,
            test_only_columns=test_only,
            common_columns=common,
            dtype_mismatches=sorted(mismatches),
            columns=details,
            duplicate_common_rows=duplicate_common,
            warnings=sorted(comparison_warnings, key=lambda item: (item.code, item.message)),
        )

    def _resolve_source(
        self, source: Path | str | StoredArtifact, artifact_id: str | None
    ) -> tuple[Path, str, str | None]:
        if isinstance(source, StoredArtifact):
            return source.content_path, artifact_id or source.artifact_id, getattr(source.manifest, "format", None)
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, artifact_id or sha256_file(path), None

    def _read_csv(self, path: Path) -> tuple[pd.DataFrame, int, int, int, list[str], bool]:
        sample_parts: list[pd.DataFrame] = []
        sampled = 0
        row_count = 0
        batches = 0
        names: list[str] = []
        try:
            names = [str(name) for name in pd.read_csv(path, nrows=0).columns]
            chunks = pd.read_csv(path, chunksize=self.chunk_size, low_memory=False)
            for chunk in chunks:
                batches += 1
                row_count += len(chunk)
                if sampled < self.sample_limit:
                    part = chunk.iloc[: self.sample_limit - sampled].copy()
                    sample_parts.append(part)
                    sampled += len(part)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(), 0, 0, 0, [], False
        sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame(columns=names)
        return sample, row_count, row_count, batches, names, False

    def _read_parquet(self, path: Path) -> tuple[pd.DataFrame, int, int, int, list[str], bool]:
        parquet_file = parquet.ParquetFile(path)
        row_count = int(parquet_file.metadata.num_rows)
        names = [str(name) for name in parquet_file.schema_arrow.names]
        parts: list[pd.DataFrame] = []
        sampled = 0
        batches = 0
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            batches += 1
            frame = batch.to_pandas()
            if sampled < self.sample_limit:
                part = frame.iloc[: self.sample_limit - sampled].copy()
                parts.append(part)
                sampled += len(part)
            if sampled >= self.sample_limit:
                break
        sample = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=names)
        return sample, row_count, len(sample), batches, names, True

    @staticmethod
    def _column_profiles(sample: pd.DataFrame, names: list[str]) -> list[ColumnProfile]:
        profiles: list[ColumnProfile] = []
        for name in names:
            series = sample[name] if name in sample else pd.Series(dtype="object")
            missing = int(series.isna().sum())
            observed = series.dropna()
            cardinality = int(observed.nunique(dropna=True))
            denominator = len(sample)
            profiles.append(
                ColumnProfile(
                    name=name,
                    dtype=_normalise_dtype(series.dtype),
                    nullable=missing > 0,
                    missing_count=missing,
                    missing_fraction=missing / denominator if denominator else 0.0,
                    cardinality=cardinality,
                    cardinality_fraction=cardinality / denominator if denominator else 0.0,
                    is_constant=cardinality <= 1 and denominator > 0,
                )
            )
        return profiles

    @staticmethod
    def _suggest_columns(sample: pd.DataFrame, names: list[str], target_column: str | None) -> ColumnSuggestions:
        target_names = {"target", "label", "y", "outcome", "response", "class"}
        targets = [target_column] if target_column in names else [name for name in names if name.lower() in target_names]
        ids: list[str] = []
        groups: list[str] = []
        times: list[str] = []
        for name in names:
            lower = name.lower()
            series = sample[name] if name in sample else pd.Series(dtype="object")
            unique_fraction = series.nunique(dropna=True) / len(series) if len(series) else 0.0
            group_like = any(token in lower for token in ("group", "customer", "account", "user"))
            time_like = any(token in lower for token in ("time", "date", "timestamp"))
            id_name_like = lower in {"id", "uuid", "identifier"} or lower.endswith("_id")
            if id_name_like and (unique_fraction >= 0.99 or lower in {"id", "uuid", "identifier"}):
                ids.append(name)
            if group_like:
                groups.append(name)
            if time_like or pd.api.types.is_datetime64_any_dtype(series):
                times.append(name)
        return ColumnSuggestions(
            target_columns=sorted(set(targets)),
            id_columns=sorted(set(ids)),
            group_columns=sorted(set(groups)),
            time_columns=sorted(set(times)),
        )

    @staticmethod
    def _target_distribution(sample: pd.DataFrame, target: str | None) -> TargetDistribution | None:
        if not target or target not in sample:
            return None
        values = sample[target].dropna().map(_canonical_value)
        counts = values.value_counts().to_dict()
        ordered = {str(key): int(counts[key]) for key in sorted(counts)}
        total = sum(ordered.values())
        proportions = {key: ordered[key] / total for key in ordered} if total else {}
        return TargetDistribution(column=target, counts=ordered, proportions=proportions, observed_rows=int(total))

    @staticmethod
    def _profile_warnings(
        sample: pd.DataFrame,
        columns: list[ColumnProfile],
        duplicate_count: int,
        resolved_target: str | None,
        requested_target: str | None,
        suggestions: ColumnSuggestions,
        *,
        sampled: bool,
    ) -> list[ProfileWarning]:
        warnings: list[ProfileWarning] = []
        missing = [column.name for column in columns if column.missing_count]
        if missing:
            warnings.append(_warning("missing_values", "One or more columns contain missing values.", columns=missing))
        constants = [column.name for column in columns if column.is_constant]
        if constants:
            warnings.append(_warning("constant_columns", "One or more columns are constant in the observed data.", columns=constants))
        if duplicate_count:
            warnings.append(_warning("duplicate_rows", "Duplicate rows were observed in the profiling sample."))
        if sampled:
            warnings.append(_warning("sampled_profile", "Column metrics were calculated from a bounded sample.", severity=WarningSeverity.INFO))
        if requested_target and resolved_target is None:
            warnings.append(_warning("target_not_found", "The requested target column was not found."))
        if resolved_target and resolved_target in sample:
            target_values = sample[resolved_target].dropna()
            if target_values.nunique() <= 1 and len(target_values):
                warnings.append(_warning("single_target_value", "The target has one observed value.", columns=[resolved_target]))
            if len(target_values) and target_values.value_counts(normalize=True).iloc[0] >= 0.8 and target_values.nunique() > 1:
                warnings.append(_warning("target_imbalance", "The target distribution is strongly imbalanced.", columns=[resolved_target]))
            for column in sample.columns:
                if column == resolved_target:
                    continue
                if sample[column].equals(sample[resolved_target]):
                    warnings.append(_warning("possible_target_leakage", "A feature exactly matches the target in the observed data.", columns=[column, resolved_target]))
        if suggestions.id_columns:
            warnings.append(_warning("identifier_column", "Identifier-like columns may require exclusion or careful validation.", columns=suggestions.id_columns, severity=WarningSeverity.INFO))
        if suggestions.group_columns:
            warnings.append(_warning("group_validation", "Group-like columns suggest group-aware validation.", columns=suggestions.group_columns, severity=WarningSeverity.INFO))
        if suggestions.time_columns:
            warnings.append(_warning("time_validation", "Time-like columns suggest temporal ordering or leakage checks.", columns=suggestions.time_columns, severity=WarningSeverity.INFO))
        return warnings


def _row_key(row: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(_canonical_value(value) for value in row)
