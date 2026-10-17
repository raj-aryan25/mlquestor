"""Reproducible train/validation split plans for tabular evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)

from mlquest.domain import SplitSpecification, SplitStrategy, TaskType


@dataclass(frozen=True)
class SplitFold:
    """One persisted training/evaluation fold."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]

    @property
    def test_indices(self) -> tuple[int, ...]:
        return self.validation_indices

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
        }


@dataclass(frozen=True)
class SplitPlan:
    """A split plan whose identity includes the exact assignments."""

    strategy: SplitStrategy
    split_identity: str
    seed: int | None
    folds: tuple[SplitFold, ...]
    assignments: tuple[int, ...]
    specification: SplitSpecification

    @property
    def identity(self) -> str:
        return self.split_identity

    @property
    def split_id(self) -> str:
        """Alias matching the domain split specification terminology."""

        return self.split_identity

    @property
    def train_indices(self) -> tuple[int, ...]:
        if len(self.folds) != 1:
            raise AttributeError("cross-validation has multiple training folds")
        return self.folds[0].train_indices

    @property
    def validation_indices(self) -> tuple[int, ...]:
        if len(self.folds) != 1:
            raise AttributeError("cross-validation has multiple validation folds")
        return self.folds[0].validation_indices

    @property
    def test_indices(self) -> tuple[int, ...]:
        return self.validation_indices

    @property
    def fold_assignments(self) -> tuple[int, ...]:
        return self.assignments

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "split_identity": self.split_identity,
            "seed": self.seed,
            "assignments": list(self.assignments),
            "folds": [fold.as_dict() for fold in self.folds],
            "specification": self.specification.model_dump(
                mode="json", exclude={"id", "schema_version", "created_at"}, exclude_none=True
            ),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


def _column(
    data: Any, name: str | None, argument: Iterable[Any] | None, kind: str
) -> np.ndarray | None:
    if argument is not None:
        return np.asarray(argument)
    if name is None:
        return None
    if not hasattr(data, "columns") or name not in data.columns:
        raise ValueError(f"{kind} column {name!r} is not present in the input data")
    return np.asarray(data[name])


def _size(data: Any) -> int:
    try:
        size = len(data)
    except TypeError as exc:
        raise ValueError("split input must be a sized table or array") from exc
    if size < 2:
        raise ValueError("at least two rows are required for a split")
    return size


def _target(y: Iterable[Any] | None, n_rows: int) -> np.ndarray | None:
    if y is None:
        return None
    values = np.asarray(y)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError("split target must be one-dimensional and match the input row count")
    return values


def _split_sizes(specification: SplitSpecification) -> tuple[float, float | None]:
    test_size = specification.test_size if specification.test_size is not None else 0.2
    return test_size, specification.validation_size


def _effective_seed(specification: SplitSpecification) -> int:
    return specification.random_seed if specification.random_seed is not None else 0


def _random_holdout(
    indices: np.ndarray,
    spec: SplitSpecification,
    *,
    y: np.ndarray | None,
    groups: np.ndarray | None,
    stratified: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    test_size, validation_size = _split_sizes(spec)
    seed = _effective_seed(spec)
    if not spec.shuffle and groups is None and not stratified:
        test_count = max(1, int(np.ceil(len(indices) * test_size)))
        test = indices[-test_count:]
        train = indices[:-test_count]
        if validation_size is None:
            return train, None, test
        validation_count = max(1, int(np.ceil(len(indices) * validation_size)))
        if validation_count >= len(train):
            raise ValueError("split sizes leave no training rows")
        return train[:-validation_count], train[-validation_count:], test
    if groups is not None:
        if len(np.unique(groups)) < 2:
            raise ValueError("group split requires at least two distinct groups")
        first = next(
            GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(
                indices, groups=groups
            )
        )
    elif stratified:
        if y is None:
            raise ValueError("stratified split requires target labels")
        first = next(
            StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(
                indices, y
            )
        )
    else:
        first = next(
            ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(indices)
        )
    train_part, test_part = first
    train = indices[train_part]
    test = indices[test_part]
    if validation_size is None:
        return train, None, test
    relative_validation = validation_size / (1.0 - test_size)
    if groups is not None:
        second = next(
            GroupShuffleSplit(n_splits=1, test_size=relative_validation, random_state=seed).split(
                train, groups=groups[train]
            )
        )
    elif stratified:
        if y is None:
            raise ValueError("stratified split requires target labels")
        second = next(
            StratifiedShuffleSplit(
                n_splits=1, test_size=relative_validation, random_state=seed
            ).split(train, y[train])
        )
    else:
        second = next(
            ShuffleSplit(n_splits=1, test_size=relative_validation, random_state=seed).split(train)
        )
    remaining, validation_part = second
    return train[remaining], train[validation_part], test


def _time_holdout(
    indices: np.ndarray, spec: SplitSpecification, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if len(times) != len(indices):
        raise ValueError("time values must match the input row count")
    ordered = indices[np.argsort(times, kind="mergesort")]
    test_size, validation_size = _split_sizes(spec)
    test_count = max(1, int(np.ceil(len(ordered) * test_size)))
    test = ordered[-test_count:]
    remaining = ordered[:-test_count]
    if validation_size is None:
        return remaining, None, test
    validation_count = max(1, int(np.ceil(len(ordered) * validation_size)))
    if validation_count >= len(remaining):
        raise ValueError("time split sizes leave no training rows")
    validation = remaining[-validation_count:]
    return remaining[:-validation_count], validation, test


def _identity(
    spec: SplitSpecification,
    folds: tuple[SplitFold, ...],
    assignments: tuple[int, ...],
    seed: int | None,
) -> str:
    if spec.split_id is not None:
        return spec.split_id
    payload = {
        "specification": spec.model_dump(
            mode="json",
            exclude={"id", "schema_version", "created_at", "split_id"},
            exclude_none=True,
        ),
        "seed": seed,
        "assignments": assignments,
        "folds": [fold.as_dict() for fold in folds],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def create_split_plan(
    data: Any,
    specification: SplitSpecification,
    *,
    y: Iterable[Any] | None = None,
    groups: Iterable[Any] | None = None,
    times: Iterable[Any] | None = None,
    task_type: TaskType | None = None,
) -> SplitPlan:
    """Create a deterministic plan for random, grouped, temporal, or CV evaluation."""

    n_rows = _size(data)
    target = _target(y, n_rows)
    group_values = _column(data, specification.group_column, groups, "group")
    time_values = _column(data, specification.time_column, times, "time")
    if group_values is not None and len(group_values) != n_rows:
        raise ValueError("group values must match the input row count")
    if time_values is not None and len(time_values) != n_rows:
        raise ValueError("time values must match the input row count")
    indices = np.arange(n_rows, dtype=int)
    strategy = specification.strategy
    folds: list[SplitFold]
    assignments = np.full(n_rows, -1, dtype=int)
    if strategy is SplitStrategy.CROSS_VALIDATION:
        iterator: Iterator[tuple[np.ndarray, np.ndarray]]
        splitter: Any
        if group_values is not None:
            splitter = GroupKFold(n_splits=specification.n_splits or 2)
            iterator = splitter.split(indices, target, group_values)
        elif task_type is TaskType.CLASSIFICATION and target is not None:
            splitter = StratifiedKFold(
                n_splits=specification.n_splits or 2,
                shuffle=specification.shuffle,
                random_state=_effective_seed(specification) if specification.shuffle else None,
            )
            iterator = splitter.split(indices, target)
        else:
            splitter = KFold(
                n_splits=specification.n_splits or 2,
                shuffle=specification.shuffle,
                random_state=_effective_seed(specification) if specification.shuffle else None,
            )
            iterator = splitter.split(indices)
        folds = []
        for fold_number, (train_part, validation_part) in enumerate(iterator):
            validation_indices = tuple(indices[validation_part].tolist())
            folds.append(SplitFold(tuple(indices[train_part].tolist()), validation_indices))
            assignments[list(validation_indices)] = fold_number
    elif strategy is SplitStrategy.TIME:
        if time_values is None:
            raise ValueError("time split requires time values or the configured time column")
        ordered = indices[np.argsort(time_values, kind="mergesort")]
        if specification.n_splits is not None:
            folds = []
            splitter = TimeSeriesSplit(n_splits=specification.n_splits)
            for fold_number, (train_part, validation_part) in enumerate(splitter.split(ordered)):
                train_indices = tuple(ordered[train_part].tolist())
                validation_indices = tuple(ordered[validation_part].tolist())
                folds.append(SplitFold(train_indices, validation_indices))
                assignments[list(validation_indices)] = fold_number
        else:
            train, validation_rows, test = _time_holdout(indices, specification, time_values)
            evaluation = validation_rows if validation_rows is not None else test
            folds = [SplitFold(tuple(train.tolist()), tuple(evaluation.tolist()))]
            assignments[train] = 0
            assignments[test] = 1
            if validation_rows is not None:
                assignments[validation_rows] = 0
    else:
        train, validation_rows, test = _random_holdout(
            indices,
            specification,
            y=target,
            groups=group_values if strategy is SplitStrategy.GROUP else None,
            stratified=strategy is SplitStrategy.STRATIFIED,
        )
        evaluation = validation_rows if validation_rows is not None else test
        folds = [SplitFold(tuple(train.tolist()), tuple(evaluation.tolist()))]
        assignments[train] = 0
        assignments[test] = 1
        if validation_rows is not None:
            assignments[validation_rows] = 0
    frozen_folds = tuple(folds)
    frozen_assignments = tuple(int(value) for value in assignments.tolist())
    return SplitPlan(
        strategy=strategy,
        split_identity=_identity(
            specification,
            frozen_folds,
            frozen_assignments,
            _effective_seed(specification)
            if strategy is not SplitStrategy.TIME
            else specification.random_seed,
        ),
        seed=(
            _effective_seed(specification)
            if strategy is not SplitStrategy.TIME
            else specification.random_seed
        ),
        folds=frozen_folds,
        assignments=frozen_assignments,
        specification=specification,
    )


def generate_splits(
    data: Any,
    y_or_specification: Iterable[Any] | SplitSpecification,
    specification: SplitSpecification | None = None,
    *,
    groups: Iterable[Any] | None = None,
    times: Iterable[Any] | None = None,
    task_type: TaskType | None = None,
) -> SplitPlan:
    """Accept either ``(data, spec)`` or ``(data, y, spec)``."""

    if specification is None:
        if not isinstance(y_or_specification, SplitSpecification):
            raise TypeError("specification is required when y is supplied")
        return create_split_plan(
            data, y_or_specification, groups=groups, times=times, task_type=task_type
        )
    if isinstance(y_or_specification, SplitSpecification):
        raise TypeError("provide y before specification")
    return create_split_plan(
        data, specification, y=y_or_specification, groups=groups, times=times, task_type=task_type
    )


class DeterministicSplitter:
    """Reusable splitter wrapper suitable for a baseline runner."""

    def __init__(
        self, specification: SplitSpecification, *, task_type: TaskType | None = None
    ) -> None:
        self.specification = specification
        self.task_type = task_type

    def split(
        self,
        data: Any,
        y: Iterable[Any] | None = None,
        *,
        groups: Iterable[Any] | None = None,
        times: Iterable[Any] | None = None,
    ) -> SplitPlan:
        return create_split_plan(
            data,
            self.specification,
            y=y,
            groups=groups,
            times=times,
            task_type=self.task_type,
        )


__all__ = [
    "DeterministicSplitter",
    "SplitFold",
    "SplitPlan",
    "create_split_plan",
    "generate_splits",
]
