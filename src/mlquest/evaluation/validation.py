"""Validation of model predictions before they reach the metric layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from mlquest.domain import TaskType


class PredictionValidationError(ValueError):
    """Raised when predictions cannot be safely evaluated."""


@dataclass(frozen=True)
class PredictionValidationResult:
    """Validated prediction metadata returned to evaluation callers."""

    values: np.ndarray
    task_type: TaskType
    prediction_kind: str
    label_domain: tuple[Any, ...] = ()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])


def _as_target(values: Iterable[Any], expected_size: int | None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise PredictionValidationError("target values must be one-dimensional")
    if expected_size is not None and len(array) != expected_size:
        raise PredictionValidationError(
            f"target and prediction row counts differ ({len(array)} != {expected_size})"
        )
    if len(array) == 0:
        raise PredictionValidationError("target and predictions must not be empty")
    return array


def _domain(values: np.ndarray, labels: Iterable[Any] | None) -> tuple[Any, ...]:
    supplied = tuple(labels) if labels is not None else tuple(np.unique(values).tolist())
    if not supplied:
        raise PredictionValidationError("classification label domain must not be empty")
    missing = [
        value for value in np.unique(values) if not any(value == known for known in supplied)
    ]
    if missing:
        raise PredictionValidationError(
            f"target contains labels outside the label domain: {missing!r}"
        )
    return supplied


def _numeric(values: np.ndarray, name: str) -> np.ndarray:
    if values.dtype.kind not in "biufc":
        raise PredictionValidationError(f"{name} predictions must be numeric")
    if values.dtype.kind == "b":
        raise PredictionValidationError(f"{name} predictions must be numeric, not boolean")
    if not np.isfinite(values).all():
        raise PredictionValidationError(f"{name} predictions must contain only finite values")
    return values


def validate_predictions(
    y_true: Iterable[Any],
    predictions: Any,
    task_type: TaskType,
    *,
    prediction_kind: str = "label",
    labels: Iterable[Any] | None = None,
    label_domain: Iterable[Any] | None = None,
    expected_size: int | None = None,
) -> PredictionValidationResult:
    """Validate shape, type, finiteness, and classification label membership.

    ``prediction_kind`` is ``label`` for hard labels, ``probability`` for
    probabilities/scores, ``probability_matrix`` for row-normalised class
    probabilities, and ``value`` for regression predictions. Binary
    probabilities may be one-dimensional; multiclass probabilities must be a
    row-per-class matrix.
    """

    try:
        task = task_type if isinstance(task_type, TaskType) else TaskType(task_type)
    except ValueError as exc:
        raise PredictionValidationError(f"unsupported task type: {task_type!r}") from exc
    if prediction_kind not in {"label", "probability", "probability_matrix", "value"}:
        raise PredictionValidationError(f"unsupported prediction kind: {prediction_kind!r}")
    if labels is not None and label_domain is not None:
        raise PredictionValidationError("provide only one of labels and label_domain")
    labels = label_domain if label_domain is not None else labels

    try:
        raw = np.asarray(predictions)
    except (TypeError, ValueError) as exc:
        raise PredictionValidationError("predictions must be a rectangular array") from exc
    valid_dimensions = {
        "label": (1,),
        "value": (1,),
        "probability": (1, 2),
        "probability_matrix": (2,),
    }
    if raw.ndim not in valid_dimensions[prediction_kind]:
        dimensionality = (
            "one-dimensional"
            if prediction_kind not in {"probability"}
            else "one- or two-dimensional"
        )
        raise PredictionValidationError(
            f"{prediction_kind} predictions must be {dimensionality}; received shape {raw.shape}"
        )
    if raw.ndim == 0:
        raise PredictionValidationError("predictions must contain one value per row")
    if expected_size is None:
        expected_size = len(_as_target(y_true, None))
    target = _as_target(y_true, expected_size)

    if task is TaskType.REGRESSION:
        if prediction_kind != "value":
            raise PredictionValidationError(
                "regression predictions must use prediction_kind='value'"
            )
        values = _numeric(raw, "regression")
        return PredictionValidationResult(values, task, prediction_kind)

    domain = _domain(target, labels)
    if prediction_kind == "label":
        if len(raw) != expected_size:
            raise PredictionValidationError("classification predictions have the wrong row count")
        unknown = [value for value in raw.tolist() if not any(value == known for known in domain)]
        if unknown:
            raise PredictionValidationError(
                f"predicted labels outside the label domain: {unknown!r}"
            )
        return PredictionValidationResult(raw, task, prediction_kind, domain)

    try:
        numeric = raw.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise PredictionValidationError(
            "classification probability predictions must be numeric"
        ) from exc
    values = _numeric(numeric, "classification probability")
    if len(values) != expected_size:
        raise PredictionValidationError("classification probabilities have the wrong row count")
    if values.ndim == 1:
        if len(domain) != 2:
            raise PredictionValidationError(
                "one-dimensional probabilities require exactly two labels"
            )
    elif values.shape[1] != len(domain):
        raise PredictionValidationError(
            f"probability columns ({values.shape[1]}) do not match label domain ({len(domain)})"
        )
    if ((values < 0) | (values > 1)).any():
        raise PredictionValidationError("classification probabilities must be between zero and one")
    if values.ndim == 2 and not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise PredictionValidationError("multiclass probability rows must sum to one")
    return PredictionValidationResult(values, task, prediction_kind, domain)


def validate_classification_predictions(*args: Any, **kwargs: Any) -> PredictionValidationResult:
    """Convenience wrapper for classification validation."""

    kwargs["task_type"] = TaskType.CLASSIFICATION
    return validate_predictions(*args, **kwargs)


def validate_regression_predictions(*args: Any, **kwargs: Any) -> PredictionValidationResult:
    """Convenience wrapper for regression validation."""

    kwargs["task_type"] = TaskType.REGRESSION
    kwargs.setdefault("prediction_kind", "value")
    return validate_predictions(*args, **kwargs)


class PredictionValidator:
    """Reusable validator facade for a baseline or validation runner."""

    def __init__(self, task_type: TaskType, *, labels: Iterable[Any] | None = None) -> None:
        self.task_type = task_type
        self.labels = labels

    def validate(
        self,
        y_true: Iterable[Any],
        predictions: Any,
        *,
        prediction_kind: str = "label",
        labels: Iterable[Any] | None = None,
    ) -> PredictionValidationResult:
        return validate_predictions(
            y_true,
            predictions,
            self.task_type,
            prediction_kind=prediction_kind,
            labels=self.labels if labels is None else labels,
        )


def validate_submission_schema(
    submission: Any,
    *,
    expected_rows: int,
    expected_columns: Iterable[str],
    finite_columns: Iterable[str] = (),
    label_columns: Iterable[str] = (),
    label_domain: Iterable[Any] | None = None,
) -> pd.DataFrame:
    """Validate a tabular prediction artifact before it leaves the application."""

    if not isinstance(submission, pd.DataFrame):
        raise PredictionValidationError("submission must be a pandas DataFrame")
    if expected_rows < 1:
        raise PredictionValidationError("expected_rows must be positive")
    expected = list(expected_columns)
    actual = list(submission.columns)
    if actual != expected:
        raise PredictionValidationError(
            f"submission columns {actual!r} do not match expected columns {expected!r}"
        )
    if len(submission) != expected_rows:
        raise PredictionValidationError(
            f"submission row count {len(submission)} does not match expected {expected_rows}"
        )
    for column in finite_columns:
        if column not in submission.columns:
            raise PredictionValidationError(f"submission column {column!r} is missing")
        values = submission[column].to_numpy()
        try:
            numeric = values.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise PredictionValidationError(
                f"submission column {column!r} must be numeric"
            ) from exc
        if not np.isfinite(numeric).all():
            raise PredictionValidationError(
                f"submission column {column!r} must contain only finite values"
            )
    if label_domain is not None:
        domain = tuple(label_domain)
        for column in label_columns:
            if column not in submission.columns:
                raise PredictionValidationError(f"submission column {column!r} is missing")
            unknown = [
                value
                for value in submission[column].tolist()
                if not any(value == known for known in domain)
            ]
            if unknown:
                raise PredictionValidationError(
                    f"submission column {column!r} contains labels outside the label "
                    f"domain: {unknown!r}"
                )
    return submission


__all__ = [
    "PredictionValidationError",
    "PredictionValidationResult",
    "PredictionValidator",
    "validate_classification_predictions",
    "validate_predictions",
    "validate_regression_predictions",
    "validate_submission",
    "validate_submission_schema",
]


validate_submission = validate_submission_schema
