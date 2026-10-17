"""The explicit, allow-listed metric registry used for local evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from mlquest.domain import MetricDirection, TaskType

from .validation import validate_predictions


class UnknownMetricError(ValueError):
    """Raised when a metric is not in the explicit initial registry."""


class MetricTaskError(ValueError):
    """Raised when a metric is used for an unsupported task type."""


MetricFunction = Callable[[np.ndarray, np.ndarray, tuple[Any, ...]], float]


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    task_type: TaskType
    direction: MetricDirection
    prediction_kind: str
    function: MetricFunction

    @property
    def higher_is_better(self) -> bool:
        return self.direction is MetricDirection.MAXIMIZE


@dataclass(frozen=True)
class MetricScore:
    metric_name: str
    value: float
    direction: MetricDirection

    @property
    def higher_is_better(self) -> bool:
        return self.direction is MetricDirection.MAXIMIZE

    def better_than(self, other: "MetricScore") -> bool:
        if self.metric_name != other.metric_name or self.direction is not other.direction:
            raise ValueError("metric scores must use the same metric and direction")
        return self.value > other.value if self.higher_is_better else self.value < other.value


def _classification_average(y_true: np.ndarray) -> Literal["binary", "weighted"]:
    return "binary" if len(np.unique(y_true)) == 2 else "weighted"


def _accuracy(y: np.ndarray, p: np.ndarray, _: tuple[Any, ...]) -> float:
    return float(accuracy_score(y, p))


def _balanced_accuracy(y: np.ndarray, p: np.ndarray, _: tuple[Any, ...]) -> float:
    return float(balanced_accuracy_score(y, p))


def _f1(y: np.ndarray, p: np.ndarray, _: tuple[Any, ...]) -> float:
    return float(f1_score(y, p, average=_classification_average(y), zero_division=0))


def _precision(y: np.ndarray, p: np.ndarray, _: tuple[Any, ...]) -> float:
    return float(precision_score(y, p, average=_classification_average(y), zero_division=0))


def _recall(y: np.ndarray, p: np.ndarray, _: tuple[Any, ...]) -> float:
    return float(recall_score(y, p, average=_classification_average(y), zero_division=0))


def _roc_auc(y: np.ndarray, p: np.ndarray, _: tuple[Any, ...]) -> float:
    if p.ndim == 1:
        scores = p
    elif len(np.unique(y)) == 2:
        scores = p[:, 1]
    else:
        scores = p
    return float(roc_auc_score(y, scores))


def _log_loss(y: np.ndarray, p: np.ndarray, labels: tuple[Any, ...]) -> float:
    return float(log_loss(y, p, labels=list(labels)))


def _mae(y: np.ndarray, p: np.ndarray, _: tuple[Any, ...]) -> float:
    return float(mean_absolute_error(y, p))


def _mse(y: np.ndarray, p: np.ndarray, _: tuple[Any, ...]) -> float:
    return float(mean_squared_error(y, p))


def _rmse(y: np.ndarray, p: np.ndarray, _: tuple[Any, ...]) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def _r2(y: np.ndarray, p: np.ndarray, _: tuple[Any, ...]) -> float:
    return float(r2_score(y, p))


def _classification(
    name: str, direction: MetricDirection, kind: str, function: MetricFunction
) -> MetricDefinition:
    return MetricDefinition(name, TaskType.CLASSIFICATION, direction, kind, function)


def _regression(name: str, function: MetricFunction) -> MetricDefinition:
    direction = MetricDirection.MINIMIZE if name != "r2" else MetricDirection.MAXIMIZE
    return MetricDefinition(name, TaskType.REGRESSION, direction, "value", function)


_INITIAL_DEFINITIONS = (
    _classification("accuracy", MetricDirection.MAXIMIZE, "label", _accuracy),
    _classification("balanced_accuracy", MetricDirection.MAXIMIZE, "label", _balanced_accuracy),
    _classification("f1", MetricDirection.MAXIMIZE, "label", _f1),
    _classification("precision", MetricDirection.MAXIMIZE, "label", _precision),
    _classification("recall", MetricDirection.MAXIMIZE, "label", _recall),
    _classification("roc_auc", MetricDirection.MAXIMIZE, "probability", _roc_auc),
    _classification("log_loss", MetricDirection.MINIMIZE, "probability_matrix", _log_loss),
    _regression("mae", _mae),
    _regression("mse", _mse),
    _regression("rmse", _rmse),
    _regression("r2", _r2),
)


class MetricRegistry:
    """An immutable allow-list of metrics and their scoring contracts."""

    def __init__(self, definitions: Iterable[MetricDefinition] = _INITIAL_DEFINITIONS) -> None:
        values = tuple(definitions)
        names = [definition.name for definition in values]
        if len(names) != len(set(names)):
            raise ValueError("metric names must be unique")
        self._definitions: Mapping[str, MetricDefinition] = {
            name: definition for name, definition in zip(names, values)
        }

    @property
    def definitions(self) -> Mapping[str, MetricDefinition]:
        return self._definitions

    def get(self, name: str) -> MetricDefinition:
        key = name.strip().lower() if isinstance(name, str) else ""
        try:
            return self._definitions[key]
        except KeyError as exc:
            choices = ", ".join(sorted(self._definitions))
            raise UnknownMetricError(f"unknown metric {name!r}; choose from {choices}") from exc

    def direction(self, name: str) -> MetricDirection:
        return self.get(name).direction

    def score(
        self,
        name: str,
        y_true: Iterable[Any],
        predictions: Any,
        task_type: TaskType,
        *,
        labels: Iterable[Any] | None = None,
    ) -> MetricScore:
        definition = self.get(name)
        task = task_type if isinstance(task_type, TaskType) else TaskType(task_type)
        if task is not definition.task_type:
            raise MetricTaskError(
                f"metric {definition.name!r} is for {definition.task_type.value}, not {task.value}"
            )
        validated = validate_predictions(
            y_true,
            predictions,
            task,
            prediction_kind=definition.prediction_kind,
            labels=labels,
        )
        target = np.asarray(y_true)
        value = definition.function(target, validated.values, validated.label_domain)
        if not np.isfinite(value):
            raise ValueError(f"metric {definition.name!r} produced a non-finite value")
        return MetricScore(definition.name, value, definition.direction)

    def evaluate(self, *args: Any, **kwargs: Any) -> float:
        return self.score(*args, **kwargs).value


DEFAULT_METRIC_REGISTRY = MetricRegistry()


def evaluate_metric(
    name: str,
    y_true: Iterable[Any],
    predictions: Any,
    task_type: TaskType,
    *,
    labels: Iterable[Any] | None = None,
    registry: MetricRegistry = DEFAULT_METRIC_REGISTRY,
) -> MetricScore:
    """Evaluate one registered metric and return its value plus direction."""

    return registry.score(name, y_true, predictions, task_type, labels=labels)


__all__ = [
    "DEFAULT_METRIC_REGISTRY",
    "MetricDefinition",
    "MetricRegistry",
    "MetricScore",
    "UnknownMetricError",
    "evaluate_metric",
]
