from __future__ import annotations

import numpy as np
import pytest

from mlquest.domain import MetricDirection, TaskType
from mlquest.evaluation.metrics import (
    DEFAULT_METRIC_REGISTRY,
    UnknownMetricError,
    evaluate_metric,
)
from mlquest.evaluation.validation import PredictionValidationError


def test_initial_registry_is_explicit_with_expected_directions() -> None:
    expected = {
        "accuracy": MetricDirection.MAXIMIZE,
        "balanced_accuracy": MetricDirection.MAXIMIZE,
        "f1": MetricDirection.MAXIMIZE,
        "precision": MetricDirection.MAXIMIZE,
        "recall": MetricDirection.MAXIMIZE,
        "roc_auc": MetricDirection.MAXIMIZE,
        "log_loss": MetricDirection.MINIMIZE,
        "mae": MetricDirection.MINIMIZE,
        "mse": MetricDirection.MINIMIZE,
        "rmse": MetricDirection.MINIMIZE,
        "r2": MetricDirection.MAXIMIZE,
    }

    assert set(DEFAULT_METRIC_REGISTRY.definitions) == set(expected)
    assert {name: DEFAULT_METRIC_REGISTRY.direction(name) for name in expected} == expected


def test_unknown_metrics_are_rejected_instead_of_guessed() -> None:
    with pytest.raises(UnknownMetricError):
        DEFAULT_METRIC_REGISTRY.get("made_up_metric")


def test_metrics_validate_prediction_contract_and_return_direction() -> None:
    accuracy = evaluate_metric("accuracy", [0, 1, 1], [0, 1, 0], TaskType.CLASSIFICATION)
    rmse = evaluate_metric("rmse", [1.0, 3.0], [2.0, 3.0], TaskType.REGRESSION)
    logloss = evaluate_metric(
        "log_loss",
        [0, 1],
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        TaskType.CLASSIFICATION,
    )

    assert accuracy.value == pytest.approx(2 / 3)
    assert accuracy.direction is MetricDirection.MAXIMIZE
    assert rmse.value == pytest.approx(1 / np.sqrt(2))
    assert rmse.direction is MetricDirection.MINIMIZE
    assert logloss.value > 0


def test_score_comparison_uses_metric_direction() -> None:
    better_accuracy = DEFAULT_METRIC_REGISTRY.score(
        "accuracy", [0, 1], [0, 1], TaskType.CLASSIFICATION
    )
    worse_accuracy = DEFAULT_METRIC_REGISTRY.score(
        "accuracy", [0, 1], [0, 0], TaskType.CLASSIFICATION
    )
    better_mae = DEFAULT_METRIC_REGISTRY.score("mae", [0.0, 1.0], [0.0, 1.0], TaskType.REGRESSION)
    worse_mae = DEFAULT_METRIC_REGISTRY.score("mae", [0.0, 1.0], [1.0, 1.0], TaskType.REGRESSION)

    assert better_accuracy.better_than(worse_accuracy)
    assert better_mae.better_than(worse_mae)


def test_log_loss_requires_a_probability_matrix() -> None:
    with pytest.raises(PredictionValidationError, match="one-dimensional"):
        evaluate_metric("log_loss", [0, 1], [0.1, 0.9], TaskType.CLASSIFICATION)
