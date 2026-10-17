from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlquest.domain import (
    MetricDirection,
    SplitSpecification,
    SplitStrategy,
    TaskType,
    ValidationSpecification,
)
from mlquest.evaluation import BaselineRunner, BaselineTier


def _classification_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "numeric": [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6, 0.15, 0.85, 0.25, 0.75],
            "category": ["a", "a", "b", "b", "a", "b", "a", "b", "unseen", "b", "a", "b"],
        }
    )
    y = pd.Series([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], name="target")
    return X, y


def _validation(metric: str, direction: MetricDirection) -> ValidationSpecification:
    return ValidationSpecification(
        split=SplitSpecification(
            strategy=SplitStrategy.CROSS_VALIDATION,
            n_splits=3,
            random_seed=7,
            shuffle=True,
        ),
        primary_metric=metric,
        metric_direction=direction,
    )


def test_trivial_baseline_returns_fold_and_aggregate_contracts() -> None:
    X, y = _classification_data()
    result = BaselineRunner(n_threads=1).run(
        X,
        y,
        baseline=BaselineTier.TRIVIAL,
        task_type=TaskType.CLASSIFICATION,
        validation=_validation("accuracy", MetricDirection.MAXIMIZE),
    )

    assert result.baseline_tier == "trivial"
    assert len(result.fold_results) == 3
    assert result.aggregate_metrics["accuracy"] == result.primary_value
    assert all(
        fold.prediction_validation.prediction_kind == "label" for fold in result.fold_results
    )
    assert result.resource_metadata.thread_count == 1
    assert result.split_identity


def test_simple_sklearn_baseline_fits_preprocessing_inside_each_fold() -> None:
    X, y = _classification_data()
    folds = [
        (np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([8, 9, 10, 11])),
        (np.array([4, 5, 6, 7, 8, 9, 10, 11]), np.array([0, 1, 2, 3])),
    ]
    result = BaselineRunner(n_threads=1).run(
        X,
        y,
        baseline="simple",
        task_type="classification",
        primary_metric="roc_auc",
        metric_direction="maximize",
        folds=folds,
    )

    assert result.baseline_tier == "simple"
    assert len(result.fold_results) == 2
    assert all(0.0 <= fold.primary_value <= 1.0 for fold in result.fold_results)


def test_strong_lightgbm_baseline_is_cpu_controlled() -> None:
    X, y = _classification_data()
    result = BaselineRunner(n_threads=1).run(
        X,
        y,
        baseline=BaselineTier.STRONG,
        task_type=TaskType.CLASSIFICATION,
        validation=_validation("log_loss", MetricDirection.MINIMIZE),
        model_params={"n_estimators": 12, "num_leaves": 7},
    )

    assert result.baseline_tier == "strong"
    assert result.metric_direction is MetricDirection.MINIMIZE
    assert result.ranking_value == -result.primary_value
    assert result.resource_metadata.thread_count == 1


def test_baseline_rejects_a_metric_direction_that_changes_the_registry_contract() -> None:
    X, y = _classification_data()
    with pytest.raises(ValueError, match="metric direction"):
        BaselineRunner(n_threads=1).run(
            X,
            y,
            baseline=BaselineTier.TRIVIAL,
            task_type=TaskType.CLASSIFICATION,
            primary_metric="accuracy",
            metric_direction=MetricDirection.MINIMIZE,
            folds=[(np.arange(8), np.arange(8, 12))],
        )
