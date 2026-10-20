"""Fold-safe, CPU-controlled baselines for tabular supervised tasks."""

from __future__ import annotations

import os
import resource
import time
from contextlib import nullcontext
from enum import StrEnum
from typing import Any, Iterable, Sequence, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlquest.domain import (
    MetricDirection,
    SplitSpecification,
    SplitStrategy,
    TaskSpecification,
    TaskType,
    ValidationSpecification,
)

from .metrics import DEFAULT_METRIC_REGISTRY
from .results import (
    AggregateEvaluationResult,
    FoldEvaluationResult,
    PredictionValidationSummary,
    ResourceMetadata,
    validate_predictions,
)
from .splits import create_split_plan

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - sklearn normally brings this dependency
    threadpool_limits = None


class BaselineTier(StrEnum):
    """The deliberately ordered baseline levels used in comparison reports."""

    TRIVIAL = "trivial"
    SIMPLE = "simple"
    STRONG = "strong"


class BaselineUnavailableError(RuntimeError):
    """Raised when an optional baseline implementation cannot be constructed."""


def _preprocessor(X: Any) -> ColumnTransformer:
    if isinstance(X, pd.DataFrame):
        numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical = [column for column in X.columns if column not in numeric]
        numeric_selector: list[str] | list[int] = numeric
        categorical_selector: list[str] | list[int] = categorical
    else:
        matrix = np.asarray(X)
        if matrix.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        numeric_selector = [
            index
            for index in range(matrix.shape[1])
            if np.issubdtype(matrix[:, index].dtype, np.number)
        ]
        categorical_selector = [
            index for index in range(matrix.shape[1]) if index not in numeric_selector
        ]
    transformers: list[tuple[str, Any, Any]] = []
    if numeric_selector:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_selector,
            )
        )
    if categorical_selector:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical_selector,
            )
        )
    if not transformers:
        raise ValueError("X must contain at least one feature column")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_baseline(
    tier: BaselineTier | str,
    task_type: TaskType | str,
    *,
    X: Any | None = None,
    random_seed: int = 0,
    n_threads: int = 1,
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build a named baseline pipeline without fitting it.

    The preprocessor is part of every tier so each fold gets an independently
    fitted transformer, including the trivial baseline.
    """

    resolved_tier = BaselineTier(tier)
    task = TaskType(task_type)
    if n_threads < 1:
        raise ValueError("n_threads must be positive")
    params = dict(model_params or {})
    if resolved_tier is BaselineTier.TRIVIAL:
        model: Any = (
            DummyClassifier(strategy="prior")
            if task is TaskType.CLASSIFICATION
            else DummyRegressor(strategy="mean")
        )
    elif resolved_tier is BaselineTier.SIMPLE:
        if task is TaskType.CLASSIFICATION:
            defaults: dict[str, Any] = {
                "max_iter": 1000,
                "random_state": random_seed,
            }
            defaults.update(params)
            model = LogisticRegression(**defaults)
        else:
            model = Ridge(**params)
    else:
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:  # pragma: no cover - dependency is in project requirements
            raise BaselineUnavailableError("LightGBM is required for the strong baseline") from exc
        common: dict[str, Any] = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": random_seed,
            "n_jobs": n_threads,
            "verbosity": -1,
        }
        common.update(params)
        model = (
            LGBMClassifier(**common) if task is TaskType.CLASSIFICATION else LGBMRegressor(**common)
        )
    preprocessor = _preprocessor(X) if X is not None else "passthrough"
    return Pipeline([("preprocess", preprocessor), ("model", model)])


class BaselineRunner:
    """Evaluate trivial, simple, or strong baselines using fold-local fitting."""

    def __init__(self, *, n_threads: int = 1, random_seed: int = 0) -> None:
        if n_threads < 1:
            raise ValueError("n_threads must be positive")
        self.n_threads = n_threads
        self.random_seed = random_seed

    def run(
        self,
        X: Any,
        y: Any,
        *,
        baseline: BaselineTier | str = BaselineTier.TRIVIAL,
        task: TaskSpecification | None = None,
        task_type: TaskType | str | None = None,
        validation: ValidationSpecification | SplitSpecification | None = None,
        primary_metric: str | None = None,
        metric_direction: MetricDirection | str | None = None,
        secondary_metrics: Sequence[str] = (),
        folds: Iterable[tuple[Iterable[int], Iterable[int]]] | None = None,
        groups: Any | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> AggregateEvaluationResult:
        """Run one baseline over all folds and return fold and aggregate results."""

        resolved_task_type, metric, direction, secondary = self._resolve_contract(
            task=task,
            task_type=task_type,
            validation=validation,
            primary_metric=primary_metric,
            metric_direction=metric_direction,
            secondary_metrics=secondary_metrics,
        )
        X_matrix = self._validate_features(X)
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(X_matrix):
            raise ValueError("y must be a one-dimensional vector matching X rows")
        fold_source, split_identity = self._folds(
            X_matrix,
            target,
            validation,
            folds=folds,
            groups=groups,
            task_type=resolved_task_type,
        )
        resolved_folds = list(fold_source)
        if not resolved_folds:
            raise ValueError("at least one validation fold is required")
        fold_results: list[FoldEvaluationResult] = []
        all_metrics: dict[str, list[float]] = {metric: [] for metric in [metric, *secondary]}
        total_started = time.perf_counter()
        total_cpu_started = time.process_time()
        for fold_index, (train_index, valid_index) in enumerate(resolved_folds):
            train_index = np.asarray(list(train_index), dtype=int)
            valid_index = np.asarray(list(valid_index), dtype=int)
            self._check_indices(train_index, valid_index, len(target))
            pipeline = self._pipeline(
                baseline,
                resolved_task_type,
                X_matrix,
                model_params=model_params,
            )
            started = time.perf_counter()
            cpu_started = time.process_time()
            with self._thread_limit():
                pipeline.fit(_take_rows(X_matrix, train_index), target[train_index])
                predictions = cast(np.ndarray, pipeline.predict(_take_rows(X_matrix, valid_index)))
                labels = np.asarray(
                    getattr(pipeline[-1], "classes_", np.unique(target[train_index]))
                )
                prediction_validation = validate_predictions(
                    target[valid_index],
                    predictions,
                    resolved_task_type,
                    prediction_kind=(
                        "value" if resolved_task_type is TaskType.REGRESSION else "label"
                    ),
                    expected_size=len(valid_index),
                    labels=labels if resolved_task_type is TaskType.CLASSIFICATION else None,
                )
                prediction_summary = PredictionValidationSummary(
                    n_rows=prediction_validation.n_rows,
                    shape=prediction_validation.shape,
                    prediction_kind=prediction_validation.prediction_kind,
                    label_domain=list(prediction_validation.label_domain),
                )
                probabilities = None
                if resolved_task_type is TaskType.CLASSIFICATION and hasattr(
                    pipeline, "predict_proba"
                ):
                    probabilities = pipeline.predict_proba(_take_rows(X_matrix, valid_index))
                    validate_predictions(
                        target[valid_index],
                        probabilities,
                        resolved_task_type,
                        prediction_kind="probability",
                        expected_size=len(valid_index),
                        labels=labels,
                    )
                metrics = self._metrics(
                    metric_names=[metric, *secondary],
                    task_type=resolved_task_type,
                    y_true=target[valid_index],
                    predictions=predictions,
                    probabilities=probabilities,
                    labels=labels,
                )
            fold_runtime = time.perf_counter() - started
            fold_cpu = time.process_time() - cpu_started
            fold_resource = self._resource_metadata(fold_runtime, fold_cpu)
            fold_results.append(
                FoldEvaluationResult(
                    fold_index=fold_index,
                    train_size=len(train_index),
                    validation_size=len(valid_index),
                    metrics=metrics,
                    primary_metric=metric,
                    primary_value=metrics[metric],
                    prediction_validation=prediction_summary,
                    resource_metadata=fold_resource,
                )
            )
            for name, value in metrics.items():
                all_metrics[name].append(value)
        aggregate = {name: float(np.mean(values)) for name, values in all_metrics.items()}
        stddev = {name: float(np.std(values, ddof=0)) for name, values in all_metrics.items()}
        total_runtime = time.perf_counter() - total_started
        total_cpu = time.process_time() - total_cpu_started
        primary_value = aggregate[metric]
        return AggregateEvaluationResult(
            baseline_name=BaselineTier(baseline).value,
            baseline_tier=BaselineTier(baseline).value,
            task_type=resolved_task_type,
            primary_metric=metric,
            metric_direction=direction,
            fold_results=fold_results,
            aggregate_metrics=aggregate,
            metric_stddev=stddev,
            primary_value=primary_value,
            ranking_value=(
                primary_value if direction is MetricDirection.MAXIMIZE else -primary_value
            ),
            resource_metadata=self._resource_metadata(total_runtime, total_cpu),
            comparison_context_id=(
                validation.comparison_context_id
                if isinstance(validation, ValidationSpecification)
                else None
            ),
            split_identity=split_identity,
        )

    evaluate = run

    def _pipeline(
        self,
        baseline: BaselineTier | str,
        task_type: TaskType,
        X: Any,
        *,
        model_params: dict[str, Any] | None,
    ) -> Pipeline:
        pipeline = build_baseline(
            baseline,
            task_type,
            X=X,
            random_seed=self.random_seed,
            n_threads=self.n_threads,
            model_params=model_params,
        )
        return pipeline

    def _resolve_contract(
        self,
        *,
        task: TaskSpecification | None,
        task_type: TaskType | str | None,
        validation: ValidationSpecification | SplitSpecification | None,
        primary_metric: str | None,
        metric_direction: MetricDirection | str | None,
        secondary_metrics: Sequence[str],
    ) -> tuple[TaskType, str, MetricDirection, list[str]]:
        if task is not None:
            task_type = task.task_type
            primary_metric = primary_metric or task.primary_metric
            metric_direction = metric_direction or task.metric_direction
            if not secondary_metrics:
                secondary_metrics = task.secondary_metrics
        if task_type is None:
            raise ValueError("task_type or task is required")
        if isinstance(validation, ValidationSpecification):
            primary_metric = primary_metric or validation.primary_metric
            metric_direction = metric_direction or validation.metric_direction
            if not secondary_metrics:
                secondary_metrics = validation.secondary_metrics
        if not primary_metric:
            raise ValueError("primary_metric or validation.primary_metric is required")
        primary_definition = DEFAULT_METRIC_REGISTRY.get(primary_metric)
        direction = (
            MetricDirection(metric_direction)
            if metric_direction is not None
            else primary_definition.direction
        )
        if direction is not primary_definition.direction:
            raise ValueError(
                f"metric direction for {primary_definition.name!r} must be "
                f"{primary_definition.direction.value}"
            )
        metrics = [str(primary_metric), *(str(value) for value in secondary_metrics)]
        return TaskType(task_type), str(primary_metric), direction, list(dict.fromkeys(metrics[1:]))

    def _folds(
        self,
        X: Any,
        y: np.ndarray,
        validation: ValidationSpecification | SplitSpecification | None,
        *,
        folds: Iterable[tuple[Iterable[int], Iterable[int]]] | None,
        groups: Any | None,
        task_type: TaskType,
    ) -> tuple[Iterable[tuple[Iterable[int], Iterable[int]]], str | None]:
        if folds is not None:
            return folds, None
        split = validation.split if isinstance(validation, ValidationSpecification) else validation
        split = split or SplitSpecification(
            strategy=(
                SplitStrategy.STRATIFIED
                if task_type is TaskType.CLASSIFICATION
                else SplitStrategy.RANDOM
            ),
            n_splits=5,
            random_seed=self.random_seed,
        )
        plan = create_split_plan(X, split, y=y, groups=groups, task_type=task_type)
        return (
            [
                (
                    np.asarray(fold.train_indices, dtype=int),
                    np.asarray(fold.validation_indices, dtype=int),
                )
                for fold in plan.folds
            ],
            plan.identity,
        )

    @staticmethod
    def _validate_features(X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            if len(X.columns) == 0:
                raise ValueError("X must contain at least one feature column")
            return X.reset_index(drop=True)
        matrix = np.asarray(X)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("X must be a two-dimensional feature matrix with columns")
        return matrix

    @staticmethod
    def _check_indices(train: np.ndarray, valid: np.ndarray, size: int) -> None:
        if len(train) == 0 or len(valid) == 0:
            raise ValueError("validation folds must contain training and validation rows")
        if np.any(train < 0) or np.any(valid < 0) or np.any(train >= size) or np.any(valid >= size):
            raise ValueError("fold indices are outside the dataset")
        if np.intersect1d(train, valid).size:
            raise ValueError("training and validation rows must be disjoint")

    def _metrics(
        self,
        *,
        metric_names: Sequence[str],
        task_type: TaskType,
        y_true: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray | None,
        labels: np.ndarray,
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for name in metric_names:
            definition = DEFAULT_METRIC_REGISTRY.get(name)
            metric_predictions = (
                probabilities
                if definition.prediction_kind in {"probability", "probability_matrix"}
                else predictions
            )
            if metric_predictions is None:
                raise ValueError(f"metric {name} requires probability predictions")
            values[name] = DEFAULT_METRIC_REGISTRY.evaluate(
                name,
                y_true,
                metric_predictions,
                task_type,
                labels=labels if task_type is TaskType.CLASSIFICATION else None,
            )
        return values

    def _thread_limit(self) -> Any:
        return (
            threadpool_limits(limits=self.n_threads)
            if threadpool_limits is not None
            else nullcontext()
        )

    def _resource_metadata(
        self,
        runtime_seconds: float,
        cpu_time_seconds: float,
    ) -> ResourceMetadata:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return ResourceMetadata(
            runtime_seconds=max(0.0, runtime_seconds),
            cpu_time_seconds=max(0.0, cpu_time_seconds),
            peak_memory_mb=max(0.0, float(usage.ru_maxrss) / 1024.0),
            thread_count=self.n_threads,
            cpu_count=max(1, os.cpu_count() or 1),
        )


def evaluate_baseline(
    X: Any,
    y: Any,
    *,
    baseline: BaselineTier | str = BaselineTier.TRIVIAL,
    task: TaskSpecification | None = None,
    task_type: TaskType | str | None = None,
    validation: ValidationSpecification | SplitSpecification | None = None,
    primary_metric: str | None = None,
    metric_direction: MetricDirection | str | None = None,
    secondary_metrics: Sequence[str] = (),
    folds: Iterable[tuple[Iterable[int], Iterable[int]]] | None = None,
    groups: Any | None = None,
    n_threads: int = 1,
    random_seed: int = 0,
    model_params: dict[str, Any] | None = None,
) -> AggregateEvaluationResult:
    """Convenience wrapper around :class:`BaselineRunner`."""

    return BaselineRunner(n_threads=n_threads, random_seed=random_seed).run(
        X,
        y,
        baseline=baseline,
        task=task,
        task_type=task_type,
        validation=validation,
        primary_metric=primary_metric,
        metric_direction=metric_direction,
        secondary_metrics=secondary_metrics,
        folds=folds,
        groups=groups,
        model_params=model_params,
    )


def _take_rows(X: Any, indices: np.ndarray) -> Any:
    return X.iloc[indices] if isinstance(X, pd.DataFrame) else X[indices]
