from __future__ import annotations

import pytest

from mlquest.domain import MetricDirection, TaskSpecification, TaskType
from mlquest.planning import (
    AllowlistedExperimentPlanner,
    OptunaExperimentSearch,
    SearchParameter,
)


def _specification():
    task = TaskSpecification(
        name="tiny",
        problem_statement="predict the target",
        task_type=TaskType.CLASSIFICATION,
        target_column="target",
        primary_metric="accuracy",
        metric_direction=MetricDirection.MAXIMIZE,
        source_artifact_ids=["dataset-1"],
    )
    result = AllowlistedExperimentPlanner().plan(
        {"model_family": "logistic", "rationale": "bounded search"}, task=task
    )
    assert result.specification is not None
    return result.specification


def test_optuna_search_is_bounded_and_deterministic() -> None:
    parameters = [
        SearchParameter(name="C", kind="float", low=0.1, high=2.0),
        SearchParameter(name="solver", kind="categorical", choices=("lbfgs", "liblinear")),
    ]

    def evaluate(specification):
        return float(specification.model_config_values["C"])

    first = OptunaExperimentSearch(seed=7, max_trials=4).search(
        _specification(), parameters, evaluate
    )
    second = OptunaExperimentSearch(seed=7, max_trials=4).search(
        _specification(), parameters, evaluate
    )

    assert len(first.trials) == 4
    assert first.trials == second.trials
    assert first.best_specification.model_config_values["C"] == max(
        trial.parameters["C"] for trial in first.trials
    )


def test_search_parameter_rejects_unsafe_or_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        SearchParameter(name="shell", kind="categorical", choices=("x",))
    with pytest.raises(ValueError, match="ordered"):
        SearchParameter(name="depth", kind="int", low=3, high=3)
    with pytest.raises(ValueError, match="choices"):
        SearchParameter(name="solver", kind="categorical")
