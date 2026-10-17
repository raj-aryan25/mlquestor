from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlquest.domain import (
    ExperimentSpecificationProposal,
    MetricDirection,
    ResearchQuery,
    TaskInterpretation,
    TaskType,
)


def test_planning_models_accept_structured_advisory_values() -> None:
    interpretation = TaskInterpretation(
        task_type=TaskType.CLASSIFICATION,
        target_column="target",
        primary_metric="roc_auc",
        metric_direction=MetricDirection.MAXIMIZE,
        summary="A binary tabular classification task.",
    )
    proposal = ExperimentSpecificationProposal(
        model_family="lightgbm",
        parameters={"num_leaves": 16, "learning_rate": 0.05},
        random_seeds=[42],
        rationale="The baseline supports a controlled tree-depth test.",
    )
    assert interpretation.target_column == "target"
    assert proposal.parameters["num_leaves"] == 16


def test_planning_models_reject_unknown_or_execution_fields() -> None:
    with pytest.raises(ValidationError):
        ResearchQuery(query="tabular imputation", rationale="find comparable methods", command="rm")
    with pytest.raises(ValidationError):
        ExperimentSpecificationProposal(
            model_family="linear",
            rationale="compare a simple baseline",
            command=["sh", "-c", "echo unsafe"],
        )
    with pytest.raises(ValidationError):
        ExperimentSpecificationProposal(
            model_family="linear",
            rationale="compare a simple baseline",
            parameters={"shell": "echo unsafe"},
        )
