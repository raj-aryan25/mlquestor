from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlquest.domain import (
    EvidenceStrength,
    MetricDirection,
    SplitSpecification,
    SplitStrategy,
    TaskSpecification,
    TaskType,
)
from mlquest.planning import (
    AllowlistedExperimentPlanner,
    CandidateProposal,
    EvidenceAssessment,
    ExperimentResult,
    ExperimentStatus,
    SerialExperimentScheduler,
    assess_evidence_strength,
)


def _classification_task() -> TaskSpecification:
    return TaskSpecification(
        name="tiny",
        problem_statement="predict the target",
        task_type=TaskType.CLASSIFICATION,
        target_column="target",
        primary_metric="accuracy",
        metric_direction=MetricDirection.MAXIMIZE,
        source_artifact_ids=["dataset-1"],
    )


def test_planner_rejects_task_metric_and_split_mismatch() -> None:
    task = TaskSpecification(
        name="regression",
        problem_statement="predict a value",
        task_type=TaskType.REGRESSION,
        target_column="target",
        primary_metric="accuracy",
        metric_direction=MetricDirection.MAXIMIZE,
        source_artifact_ids=["dataset-1"],
    )
    result = AllowlistedExperimentPlanner().plan(
        {"model_family": "linear", "rationale": "safe"},
        task=task,
        validation=SplitSpecification(strategy=SplitStrategy.STRATIFIED, test_size=0.2),
        use_fallback=False,
    )
    assert not result.accepted
    assert {item.code for item in result.diagnostics} >= {
        "metric_task_mismatch",
        "split_task_mismatch",
    }


def test_malformed_and_forbidden_proposals_use_safe_fallback() -> None:
    planner = AllowlistedExperimentPlanner()
    result = planner.plan(
        {
            "model_family": "lightgbm",
            "rationale": "unsafe",
            "parameters": {"shell": "rm -rf /"},
        },
        task=_classification_task(),
    )
    assert result.accepted
    assert result.used_fallback
    assert result.specification is not None
    assert result.specification.model_family == "logistic"
    assert any(item.code == "malformed_proposal" for item in result.diagnostics)
    assert "shell" not in result.specification.model_config_values


def test_fallback_proposal_is_deterministic_and_context_is_preserved() -> None:
    planner = AllowlistedExperimentPlanner()
    task = _classification_task()
    first = planner.fallback_proposal(task)
    second = planner.fallback_proposal(task)
    assert first.canonical_json() == second.canonical_json()
    result = planner.plan(first, task=task, comparison_context_id="context-1")
    assert result.specification is not None
    assert result.specification.comparison_context_id == "context-1"
    assert result.specification.validation.comparison_context_id == "context-1"


def test_serial_scheduler_ranks_value_cost_novelty_and_risk_and_limits_count() -> None:
    proposal_a = {"model_family": "logistic", "rationale": "a"}
    proposal_b = {"model_family": "logistic", "rationale": "b"}
    candidates = [
        CandidateProposal(
            proposal=proposal_a,
            expected_value=0.9,
            estimated_cost=1,
            novelty=0.6,
            risk=0.1,
        ),
        CandidateProposal(
            proposal=proposal_b,
            expected_value=0.8,
            estimated_cost=100,
            novelty=0.9,
            risk=0.0,
        ),
    ]
    result = SerialExperimentScheduler(max_count=1).schedule(candidates)
    assert len(result.selected) == 1
    assert result.selected[0].candidate.proposal.rationale == "a"
    assert result.selected[0].rank == 1


def test_one_score_cannot_be_confirmed_evidence() -> None:
    evidence = assess_evidence_strength([0.81], comparison_context_id="context-1")
    assert evidence.strength is EvidenceStrength.EXPLORATORY
    result = ExperimentResult(
        experiment_id="experiment-1",
        status=ExperimentStatus.SUCCEEDED,
        comparison_context_id="context-1",
        scores=[0.81],
        evidence=evidence,
    )
    assert result.evidence.strength is EvidenceStrength.EXPLORATORY
    with pytest.raises(ValidationError):
        EvidenceAssessment(
            strength=EvidenceStrength.CONFIRMED,
            score_count=1,
            successful_run_count=1,
            comparison_context_id="context-1",
        )
