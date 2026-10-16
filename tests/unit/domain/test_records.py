import pytest
from pydantic import ValidationError

from mlquest.domain import (
    ApprovalDecision,
    ApprovalDecisionType,
    ApprovalRequest,
    BudgetConfiguration,
    EvidenceClaim,
    EvidenceClaimType,
    EvidenceStrength,
    ExternalScoreObservation,
    ExperimentSpecification,
    Hypothesis,
    HypothesisStatus,
    MetricDirection,
    MetricResult,
    ResearchRecord,
    SourceType,
    SplitSpecification,
    SplitStrategy,
    TaskSpecification,
    TaskType,
    ValidationSpecification,
)


def test_task_and_experiment_specs_reject_invalid_values() -> None:
    with pytest.raises(ValidationError):
        TaskSpecification(
            name="task",
            problem_statement="predict a target",
            task_type=TaskType.CLASSIFICATION,
            target_column="target",
            primary_metric="accuracy",
            metric_direction=MetricDirection.MAXIMIZE,
        )
    with pytest.raises(ValidationError):
        SplitSpecification(strategy=SplitStrategy.GROUP)


def test_budget_and_metric_boundaries_are_enforced() -> None:
    with pytest.raises(ValidationError):
        BudgetConfiguration(session_seconds=60, model_execution_seconds=50, finalisation_reserve_seconds=20)
    with pytest.raises(ValidationError):
        MetricResult(
            metric_name="accuracy",
            value=float("nan"),
            direction=MetricDirection.MAXIMIZE,
            run_attempt_id="run-1",
            comparison_context_id="context-1",
        )


def test_record_serialization_is_stable_for_same_record() -> None:
    validation = ValidationSpecification(
        split=SplitSpecification(strategy=SplitStrategy.RANDOM, test_size=0.2, random_seed=1),
        primary_metric="rmse",
        metric_direction=MetricDirection.MINIMIZE,
    )
    experiment = ExperimentSpecification(name="baseline", model_family="linear", validation=validation)
    assert experiment.model_dump_json() == experiment.model_dump_json()
    assert experiment.canonical_json() == experiment.canonical_json()


def test_approval_research_and_external_evidence_records_link_by_id() -> None:
    request = ApprovalRequest(
        phase="experiment",
        requested_action="run baseline",
        rationale="establish a comparison point",
    )
    request.resolve(
        ApprovalDecision(
            request_id=request.id,
            decision=ApprovalDecisionType.APPROVE,
            decided_by="researcher",
        )
    )
    source = ResearchRecord(
        title="A useful paper",
        url="https://example.test/paper",
        source_type=SourceType.PAPER,
        accessed_at="2026-01-01T00:00:00Z",
    )
    hypothesis = Hypothesis(
        statement="feature X improves the metric",
        rationale="the source describes a matching data pattern",
        source_record_ids=[source.id],
        status=HypothesisStatus.TESTED,
    )
    claim = EvidenceClaim(
        claim="the observed gain is supported by a measured run",
        claim_type=EvidenceClaimType.MEASURED_RESULT,
        strength=EvidenceStrength.PROVISIONAL,
        evidence_record_ids=[hypothesis.id],
    )
    score = ExternalScoreObservation(
        source="manual leaderboard entry",
        metric_name="accuracy",
        score=0.81,
        direction=MetricDirection.MAXIMIZE,
        observed_at="2026-01-02T00:00:00Z",
        reported_by="researcher",
    )
    assert request.decision is ApprovalDecisionType.APPROVE
    assert claim.evidence_record_ids == [hypothesis.id]
    assert score.reported_by == "researcher"
