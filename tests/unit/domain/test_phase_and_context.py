import pytest

from mlquest.domain import (
    ComparisonContext,
    MetricDirection,
    PhaseName,
    PhaseState,
    PhaseStatus,
    PhaseTransitionError,
    SplitSpecification,
    SplitStrategy,
)


def test_phase_state_rejects_illegal_transition() -> None:
    state = PhaseState(phase=PhaseName.INTAKE)
    state.transition_to(PhaseStatus.RUNNING)
    state.transition_to(PhaseStatus.COMPLETED)
    with pytest.raises(PhaseTransitionError):
        state.transition_to(PhaseStatus.RUNNING)


def test_comparison_context_identity_changes_for_comparison_inputs() -> None:
    split = SplitSpecification(strategy=SplitStrategy.STRATIFIED, test_size=0.2, random_seed=7)
    base = ComparisonContext(
        dataset_id="dataset-v1",
        split_specification=split,
        metric_name="accuracy",
        metric_direction=MetricDirection.MAXIMIZE,
        target_column="label",
        target_definition="the observed class",
    )
    changed = base.model_copy(update={"target_definition": "the adjudicated class"})
    assert base.identity != changed.identity
    equivalent = ComparisonContext(
        dataset_id="dataset-v1",
        split_specification=SplitSpecification(
            strategy=SplitStrategy.STRATIFIED, test_size=0.2, random_seed=7
        ),
        metric_name="accuracy",
        metric_direction=MetricDirection.MAXIMIZE,
        target_column="label",
        target_definition="the observed class",
    )
    assert base.identity == equivalent.identity
