from __future__ import annotations

from mlquest.budgets import BudgetManager
from mlquest.domain import (
    BudgetConfiguration,
    EvidenceStrength,
    MetricDirection,
    TaskSpecification,
)
from mlquest.persistence import LocalLedger
from mlquest.planning import (
    ExperimentLoop,
    ExperimentResult,
    ExperimentStatus,
    LoopMode,
    LoopStatus,
    assess_evidence_strength,
)


def _task() -> TaskSpecification:
    return TaskSpecification(
        name="tiny",
        problem_statement="predict the target",
        task_type="classification",
        target_column="target",
        primary_metric="accuracy",
        metric_direction=MetricDirection.MAXIMIZE,
        source_artifact_ids=["dataset-1"],
    )


class _DockerFixture:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def execute(self, specification):
        self.calls += 1
        if self.fail:
            raise RuntimeError("fixture Docker failure")
        context = specification.comparison_context_id
        scores = [0.8]
        return ExperimentResult(
            experiment_id=specification.id,
            status=ExperimentStatus.SUCCEEDED,
            comparison_context_id=context,
            scores=scores,
            evidence=assess_evidence_strength(scores, comparison_context_id=context),
        )


def _loop(ledger: LocalLedger, executor: _DockerFixture) -> ExperimentLoop:
    manager = BudgetManager(
        "session-1",
        BudgetConfiguration(
            session_seconds=600,
            model_execution_seconds=300,
            finalisation_reserve_seconds=30,
        ),
    )
    return ExperimentLoop(
        ledger=ledger,
        session_id="session-1",
        task=_task(),
        budget_manager=manager,
        executor=executor,
        dataset_size=100,
    )


def test_autonomous_loop_executes_fallback_and_records_phase_and_result() -> None:
    with LocalLedger(":memory:") as ledger:
        executor = _DockerFixture()
        result = _loop(ledger, executor).run([], mode=LoopMode.AUTONOMOUS)

        assert result.status is LoopStatus.COMPLETED
        assert result.phase_status.value == "completed"
        assert executor.calls == 1
        assert result.results[0].evidence.strength is EvidenceStrength.EXPLORATORY
        assert ledger.list_records("experiments", session_id="session-1")
        assert any(
            event.event_type == "experiment.result_recorded"
            for event in ledger.list_events("session-1")
        )
        assert any(
            event.event_type == "approval.auto_approved"
            for event in ledger.list_events("session-1")
        )


def test_approval_mode_pauses_before_execution_and_resumes_after_approval() -> None:
    with LocalLedger(":memory:") as ledger:
        executor = _DockerFixture()
        loop = _loop(ledger, executor)
        pending = loop.run([], mode=LoopMode.APPROVAL)

        assert pending.status is LoopStatus.AWAITING_APPROVAL
        assert pending.approval_request_id is not None
        assert executor.calls == 0

        completed = loop.run([], mode=LoopMode.APPROVAL, approval_granted=True)
        assert completed.status is LoopStatus.COMPLETED
        assert executor.calls == 1
        assert ledger.list_records("approvals", session_id="session-1")[0]["status"] != "pending"


def test_executor_failure_is_retained_as_diagnostic() -> None:
    with LocalLedger(":memory:") as ledger:
        result = _loop(ledger, _DockerFixture(fail=True)).run([], mode=LoopMode.AUTONOMOUS)

        assert result.status is LoopStatus.COMPLETED
        assert result.results[0].status is ExperimentStatus.FAILED
        assert result.results[0].diagnostics[0].code == "executor_failed"
        recorded = [
            event
            for event in ledger.list_events("session-1")
            if event.event_type == "experiment.result_recorded"
        ]
        assert recorded[0].payload["diagnostics"][0]["code"] == "executor_failed"
