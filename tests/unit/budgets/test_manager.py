from __future__ import annotations

import json

import pytest

from mlquest.budgets import (
    AttemptOutcome,
    BudgetAdmissionError,
    BudgetManager,
    BudgetSnapshot,
    CostObservation,
    RunRequest,
    RunStateError,
)
from mlquest.budgets.limits import ExecutionLimits, run_limited_command
from mlquest.domain import BudgetConfiguration


class FakeClock:
    def __init__(self, *, monotonic: float = 0.0, wall: float = 1_000.0) -> None:
        self.monotonic_value = monotonic
        self.wall_value = wall

    def monotonic(self) -> float:
        return self.monotonic_value

    def time(self) -> float:
        return self.wall_value

    def advance(self, seconds: float) -> None:
        self.monotonic_value += seconds
        self.wall_value += seconds


def configuration(**overrides: int) -> BudgetConfiguration:
    values = {
        "session_seconds": 100,
        "model_execution_seconds": 80,
        "finalisation_reserve_seconds": 20,
    }
    values.update(overrides)
    return BudgetConfiguration(**values)


def test_elapsed_time_is_monotonic_and_restore_preserves_it() -> None:
    clock = FakeClock()
    manager = BudgetManager("session-1", configuration(), clock=clock)
    clock.advance(12)
    assert manager.status.session_elapsed_seconds == 12

    snapshot = manager.snapshot()
    resumed_clock = FakeClock(monotonic=50, wall=1_050)
    resumed = BudgetManager.from_snapshot(snapshot, clock=resumed_clock)
    assert resumed.status.session_elapsed_seconds == 50
    assert resumed.session_deadline_monotonic == 100
    resumed_clock.advance(3)
    assert resumed.status.session_elapsed_seconds == 53


def test_preflight_rejects_a_run_that_threatens_finalisation_reserve() -> None:
    clock = FakeClock()
    manager = BudgetManager("session-1", configuration(), clock=clock)
    clock.advance(70)

    result = manager.preflight(
        RunRequest("linear", 1, {"n_estimators": 100}),
        estimate=manager.estimate(model_family="linear", dataset_size=1),
    )
    assert not result.accepted
    assert result.reason is not None
    assert "schedulable session" in result.reason
    with pytest.raises(BudgetAdmissionError) as error:
        manager.admit(
            RunRequest("linear", 1),
            estimate=manager.estimate(model_family="linear", dataset_size=1),
        )
    assert error.value.result.accepted is False


def test_estimate_uses_prior_family_observation_and_configuration() -> None:
    manager = BudgetManager("session-1", configuration(), clock=FakeClock())
    manager.record_observation(
        CostObservation(
            model_family="custom-model",
            dataset_size=100,
            runtime_seconds=40,
        )
    )
    estimate = manager.estimate(
        model_family="custom-model",
        dataset_size=400,
        configuration={"epochs": 200},
    )
    assert estimate.based_on_observations == 1
    assert estimate.estimated_seconds > 80


@pytest.mark.parametrize(
    "outcome", [AttemptOutcome.SUCCEEDED, AttemptOutcome.FAILED, AttemptOutcome.TIMED_OUT]
)
def test_actual_runtime_is_charged_for_every_terminal_outcome(outcome: AttemptOutcome) -> None:
    manager = BudgetManager("session-1", configuration(), clock=FakeClock())
    request = RunRequest("linear", 10)
    reservation = manager.admit(request, estimate=manager.estimate(request))
    result = manager.charge(reservation, outcome, runtime_seconds=7)

    assert result.outcome is outcome
    assert result.runtime_seconds == 7
    assert result.status.session_elapsed_seconds == 7
    assert result.status.model_execution_elapsed_seconds == 7
    assert manager.snapshot().observations[-1].outcome is outcome
    with pytest.raises(RunStateError):
        manager.charge(reservation, outcome, runtime_seconds=1)


def test_string_timeout_outcome_is_typed_and_charged() -> None:
    manager = BudgetManager("session-1", configuration(), clock=FakeClock())
    reservation = manager.admit(
        RunRequest("linear", 10),
        estimate=manager.estimate(model_family="linear", dataset_size=10),
    )
    result = manager.charge(reservation, "timeout", runtime_seconds=3)
    assert result.outcome is AttemptOutcome.TIMED_OUT


def test_session_work_is_separate_from_model_execution_accounting() -> None:
    manager = BudgetManager("session-1", configuration(), clock=FakeClock())
    manager.charge_session_runtime(11)
    assert manager.status.session_elapsed_seconds == 11
    assert manager.status.model_execution_elapsed_seconds == 0


def test_snapshot_can_be_saved_and_loaded_without_losing_elapsed_time(tmp_path) -> None:
    clock = FakeClock()
    manager = BudgetManager("session-1", configuration(), clock=clock)
    clock.advance(9)
    path = tmp_path / "budget.json"
    manager.save_snapshot(str(path))

    loaded = BudgetSnapshot.load(path)
    assert json.loads(path.read_text())["session_elapsed_seconds"] == 9
    restored = BudgetManager.from_snapshot(loaded, clock=clock)
    assert restored.status.session_elapsed_seconds == 9


def test_maximum_experiment_count_is_enforced_before_admission() -> None:
    manager = BudgetManager("session-1", configuration(max_experiments=1), clock=FakeClock())
    reservation = manager.admit(RunRequest("linear", 10))
    manager.charge(reservation, AttemptOutcome.FAILED, runtime_seconds=1)

    result = manager.preflight(RunRequest("linear", 10))
    assert result.accepted is False
    assert result.reason == "maximum experiment count has been reached"


def test_resource_limits_are_checked_before_admission() -> None:
    manager = BudgetManager(
        "session-1",
        configuration(max_memory_bytes=100),
        clock=FakeClock(),
    )
    result = manager.preflight(RunRequest("linear", 10, estimated_memory_bytes=101))
    assert result.accepted is False
    assert result.reason == "estimated memory exceeds the configured memory limit"


def test_limited_command_terminates_a_process_group_on_timeout() -> None:
    result = run_limited_command(
        ["python", "-c", "import time; time.sleep(10)"],
        ExecutionLimits(timeout_seconds=0.05),
    )
    assert result.timed_out is True


def test_limited_command_enforces_output_quota() -> None:
    result = run_limited_command(
        ["python", "-c", "print('x' * 10000)"],
        ExecutionLimits(timeout_seconds=2, max_output_bytes=128),
    )
    assert result.output_limited is True
    assert len(result.stdout) <= 128


def test_manager_records_a_timed_out_command_as_a_charged_attempt() -> None:
    manager = BudgetManager("session-1", configuration(), clock=FakeClock())
    result, charge = manager.run_limited_command(
        RunRequest("linear", 10),
        ["python", "-c", "import time; time.sleep(10)"],
        ExecutionLimits(timeout_seconds=0.05),
    )
    assert result.timed_out is True
    assert charge.outcome is AttemptOutcome.TIMED_OUT
    assert manager.snapshot().observations[-1].outcome is AttemptOutcome.TIMED_OUT
