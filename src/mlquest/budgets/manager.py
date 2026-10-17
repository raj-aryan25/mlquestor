"""Budget accounting service for durable MLQuest sessions."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from mlquest.domain import BudgetConfiguration

from .estimator import CostEstimator
from .limits import ExecutionLimits, ExecutionResult, run_limited_command
from .models import (
    AttemptOutcome,
    BudgetAdmissionError,
    BudgetSnapshot,
    BudgetStatus,
    ChargeResult,
    CostEstimate,
    CostObservation,
    PreflightResult,
    RunRequest,
    RunReservation,
    RunStateError,
)


class Clock(Protocol):
    """Clock interface injectable by tests and durable session adapters."""

    def monotonic(self) -> float: ...

    def time(self) -> float: ...


class SystemClock:
    """Production clock using monotonic elapsed time and epoch persistence."""

    def monotonic(self) -> float:
        return time.monotonic()

    def time(self) -> float:
        return time.time()


@dataclass(slots=True)
class _MutableState:
    session_elapsed_seconds: float
    model_execution_elapsed_seconds: float
    reserved_session_seconds: float = 0.0
    reserved_model_execution_seconds: float = 0.0


class BudgetManager:
    """Enforce session, model, and finalisation budgets before execution.

    A manager is scoped to one durable session.  The session service can store
    :meth:`snapshot` alongside its session record and restore it on a later
    process invocation.  The manager never cancels a process; its reservation
    receipt is the integration point for the Step 8 runner.
    """

    def __init__(
        self,
        session_id: str,
        configuration: BudgetConfiguration,
        *,
        clock: Clock | None = None,
        started_wall_time: float | None = None,
        session_elapsed_seconds: float = 0.0,
        model_execution_elapsed_seconds: float = 0.0,
        observations: tuple[CostObservation, ...] = (),
    ) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        if session_elapsed_seconds < 0 or model_execution_elapsed_seconds < 0:
            raise ValueError("elapsed budget values must be non-negative")
        self.session_id = session_id
        self.configuration = configuration
        self.clock = clock or SystemClock()
        now_wall = self.clock.time()
        self._started_wall_time = now_wall if started_wall_time is None else started_wall_time
        if self._started_wall_time > now_wall:
            raise ValueError("started_wall_time cannot be in the future")
        self._session_deadline_wall_time = self._started_wall_time + configuration.session_seconds
        self._anchor_monotonic = self.clock.monotonic() - session_elapsed_seconds
        self._state = _MutableState(session_elapsed_seconds, model_execution_elapsed_seconds)
        self._observations = list(observations)
        self._reservations: dict[str, RunReservation] = {}
        self._estimator = CostEstimator()

    @classmethod
    def from_snapshot(
        cls,
        snapshot: BudgetSnapshot | Mapping[str, object],
        *,
        clock: Clock | None = None,
    ) -> "BudgetManager":
        if not isinstance(snapshot, BudgetSnapshot):
            snapshot = BudgetSnapshot.from_dict(snapshot)
        manager = cls(
            snapshot.session_id,
            snapshot.configuration,
            clock=clock,
            started_wall_time=snapshot.started_wall_time,
            session_elapsed_seconds=snapshot.session_elapsed_seconds,
            model_execution_elapsed_seconds=snapshot.model_execution_elapsed_seconds,
            observations=snapshot.observations,
        )
        manager._session_deadline_wall_time = snapshot.session_deadline_wall_time
        return manager

    def snapshot(self) -> BudgetSnapshot:
        """Return a durable checkpoint without resetting elapsed accounting."""

        self._sync_elapsed()
        return BudgetSnapshot(
            session_id=self.session_id,
            configuration=self.configuration,
            started_wall_time=self._started_wall_time,
            session_deadline_wall_time=self._session_deadline_wall_time,
            session_elapsed_seconds=self._state.session_elapsed_seconds,
            model_execution_elapsed_seconds=self._state.model_execution_elapsed_seconds,
            observations=tuple(self._observations),
        )

    def save_snapshot(self, path: str | Path) -> BudgetSnapshot:
        snapshot = self.snapshot()
        snapshot.save(path)
        return snapshot

    @property
    def status(self) -> BudgetStatus:
        self._sync_elapsed()
        session_remaining = max(
            0.0, self.configuration.session_seconds - self._state.session_elapsed_seconds
        )
        model_remaining = max(
            0.0,
            self.configuration.model_execution_seconds
            - self._state.model_execution_elapsed_seconds,
        )
        return BudgetStatus(
            session_elapsed_seconds=round(self._state.session_elapsed_seconds, 6),
            session_remaining_seconds=round(session_remaining, 6),
            model_execution_elapsed_seconds=round(self._state.model_execution_elapsed_seconds, 6),
            model_execution_remaining_seconds=round(model_remaining, 6),
            finalisation_reserve_seconds=float(self.configuration.finalisation_reserve_seconds),
            schedulable_session_seconds=round(
                max(
                    0.0,
                    session_remaining
                    - self.configuration.finalisation_reserve_seconds
                    - self._state.reserved_session_seconds,
                ),
                6,
            ),
            reserved_session_seconds=round(self._state.reserved_session_seconds, 6),
            reserved_model_execution_seconds=round(self._state.reserved_model_execution_seconds, 6),
        )

    @property
    def session_deadline_wall_time(self) -> float:
        """Persisted epoch deadline for inspection and durable integrations."""

        return self._session_deadline_wall_time

    @property
    def session_deadline_monotonic(self) -> float:
        """Process-local monotonic deadline derived from the persisted elapsed base."""

        return self._anchor_monotonic + self.configuration.session_seconds

    def estimate(
        self,
        request: RunRequest | None = None,
        *,
        model_family: str | None = None,
        dataset_size: int | None = None,
        configuration: Mapping[str, Any] | None = None,
    ) -> CostEstimate:
        if request is None:
            if model_family is None or dataset_size is None:
                raise ValueError("request or model_family and dataset_size is required")
            request = RunRequest(model_family, dataset_size, configuration or {})
        return self._estimator.estimate(request, self._observations)

    estimate_cost = estimate

    def record_observation(self, observation: CostObservation) -> None:
        """Add an external prior observation to the estimator history."""

        self._observations.append(observation)

    def preflight(
        self,
        request: RunRequest,
        *,
        estimate: CostEstimate | None = None,
    ) -> PreflightResult:
        """Check whether a run can fit while preserving the finalisation reserve."""

        cost = estimate or self.estimate(request)
        status = self.status
        reason: str | None = None
        if status.session_remaining_seconds <= self.configuration.finalisation_reserve_seconds:
            reason = "session budget is at or inside the protected finalisation reserve"
        elif (
            self.configuration.max_experiments is not None
            and len(self._observations) + len(self._reservations)
            >= self.configuration.max_experiments
        ):
            reason = "maximum experiment count has been reached"
        elif (
            self.configuration.max_memory_bytes is not None
            and request.estimated_memory_bytes is not None
            and request.estimated_memory_bytes > self.configuration.max_memory_bytes
        ):
            reason = "estimated memory exceeds the configured memory limit"
        elif (
            self.configuration.max_storage_bytes is not None
            and request.estimated_storage_bytes is not None
            and request.estimated_storage_bytes > self.configuration.max_storage_bytes
        ):
            reason = "estimated storage exceeds the configured storage limit"
        elif cost.estimated_seconds > status.schedulable_session_seconds:
            reason = (
                f"estimated runtime {cost.estimated_seconds:.3f}s exceeds schedulable session "
                f"time {status.schedulable_session_seconds:.3f}s"
            )
        elif cost.estimated_seconds > max(
            0.0,
            status.model_execution_remaining_seconds - status.reserved_model_execution_seconds,
        ):
            reason = (
                f"estimated runtime {cost.estimated_seconds:.3f}s exceeds remaining model "
                "execution budget"
            )
        return PreflightResult(accepted=reason is None, estimate=cost, status=status, reason=reason)

    def admit(
        self,
        request: RunRequest,
        *,
        estimate: CostEstimate | None = None,
    ) -> RunReservation:
        """Reserve estimated capacity and return a receipt for the runner."""

        result = self.preflight(request, estimate=estimate)
        if not result.accepted:
            raise BudgetAdmissionError(result)
        now_mono = self.clock.monotonic()
        reservation = RunReservation(
            reservation_id=str(uuid4()),
            session_id=self.session_id,
            request=request,
            estimate=result.estimate,
            started_monotonic=now_mono,
            started_wall_time=self.clock.time(),
            session_elapsed_at_start=self._state.session_elapsed_seconds,
        )
        self._reservations[reservation.reservation_id] = reservation
        self._state.reserved_session_seconds += reservation.estimate.estimated_seconds
        self._state.reserved_model_execution_seconds += reservation.estimate.estimated_seconds
        return reservation

    start_run = admit

    def charge(
        self,
        reservation: RunReservation,
        outcome: AttemptOutcome | str,
        *,
        runtime_seconds: float | None = None,
    ) -> ChargeResult:
        """Charge actual runtime for success, failure, timeout, or cancellation."""

        active = self._reservations.get(reservation.reservation_id)
        if active is None or active != reservation:
            raise RunStateError(
                f"unknown or already charged reservation: {reservation.reservation_id}"
            )
        terminal_outcome = _coerce_outcome(outcome)
        if runtime_seconds is None:
            runtime_seconds = max(0.0, self.clock.monotonic() - reservation.started_monotonic)
        if runtime_seconds < 0:
            raise ValueError("runtime_seconds must be non-negative")

        self._reservations.pop(reservation.reservation_id)
        self._sync_elapsed()
        self._state.reserved_session_seconds = max(
            0.0, self._state.reserved_session_seconds - reservation.estimate.estimated_seconds
        )
        self._state.reserved_model_execution_seconds = max(
            0.0,
            self._state.reserved_model_execution_seconds - reservation.estimate.estimated_seconds,
        )
        self._state.session_elapsed_seconds = max(
            self._state.session_elapsed_seconds,
            reservation.session_elapsed_at_start + runtime_seconds,
        )
        self._anchor_monotonic = self.clock.monotonic() - self._state.session_elapsed_seconds
        self._state.model_execution_elapsed_seconds += runtime_seconds
        self._observations.append(
            CostObservation(
                model_family=reservation.request.model_family,
                dataset_size=reservation.request.dataset_size,
                runtime_seconds=runtime_seconds,
                configuration=reservation.request.configuration,
                outcome=terminal_outcome,
            )
        )
        status = self.status
        return ChargeResult(
            reservation_id=reservation.reservation_id,
            outcome=terminal_outcome,
            runtime_seconds=runtime_seconds,
            status=status,
            exceeded_session_budget=status.session_elapsed_seconds
            > self.configuration.session_seconds,
            exceeded_model_execution_budget=status.model_execution_elapsed_seconds
            > self.configuration.model_execution_seconds,
        )

    finish_run = charge

    def charge_session_runtime(self, runtime_seconds: float) -> BudgetStatus:
        """Charge non-model session work such as analysis or finalisation."""

        if runtime_seconds < 0:
            raise ValueError("runtime_seconds must be non-negative")
        self._sync_elapsed()
        self._state.session_elapsed_seconds += runtime_seconds
        self._anchor_monotonic = self.clock.monotonic() - self._state.session_elapsed_seconds
        return self.status

    def run_limited_command(
        self,
        request: RunRequest,
        command: list[str] | tuple[str, ...],
        limits: ExecutionLimits,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[ExecutionResult, ChargeResult]:
        """Admit, execute, and charge one bounded command as a run attempt."""

        reservation = self.admit(request)
        try:
            result = run_limited_command(command, limits, cwd=cwd, env=env)
        except Exception:
            self.charge(reservation, AttemptOutcome.FAILED)
            raise
        outcome = (
            AttemptOutcome.TIMED_OUT
            if result.timed_out
            else AttemptOutcome.FAILED
            if result.returncode != 0 or result.output_limited
            else AttemptOutcome.SUCCEEDED
        )
        charge = self.charge(reservation, outcome)
        return result, charge

    def _sync_elapsed(self) -> None:
        now_monotonic = self.clock.monotonic()
        monotonic_elapsed = max(0.0, now_monotonic - self._anchor_monotonic)
        persisted_wall_elapsed = max(0.0, self.clock.time() - self._started_wall_time)
        self._state.session_elapsed_seconds = max(
            self._state.session_elapsed_seconds,
            monotonic_elapsed,
            persisted_wall_elapsed,
        )
        self._anchor_monotonic = now_monotonic - self._state.session_elapsed_seconds


def _coerce_outcome(value: AttemptOutcome | str) -> AttemptOutcome:
    if isinstance(value, AttemptOutcome):
        return value
    aliases = {
        "success": AttemptOutcome.SUCCEEDED,
        "failure": AttemptOutcome.FAILED,
        "timeout": AttemptOutcome.TIMED_OUT,
    }
    if value in aliases:
        return aliases[value]
    try:
        return AttemptOutcome(value)
    except ValueError as exc:
        raise ValueError(f"unsupported attempt outcome: {value}") from exc
