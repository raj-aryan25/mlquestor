"""Typed contracts for budget accounting and run admission.

The contracts in this module are deliberately independent of an experiment
runner.  Step 8 can use a :class:`RunReservation` to attach process
cancellation and output limits without changing budget accounting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from mlquest.domain import BudgetConfiguration


class AttemptOutcome(StrEnum):
    """Terminal states that are charged equally for their actual runtime."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class BudgetError(RuntimeError):
    """Base error for invalid or unsafe budget operations."""


class BudgetAdmissionError(BudgetError):
    """Raised when a run cannot be admitted without violating a budget."""

    def __init__(self, result: "PreflightResult") -> None:
        self.result = result
        super().__init__(result.reason or "run rejected by budget preflight")


class InvalidSnapshotError(BudgetError):
    """Raised when persisted budget state cannot be restored safely."""


class RunStateError(BudgetError):
    """Raised when a reservation is charged more than once or is unknown."""


@dataclass(frozen=True, slots=True)
class RunRequest:
    """The inputs that affect an execution cost estimate."""

    model_family: str
    dataset_size: int
    configuration: Mapping[str, Any] = field(default_factory=dict)
    estimated_memory_bytes: int | None = None
    estimated_storage_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.model_family.strip():
            raise ValueError("model_family must not be empty")
        if self.dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        if self.estimated_memory_bytes is not None and self.estimated_memory_bytes < 0:
            raise ValueError("estimated_memory_bytes must be non-negative")
        if self.estimated_storage_bytes is not None and self.estimated_storage_bytes < 0:
            raise ValueError("estimated_storage_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class CostObservation:
    """An actual completed attempt used to improve future estimates."""

    model_family: str
    dataset_size: int
    runtime_seconds: float
    configuration: Mapping[str, Any] = field(default_factory=dict)
    outcome: AttemptOutcome = AttemptOutcome.SUCCEEDED

    def __post_init__(self) -> None:
        if not self.model_family.strip():
            raise ValueError("model_family must not be empty")
        if self.dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        if self.runtime_seconds < 0:
            raise ValueError("runtime_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """An explainable runtime estimate in seconds."""

    model_family: str
    dataset_size: int
    estimated_seconds: float
    based_on_observations: int
    dataset_factor: float
    configuration_factor: float


@dataclass(frozen=True, slots=True)
class BudgetStatus:
    """Current accounting view, suitable for a status command or UI."""

    session_elapsed_seconds: float
    session_remaining_seconds: float
    model_execution_elapsed_seconds: float
    model_execution_remaining_seconds: float
    finalisation_reserve_seconds: float
    schedulable_session_seconds: float
    reserved_session_seconds: float
    reserved_model_execution_seconds: float


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """The typed answer to a run admission check."""

    accepted: bool
    estimate: CostEstimate
    status: BudgetStatus
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RunReservation:
    """A preflight-approved run handed to the future experiment runner."""

    reservation_id: str
    session_id: str
    request: RunRequest
    estimate: CostEstimate
    started_monotonic: float
    started_wall_time: float
    session_elapsed_at_start: float


@dataclass(frozen=True, slots=True)
class ChargeResult:
    """Receipt for charging a terminal attempt, including failures/timeouts."""

    reservation_id: str
    outcome: AttemptOutcome
    runtime_seconds: float
    status: BudgetStatus
    exceeded_session_budget: bool
    exceeded_model_execution_budget: bool


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Durable budget state.

    ``started_wall_time`` and ``session_deadline_wall_time`` bridge process
    restarts.  ``session_elapsed_seconds`` is monotonic within a process and
    is never replaced with zero during restore.
    """

    session_id: str
    configuration: BudgetConfiguration
    started_wall_time: float
    session_deadline_wall_time: float
    session_elapsed_seconds: float
    model_execution_elapsed_seconds: float
    observations: tuple[CostObservation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "configuration": self.configuration.model_dump(mode="json", exclude_none=True),
            "started_wall_time": self.started_wall_time,
            "session_deadline_wall_time": self.session_deadline_wall_time,
            "session_elapsed_seconds": self.session_elapsed_seconds,
            "model_execution_elapsed_seconds": self.model_execution_elapsed_seconds,
            "observations": [
                {
                    "model_family": item.model_family,
                    "dataset_size": item.dataset_size,
                    "runtime_seconds": item.runtime_seconds,
                    "configuration": dict(item.configuration),
                    "outcome": item.outcome.value,
                }
                for item in self.observations
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    def save(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BudgetSnapshot":
        try:
            configuration = BudgetConfiguration.model_validate(value["configuration"])
            observations = tuple(
                CostObservation(
                    model_family=str(item["model_family"]),
                    dataset_size=int(item["dataset_size"]),
                    runtime_seconds=float(item["runtime_seconds"]),
                    configuration=dict(item.get("configuration", {})),
                    outcome=AttemptOutcome(
                        str(item.get("outcome", AttemptOutcome.SUCCEEDED.value))
                    ),
                )
                for item in value.get("observations", [])
            )
            snapshot = cls(
                session_id=str(value["session_id"]),
                configuration=configuration,
                started_wall_time=float(value["started_wall_time"]),
                session_deadline_wall_time=float(value["session_deadline_wall_time"]),
                session_elapsed_seconds=float(value["session_elapsed_seconds"]),
                model_execution_elapsed_seconds=float(value["model_execution_elapsed_seconds"]),
                observations=observations,
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise InvalidSnapshotError(f"invalid budget snapshot: {exc}") from exc
        if snapshot.session_elapsed_seconds < 0 or snapshot.model_execution_elapsed_seconds < 0:
            raise InvalidSnapshotError("snapshot elapsed values must be non-negative")
        if snapshot.session_deadline_wall_time < snapshot.started_wall_time:
            raise InvalidSnapshotError("snapshot deadline precedes session start")
        return snapshot

    @classmethod
    def load(cls, path: Path | str) -> "BudgetSnapshot":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise TypeError("snapshot JSON must contain an object")
            return cls.from_dict(value)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise InvalidSnapshotError(f"could not read budget snapshot: {exc}") from exc
