"""Deterministic serial ranking and admission of experiment candidates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .contracts import (
    CandidateProposal,
    DiagnosticSeverity,
    PlanningDiagnostic,
    RankedCandidate,
    ScheduleResult,
)


@dataclass(frozen=True, slots=True)
class SchedulerWeights:
    """Weights for the explainable scheduler utility function."""

    expected_value: float = 0.50
    novelty: float = 0.20
    risk: float = 0.20
    cost: float = 0.10

    def __post_init__(self) -> None:
        if any(value < 0 for value in (self.expected_value, self.novelty, self.risk, self.cost)):
            raise ValueError("scheduler weights must be non-negative")
        if self.expected_value + self.novelty + self.risk + self.cost == 0:
            raise ValueError("scheduler weights must not all be zero")


class SerialExperimentScheduler:
    """Rank candidates and return at most ``max_count`` for serial execution."""

    def __init__(
        self,
        *,
        max_count: int = 1,
        max_estimated_cost: float | None = None,
        weights: SchedulerWeights = SchedulerWeights(),
    ) -> None:
        if max_count < 1:
            raise ValueError("max_count must be positive")
        if max_estimated_cost is not None and max_estimated_cost < 0:
            raise ValueError("max_estimated_cost must be non-negative")
        self.max_count = max_count
        self.max_estimated_cost = max_estimated_cost
        self.weights = weights

    def score(self, candidate: CandidateProposal) -> float:
        """Calculate a higher-is-better utility from all four ranking signals."""

        weights = self.weights
        total = weights.expected_value + weights.novelty + weights.risk + weights.cost
        cost_penalty = candidate.estimated_cost / (candidate.estimated_cost + 1.0)
        return round(
            (
                weights.expected_value * candidate.expected_value
                + weights.novelty * candidate.novelty
                - weights.risk * candidate.risk
                - weights.cost * cost_penalty
            )
            / total,
            12,
        )

    def rank(self, candidates: Iterable[CandidateProposal]) -> list[RankedCandidate]:
        """Return a deterministic ranking, with proposal digest as the final tie-breaker."""

        ranked = [
            (candidate, self.score(candidate), candidate.proposal.content_digest())
            for candidate in candidates
        ]
        ranked.sort(
            key=lambda item: (
                -item[1],
                -item[0].expected_value,
                -item[0].novelty,
                item[0].risk,
                item[0].estimated_cost,
                item[2],
            )
        )
        return [
            RankedCandidate(candidate=candidate, ranking_score=score, rank=index)
            for index, (candidate, score, _) in enumerate(ranked, start=1)
        ]

    def schedule(
        self, candidates: Iterable[CandidateProposal], *, max_count: int | None = None
    ) -> ScheduleResult:
        """Select a bounded, serializable prefix subject to the cost ceiling."""

        ranked = self.rank(candidates)
        limit = self.max_count if max_count is None else max_count
        if limit < 1:
            raise ValueError("max_count must be positive")
        selected: list[RankedCandidate] = []
        spent = 0.0
        diagnostics: list[PlanningDiagnostic] = []
        for candidate in ranked:
            if len(selected) >= limit:
                break
            cost = candidate.candidate.estimated_cost
            if self.max_estimated_cost is not None and spent + cost > self.max_estimated_cost:
                diagnostics.append(
                    PlanningDiagnostic(
                        code="candidate_over_cost_ceiling",
                        message=(
                            "candidate was not selected because the serial cost ceiling "
                            "would be exceeded"
                        ),
                        severity=DiagnosticSeverity.INFO,
                    )
                )
                continue
            selected.append(candidate)
            spent += cost
        return ScheduleResult(selected=selected, ranked=ranked, diagnostics=diagnostics)

    rank_candidates = rank
    rank_proposals = rank
    select = schedule


ExperimentScheduler = SerialExperimentScheduler
SerialScheduler = SerialExperimentScheduler


__all__ = [
    "ExperimentScheduler",
    "SchedulerWeights",
    "SerialExperimentScheduler",
    "SerialScheduler",
]
