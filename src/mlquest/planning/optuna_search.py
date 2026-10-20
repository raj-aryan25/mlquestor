"""Bounded Optuna search over approved experiment parameter spaces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import optuna

from mlquest.domain import ExperimentSpecification, MetricDirection

ParameterKind = Literal["float", "int", "categorical"]


@dataclass(frozen=True, slots=True)
class SearchParameter:
    """One allowlisted model parameter exposed to the searcher."""

    name: str
    kind: ParameterKind
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    choices: tuple[str | int | float | bool, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or self.name in {"command", "shell", "docker", "env"}:
            raise ValueError("search parameter name is unsafe or empty")
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError("categorical parameters require choices")
            if self.low is not None or self.high is not None or self.step is not None:
                raise ValueError("categorical parameters do not accept numeric bounds")
            return
        if self.low is None or self.high is None or self.low >= self.high:
            raise ValueError("numeric parameters require ordered low and high bounds")
        if self.kind == "int" and any(
            value is not None and not isinstance(value, int)
            for value in (self.low, self.high, self.step)
        ):
            raise ValueError("integer search bounds must be integers")
        if self.step is not None and self.step <= 0:
            raise ValueError("search parameter step must be positive")


@dataclass(frozen=True, slots=True)
class SearchTrial:
    number: int
    parameters: dict[str, str | int | float | bool]
    score: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    best_specification: ExperimentSpecification
    trials: tuple[SearchTrial, ...]


class OptunaExperimentSearch:
    """Run a bounded, deterministic serial search around one approved spec."""

    def __init__(self, *, seed: int = 0, max_trials: int = 20) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if max_trials < 1:
            raise ValueError("max_trials must be positive")
        self.seed = seed
        self.max_trials = max_trials

    def search(
        self,
        specification: ExperimentSpecification,
        parameters: Sequence[SearchParameter],
        evaluate: Callable[[ExperimentSpecification], float],
    ) -> SearchResult:
        """Evaluate at most ``max_trials`` specs and return the best observed one."""

        if not parameters:
            raise ValueError("at least one search parameter is required")
        names = [parameter.name for parameter in parameters]
        if len(set(names)) != len(names):
            raise ValueError("search parameter names must be unique")
        direction = (
            "maximize"
            if specification.validation.metric_direction is MetricDirection.MAXIMIZE
            else "minimize"
        )
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.RandomSampler(seed=self.seed),
        )
        trials: list[SearchTrial] = []
        best_specification = specification
        best_score: float | None = None
        for _ in range(self.max_trials):
            trial = study.ask()
            values: dict[str, str | int | float | bool] = {}
            for parameter in parameters:
                values[parameter.name] = self._suggest(trial, parameter)
            candidate = specification.model_copy(
                update={
                    "model_config_values": {
                        **specification.model_config_values,
                        **values,
                    }
                }
            )
            score = float(evaluate(candidate))
            if score != score or score in {float("inf"), float("-inf")}:
                raise ValueError("search evaluation must return finite scores")
            study.tell(trial, score)
            trials.append(SearchTrial(trial.number, values, score))
            if best_score is None or (
                direction == "maximize" and score > best_score
            ) or (direction == "minimize" and score < best_score):
                best_score = score
                best_specification = candidate
        return SearchResult(best_specification=best_specification, trials=tuple(trials))

    @staticmethod
    def _suggest(
        trial: optuna.Trial, parameter: SearchParameter
    ) -> str | int | float | bool:
        if parameter.kind == "float":
            assert isinstance(parameter.low, (float, int))
            assert isinstance(parameter.high, (float, int))
            return trial.suggest_float(
                parameter.name,
                float(parameter.low),
                float(parameter.high),
                step=float(parameter.step) if parameter.step is not None else None,
            )
        if parameter.kind == "int":
            assert isinstance(parameter.low, int)
            assert isinstance(parameter.high, int)
            return trial.suggest_int(
                parameter.name,
                parameter.low,
                parameter.high,
                step=int(parameter.step) if parameter.step is not None else 1,
            )
        return cast(
            str | int | float | bool,
            trial.suggest_categorical(parameter.name, list(parameter.choices)),
        )


__all__ = ["OptunaExperimentSearch", "SearchParameter", "SearchResult", "SearchTrial"]
