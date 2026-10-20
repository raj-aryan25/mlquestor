"""Validation and deterministic fallback planning for experiment proposals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from mlquest.domain import (
    ExperimentSpecification,
    ExperimentSpecificationProposal,
    SplitSpecification,
    SplitStrategy,
    TaskSpecification,
    TaskType,
    ValidationSpecification,
)
from mlquest.evaluation.metrics import DEFAULT_METRIC_REGISTRY, MetricRegistry

from .contracts import (
    DiagnosticSeverity,
    PlanningDiagnostic,
    PlanningResult,
)

DEFAULT_MODEL_FAMILIES: dict[str, frozenset[TaskType]] = {
    "gradient_boosting": frozenset(TaskType),
    "lightgbm": frozenset(TaskType),
    "linear": frozenset({TaskType.REGRESSION}),
    "logistic": frozenset({TaskType.CLASSIFICATION}),
    "neural_network": frozenset(TaskType),
    "random_forest": frozenset(TaskType),
    "tree": frozenset(TaskType),
    "xgboost": frozenset(TaskType),
}


class AllowlistedExperimentPlanner:
    """Turn advisory proposals into execution-neutral specifications.

    The planner only emits declarative domain data.  It never accepts or
    generates commands, images, mounts, environment variables, or other
    process controls.
    """

    def __init__(
        self,
        *,
        model_families: Mapping[str, frozenset[TaskType]] = DEFAULT_MODEL_FAMILIES,
        metric_registry: MetricRegistry = DEFAULT_METRIC_REGISTRY,
    ) -> None:
        self.model_families = {key.lower(): value for key, value in model_families.items()}
        self.metric_registry = metric_registry

    def fallback_proposal(self, task: TaskSpecification) -> ExperimentSpecificationProposal:
        """Return a stable, safe baseline proposal for a task."""

        family = "logistic" if task.task_type is TaskType.CLASSIFICATION else "linear"
        return ExperimentSpecificationProposal(
            model_family=family,
            parameters={},
            random_seeds=[0],
            rationale="Deterministic fallback baseline for a validated task.",
            expected_effect="Establish a reproducible reference score.",
        )

    fallback = fallback_proposal

    def plan(
        self,
        proposal: ExperimentSpecificationProposal | Mapping[str, Any] | object,
        *,
        task: TaskSpecification,
        validation: ValidationSpecification | SplitSpecification | None = None,
        comparison_context_id: str | None = None,
        experiment_name: str | None = None,
        dataset_artifact_id: str | None = None,
        parent_experiment_id: str | None = None,
        hypothesis_id: str | None = None,
        use_fallback: bool = True,
    ) -> PlanningResult:
        """Validate a proposal and convert it to an ``ExperimentSpecification``."""

        diagnostics: list[PlanningDiagnostic] = []
        parsed, parse_diagnostics = self._parse_proposal(proposal)
        diagnostics.extend(parse_diagnostics)
        proposal_was_invalid = parsed is None
        if parsed is None:
            if not use_fallback:
                return PlanningResult(diagnostics=diagnostics)
            parsed = self.fallback_proposal(task)
            diagnostics.append(
                PlanningDiagnostic(
                    code="fallback_used",
                    message="invalid proposal replaced with the deterministic fallback",
                    severity=DiagnosticSeverity.WARNING,
                )
            )
            used_fallback = True
        else:
            used_fallback = False

        split, metric_diagnostics = self._validation(task, validation)
        diagnostics.extend(metric_diagnostics)
        if any(item.severity is DiagnosticSeverity.ERROR for item in metric_diagnostics):
            return PlanningResult(
                proposal=parsed,
                diagnostics=diagnostics,
                used_fallback=used_fallback,
            )
        model_diagnostics = self._validate_model(parsed, task.task_type)
        diagnostics.extend(model_diagnostics)
        if any(item.severity is DiagnosticSeverity.ERROR for item in model_diagnostics):
            if not use_fallback or proposal_was_invalid:
                return PlanningResult(
                    proposal=parsed,
                    diagnostics=diagnostics,
                    used_fallback=used_fallback,
                )
            parsed = self.fallback_proposal(task)
            used_fallback = True
            diagnostics.append(
                PlanningDiagnostic(
                    code="fallback_used",
                    message=(
                        "unsafe or unsupported proposal replaced with the deterministic "
                        "fallback"
                    ),
                    severity=DiagnosticSeverity.WARNING,
                )
            )
            model_diagnostics = self._validate_model(parsed, task.task_type)
            diagnostics.extend(model_diagnostics)
            if any(item.severity is DiagnosticSeverity.ERROR for item in model_diagnostics):
                return PlanningResult(
                    proposal=parsed,
                    diagnostics=diagnostics,
                    used_fallback=True,
                )

        context = comparison_context_id
        if isinstance(validation, ValidationSpecification):
            if validation.comparison_context_id is not None:
                if context is not None and context != validation.comparison_context_id:
                    diagnostics.append(
                        PlanningDiagnostic(
                            code="comparison_context_mismatch",
                            message="requested and validation comparison contexts differ",
                            field="comparison_context_id",
                        )
                    )
                context = validation.comparison_context_id
        validation_spec = ValidationSpecification(
            split=split,
            primary_metric=task.primary_metric,
            metric_direction=task.metric_direction,
            secondary_metrics=task.secondary_metrics,
            comparison_context_id=context,
        )
        if any(item.code == "comparison_context_mismatch" for item in diagnostics):
            return PlanningResult(
                proposal=parsed,
                diagnostics=diagnostics,
                used_fallback=used_fallback,
            )
        specification = ExperimentSpecification(
            name=experiment_name or self._name(parsed),
            model_family=parsed.model_family.lower(),
            validation=validation_spec,
            dataset_artifact_id=dataset_artifact_id
            or (task.source_artifact_ids[0] if task.source_artifact_ids else None),
            parent_experiment_id=parent_experiment_id,
            hypothesis_id=hypothesis_id,
            model_config_values=parsed.parameters,
            random_seeds=parsed.random_seeds,
            comparison_context_id=context,
        )
        return PlanningResult(
            specification=specification,
            proposal=parsed,
            diagnostics=diagnostics,
            used_fallback=used_fallback,
            accepted=True,
        )

    convert = plan
    validate_and_convert = plan

    def _parse_proposal(
        self, proposal: ExperimentSpecificationProposal | Mapping[str, Any] | object
    ) -> tuple[ExperimentSpecificationProposal | None, list[PlanningDiagnostic]]:
        if isinstance(proposal, ExperimentSpecificationProposal):
            return proposal, []
        try:
            return ExperimentSpecificationProposal.model_validate(proposal), []
        except (ValidationError, TypeError, ValueError) as exc:
            return None, [
                PlanningDiagnostic(
                    code="malformed_proposal",
                    message=f"proposal failed schema validation: {exc}",
                    field="proposal",
                )
            ]

    def _validation(
        self,
        task: TaskSpecification,
        validation: ValidationSpecification | SplitSpecification | None,
    ) -> tuple[SplitSpecification, list[PlanningDiagnostic]]:
        diagnostics: list[PlanningDiagnostic] = []
        if validation is None:
            split = SplitSpecification(
                strategy=(
                    SplitStrategy.STRATIFIED
                    if task.task_type is TaskType.CLASSIFICATION
                    else SplitStrategy.RANDOM
                ),
                test_size=0.2,
                random_seed=0,
            )
        elif isinstance(validation, ValidationSpecification):
            split = validation.split
            if validation.primary_metric.lower() != task.primary_metric.lower():
                diagnostics.append(
                    PlanningDiagnostic(
                        code="metric_mismatch",
                        message="validation primary metric does not match the task",
                        field="validation.primary_metric",
                    )
                )
            if validation.metric_direction is not task.metric_direction:
                diagnostics.append(
                    PlanningDiagnostic(
                        code="metric_direction_mismatch",
                        message="validation metric direction does not match the task",
                        field="validation.metric_direction",
                    )
                )
        else:
            split = validation
        try:
            definition = self.metric_registry.get(task.primary_metric)
            if definition.task_type is not task.task_type:
                diagnostics.append(
                    PlanningDiagnostic(
                        code="metric_task_mismatch",
                        message="primary metric is not valid for the task type",
                        field="primary_metric",
                    )
                )
            if definition.direction is not task.metric_direction:
                diagnostics.append(
                    PlanningDiagnostic(
                        code="metric_direction_mismatch",
                        message="task metric direction disagrees with the metric registry",
                        field="metric_direction",
                    )
                )
        except ValueError as exc:
            diagnostics.append(
                PlanningDiagnostic(
                    code="unknown_metric",
                    message=str(exc),
                    field="primary_metric",
                )
            )
        for metric_name in task.secondary_metrics:
            try:
                secondary = self.metric_registry.get(metric_name)
            except ValueError as exc:
                diagnostics.append(
                    PlanningDiagnostic(
                        code="unknown_secondary_metric",
                        message=str(exc),
                        field="secondary_metrics",
                    )
                )
                continue
            if secondary.task_type is not task.task_type:
                diagnostics.append(
                    PlanningDiagnostic(
                        code="secondary_metric_task_mismatch",
                        message=f"secondary metric {metric_name!r} is not valid for the task type",
                        field="secondary_metrics",
                    )
                )
        if (
            split.strategy is SplitStrategy.STRATIFIED
            and task.task_type is not TaskType.CLASSIFICATION
        ):
            diagnostics.append(
                PlanningDiagnostic(
                    code="split_task_mismatch",
                    message="stratified splits require a classification task",
                    field="validation.split.strategy",
                )
            )
        return split, diagnostics

    def _validate_model(
        self, proposal: ExperimentSpecificationProposal, task_type: TaskType
    ) -> list[PlanningDiagnostic]:
        family = proposal.model_family.lower()
        allowed_tasks = self.model_families.get(family)
        if allowed_tasks is None:
            return [
                PlanningDiagnostic(
                    code="model_family_not_allowlisted",
                    message=f"model family {proposal.model_family!r} is not allowlisted",
                    field="model_family",
                )
            ]
        if task_type not in allowed_tasks:
            return [
                PlanningDiagnostic(
                    code="model_family_task_mismatch",
                    message=(
                        f"model family {proposal.model_family!r} is not valid for "
                        f"{task_type.value}"
                    ),
                    field="model_family",
                )
            ]
        return []

    @staticmethod
    def _name(proposal: ExperimentSpecificationProposal) -> str:
        return f"{proposal.model_family.lower()}-experiment"


ExperimentPlanner = AllowlistedExperimentPlanner


__all__ = ["AllowlistedExperimentPlanner", "DEFAULT_MODEL_FAMILIES", "ExperimentPlanner"]
