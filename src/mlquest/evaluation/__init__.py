"""Evaluation contracts, split/metric services, and reproducible baselines."""

from .baselines import (
    BaselineRunner,
    BaselineTier,
    BaselineUnavailableError,
    build_baseline,
    evaluate_baseline,
)
from .metrics import (
    DEFAULT_METRIC_REGISTRY,
    MetricDefinition,
    MetricRegistry,
    MetricScore,
    UnknownMetricError,
    evaluate_metric,
)
from .results import (
    AggregateEvaluationResult,
    BaselineEvaluationResult,
    FoldEvaluationResult,
    PredictionValidationSummary,
    ResourceMetadata,
)
from .splits import DeterministicSplitter, SplitFold, SplitPlan, create_split_plan, generate_splits
from .validation import (
    PredictionValidationError,
    PredictionValidationResult,
    PredictionValidator,
    validate_predictions,
    validate_submission,
    validate_submission_schema,
)

__all__ = [
    "BaselineRunner",
    "BaselineTier",
    "BaselineUnavailableError",
    "AggregateEvaluationResult",
    "BaselineEvaluationResult",
    "FoldEvaluationResult",
    "PredictionValidationError",
    "PredictionValidationResult",
    "PredictionValidator",
    "PredictionValidationSummary",
    "ResourceMetadata",
    "build_baseline",
    "evaluate_baseline",
    "validate_predictions",
    "validate_submission",
    "validate_submission_schema",
    "DEFAULT_METRIC_REGISTRY",
    "MetricDefinition",
    "MetricRegistry",
    "MetricScore",
    "UnknownMetricError",
    "evaluate_metric",
    "DeterministicSplitter",
    "SplitFold",
    "SplitPlan",
    "create_split_plan",
    "generate_splits",
]
