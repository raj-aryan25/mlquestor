"""Small, deterministic cost estimator for preflight admission."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from .models import CostEstimate, CostObservation, RunRequest


class CostEstimator:
    """Estimate runtime from history, model family, data volume, and config.

    This is intentionally conservative and explainable.  A later planner can
    replace it with a richer model while retaining the same result contract.
    """

    _FAMILY_BASE_SECONDS = {
        "linear": 20.0,
        "logistic": 30.0,
        "tree": 75.0,
        "random_forest": 90.0,
        "gradient_boosting": 110.0,
        "lightgbm": 100.0,
        "xgboost": 120.0,
        "neural_network": 180.0,
    }

    def estimate(
        self,
        request: RunRequest,
        observations: Iterable[CostObservation] = (),
    ) -> CostEstimate:
        family = request.model_family.lower()
        family_observations = [item for item in observations if item.model_family.lower() == family]
        baseline = self._FAMILY_BASE_SECONDS.get(family, 60.0)
        dataset_factor = 1.0 + min(math.log10(max(request.dataset_size, 1)) / 10.0, 2.0)
        configuration_factor = self._configuration_factor(request.configuration)

        if family_observations:
            # Runtime scales sub-linearly with data size, avoiding a sharp jump
            # when a resumed session sees a slightly larger derived dataset.
            observed = sum(item.runtime_seconds for item in family_observations) / len(
                family_observations
            )
            reference_size = sum(item.dataset_size for item in family_observations) / len(
                family_observations
            )
            observed_dataset_factor = (max(request.dataset_size, 1) / max(reference_size, 1)) ** 0.5
            seconds = observed * observed_dataset_factor * configuration_factor
        else:
            seconds = baseline * dataset_factor * configuration_factor

        return CostEstimate(
            model_family=request.model_family,
            dataset_size=request.dataset_size,
            estimated_seconds=max(1.0, round(seconds, 3)),
            based_on_observations=len(family_observations),
            dataset_factor=round(dataset_factor, 6),
            configuration_factor=round(configuration_factor, 6),
        )

    @staticmethod
    def _configuration_factor(configuration: Mapping[str, Any]) -> float:
        factor = 1.0
        for key, value in configuration.items():
            normalized = key.lower()
            if normalized in {"n_estimators", "estimators", "trees", "iterations", "epochs"}:
                try:
                    factor *= max(0.5, min(float(value) / 100.0, 5.0))
                except (TypeError, ValueError):
                    continue
            elif normalized in {"cv", "folds", "n_splits"}:
                try:
                    factor *= max(1.0, min(float(value), 10.0) / 3.0)
                except (TypeError, ValueError):
                    continue
            elif normalized in {"max_depth", "depth"}:
                try:
                    factor *= 1.0 + max(0.0, min(float(value), 20.0)) / 20.0
                except (TypeError, ValueError):
                    continue
            elif normalized in {"seeds", "random_seeds", "n_seeds"}:
                try:
                    factor *= max(1.0, min(float(value), 20.0))
                except (TypeError, ValueError):
                    continue
        return max(0.25, min(factor, 20.0))
