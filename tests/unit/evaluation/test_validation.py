from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlquest.domain import TaskType
from mlquest.evaluation.validation import (
    PredictionValidationError,
    validate_predictions,
    validate_submission_schema,
)


def test_label_predictions_must_match_shape_and_label_domain() -> None:
    with pytest.raises(PredictionValidationError, match="row count"):
        validate_predictions(["yes", "no"], ["yes"], TaskType.CLASSIFICATION)
    with pytest.raises(PredictionValidationError, match="outside the label domain"):
        validate_predictions(["yes", "no"], ["maybe", "no"], TaskType.CLASSIFICATION)


def test_probability_predictions_require_finite_valid_rows_and_matching_classes() -> None:
    good = validate_predictions(
        ["no", "yes"],
        [[0.8, 0.2], [0.1, 0.9]],
        TaskType.CLASSIFICATION,
        prediction_kind="probability",
    )
    assert good.shape == (2, 2)
    assert good.label_domain == ("no", "yes")

    with pytest.raises(PredictionValidationError, match="finite"):
        validate_predictions(
            [0, 1], [np.nan, 0.9], TaskType.CLASSIFICATION, prediction_kind="probability"
        )
    with pytest.raises(PredictionValidationError, match="sum to one"):
        validate_predictions(
            [0, 1], [[0.8, 0.8], [0.1, 0.9]], TaskType.CLASSIFICATION, prediction_kind="probability"
        )
    with pytest.raises(PredictionValidationError, match="columns"):
        validate_predictions(
            [0, 1],
            [[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]],
            TaskType.CLASSIFICATION,
            prediction_kind="probability",
        )


def test_regression_predictions_are_numeric_one_dimensional_and_finite() -> None:
    result = validate_predictions([1, 2], [1.5, 2.5], TaskType.REGRESSION, prediction_kind="value")
    assert result.n_rows == 2
    with pytest.raises(PredictionValidationError, match="one-dimensional"):
        validate_predictions([1, 2], [[1], [2]], TaskType.REGRESSION, prediction_kind="value")
    with pytest.raises(PredictionValidationError, match="numeric"):
        validate_predictions(
            [1, 2], ["bad", "values"], TaskType.REGRESSION, prediction_kind="value"
        )


def test_submission_schema_checks_columns_rows_finite_values_and_labels() -> None:
    submission = pd.DataFrame({"row_id": [1, 2], "prediction": ["yes", "no"]})
    assert validate_submission_schema(
        submission,
        expected_rows=2,
        expected_columns=["row_id", "prediction"],
        label_columns=["prediction"],
        label_domain=["yes", "no"],
    ).equals(submission)

    with pytest.raises(PredictionValidationError, match="columns"):
        validate_submission_schema(
            submission,
            expected_rows=2,
            expected_columns=["prediction"],
        )
