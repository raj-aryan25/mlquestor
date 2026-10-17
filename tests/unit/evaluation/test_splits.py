from __future__ import annotations

import numpy as np
import pandas as pd

from mlquest.domain import SplitSpecification, SplitStrategy, TaskType
from mlquest.evaluation.splits import create_split_plan


def _data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [0, 1] * 10,
            "group": ["a"] * 5 + ["b"] * 5 + ["c"] * 5 + ["d"] * 5,
            "event_time": pd.date_range("2026-01-01", periods=20, freq="D"),
        }
    )


def test_random_and_stratified_plans_are_deterministic_and_persist_assignments() -> None:
    frame = _data()
    for strategy in (SplitStrategy.RANDOM, SplitStrategy.STRATIFIED, SplitStrategy.HOLDOUT):
        spec = SplitSpecification(strategy=strategy, test_size=0.25, random_seed=17)
        first = create_split_plan(frame, spec, y=frame["label"], task_type=TaskType.CLASSIFICATION)
        second = create_split_plan(frame, spec, y=frame["label"], task_type=TaskType.CLASSIFICATION)

        assert first.split_identity == second.split_identity
        assert first.assignments == second.assignments
        assert first.to_json() == second.to_json()


def test_holdout_assignments_distinguish_training_and_evaluation_rows() -> None:
    frame = _data()
    spec = SplitSpecification(strategy=SplitStrategy.HOLDOUT, test_size=0.25, random_seed=17)
    plan = create_split_plan(frame, spec)

    assert set(plan.assignments) == {0, 1}
    assert set(np.where(np.asarray(plan.assignments) == 1)[0]) == set(plan.validation_indices)


def test_group_plan_keeps_groups_wholly_in_one_side() -> None:
    frame = _data()
    spec = SplitSpecification(
        strategy=SplitStrategy.GROUP, group_column="group", test_size=0.25, random_seed=3
    )
    plan = create_split_plan(frame, spec)
    train_groups = set(frame.iloc[list(plan.train_indices)]["group"])
    test_groups = set(frame.iloc[list(plan.test_indices)]["group"])

    assert train_groups.isdisjoint(test_groups)


def test_time_plan_is_ordered_and_cross_validation_covers_each_row_once() -> None:
    frame = _data()
    time_spec = SplitSpecification(
        strategy=SplitStrategy.TIME, time_column="event_time", test_size=0.2
    )
    time_plan = create_split_plan(frame, time_spec)
    assert max(time_plan.train_indices) < min(time_plan.test_indices)

    cv_spec = SplitSpecification(
        strategy=SplitStrategy.CROSS_VALIDATION, n_splits=4, random_seed=11, shuffle=True
    )
    cv_plan = create_split_plan(frame, cv_spec, y=frame["label"], task_type=TaskType.CLASSIFICATION)
    assert len(cv_plan.folds) == 4
    assert sorted(cv_plan.assignments) == [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5


def test_time_cross_validation_keeps_each_validation_window_in_the_future() -> None:
    frame = _data()
    spec = SplitSpecification(strategy=SplitStrategy.TIME, time_column="event_time", n_splits=3)
    plan = create_split_plan(frame, spec)

    assert len(plan.folds) == 3
    for fold in plan.folds:
        assert max(fold.train_indices) < min(fold.validation_indices)
