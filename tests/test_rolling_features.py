import numpy as np
import pandas as pd

from src.models.trainer import RollingNationalTeamFeatureBuilder


def _df(second_score=(0, 2), first_score=(1, 0)):
    return pd.DataFrame([
        {
            "date": pd.Timestamp("2024-01-01"),
            "team_a": "Alpha",
            "team_b": "Beta",
            "goals_a": first_score[0],
            "goals_b": first_score[1],
            "result": "W" if first_score[0] > first_score[1] else "L",
            "neutral": True,
            "tournament": "Friendly",
        },
        {
            "date": pd.Timestamp("2024-02-01"),
            "team_a": "Alpha",
            "team_b": "Beta",
            "goals_a": second_score[0],
            "goals_b": second_score[1],
            "result": "W" if second_score[0] > second_score[1] else "L",
            "neutral": True,
            "tournament": "Friendly",
        },
        {
            "date": pd.Timestamp("2024-03-01"),
            "team_a": "Gamma",
            "team_b": "Delta",
            "goals_a": 0,
            "goals_b": 0,
            "result": "D",
            "neutral": True,
            "tournament": "Friendly",
        },
        {
            "date": pd.Timestamp("2024-04-01"),
            "team_a": "Gamma",
            "team_b": "Delta",
            "goals_a": 1,
            "goals_b": 0,
            "result": "W",
            "neutral": True,
            "tournament": "Friendly",
        },
    ])


def test_rolling_features_no_future_leakage():
    builder_a = RollingNationalTeamFeatureBuilder()
    x_train_a, *_ = builder_a.build_split_matrices(_df(second_score=(0, 5)), train_end=1, val_end=3)

    builder_b = RollingNationalTeamFeatureBuilder()
    x_train_b, *_ = builder_b.build_split_matrices(_df(second_score=(5, 0)), train_end=1, val_end=3)

    np.testing.assert_allclose(x_train_a[0], x_train_b[0])


def test_rolling_features_current_match_result_is_not_in_features():
    builder_a = RollingNationalTeamFeatureBuilder()
    x_train_a, *_ = builder_a.build_split_matrices(_df(first_score=(1, 0)), train_end=1, val_end=3)

    builder_b = RollingNationalTeamFeatureBuilder()
    x_train_b, *_ = builder_b.build_split_matrices(_df(first_score=(8, 0)), train_end=1, val_end=3)

    np.testing.assert_allclose(x_train_a[0], x_train_b[0])


def test_reverse_perspective_only_in_train_split():
    builder = RollingNationalTeamFeatureBuilder()
    x_train, y_train, x_val, y_val, x_test, y_test, *_ = builder.build_split_matrices(
        _df(),
        train_end=2,
        val_end=3,
    )

    assert len(x_train) == 4
    assert len(y_train) == 4
    assert len(x_val) == 1
    assert len(y_val) == 1
    assert len(x_test) == 1
    assert len(y_test) == 1
