import pandas as pd

from src.models.poisson_model import PoissonGoalModel


def _match(team_a, team_b, goals_a, goals_b, date="2024-01-01"):
    return {
        "date": pd.Timestamp(date),
        "team_a": team_a,
        "team_b": team_b,
        "goals_a": goals_a,
        "goals_b": goals_b,
        "neutral": True,
    }


def test_poisson_empirical_defense_sign():
    rows = []
    for i in range(12):
        rows.append(_match("Strong", "Weak", 3, 0, f"2024-01-{i + 1:02d}"))
        rows.append(_match("Strong", "Average", 2, 0, f"2024-02-{i + 1:02d}"))
        rows.append(_match("Average", "Weak", 2, 0, f"2024-03-{i + 1:02d}"))

    model = PoissonGoalModel().fit(pd.DataFrame(rows))

    strong_xg, _ = model.predict_goals("Strong", "Weak")
    weak_xg, _ = model.predict_goals("Weak", "Strong")
    assert strong_xg > weak_xg


def test_poisson_bad_defense_increases_opponent_xg():
    rows = []
    for i in range(12):
        rows.append(_match("Attack", "StrongDef", 1, 0, f"2024-01-{i + 1:02d}"))
        rows.append(_match("Attack", "WeakDef", 3, 1, f"2024-02-{i + 1:02d}"))
        rows.append(_match("Neutral", "StrongDef", 0, 0, f"2024-03-{i + 1:02d}"))
        rows.append(_match("Neutral", "WeakDef", 3, 1, f"2024-04-{i + 1:02d}"))

    model = PoissonGoalModel().fit(pd.DataFrame(rows))

    xg_vs_strong_def, _ = model.predict_goals("Attack", "StrongDef")
    xg_vs_weak_def, _ = model.predict_goals("Attack", "WeakDef")
    assert xg_vs_weak_def > xg_vs_strong_def


def test_prediction_probabilities_sum_to_one():
    model = PoissonGoalModel()
    model.attack_ = {"A": 0.6, "B": 0.1}
    model.defense_ = {"A": 0.4, "B": -0.2}
    model.is_fitted_ = True

    probs = model.predict_outcome_probs("A", "B")
    assert 0.999 <= sum(probs) <= 1.001


def test_top_scoreline_compatible_with_reasonable_xg():
    model = PoissonGoalModel()
    model.attack_ = {"A": 0.3, "B": 0.0}
    model.defense_ = {"A": 0.1, "B": 0.0}
    model.is_fitted_ = True

    xg_a, xg_b = model.predict_goals("A", "B")
    goals_a, goals_b, prob = model.get_top_scorelines("A", "B", n=1)[0]

    assert prob > 0
    assert abs(goals_a - xg_a) <= 2
    assert abs(goals_b - xg_b) <= 2


def test_representative_scoreline_conditions_on_outcome_and_xg_volume():
    model = PoissonGoalModel()
    model.attack_ = {"A": 0.8, "B": 0.0}
    model.defense_ = {"A": 0.4, "B": -0.2}
    model.is_fitted_ = True

    goals_a, goals_b, prob = model.get_representative_scoreline("A", "B", "win_a")

    assert goals_a > goals_b
    assert goals_a >= 2
    assert prob > 0
