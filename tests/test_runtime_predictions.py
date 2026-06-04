from pathlib import Path

import pytest

from src.runtime import load_prediction_runtime, predict_match


@pytest.mark.skipif(not Path("models/poisson_model.pkl").exists(), reason="trained model artifacts not present")
def test_runtime_prediction_probabilities_sum_to_one():
    runtime = load_prediction_runtime()
    for team_a, team_b in [("Argentina", "Austria"), ("Brazil", "Morocco"), ("France", "Senegal")]:
        result, _ = predict_match(runtime, team_a, team_b)
        total = result["p_win_a"] + result["p_draw"] + result["p_win_b"]
        assert 0.999 <= total <= 1.001
