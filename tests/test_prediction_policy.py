from src.prediction_policy import enrich_prediction_result, outcome_scoreline


def test_win_prediction_uses_compatible_scoreline_not_global_draw():
    result = {
        "p_win_a": 0.47,
        "p_draw": 0.32,
        "p_win_b": 0.21,
        "xg_a": 0.92,
        "xg_b": 0.82,
        "top_scorelines": [
            (0, 0, 0.20),
            (1, 1, 0.16),
            (1, 0, 0.14),
            (0, 1, 0.12),
        ],
    }

    scoreline = outcome_scoreline(result)

    assert scoreline == (1, 0, 0.14)


def test_enriched_result_keeps_exact_mode_as_alternative():
    result = enrich_prediction_result({
        "p_win_a": 0.47,
        "p_draw": 0.32,
        "p_win_b": 0.21,
        "xg_a": 0.92,
        "xg_b": 0.82,
        "top_scorelines": [(0, 0, 0.20), (1, 0, 0.14)],
    })

    assert result["predicted_outcome"] == "win_a"
    assert result["outcome_scoreline"] == (1, 0, 0.14)
    assert result["most_likely_scoreline"] == (1, 0, 0.14)
    assert result["exact_most_likely_scoreline"] == (0, 0, 0.20)


def test_draw_prediction_uses_draw_scoreline():
    result = {
        "p_win_a": 0.31,
        "p_draw": 0.38,
        "p_win_b": 0.31,
        "xg_a": 1.1,
        "xg_b": 1.0,
        "top_scorelines": [(1, 0, 0.13), (1, 1, 0.12)],
    }

    assert outcome_scoreline(result) == (1, 1, 0.12)


def test_high_favorite_xg_prefers_two_goal_scoreline_over_modal_one_nil():
    result = {
        "p_win_a": 0.62,
        "p_draw": 0.22,
        "p_win_b": 0.16,
        "xg_a": 2.2,
        "xg_b": 0.7,
        "top_scorelines": [
            (1, 0, 0.15),
            (2, 0, 0.14),
            (2, 1, 0.10),
            (3, 0, 0.09),
        ],
    }

    assert outcome_scoreline(result) in {(2, 0, 0.14), (2, 1, 0.10), (3, 0, 0.09)}


def test_both_teams_with_good_xg_prefers_two_one_over_one_nil():
    result = {
        "p_win_a": 0.48,
        "p_draw": 0.28,
        "p_win_b": 0.24,
        "xg_a": 1.8,
        "xg_b": 1.4,
        "top_scorelines": [
            (1, 0, 0.15),
            (2, 1, 0.13),
            (2, 0, 0.10),
            (3, 1, 0.08),
        ],
    }

    assert outcome_scoreline(result) == (2, 1, 0.13)


def test_low_xg_favorite_can_remain_one_nil():
    result = {
        "p_win_a": 0.46,
        "p_draw": 0.34,
        "p_win_b": 0.20,
        "xg_a": 0.8,
        "xg_b": 0.6,
        "top_scorelines": [
            (1, 0, 0.16),
            (2, 0, 0.05),
            (2, 1, 0.03),
        ],
    }

    assert outcome_scoreline(result) == (1, 0, 0.16)


def test_high_xg_draw_prefers_scored_draw_over_nil_nil():
    result = {
        "p_win_a": 0.31,
        "p_draw": 0.38,
        "p_win_b": 0.31,
        "xg_a": 1.6,
        "xg_b": 1.5,
        "top_scorelines": [
            (0, 0, 0.13),
            (1, 1, 0.12),
            (2, 2, 0.05),
        ],
    }

    assert outcome_scoreline(result) in {(1, 1, 0.12), (2, 2, 0.05)}
