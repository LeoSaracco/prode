from src.simulation.tournament_simulator import TournamentSimulator


def test_tournament_sets_round_32_probability_and_title_tier():
    strength = {
        "England": 10,
        "Brazil": 9,
        "Argentina": 8,
        "Spain": 7,
    }

    def predictor(team_a, team_b):
        diff = strength.get(team_a, 1) - strength.get(team_b, 1)
        if diff > 0:
            return {"p_win_a": 0.85, "p_draw": 0.10, "p_win_b": 0.05, "xg_a": 2.0, "xg_b": 0.7}
        if diff < 0:
            return {"p_win_a": 0.05, "p_draw": 0.10, "p_win_b": 0.85, "xg_a": 0.7, "xg_b": 2.0}
        return {"p_win_a": 0.35, "p_draw": 0.30, "p_win_b": 0.35, "xg_a": 1.0, "xg_b": 1.0}

    df = TournamentSimulator(predictor=predictor).simulate(n_sims=80)

    assert df["p_round_32"].max() > 0
    assert (df["p_round_32"] >= df["p_round_16"]).all()
    assert set(df["title_tier"].unique()) <= {"candidate", "aspirant"}
    assert (df.loc[df["p_champion"] >= 0.08, "title_tier"] == "candidate").all()
    assert (df.loc[df["p_champion"] < 0.08, "title_tier"] == "aspirant").all()
