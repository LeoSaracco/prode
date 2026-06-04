from src.simulation.group_simulator import GroupSimulator


def test_group_simulator_uses_predictor_probs():
    teams = ["Favorite", "B", "C", "D"]

    def predictor(team_a, team_b):
        if team_a == "Favorite":
            return {"p_win_a": 0.9, "p_draw": 0.08, "p_win_b": 0.02}
        if team_b == "Favorite":
            return {"p_win_a": 0.02, "p_draw": 0.08, "p_win_b": 0.9}
        return {"p_win_a": 0.35, "p_draw": 0.3, "p_win_b": 0.35}

    df = GroupSimulator(predictor=predictor).simulate_group("X", teams=teams, n_sims=1000)
    favorite = df.set_index("team").loc["Favorite"]

    assert favorite["prob_1st"] == df["prob_1st"].max()
    assert favorite["qualify_direct_prob"] == df["qualify_direct_prob"].max()


def test_group_simulator_argentina_like_favorite():
    teams = ["Argentina", "Austria", "Algeria", "Jordan"]
    strength = {"Argentina": 4, "Austria": 3, "Algeria": 2, "Jordan": 1}

    def predictor(team_a, team_b):
        diff = strength[team_a] - strength[team_b]
        if diff > 0:
            return {"p_win_a": 0.72, "p_draw": 0.18, "p_win_b": 0.10}
        return {"p_win_a": 0.10, "p_draw": 0.18, "p_win_b": 0.72}

    df = GroupSimulator(predictor=predictor).simulate_group("J", teams=teams, n_sims=1500)
    rows = df.set_index("team")

    assert rows.loc["Argentina", "prob_1st"] > rows.loc["Austria", "prob_1st"]
    assert rows.loc["Argentina", "qualify_direct_prob"] > rows.loc["Algeria", "qualify_direct_prob"]


def test_group_simulator_does_not_depend_only_on_poisson_when_predictor_probs_exist():
    teams = ["A", "B", "C", "D"]

    class FlatPoisson:
        def predict_goals(self, team_a, team_b):
            return 1.0, 1.0

    def predictor(team_a, team_b):
        if team_a == "D":
            return {"p_win_a": 0.88, "p_draw": 0.08, "p_win_b": 0.04}
        if team_b == "D":
            return {"p_win_a": 0.04, "p_draw": 0.08, "p_win_b": 0.88}
        return {"p_win_a": 0.35, "p_draw": 0.3, "p_win_b": 0.35}

    df = GroupSimulator(poisson_model=FlatPoisson(), predictor=predictor).simulate_group(
        "X", teams=teams, n_sims=1000
    )
    assert df.iloc[0]["team"] == "D"
