from api.schemas import FeatureImpact, MatchResult, Scoreline


def scoreline_tuple_to_schema(item) -> Scoreline:
    goals_a, goals_b, probability = item
    return Scoreline(goals_a=int(goals_a), goals_b=int(goals_b), probability=round(float(probability), 4))


def format_prediction(result: dict, shap_features: list[tuple[str, float]]) -> MatchResult:
    scorelines = [scoreline_tuple_to_schema(item) for item in result.get("top_scorelines", [])]
    outcome_scoreline = result.get("outcome_scoreline")
    exact_scoreline = result.get("exact_most_likely_scoreline")
    team_a = result["team_a"]
    team_b = result["team_b"]
    top_features = [
        FeatureImpact(
            name=name,
            value=round(float(value), 4),
            direction=team_a if value >= 0 else team_b,
        )
        for name, value in shap_features[:5]
    ]
    return MatchResult(
        team_a=team_a,
        team_b=team_b,
        probabilities={
            "win_a": result["p_win_a"],
            "draw": result["p_draw"],
            "win_b": result["p_win_b"],
        },
        expected_goals={"team_a": result["xg_a"], "team_b": result["xg_b"]},
        confidence=result["confidence"],
        predicted_outcome=result.get("predicted_outcome", "draw"),
        outcome_scoreline=scoreline_tuple_to_schema(outcome_scoreline) if outcome_scoreline else None,
        exact_most_likely_scoreline=scoreline_tuple_to_schema(exact_scoreline) if exact_scoreline else None,
        most_likely_scoreline=scoreline_tuple_to_schema(outcome_scoreline) if outcome_scoreline else (scorelines[0] if scorelines else None),
        top_scorelines=scorelines,
        upset_risk=result["upset_risk"],
        top_features=top_features,
        elo={"team_a": result["elo_a"], "team_b": result["elo_b"], "diff": result["elo_diff"]},
        model_breakdown={
            key: [round(float(v), 4) for v in value]
            for key, value in result.get("model_breakdown", {}).items()
        },
    )
