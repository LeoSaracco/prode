from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_runtime
from api.schemas import (
    FeatureImpact,
    GroupsResponse,
    MatchResult,
    PredictRequest,
    Scoreline,
    TeamInfo,
    TeamsResponse,
)
from config.team_aliases import TEAM_ALIASES
from config.wc2026_groups import CONFEDERATION, GROUPS, get_group_for_team
from src.features.elo_features import get_elo_rating
from src.runtime import PredictionRuntime, predict_match

router = APIRouter()


def _scoreline_tuple_to_schema(item) -> Scoreline:
    goals_a, goals_b, probability = item
    return Scoreline(goals_a=int(goals_a), goals_b=int(goals_b), probability=round(float(probability), 4))


def _format_prediction(result: dict, shap_features: list[tuple[str, float]]) -> MatchResult:
    scorelines = [_scoreline_tuple_to_schema(item) for item in result.get("top_scorelines", [])]
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
        most_likely_scoreline=scorelines[0] if scorelines else None,
        top_scorelines=scorelines,
        upset_risk=result["upset_risk"],
        top_features=top_features,
        elo={"team_a": result["elo_a"], "team_b": result["elo_b"], "diff": result["elo_diff"]},
        model_breakdown={
            key: [round(float(v), 4) for v in value]
            for key, value in result.get("model_breakdown", {}).items()
        },
    )


@router.get("/teams", response_model=TeamsResponse)
def teams(runtime: PredictionRuntime = Depends(get_runtime)) -> TeamsResponse:
    items = []
    for team in sorted(TEAM_ALIASES):
        items.append(TeamInfo(
            name=team,
            group=get_group_for_team(team),
            elo=get_elo_rating(team, runtime.elo_df),
            confederation=CONFEDERATION.get(team),
        ))
    return TeamsResponse(teams=items)


@router.get("/groups", response_model=GroupsResponse)
def groups() -> GroupsResponse:
    return GroupsResponse(groups=GROUPS)


@router.post("/predict", response_model=MatchResult)
def predict(payload: PredictRequest, runtime: PredictionRuntime = Depends(get_runtime)) -> MatchResult:
    try:
        result, shap_features = predict_match(runtime, payload.team_a, payload.team_b)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Prediction failed: {e}") from e
    return _format_prediction(result, shap_features)
