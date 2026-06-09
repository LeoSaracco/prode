from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_runtime
from api.schemas import (
    FeatureImpact,
    FixturePrediction,
    FixturesResponse,
    GroupsResponse,
    MatchResult,
    PredictRequest,
    Scoreline,
    TeamInfo,
    TeamsResponse,
)
from config.team_aliases import TEAM_ALIASES, resolve_team_name
from config.wc2026_groups import CONFEDERATION, GROUPS, get_group_for_team
from src.features.elo_features import get_elo_rating
from src.runtime import PredictionRuntime, predict_match

router = APIRouter()


def _scoreline_tuple_to_schema(item) -> Scoreline:
    goals_a, goals_b, probability = item
    return Scoreline(goals_a=int(goals_a), goals_b=int(goals_b), probability=round(float(probability), 4))


def _format_prediction(result: dict, shap_features: list[tuple[str, float]]) -> MatchResult:
    scorelines = [_scoreline_tuple_to_schema(item) for item in result.get("top_scorelines", [])]
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
        outcome_scoreline=_scoreline_tuple_to_schema(outcome_scoreline) if outcome_scoreline else None,
        exact_most_likely_scoreline=_scoreline_tuple_to_schema(exact_scoreline) if exact_scoreline else None,
        most_likely_scoreline=_scoreline_tuple_to_schema(outcome_scoreline) if outcome_scoreline else (scorelines[0] if scorelines else None),
        top_scorelines=scorelines,
        upset_risk=result["upset_risk"],
        top_features=top_features,
        elo={"team_a": result["elo_a"], "team_b": result["elo_b"], "diff": result["elo_diff"]},
        model_breakdown={
            key: [round(float(v), 4) for v in value]
            for key, value in result.get("model_breakdown", {}).items()
        },
    )


# ── Fixture helpers ──────────────────────────────────────────────────────────

_MD1_END = pd.Timestamp("2026-06-17")
_MD2_END = pd.Timestamp("2026-06-23")


def _compute_matchday(date_val: pd.Timestamp) -> int:
    if date_val <= _MD1_END:
        return 1
    if date_val <= _MD2_END:
        return 2
    return 3


@lru_cache(maxsize=1)
def _load_wc_fixtures() -> pd.DataFrame:
    csv_path = "data/raw/international/results.csv"
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-06-01")].copy()
    wc["team_a"] = wc["home_team"].apply(lambda x: resolve_team_name(str(x)))
    wc["team_b"] = wc["away_team"].apply(lambda x: resolve_team_name(str(x)))
    wc = wc[wc["team_a"].notna() & wc["team_b"].notna()]
    wc["group"] = wc["team_a"].apply(get_group_for_team)
    wc["matchday"] = wc["date"].apply(_compute_matchday)
    wc["venue"] = wc.apply(lambda r: f"{r['city']}, {r['country']}" if pd.notna(r["city"]) else None, axis=1)
    return wc


# ── Endpoints ────────────────────────────────────────────────────────────────


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


@router.get("/fixtures", response_model=FixturesResponse)
def fixtures(runtime: PredictionRuntime = Depends(get_runtime)) -> FixturesResponse:
    try:
        df = _load_wc_fixtures()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cannot load fixtures: {e}") from e

    rows = df.to_dict(orient="records")

    pair_to_row: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["team_a"], row["team_b"])
        if key not in pair_to_row:
            pair_to_row[key] = row

    unique_pairs = list(pair_to_row.keys())

    unique_keys = [(k[0], k[1]) for k in unique_pairs]
    for team_a, team_b in unique_keys:
        runtime.cached_predict(team_a, team_b)

    items: list[FixturePrediction] = []
    for row in rows:
        p = runtime.cached_predict(row["team_a"], row["team_b"])
        scoreline = p.get("outcome_scoreline")
        items.append(FixturePrediction(
            date=row["date"].strftime("%Y-%m-%d"),
            matchday=int(row["matchday"]),
            group=row["group"] or "",
            team_a=row["team_a"],
            team_b=row["team_b"],
            venue=row.get("venue"),
            predicted_outcome=p.get("predicted_outcome", "draw"),
            outcome_scoreline=_scoreline_tuple_to_schema(scoreline) if scoreline else None,
            win_a=round(float(p.get("p_win_a", 0)), 4),
            draw=round(float(p.get("p_draw", 0)), 4),
            win_b=round(float(p.get("p_win_b", 0)), 4),
            confidence=p.get("confidence", "BAJO"),
            xg_a=round(float(p.get("xg_a", 0)), 2),
            xg_b=round(float(p.get("xg_b", 0)), 2),
        ))
    return FixturesResponse(fixtures=items)
