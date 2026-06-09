from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_runtime
from api.schemas import (
    BracketResponse,
    GroupSimulationResponse,
    TournamentSimulationResponse,
)
from config.wc2026_groups import GROUPS
from src.runtime import PredictionRuntime
from src.simulation.group_simulator import GroupSimulator
from src.simulation.tournament_simulator import TournamentSimulator

router = APIRouter()


def _cached_predictor(runtime: PredictionRuntime):
    def predictor(team_a: str, team_b: str) -> dict:
        return runtime.cached_predict(team_a, team_b)
    return predictor


@router.get("/simulate/group/{group_name}", response_model=GroupSimulationResponse)
def simulate_group(
    group_name: str,
    n_sims: int = Query(default=10000, ge=100, le=100000),
    runtime: PredictionRuntime = Depends(get_runtime),
) -> GroupSimulationResponse:
    group = group_name.upper()
    if group not in GROUPS:
        raise HTTPException(status_code=400, detail=f"Unknown group: {group_name}")
    simulator = GroupSimulator(poisson_model=runtime.poisson, predictor=_cached_predictor(runtime))
    df = simulator.simulate_group(group, n_sims=n_sims)
    fixtures = simulator.simulate_group_fixtures(group, n_sims=n_sims)
    return GroupSimulationResponse(
        group=group,
        n_sims=n_sims,
        results=df.to_dict(orient="records"),
        fixtures=fixtures,
    )


@router.get("/simulate/tournament")
def simulate_tournament(
    n_sims: int = Query(default=5000, ge=100, le=100000),
    top_n: int = Query(default=20, ge=1, le=48),
    output: str = Query(default="list", pattern="^(list|bracket)$"),
    runtime: PredictionRuntime = Depends(get_runtime),
):
    simulator = TournamentSimulator(
        poisson_model=runtime.poisson,
        predictor=_cached_predictor(runtime),
    )
    if output == "bracket":
        bracket_data = simulator.simulate_bracket(n_sims=n_sims)
        return BracketResponse(n_sims=n_sims, rounds=bracket_data)
    else:
        df = simulator.simulate(n_sims=n_sims).head(top_n)
        return TournamentSimulationResponse(n_sims=n_sims, results=df.to_dict(orient="records"))
