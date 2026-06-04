from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_runtime
from api.schemas import GroupSimulationResponse, TournamentSimulationResponse
from config.wc2026_groups import GROUPS
from src.runtime import PredictionRuntime
from src.simulation.group_simulator import GroupSimulator
from src.simulation.tournament_simulator import TournamentSimulator

router = APIRouter()


@router.get("/simulate/group/{group_name}", response_model=GroupSimulationResponse)
def simulate_group(
    group_name: str,
    n_sims: int = Query(default=10000, ge=100, le=100000),
    runtime: PredictionRuntime = Depends(get_runtime),
) -> GroupSimulationResponse:
    group = group_name.upper()
    if group not in GROUPS:
        raise HTTPException(status_code=400, detail=f"Unknown group: {group_name}")
    simulator = GroupSimulator(poisson_model=runtime.poisson)
    df = simulator.simulate_group(group, n_sims=n_sims)
    fixtures = simulator.simulate_group_fixtures(group, n_sims=n_sims)
    return GroupSimulationResponse(
        group=group,
        n_sims=n_sims,
        results=df.to_dict(orient="records"),
        fixtures=fixtures,
    )


@router.get("/simulate/tournament", response_model=TournamentSimulationResponse)
def simulate_tournament(
    n_sims: int = Query(default=5000, ge=100, le=100000),
    top_n: int = Query(default=20, ge=1, le=48),
    runtime: PredictionRuntime = Depends(get_runtime),
) -> TournamentSimulationResponse:
    df = TournamentSimulator(poisson_model=runtime.poisson).simulate(n_sims=n_sims).head(top_n)
    return TournamentSimulationResponse(n_sims=n_sims, results=df.to_dict(orient="records"))
