from fastapi import APIRouter, Depends

from api.dependencies import get_runtime
from api.schemas import HealthResponse
from config.team_aliases import TEAM_ALIASES
from src.runtime import PredictionRuntime

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(runtime: PredictionRuntime = Depends(get_runtime)) -> HealthResponse:
    return HealthResponse(
        status="ok" if runtime.models_loaded else "degraded",
        models_loaded=runtime.models_loaded,
        teams=len(TEAM_ALIASES),
    )
