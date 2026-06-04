from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import load_models_on_startup
from api.routers import health, predictions, simulation


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models_on_startup(app)
    yield


app = FastAPI(
    title="prode-ML API",
    description="FIFA World Cup 2026 predictions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
app.include_router(simulation.router, prefix="/api/v1", tags=["simulation"])
