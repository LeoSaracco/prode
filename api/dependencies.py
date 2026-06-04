"""FastAPI runtime dependencies."""

from fastapi import Request

from src.runtime import PredictionRuntime, load_prediction_runtime


def load_models_on_startup(app) -> None:
    app.state.runtime = load_prediction_runtime()


def get_runtime(request: Request) -> PredictionRuntime:
    return request.app.state.runtime
