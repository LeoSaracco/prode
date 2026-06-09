"""Shared runtime helpers for CLI/API prediction."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd

from config.settings import MODELS_DIR
from config.team_aliases import TEAM_ALIASES, resolve_team_name
from src.data.cache_manager import CacheManager
from src.features.feature_builder import FeatureBuilder
from src.models.catboost_model import CatBoostOutcomeClassifier
from src.models.confederation_models import ConfederationModels
from src.models.elo_model import EloModel
from src.models.ensemble import EnsemblePredictor
from src.models.lgbm_model import LGBMOutcomeClassifier
from src.models.poisson_model import PoissonGoalModel
from src.models.rf_model import RFOutcomeClassifier
from src.models.two_stage import TwoStageClassifier
from src.models.xgb_model import XGBOutcomeClassifier


@dataclass
class PredictionRuntime:
    ensemble: EnsemblePredictor
    feature_builder: FeatureBuilder
    poisson: PoissonGoalModel
    two_stage: TwoStageClassifier | None
    confederation: ConfederationModels | None
    elo_df: pd.DataFrame | None
    elo_dict: dict[str, float] | None
    models_loaded: bool
    pair_cache: dict = None

    def __post_init__(self):
        if self.pair_cache is None:
            self.pair_cache = {}

    def cached_predict(self, team_a: str, team_b: str, include_shap: bool = False, include_aux: bool = False) -> dict:
        key = (team_a, team_b)
        if key not in self.pair_cache:
            self.pair_cache[key] = predict_match(self, team_a, team_b, include_shap=include_shap, include_aux=include_aux)[0]
        return self.pair_cache[key]


def load_prediction_runtime() -> PredictionRuntime:
    cache = CacheManager()
    elo_df = cache.load("elo_ratings")
    elo_dict: dict[str, float] | None = None
    if elo_df is not None and not elo_df.empty and "team" in elo_df.columns:
        elo_dict = dict(zip(elo_df["team"], elo_df["elo_rating"]))

    profiles_path = MODELS_DIR / "inference_team_profiles.json"
    team_profiles_df = pd.read_json(profiles_path) if profiles_path.exists() else cache.load("team_profiles")
    scaler_path = MODELS_DIR / "scaler.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    poisson = PoissonGoalModel().load()
    rf = RFOutcomeClassifier().load()
    xgb = XGBOutcomeClassifier().load()
    lgbm = LGBMOutcomeClassifier().load()
    catboost = CatBoostOutcomeClassifier().load()
    elo_model = EloModel(elo_df=elo_df)

    ensemble = EnsemblePredictor(
        rf_model=rf,
        xgb_model=xgb,
        lgbm_model=lgbm,
        catboost_model=catboost,
        elo_model=elo_model,
        poisson_model=poisson,
        scaler=scaler,
    )
    ensemble.load_weights()
    ensemble.load_confidence_thresholds()
    ensemble.load_calibrators()

    h2h_stats = None
    h2h_path = MODELS_DIR / "inference_h2h_stats.json"
    if h2h_path.exists():
        import json as _json
        try:
            with open(h2h_path, encoding="utf-8") as f:
                h2h_stats = _json.load(f)
        except Exception:
            pass

    two_stage = TwoStageClassifier().load()
    confederation = ConfederationModels().load()

    return PredictionRuntime(
        ensemble=ensemble,
        feature_builder=FeatureBuilder(elo_df=elo_df, team_profiles_df=team_profiles_df, h2h_stats=h2h_stats),
        poisson=poisson,
        two_stage=two_stage if two_stage.is_fitted_ else None,
        confederation=confederation if confederation.is_fitted_ else None,
        elo_df=elo_df,
        elo_dict=elo_dict,
        models_loaded=bool(rf.is_fitted_ or xgb.is_fitted_ or lgbm.is_fitted_ or catboost.is_fitted_),
    )


def resolve_team_or_raise(name: str) -> str:
    resolved = resolve_team_name(name)
    if resolved is not None:
        return resolved
    lower = name.lower()
    for canonical in TEAM_ALIASES:
        if lower in canonical.lower():
            return canonical
    raise ValueError(f"Unknown team: {name}")


def predict_match(
    runtime: PredictionRuntime,
    team_a_raw: str,
    team_b_raw: str,
    include_shap: bool = True,
    include_aux: bool = True,
) -> tuple[dict, list[tuple[str, float]]]:
    team_a = resolve_team_or_raise(team_a_raw)
    team_b = resolve_team_or_raise(team_b_raw)
    x = runtime.feature_builder.build_match_features(team_a, team_b)
    result = runtime.ensemble.predict(team_a, team_b, x, elo_source=runtime.elo_dict or runtime.elo_df)

    if include_aux and runtime.two_stage is not None:
        ts = runtime.two_stage.predict_proba(x)
        result["two_stage_breakdown"] = {
            "p_win": round(ts[0], 4), "p_draw": round(ts[1], 4), "p_loss": round(ts[2], 4)
        }

    if include_aux and runtime.confederation is not None:
        cf = runtime.confederation.predict_proba(x, team_a, team_b)
        result["confederation_breakdown"] = {
            "p_win": round(cf[0], 4), "p_draw": round(cf[1], 4), "p_loss": round(cf[2], 4)
        }

    shap_features: list[tuple[str, float]] = []
    if include_shap:
        shap_features = runtime.ensemble.get_shap_explanation(x)
    return result, shap_features
