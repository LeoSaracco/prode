"""
Ensemble que combina XGBoost, LightGBM, Elo y Poisson.
Blending ponderado con lógica de confianza basada en CI bayesiano.
"""
import logging

import numpy as np

from config.settings import (
    ENSEMBLE_WEIGHTS,
    CONFIDENCE_HIGH_PROB, CONFIDENCE_HIGH_ELO_DIFF,
    CONFIDENCE_MEDIUM_PROB, CONFIDENCE_MEDIUM_ELO_DIFF,
)
from src.features.elo_features import get_elo_rating

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    def __init__(self, xgb_model=None, lgbm_model=None, elo_model=None, poisson_model=None, scaler=None):
        self.xgb = xgb_model
        self.lgbm = lgbm_model
        self.elo = elo_model
        self.poisson = poisson_model
        self.scaler = scaler
        self.w = ENSEMBLE_WEIGHTS

    def predict(
        self,
        team_a: str,
        team_b: str,
        feature_vector: np.ndarray,
        elo_df=None,
    ) -> dict:
        """
        Retorna dict con probabilidades finales, xG esperados, nivel de confianza,
        y el marcador más probable.
        """
        # Obtener predicciones de cada modelo
        model_vector = self._scale_features(feature_vector)
        p_xgb = self._safe_predict(self.xgb, model_vector, team_a, team_b)
        p_lgbm = self._safe_predict(self.lgbm, model_vector, team_a, team_b)
        p_elo = self.elo.predict_proba(team_a, team_b) if self.elo else (0.4, 0.25, 0.35)
        p_poisson = self.poisson.predict_outcome_probs(team_a, team_b) if self.poisson else (0.4, 0.25, 0.35)

        # Blend ponderado
        w = self.w
        p_win = (w["xgb"] * p_xgb[0] + w["lgbm"] * p_lgbm[0] +
                 w["elo"] * p_elo[0] + w["poisson"] * p_poisson[0])
        p_draw = (w["xgb"] * p_xgb[1] + w["lgbm"] * p_lgbm[1] +
                  w["elo"] * p_elo[1] + w["poisson"] * p_poisson[1])
        p_loss = (w["xgb"] * p_xgb[2] + w["lgbm"] * p_lgbm[2] +
                  w["elo"] * p_elo[2] + w["poisson"] * p_poisson[2])

        # Normalizar
        total = p_win + p_draw + p_loss
        p_win, p_draw, p_loss = p_win / total, p_draw / total, p_loss / total

        # xG del modelo Poisson
        xg_a, xg_b = (1.3, 1.1)
        if self.poisson:
            xg_a, xg_b = self.poisson.predict_goals(team_a, team_b)

        # Nivel de confianza
        elo_a = get_elo_rating(team_a, elo_df) if elo_df is not None else 1800
        elo_b = get_elo_rating(team_b, elo_df) if elo_df is not None else 1800
        elo_diff = abs(elo_a - elo_b)
        max_prob = max(p_win, p_draw, p_loss)
        confidence = self._compute_confidence(max_prob, elo_diff)

        # Marcadores más probables
        top_scorelines = []
        if self.poisson:
            top_scorelines = self.poisson.get_top_scorelines(team_a, team_b, n=5)

        # Riesgo de sorpresa (prob del underdog)
        from src.features.risk_features import compute_upset_probability
        underdog_elo = min(elo_a, elo_b)
        favorite_elo = max(elo_a, elo_b)
        upset_risk = compute_upset_probability(favorite_elo, underdog_elo)

        return {
            "team_a": team_a,
            "team_b": team_b,
            "p_win_a": round(p_win, 4),
            "p_draw": round(p_draw, 4),
            "p_win_b": round(p_loss, 4),
            "xg_a": round(xg_a, 2),
            "xg_b": round(xg_b, 2),
            "confidence": confidence,
            "top_scorelines": top_scorelines,
            "upset_risk": round(upset_risk, 4),
            "elo_a": elo_a,
            "elo_b": elo_b,
            "elo_diff": elo_diff,
            "model_breakdown": {
                "xgb": p_xgb,
                "lgbm": p_lgbm,
                "elo": p_elo,
                "poisson": p_poisson,
            },
        }

    def _safe_predict(self, model, x: np.ndarray, team_a: str, team_b: str) -> tuple[float, float, float]:
        if model is None or not getattr(model, "is_fitted_", False):
            # Fallback a Elo si el modelo no está disponible
            if self.elo:
                return self.elo.predict_proba(team_a, team_b)
            return (0.40, 0.25, 0.35)
        try:
            return model.predict_proba(x)
        except Exception as e:
            logger.debug(f"Error en predicción de modelo: {e}")
            return (0.40, 0.25, 0.35)

    def _compute_confidence(self, max_prob: float, elo_diff: float) -> str:
        if max_prob > CONFIDENCE_HIGH_PROB and elo_diff > CONFIDENCE_HIGH_ELO_DIFF:
            return "ALTO"
        elif max_prob > CONFIDENCE_MEDIUM_PROB or elo_diff > CONFIDENCE_MEDIUM_ELO_DIFF:
            return "MEDIO"
        return "BAJO"

    def _scale_features(self, x: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return x
        try:
            return self.scaler.transform(x.reshape(1, -1))[0]
        except Exception as e:
            logger.debug("Error scaling feature vector: %s", e)
            return x

    def get_shap_explanation(self, feature_vector: np.ndarray) -> list[tuple[str, float]]:
        if self.xgb and getattr(self.xgb, "is_fitted_", False):
            return self.xgb.explain(feature_vector)
        return []
