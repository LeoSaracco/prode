"""
Ensemble with accuracy-weighted voting.

Architecture:
  Base models: RF, XGBoost, LightGBM, CatBoost, Elo
  Voting:       Weighted by each model's accuracy on the val set.
                Elo always gets 0.10 as a stable baseline.
                Weights are normalized to sum to 1.0.
"""

import json
import logging
from pathlib import Path

import numpy as np

from config.settings import (
    MODELS_DIR,
    CONFIDENCE_HIGH_PROB, CONFIDENCE_HIGH_ELO_DIFF,
    CONFIDENCE_MEDIUM_PROB, CONFIDENCE_MEDIUM_ELO_DIFF,
)
from src.features.elo_features import get_elo_rating

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Weighted voting ensemble based on per-model validation accuracy."""

    def __init__(
        self,
        xgb_model=None,
        lgbm_model=None,
        elo_model=None,
        poisson_model=None,
        rf_model=None,
        catboost_model=None,
        scaler=None,
    ):
        self.xgb = xgb_model
        self.lgbm = lgbm_model
        self.elo = elo_model
        self.poisson = poisson_model
        self.rf = rf_model
        self.catboost = catboost_model
        self.scaler = scaler

        self.voting_weights = None
        self.confidence_thresholds = None
        self.calibrators = None

    def set_weights_from_accuracies(self, accuracies: dict[str, float]) -> None:
        """Compute voting weights from val-set accuracies.

        Args:
            accuracies: dict like {'rf': 0.45, 'xgb': 0.44, ...}
        """
        weights = {}
        for name, acc in accuracies.items():
            if name == "elo":
                weights[name] = 0.10
            else:
                weights[name] = max(0.05, acc - 0.33)

        self.load_or_set_weights(weights)

    def load_or_set_weights(self, weights: dict[str, float]) -> None:
        total = sum(weights.values())
        if total > 0:
            self.voting_weights = {k: v / total for k, v in weights.items()}
        else:
            self.voting_weights = {
                "rf": 0.30, "xgb": 0.20, "lgbm": 0.20,
                "catboost": 0.20, "elo": 0.10,
            }
        logger.info("Voting weights: %s",
                     {k: f"{v:.2f}" for k, v in self.voting_weights.items()})

    def predict(
        self,
        team_a: str,
        team_b: str,
        feature_vector: np.ndarray,
        elo_df=None,
    ) -> dict:
        model_vector = self._scale_features(feature_vector)
        base_probs = self._get_all_base_probs(model_vector, team_a, team_b)

        w = self.voting_weights or self._default_weights()

        p_win = sum(w.get(name, 0) * base_probs[name][0] for name in base_probs)
        p_draw = sum(w.get(name, 0) * base_probs[name][1] for name in base_probs)
        p_loss = 1.0 - p_win - p_draw

        total = p_win + p_draw + p_loss
        if total > 0:
            p_win, p_draw, p_loss = p_win / total, p_draw / total, p_loss / total
        else:
            p_win, p_draw, p_loss = 0.40, 0.25, 0.35

        if self.calibrators is not None:
            try:
                T = float(self.calibrators.get("temperature", 1.0))
                raw = np.array([[p_win, p_draw, p_loss]])
                powered = np.power(np.clip(raw, 1e-9, 1.0), 1.0 / max(0.05, T))
                s = powered.sum()
                if s > 0:
                    powered /= s
                p_win, p_draw, p_loss = float(powered[0, 0]), float(powered[0, 1]), float(powered[0, 2])
            except Exception:
                pass

        xg_a, xg_b = (1.3, 1.1)
        if self.poisson:
            xg_a, xg_b = self.poisson.predict_goals(team_a, team_b)

        elo_a = get_elo_rating(team_a, elo_df) if elo_df is not None else 1800
        elo_b = get_elo_rating(team_b, elo_df) if elo_df is not None else 1800
        elo_diff = abs(elo_a - elo_b)
        max_prob = max(p_win, p_draw, p_loss)
        confidence = self._compute_confidence(max_prob, elo_diff)

        top_scorelines = []
        if self.poisson:
            top_scorelines = self.poisson.get_top_scorelines(team_a, team_b, n=5)

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
                name: base_probs.get(name, (0.4, 0.25, 0.35))
                for name in w
            },
        }

    def predict_voting_probs(self, base_probs_dict: dict[str, np.ndarray]) -> np.ndarray:
        w = self.voting_weights or self._default_weights()
        n = len(next(iter(base_probs_dict.values())))
        out = np.zeros((n, 3))
        for name, probs in base_probs_dict.items():
            weight = w.get(name, 0)
            out += weight * probs
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums

    def _get_all_base_probs(
        self, x: np.ndarray, team_a: str, team_b: str
    ) -> dict[str, tuple[float, float, float]]:
        probs = {}
        probs["rf"] = self._safe_predict(self.rf, x, team_a, team_b)
        probs["xgb"] = self._safe_predict(self.xgb, x, team_a, team_b)
        probs["lgbm"] = self._safe_predict(self.lgbm, x, team_a, team_b)
        probs["catboost"] = self._safe_predict(self.catboost, x, team_a, team_b)
        probs["elo"] = self.elo.predict_proba(team_a, team_b) if self.elo else (0.4, 0.25, 0.35)
        return probs

    def _safe_predict(
        self, model, x: np.ndarray, team_a: str, team_b: str
    ) -> tuple[float, float, float]:
        if model is None or not getattr(model, "is_fitted_", False):
            if self.elo:
                return self.elo.predict_proba(team_a, team_b)
            return (0.40, 0.25, 0.35)
        try:
            return model.predict_proba(x)
        except Exception:
            return (0.40, 0.25, 0.35)

    def _default_weights(self) -> dict[str, float]:
        return {"rf": 0.30, "xgb": 0.20, "lgbm": 0.20, "catboost": 0.20, "elo": 0.10}

    def _compute_confidence(self, max_prob: float, elo_diff: float) -> str:
        if self.confidence_thresholds:
            high = float(self.confidence_thresholds.get("high_prob", CONFIDENCE_HIGH_PROB))
            medium = float(self.confidence_thresholds.get("medium_prob", CONFIDENCE_MEDIUM_PROB))
            if max_prob >= high:
                return "ALTO"
            if max_prob >= medium or elo_diff > CONFIDENCE_MEDIUM_ELO_DIFF:
                return "MEDIO"
            return "BAJO"
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
        except Exception:
            return x

    def get_shap_explanation(self, feature_vector: np.ndarray) -> list[tuple[str, float]]:
        if self.xgb and getattr(self.xgb, "is_fitted_", False):
            return self.xgb.explain(feature_vector)
        return []

    def save_weights(self, weights: dict[str, float]) -> None:
        path = MODELS_DIR / "voting_weights.json"
        with open(path, "w") as f:
            json.dump(weights, f, indent=2)
        logger.info("Voting weights saved to %s", path)

    def load_weights(self) -> dict[str, float] | None:
        path = MODELS_DIR / "voting_weights.json"
        if path.exists():
            try:
                with open(path) as f:
                    w = json.load(f)
                self.voting_weights = w
                logger.info("Voting weights loaded: %s",
                             {k: f"{v:.2f}" for k, v in w.items()})
                return w
            except Exception:
                pass
        return None

    def load_confidence_thresholds(self) -> dict | None:
        path = MODELS_DIR / "confidence_thresholds.json"
        if path.exists():
            try:
                with open(path) as f:
                    thresholds = json.load(f)
                self.confidence_thresholds = thresholds
                logger.info("Confidence thresholds loaded: %s", thresholds)
                return thresholds
            except Exception:
                pass
        return None

    def load_calibrators(self) -> dict | None:
        import joblib
        path = MODELS_DIR / "probability_calibrators.pkl"
        if path.exists():
            try:
                data = joblib.load(path)
                if isinstance(data, dict) and "temperature" in data:
                    self.calibrators = data
                    logger.info("Temperature calibrator loaded: T=%.4f", data["temperature"])
                    return self.calibrators
                else:
                    logger.warning("Calibrator format unrecognized, skipping.")
            except Exception as e:
                logger.warning("Could not load calibrators: %s", e)
        return None
