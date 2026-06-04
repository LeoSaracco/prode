"""
Ensemble with meta-learner (LogisticRegressionCV) + Platt scaling calibration.

Architecture:
  Layer 1 — Base models: RF, XGBoost, LightGBM, CatBoost, Elo
  Layer 2 — Meta-learner: LogisticRegressionCV trained on val-set base predictions
  Layer 3 — Calibrator: CalibratedClassifierCV with isotonic regression on test predictions
"""

import logging
from pathlib import Path

import joblib
import numpy as np

from config.settings import (
    MODELS_DIR,
    CONFIDENCE_HIGH_PROB, CONFIDENCE_HIGH_ELO_DIFF,
    CONFIDENCE_MEDIUM_PROB, CONFIDENCE_MEDIUM_ELO_DIFF,
)
from src.features.elo_features import get_elo_rating

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Stacked ensemble with meta-learner and probability calibration."""

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

        self.meta_learner = None
        self.calibrator = None
        self.is_meta_trained = False
        self.base_models_order = ["rf", "xgb", "lgbm", "catboost", "elo"]

    def fit_meta_learner(
        self,
        val_base_probs: np.ndarray,
        y_val: np.ndarray,
        calibrator_probs: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
    ) -> None:
        """Train meta-learner on base model predictions.

        Args:
            val_base_probs: (n_samples, n_models * 3) stacked base probabilities
            y_val: true labels (0=L, 1=D, 2=W)
            calibrator_probs: out-of-fold predictions for calibration
            y_cal: labels for calibration
        """
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.calibration import CalibratedClassifierCV

        self.meta_learner = LogisticRegressionCV(
            Cs=10,
            cv=3,
            max_iter=2000,
            random_state=42,
            n_jobs=-1,
        )
        self.meta_learner.fit(val_base_probs, y_val)
        self.is_meta_trained = True
        logger.info("Meta-learner (LogisticRegressionCV) entrenado en %d samples", len(y_val))

        if calibrator_probs is not None and y_cal is not None and len(y_cal) >= 20:
            self.calibrator = CalibratedClassifierCV(
                estimator=None,
                method="isotonic",
                cv=3,
            )
            base_cal = calibrator_probs if calibrator_probs.ndim == 1 else calibrator_probs
            try:
                self.calibrator.fit(calibrator_probs, y_cal)
                logger.info("Calibrator (isotonic) entrenado en %d samples", len(y_cal))
            except Exception as e:
                logger.warning("Calibrator failed: %s", e)
                self.calibrator = None

    def predict(
        self,
        team_a: str,
        team_b: str,
        feature_vector: np.ndarray,
        elo_df=None,
    ) -> dict:
        model_vector = self._scale_features(feature_vector)
        base_probs = self._get_all_base_probs(model_vector, team_a, team_b)

        if self.is_meta_trained and self.meta_learner is not None:
            stacked = self._stack_base_probs(base_probs)
            meta_probs = self.meta_learner.predict_proba(stacked)[0]
            if self.calibrator is not None:
                try:
                    meta_probs = self.calibrator.predict_proba(stacked)[0]
                except Exception:
                    pass
            p_win, p_draw, p_loss = float(meta_probs[2]), float(meta_probs[1]), float(meta_probs[0])
        else:
            rf_w, xgb_w, lgbm_w, cb_w, elo_w = 0.30, 0.20, 0.20, 0.15, 0.15
            total = rf_w + xgb_w + lgbm_w + cb_w + elo_w
            p_win = (
                rf_w * base_probs["rf"][0] + xgb_w * base_probs["xgb"][0] +
                lgbm_w * base_probs["lgbm"][0] + cb_w * base_probs["catboost"][0] +
                elo_w * base_probs["elo"][0]
            ) / total
            p_draw = (
                rf_w * base_probs["rf"][1] + xgb_w * base_probs["xgb"][1] +
                lgbm_w * base_probs["lgbm"][1] + cb_w * base_probs["catboost"][1] +
                elo_w * base_probs["elo"][1]
            ) / total
            p_loss = 1.0 - p_win - p_draw

        total = p_win + p_draw + p_loss
        if total > 0:
            p_win, p_draw, p_loss = p_win / total, p_draw / total, p_loss / total
        else:
            p_win, p_draw, p_loss = 0.40, 0.25, 0.35

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
                "rf": base_probs.get("rf", (0.4, 0.25, 0.35)),
                "xgb": base_probs.get("xgb", (0.4, 0.25, 0.35)),
                "lgbm": base_probs.get("lgbm", (0.4, 0.25, 0.35)),
                "catboost": base_probs.get("catboost", (0.4, 0.25, 0.35)),
                "elo": base_probs.get("elo", (0.4, 0.25, 0.35)),
            },
            "meta_trained": self.is_meta_trained,
        }

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

    def _stack_base_probs(
        self, base_probs: dict[str, tuple[float, float, float]]
    ) -> np.ndarray:
        vecs = []
        for model_name in self.base_models_order:
            p = base_probs.get(model_name, (0.4, 0.25, 0.35))
            vecs.extend(p)
        return np.array([vecs])

    def predict_batch_stacked(
        self, all_base_probs: list[dict[str, tuple[float, float, float]]]
    ) -> np.ndarray:
        if not self.is_meta_trained or self.meta_learner is None:
            return np.tile([0.38, 0.24, 0.38], (len(all_base_probs), 1))
        stacked = np.array([
            self._stack_base_probs(bp)[0] for bp in all_base_probs
        ])
        return self.meta_learner.predict_proba(stacked)

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
        except Exception:
            return x

    def get_shap_explanation(self, feature_vector: np.ndarray) -> list[tuple[str, float]]:
        if self.xgb and getattr(self.xgb, "is_fitted_", False):
            return self.xgb.explain(feature_vector)
        return []

    def save_meta(self) -> None:
        if self.meta_learner is not None:
            joblib.dump(self.meta_learner, MODELS_DIR / "meta_learner.pkl")
            logger.info("Meta-learner guardado")
        if self.calibrator is not None:
            joblib.dump(self.calibrator, MODELS_DIR / "calibrator.pkl")
            logger.info("Calibrator guardado")

    def load_meta(self) -> bool:
        meta_path = MODELS_DIR / "meta_learner.pkl"
        cal_path = MODELS_DIR / "calibrator.pkl"
        loaded = False
        if meta_path.exists():
            try:
                self.meta_learner = joblib.load(meta_path)
                self.is_meta_trained = True
                loaded = True
                logger.info("Meta-learner cargado")
            except Exception as e:
                logger.warning("Error cargando meta-learner: %s", e)
        if cal_path.exists():
            try:
                self.calibrator = joblib.load(cal_path)
                logger.info("Calibrator cargado")
            except Exception as e:
                logger.warning("Error cargando calibrator: %s", e)
        return loaded
