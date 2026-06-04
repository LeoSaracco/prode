"""Clasificador XGBoost para predicción de resultado (W/D/L) con SHAP."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import MODELS_DIR
from src.features.feature_builder import MATCH_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class XGBOutcomeClassifier:
    """W/D/L classifier usando XGBoost. Clase 2=Win, 1=Draw, 0=Loss (perspectiva team_a)."""

    def __init__(self, params: dict | None = None):
        self.model_ = None
        self.is_fitted_ = False
        self.params = params or {}

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None,
              y_val: np.ndarray | None = None) -> "XGBOutcomeClassifier":
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("xgboost no instalado")
            return self

        defaults = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(self.params)

        self.model_ = xgb.XGBClassifier(**defaults)
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        self.model_.fit(X, y, eval_set=eval_set, verbose=False)
        self.is_fitted_ = True
        logger.info("XGBoost entrenado.")
        return self

    def predict_proba(self, x: np.ndarray) -> tuple[float, float, float]:
        """Retorna (p_win, p_draw, p_loss) para team_a."""
        if not self.is_fitted_ or self.model_ is None:
            return 0.40, 0.25, 0.35
        x2d = x.reshape(1, -1)
        proba = self.model_.predict_proba(x2d)[0]
        # Orden: 0=Loss, 1=Draw, 2=Win
        return float(proba[2]), float(proba[1]), float(proba[0])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.model_ is None:
            n = len(X)
            return np.tile([0.38, 0.24, 0.38], (n, 1))
        raw = self.model_.predict_proba(X)
        return raw[:, [2, 1, 0]]

    def explain(self, x: np.ndarray) -> list[tuple[str, float]]:
        """Retorna los top-5 features más influyentes usando SHAP."""
        if not self.is_fitted_ or self.model_ is None:
            return []
        try:
            import shap
            explainer = shap.TreeExplainer(self.model_)
            shap_values = explainer.shap_values(x.reshape(1, -1))
            # shap_values[2] = contribución a clase Win
            win_shap = shap_values[2][0]
            pairs = list(zip(MATCH_FEATURE_COLUMNS, win_shap))
            return sorted(pairs, key=lambda p: abs(p[1]), reverse=True)[:5]
        except Exception as e:
            logger.debug(f"SHAP no disponible: {e}")
            return []

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self.is_fitted_:
            return 0.0
        from sklearn.metrics import accuracy_score
        preds = self.model_.predict(X)
        return float(accuracy_score(y, preds))

    def save(self, path: Path | None = None) -> None:
        p = path or MODELS_DIR / "xgb_outcome_classifier.json"
        if self.model_ is not None:
            self.model_.save_model(str(p))
            logger.info(f"XGBoost guardado en {p}")

    def load(self, path: Path | None = None) -> "XGBOutcomeClassifier":
        try:
            import xgboost as xgb
        except ImportError:
            return self
        p = path or MODELS_DIR / "xgb_outcome_classifier.json"
        if not Path(p).exists():
            logger.warning(f"XGBoost model no encontrado en {p}")
            return self
        self.model_ = xgb.XGBClassifier()
        self.model_.load_model(str(p))
        self.is_fitted_ = True
        return self
