"""CatBoost classifier for W/D/L prediction."""

import logging
from pathlib import Path

import numpy as np

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)


class CatBoostOutcomeClassifier:
    """3-class W/D/L classifier using CatBoost with ordered boosting."""

    def __init__(self, params: dict | None = None):
        self.model_ = None
        self.is_fitted_ = False
        self.params = params or {}

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "CatBoostOutcomeClassifier":
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            logger.error("catboost no instalado")
            self.is_fitted_ = False
            return self

        defaults = {
            "iterations": 200,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "border_count": 128,
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "random_seed": 42,
            "thread_count": -1,
            "verbose": False,
            "allow_writing_files": False,
        }
        defaults.update(self.params)
        self.model_ = CatBoostClassifier(**defaults)
        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
        self.model_.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False,
        )
        self.is_fitted_ = True
        logger.info("CatBoost entrenado.")
        return self

    def predict_proba(self, x: np.ndarray) -> tuple[float, float, float]:
        if not self.is_fitted_ or self.model_ is None:
            return 0.40, 0.25, 0.35
        x2d = x.reshape(1, -1)
        proba = self.model_.predict_proba(x2d)[0]
        return float(proba[2]), float(proba[1]), float(proba[0])

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.model_ is None:
            n = len(X)
            return np.tile([0.38, 0.24, 0.38], (n, 1))
        return self.model_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self.is_fitted_:
            return 0.0
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(y, self.model_.predict(X)))

    def save(self, path: Path | None = None) -> None:
        import joblib
        p = path or MODELS_DIR / "catboost_outcome_classifier.pkl"
        if self.model_ is not None:
            joblib.dump(self.model_, p)
            logger.info(f"CatBoost guardado en {p}")

    def load(self, path: Path | None = None) -> "CatBoostOutcomeClassifier":
        import joblib
        p = path or MODELS_DIR / "catboost_outcome_classifier.pkl"
        if not Path(p).exists():
            logger.warning(f"CatBoost model no encontrado en {p}")
            return self
        try:
            self.model_ = joblib.load(p)
            self.is_fitted_ = True
        except Exception as e:
            logger.warning(f"Error cargando CatBoost: {e}")
        return self
