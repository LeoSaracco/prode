"""RandomForest classifier for W/D/L prediction. Superior to boosting on small datasets."""

import logging
from pathlib import Path

import numpy as np

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)


class RFOutcomeClassifier:
    """3-class W/D/L classifier using RandomForest with OOB scoring."""

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
    ) -> "RFOutcomeClassifier":
        from sklearn.ensemble import RandomForestClassifier

        defaults = {
            "n_estimators": 500,
            "max_depth": 12,
            "min_samples_split": 15,
            "min_samples_leaf": 6,
            "max_features": "sqrt",
            "bootstrap": True,
            "oob_score": True,
            "class_weight": "balanced_subsample",
            "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(self.params)
        self.model_ = RandomForestClassifier(**defaults)
        self.model_.fit(X, y)
        self.is_fitted_ = True

        if hasattr(self.model_, "oob_score_"):
            logger.info("RandomForest entrenado. OOB score: %.4f", self.model_.oob_score_)
        else:
            logger.info("RandomForest entrenado.")
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
            arr = np.tile([0.38, 0.24, 0.38], (n, 1))
            return arr
        return self.model_.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self.is_fitted_:
            return 0.0
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(y, self.model_.predict(X)))

    def save(self, path: Path | None = None) -> None:
        import joblib
        p = path or MODELS_DIR / "rf_outcome_classifier.pkl"
        if self.model_ is not None:
            joblib.dump(self.model_, p)
            logger.info(f"RandomForest guardado en {p}")

    def load(self, path: Path | None = None) -> "RFOutcomeClassifier":
        import joblib
        p = path or MODELS_DIR / "rf_outcome_classifier.pkl"
        if not Path(p).exists():
            logger.warning(f"RandomForest model no encontrado en {p}")
            return self
        try:
            self.model_ = joblib.load(p)
            self.is_fitted_ = True
        except Exception as e:
            logger.warning(f"Error cargando RandomForest: {e}")
        return self
