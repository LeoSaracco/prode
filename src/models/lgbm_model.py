"""Clasificador LightGBM para predicción de resultado (W/D/L)."""
import logging
from pathlib import Path

import numpy as np

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)


class LGBMOutcomeClassifier:
    def __init__(self, params: dict | None = None):
        self.model_ = None
        self.is_fitted_ = False
        self.params = params or {}

    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None,
              y_val: np.ndarray | None = None) -> "LGBMOutcomeClassifier":
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("lightgbm no instalado")
            return self

        defaults = {
            "n_estimators": 600,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        defaults.update(self.params)
        self.model_ = lgb.LGBMClassifier(**defaults)

        callbacks = [lgb.early_stopping(50, verbose=False)] if X_val is not None else []
        eval_set = [(X_val, y_val)] if X_val is not None else None
        fit_kwargs = {"callbacks": callbacks}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
        self.model_.fit(X, y, **fit_kwargs)
        self.is_fitted_ = True
        logger.info("LightGBM entrenado.")
        return self

    def predict_proba(self, x: np.ndarray) -> tuple[float, float, float]:
        if not self.is_fitted_ or self.model_ is None:
            return 0.40, 0.25, 0.35
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = self.model_.predict_proba(x.reshape(1, -1))[0]
        return float(proba[2]), float(proba[1]), float(proba[0])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self.is_fitted_:
            return 0.0
        import warnings
        from sklearn.metrics import accuracy_score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(accuracy_score(y, self.model_.predict(X)))

    def save(self, path: Path | None = None) -> None:
        import joblib
        p = path or MODELS_DIR / "lgbm_outcome_classifier.pkl"
        if self.model_ is not None:
            joblib.dump(self.model_, p)
            logger.info(f"LightGBM guardado en {p}")

    def load(self, path: Path | None = None) -> "LGBMOutcomeClassifier":
        import joblib
        p = path or MODELS_DIR / "lgbm_outcome_classifier.pkl"
        if not Path(p).exists():
            logger.warning(f"LightGBM model no encontrado en {p}")
            return self
        try:
            self.model_ = joblib.load(p)
            self.is_fitted_ = True
        except Exception as e:
            logger.warning(f"Error cargando LightGBM: {e}")
        return self
