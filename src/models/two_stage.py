"""Two-stage prediction: draw/no-draw binary + win/loss binary.

Stage 1: P(draw) via binary classifier
Stage 2: P(win | not draw) via binary classifier on non-draw samples

Outcome:  P_win  = P(not draw) * P(win | not draw)
          P_draw = P(draw)
          P_loss = P(not draw) * (1 - P(win | not draw))
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)


class TwoStageClassifier:
    """Two-stage approach: splits 3-class problem into 2 binary problems.

    This is more reliable than direct 3-class classification because:
    - Draws are fundamentally different from non-draw outcomes
    - Binary classifiers achieve higher accuracy
    - Each stage can use different features optimized for its target
    """

    def __init__(self, draw_params: dict | None = None, win_params: dict | None = None):
        self.draw_clf: RandomForestClassifier | None = None
        self.win_clf: RandomForestClassifier | None = None
        self.draw_calibrated: CalibratedClassifierCV | None = None
        self.win_calibrated: CalibratedClassifierCV | None = None
        self.is_fitted_ = False
        self.draw_params = draw_params or {}
        self.win_params = win_params or {}

    def train(self, X: np.ndarray, y: np.ndarray) -> "TwoStageClassifier":
        """Train both stages.

        Args:
            X: feature matrix
            y: labels 0=L, 1=D, 2=W
        """
        if len(X) < 50:
            logger.warning("Not enough samples for TwoStage training")
            return self

        y_draw = (y == 1).astype(int)

        mask_not_draw = y_draw == 0
        X_nd = X[mask_not_draw]
        y_nd = y[mask_not_draw]
        y_win = (y_nd == 2).astype(int)

        draw_defaults = {
            "n_estimators": 400,
            "max_depth": 10,
            "min_samples_split": 15,
            "min_samples_leaf": 6,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        draw_defaults.update(self.draw_params)
        self.draw_clf = RandomForestClassifier(**draw_defaults)
        self.draw_clf.fit(X, y_draw)

        if len(X_nd) >= 30:
            win_defaults = {
                "n_estimators": 400,
                "max_depth": 10,
                "min_samples_split": 15,
                "min_samples_leaf": 6,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }
            win_defaults.update(self.win_params)
            self.win_clf = RandomForestClassifier(**win_defaults)
            self.win_clf.fit(X_nd, y_win)
        else:
            self.win_clf = None

        self.is_fitted_ = True
        logger.info("TwoStage trained: stage1(draw) on %d, stage2(win|not-draw) on %d",
                     len(X), len(X_nd))
        return self

    def predict_proba(self, x: np.ndarray) -> tuple[float, float, float]:
        """Returns (p_win, p_draw, p_loss) for team_a."""
        if not self.is_fitted_:
            return 0.40, 0.25, 0.35

        x2d = x.reshape(1, -1)

        if self.draw_clf is not None:
            p_draw = float(self.draw_clf.predict_proba(x2d)[0][1])
        else:
            p_draw = 0.25

        p_not_draw = 1.0 - p_draw

        if self.win_clf is not None:
            p_win_given_nd = float(self.win_clf.predict_proba(x2d)[0][1])
        else:
            p_win_given_nd = self._elo_win_prob(x)

        p_win = p_not_draw * p_win_given_nd
        p_loss = p_not_draw * (1.0 - p_win_given_nd)

        total = p_win + p_draw + p_loss
        if total > 0:
            p_win /= total
            p_draw /= total
            p_loss /= total

        return p_win, p_draw, p_loss

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            n = len(X)
            return np.tile([0.38, 0.24, 0.38], (n, 1))

        if self.draw_clf is not None:
            p_draw = self.draw_clf.predict_proba(X)[:, 1]
        else:
            p_draw = np.full(len(X), 0.25)

        p_not_draw = 1.0 - p_draw

        if self.win_clf is not None:
            p_win_nd = self.win_clf.predict_proba(X)[:, 1]
        else:
            p_win_nd = np.full(len(X), 0.55)

        p_win = p_not_draw * p_win_nd
        p_loss = p_not_draw * (1.0 - p_win_nd)

        out = np.column_stack([p_loss, p_draw, p_win])
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self.is_fitted_:
            return 0.0
        from sklearn.metrics import accuracy_score
        probs = self.predict_proba_batch(X)
        return float(accuracy_score(y, probs.argmax(axis=1)))

    def _elo_win_prob(self, x: np.ndarray) -> float:
        elo_diff = float(x.flat[0])
        p = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        return float(p)

    def save(self) -> None:
        if self.draw_clf is not None:
            joblib.dump(self.draw_clf, MODELS_DIR / "twostage_draw.pkl")
        if self.win_clf is not None:
            joblib.dump(self.win_clf, MODELS_DIR / "twostage_win.pkl")
        logger.info("TwoStage models saved")

    def load(self) -> "TwoStageClassifier":
        draw_p = MODELS_DIR / "twostage_draw.pkl"
        win_p = MODELS_DIR / "twostage_win.pkl"
        if draw_p.exists():
            try:
                self.draw_clf = joblib.load(draw_p)
                self.is_fitted_ = True
            except Exception as e:
                logger.warning("Error loading TwoStage draw: %s", e)
        if win_p.exists():
            try:
                self.win_clf = joblib.load(win_p)
            except Exception as e:
                logger.warning("Error loading TwoStage win: %s", e)
        return self
