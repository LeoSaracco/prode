"""Confederation-specific ensemble models.

Matches between teams from different confederations follow different patterns.
UEFA vs UEFA matches are tighter, CONMEBOL vs UEFA have more goals, etc.

Trains separate RF models for key confederation pairs, with a global
fallback for rare matchups.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from config.settings import MODELS_DIR
from config.wc2026_groups import CONFEDERATION

logger = logging.getLogger(__name__)

CONFED_GROUPS = {
    "UEFA": ["Austria", "Belgium", "Bosnia and Herzegovina", "Croatia", "Czechia",
             "Denmark", "England", "France", "Germany", "Italy", "Netherlands",
             "Norway", "Poland", "Portugal", "Scotland", "Serbia", "Spain",
             "Sweden", "Switzerland", "Turkiye", "Ukraine"],
    "CONMEBOL": ["Argentina", "Brazil", "Colombia", "Ecuador", "Paraguay", "Uruguay"],
    "CONCACAF": ["Canada", "Costa Rica", "Haiti", "Jamaica", "Mexico",
                 "Panama", "United States"],
    "CAF": ["Algeria", "Cameroon", "Cape Verde", "Cote d'Ivoire", "DR Congo",
            "Egypt", "Ghana", "Mali", "Morocco", "Nigeria", "Senegal",
            "South Africa", "Tunisia"],
    "AFC": ["Australia", "Iran", "Iraq", "Japan", "Jordan", "Qatar",
            "Saudi Arabia", "South Korea", "Uzbekistan"],
    "OFC": ["New Zealand"],
}

KEY_PAIRS = [
    ("UEFA", "UEFA"),
    ("UEFA", "CONMEBOL"),
    ("UEFA", "CAF"),
    ("UEFA", "AFC"),
    ("UEFA", "CONCACAF"),
    ("CONMEBOL", "CONMEBOL"),
    ("CONMEBOL", "CAF"),
    ("CONMEBOL", "AFC"),
    ("CONMEBOL", "CONCACAF"),
    ("CAF", "CAF"),
    ("CAF", "AFC"),
    ("AFC", "AFC"),
    ("AFC", "CONCACAF"),
    ("CONCACAF", "CONCACAF"),
]


class ConfederationModels:
    """Collection of confederation-pair-specific RF classifiers + global fallback."""

    def __init__(self):
        self.models: dict[str, RandomForestClassifier] = {}
        self.scaler: dict[str, StandardScaler] = {}
        self.global_model: RandomForestClassifier | None = None
        self.global_scaler: StandardScaler | None = None
        self.is_fitted_ = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        teams_a: list[str],
        teams_b: list[str],
    ) -> "ConfederationModels":
        if len(X) < 100:
            logger.warning("Not enough data for confederation models")
            self.is_fitted_ = False
            return self

        conf_a = [CONFEDERATION.get(t, "OTHER") for t in teams_a]
        conf_b = [CONFEDERATION.get(t, "OTHER") for t in teams_b]

        for pair in KEY_PAIRS:
            c1, c2 = pair
            mask = ((np.array(conf_a) == c1) & (np.array(conf_b) == c2)) | \
                   ((np.array(conf_a) == c2) & (np.array(conf_b) == c1))
            if mask.sum() < 30:
                continue

            X_pair = X[mask]
            y_pair = y[mask]

            key = f"{c1}_{c2}"
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_pair)
            self.scaler[key] = scaler

            clf = RandomForestClassifier(
                n_estimators=300, max_depth=10,
                min_samples_split=15, min_samples_leaf=6,
                class_weight="balanced_subsample",
                random_state=42, n_jobs=-1,
            )
            clf.fit(X_s, y_pair)
            self.models[key] = clf
            logger.info("Confederation model %s: %d samples", key, len(y_pair))

        self.global_scaler = StandardScaler()
        X_global_s = self.global_scaler.fit_transform(X)
        self.global_model = RandomForestClassifier(
            n_estimators=300, max_depth=10,
            min_samples_split=15, min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        )
        self.global_model.fit(X_global_s, y)

        self.is_fitted_ = True
        logger.info("Confederation models trained: %d pairs + global", len(self.models))
        return self

    def predict_proba(self, x: np.ndarray, team_a: str, team_b: str) -> tuple[float, float, float]:
        x2d = x.reshape(1, -1)
        key = self._pair_key(team_a, team_b)

        if key in self.models and key in self.scaler:
            x_s = self.scaler[key].transform(x2d)
            probs = self.models[key].predict_proba(x_s)[0]
        elif self.global_model is not None and self.global_scaler is not None:
            x_s = self.global_scaler.transform(x2d)
            probs = self.global_model.predict_proba(x_s)[0]
        else:
            return 0.40, 0.25, 0.35

        return float(probs[2]), float(probs[1]), float(probs[0])

    def predict_proba_batch(self, X: np.ndarray, teams_a: list[str], teams_b: list[str]) -> np.ndarray:
        if not self.is_fitted_:
            n = len(X)
            return np.tile([0.38, 0.24, 0.38], (n, 1))

        n = len(X)
        out = np.zeros((n, 3))
        for i in range(n):
            p = self.predict_proba(X[i], teams_a[i], teams_b[i])
            out[i] = [p[2], p[1], p[0]]
        return out

    def score(self, X: np.ndarray, y: np.ndarray, teams_a: list[str], teams_b: list[str]) -> float:
        if not self.is_fitted_:
            return 0.0
        from sklearn.metrics import accuracy_score
        probs = self.predict_proba_batch(X, teams_a, teams_b)
        return float(accuracy_score(y, probs.argmax(axis=1)))

    def _pair_key(self, team_a: str, team_b: str) -> str:
        c1 = CONFEDERATION.get(team_a, "OTHER")
        c2 = CONFEDERATION.get(team_b, "OTHER")
        return f"{min(c1, c2)}_{max(c1, c2)}"

    def save(self) -> None:
        for key, clf in self.models.items():
            joblib.dump(clf, MODELS_DIR / f"confed_{key}.pkl")
            joblib.dump(self.scaler[key], MODELS_DIR / f"confed_scaler_{key}.pkl")
        if self.global_model is not None:
            joblib.dump(self.global_model, MODELS_DIR / "confed_global.pkl")
            joblib.dump(self.global_scaler, MODELS_DIR / "confed_global_scaler.pkl")
        logger.info("Confederation models saved: %d pairs", len(self.models))

    def load(self) -> "ConfederationModels":
        loaded = 0
        for pair in KEY_PAIRS:
            key = f"{pair[0]}_{pair[1]}"
            p = MODELS_DIR / f"confed_{key}.pkl"
            s = MODELS_DIR / f"confed_scaler_{key}.pkl"
            if p.exists() and s.exists():
                try:
                    self.models[key] = joblib.load(p)
                    self.scaler[key] = joblib.load(s)
                    loaded += 1
                except Exception as e:
                    logger.warning("Error loading confederation model %s: %s", key, e)
        gp = MODELS_DIR / "confed_global.pkl"
        gs = MODELS_DIR / "confed_global_scaler.pkl"
        if gp.exists() and gs.exists():
            try:
                self.global_model = joblib.load(gp)
                self.global_scaler = joblib.load(gs)
            except Exception as e:
                logger.warning("Error loading confederation global: %s", e)
        self.is_fitted_ = loaded > 0
        logger.info("Confederation models loaded: %d pairs", loaded)
        return self
