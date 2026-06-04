"""
Construye la matriz completa de features para todas las selecciones
y vectores de features diferenciales para predicción de partidos.
"""
import logging

import numpy as np
import pandas as pd

from config.wc2026_groups import WC_HISTORY_SCORE
from src.data.national_team_proxy import FALLBACK_STATS, MARKET_VALUE_EUR_M, FALLBACK_ELO
from src.features.elo_features import get_elo_rating, compute_elo_win_probability
from src.features.attack_features import compute_offensive_power, compute_big_match_rating
from src.features.defense_features import compute_defensive_stability
from src.features.squad_features import compute_squad_depth_from_market_value
from src.features.historical_features import compute_world_cup_history_score, h2h_advantage_score
from src.features.risk_features import (
    compute_upset_probability,
    compute_consistency_score,
    compute_pressure_performance,
    compute_tactical_advantage,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "elo_rating", "xg_per_game", "xga_per_game", "form_5",
    "offensive_power", "defensive_stability", "squad_depth_score",
    "big_match_rating", "pressure_performance", "consistency_score",
    "wc_history_score", "market_value_norm",
]

MATCH_FEATURE_COLUMNS = [
    "elo_diff", "elo_win_prob_a",
    "xg_diff", "xga_diff",
    "form_5_diff", "offensive_power_diff", "defensive_stability_diff",
    "squad_depth_diff", "big_match_rating_diff", "pressure_diff",
    "consistency_diff", "wc_history_diff", "market_value_diff",
    "tactical_advantage", "h2h_advantage",
    # Features de interacción
    "form_times_elo_diff", "attack_vs_defense_clash",
]


class FeatureBuilder:
    def __init__(
        self,
        elo_df: pd.DataFrame | None = None,
        h2h_df: pd.DataFrame | None = None,
    ):
        self.elo_df = elo_df if elo_df is not None else pd.DataFrame()
        self.h2h_df = h2h_df if h2h_df is not None else pd.DataFrame()
        self._team_features: dict[str, dict] = {}

    def build_team_features(self, team: str) -> dict:
        """Construye el vector de features para una selección."""
        if team in self._team_features:
            return self._team_features[team]

        elo = get_elo_rating(team, self.elo_df)
        stats = FALLBACK_STATS.get(team, {
            "xg_pg": 1.0, "xga_pg": 1.2, "form_5": 0.50, "ppda": 14.0
        })
        xg_pg = stats["xg_pg"]
        xga_pg = stats["xga_pg"]
        form_5 = stats["form_5"]
        ppda = stats["ppda"]
        wc_history = compute_world_cup_history_score(team)
        mv = MARKET_VALUE_EUR_M.get(team, 100.0)

        offensive_power = compute_offensive_power(xg_pg)
        defensive_stability = compute_defensive_stability(xga_pg, ppda)
        squad_depth = compute_squad_depth_from_market_value(team)
        big_match = compute_big_match_rating(form_5, wc_history, elo)
        pressure = compute_pressure_performance(form_5, wc_history, elo)
        consistency = compute_consistency_score([
            3 if form_5 > 0.75 else (1 if form_5 > 0.45 else 0)
        ] * 10)

        feats = {
            "team": team,
            "elo_rating": elo,
            "xg_per_game": xg_pg,
            "xga_per_game": xga_pg,
            "form_5": form_5,
            "ppda": ppda,
            "offensive_power": offensive_power,
            "defensive_stability": defensive_stability,
            "squad_depth_score": squad_depth,
            "big_match_rating": big_match,
            "pressure_performance": pressure,
            "consistency_score": consistency,
            "wc_history_score": wc_history,
            "market_value_norm": min(mv / 1500, 1.0),
        }
        self._team_features[team] = feats
        return feats

    def build_match_features(
        self, team_a: str, team_b: str, h2h_df: pd.DataFrame | None = None
    ) -> np.ndarray:
        """
        Retorna vector de features para el partido A vs B.
        Usado por el ensemble en tiempo de inferencia.
        """
        fa = self.build_team_features(team_a)
        fb = self.build_team_features(team_b)

        elo_diff = fa["elo_rating"] - fb["elo_rating"]
        elo_win_prob = compute_elo_win_probability(fa["elo_rating"], fb["elo_rating"])

        h2h_score = 0.0
        if h2h_df is not None and not h2h_df.empty:
            from src.features.historical_features import compute_h2h_record
            h2h = compute_h2h_record(h2h_df, team_a, team_b)
            h2h_score = h2h_advantage_score(h2h, team_a)

        tactical_adv = compute_tactical_advantage(
            elo_diff,
            fa["offensive_power"], fb["defensive_stability"],
            fb["offensive_power"], fa["defensive_stability"],
        )

        # Features de interacción
        form_times_elo = fa["form_5"] * (elo_diff / 400)
        attack_vs_def = fa["offensive_power"] - fb["defensive_stability"]

        features = np.array([
            elo_diff,
            elo_win_prob,
            fa["xg_per_game"] - fb["xg_per_game"],
            fa["xga_per_game"] - fb["xga_per_game"],
            fa["form_5"] - fb["form_5"],
            fa["offensive_power"] - fb["offensive_power"],
            fa["defensive_stability"] - fb["defensive_stability"],
            fa["squad_depth_score"] - fb["squad_depth_score"],
            fa["big_match_rating"] - fb["big_match_rating"],
            fa["pressure_performance"] - fb["pressure_performance"],
            fa["consistency_score"] - fb["consistency_score"],
            fa["wc_history_score"] - fb["wc_history_score"],
            fa["market_value_norm"] - fb["market_value_norm"],
            tactical_adv,
            h2h_score,
            form_times_elo,
            attack_vs_def,
        ], dtype=np.float32)

        return features

    def build_all_team_features_df(self) -> pd.DataFrame:
        """Construye la matriz de features de todas las selecciones."""
        from config.team_aliases import TEAM_ALIASES
        records = [self.build_team_features(team) for team in TEAM_ALIASES]
        df = pd.DataFrame(records)
        logger.info(f"Feature matrix construida: {df.shape}")
        return df

    def build_training_matrix(
        self, match_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construye X, y para entrenamiento a partir de partidos históricos.
        match_df debe tener: team_a, team_b, result (W/D/L desde perspectiva de team_a).
        """
        label_map = {"W": 2, "D": 1, "L": 0}
        X_rows, y_rows = [], []
        for _, row in match_df.iterrows():
            team_a = row.get("team_a") or row.get("home_team")
            team_b = row.get("team_b") or row.get("away_team")
            result = row.get("result", "D")
            if team_a is None or team_b is None:
                continue
            try:
                x = self.build_match_features(team_a, team_b)
                X_rows.append(x)
                y_rows.append(label_map.get(result, 1))
            except Exception:
                continue
        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)
