"""
Construye la matriz completa de features para todas las selecciones
y vectores de features diferenciales para predicción de partidos.
"""
import logging

import numpy as np
import pandas as pd

from src.data.national_team_proxy import FALLBACK_STATS, MARKET_VALUE_EUR_M
from src.features.elo_features import get_elo_rating, compute_elo_win_probability
from src.features.attack_features import compute_offensive_power
from src.features.defense_features import compute_defensive_stability
from src.features.squad_features import compute_squad_depth_from_market_value
from src.features.historical_features import compute_world_cup_history_score, h2h_advantage_score
from src.features.risk_features import (
    compute_consistency_score,
    compute_tactical_advantage,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "elo_rating", "xg_per_game", "xga_per_game", "form_5",
    "offensive_power", "defensive_stability", "squad_depth_score",
    "consistency_score", "wc_history_score", "market_value_norm",
    "elo_momentum", "days_since_last_match",
]

MATCH_FEATURE_COLUMNS = [
    "elo_diff", "elo_win_prob_a",
    "xg_diff", "xga_diff",
    "form_5_diff", "offensive_power_diff", "defensive_stability_diff",
    "squad_depth_diff", "consistency_diff",
    "wc_history_diff", "market_value_diff",
    "tactical_advantage", "h2h_advantage",
    "is_tournament", "is_wc", "is_qualifier", "is_home",
    "fifa_rank_diff", "elo_momentum_diff",
    "days_since_last_match_diff", "rest_days_diff",
]

INFERENCE_CONTEXT_FEATURES = {
    "is_tournament": 1.0,
    "is_wc": 1.0,
    "is_qualifier": 0.0,
    "is_home": 0.0,
}

DEFAULT_EXTRA_FEATURES = {
    "fifa_rank_diff": 0.0,
    "elo_momentum_diff": 0.0,
    "days_since_last_match_diff": 0.0,
    "rest_days_diff": 0.0,
}


class FeatureBuilder:
    def __init__(
        self,
        elo_df: pd.DataFrame | None = None,
        h2h_df: pd.DataFrame | None = None,
        team_profiles_df: pd.DataFrame | None = None,
        h2h_stats: dict | None = None,
    ):
        self.elo_df = elo_df if elo_df is not None else pd.DataFrame()
        self.h2h_df = h2h_df if h2h_df is not None else pd.DataFrame()
        self.team_profiles_df = team_profiles_df if team_profiles_df is not None else pd.DataFrame()
        self._profile_lookup = {}
        if not self.team_profiles_df.empty and "team" in self.team_profiles_df.columns:
            self._profile_lookup = {
                str(row["team"]): row
                for _, row in self.team_profiles_df.iterrows()
            }
        self._team_features: dict[str, dict] = {}
        # h2h_stats: {"TeamA||TeamB": {"win_rate": 0.6, "n": 10}, ...}
        self._h2h: dict[tuple[str, str], float] = {}
        if h2h_stats:
            for key, val in h2h_stats.items():
                try:
                    ta, tb = key.split("||", 1)
                    self._h2h[(ta, tb)] = float(val["win_rate"])
                except Exception:
                    pass

    def build_team_features(self, team: str) -> dict:
        if team in self._team_features:
            return self._team_features[team]

        profile = self._profile_lookup.get(team)
        elo = float(profile.get("elo_rating")) if profile is not None and "elo_rating" in profile and not pd.isna(profile.get("elo_rating")) else get_elo_rating(team, self.elo_df)
        stats = FALLBACK_STATS.get(team, {
            "xg_pg": 1.0, "xga_pg": 1.2, "form_5": 0.50, "ppda": 14.0
        })
        xg_pg = float(profile.get("xg_per_game")) if profile is not None and "xg_per_game" in profile and not pd.isna(profile.get("xg_per_game")) else stats["xg_pg"]
        xga_pg = float(profile.get("xga_per_game")) if profile is not None and "xga_per_game" in profile and not pd.isna(profile.get("xga_per_game")) else stats["xga_pg"]
        form_5 = float(profile.get("form_5")) if profile is not None and "form_5" in profile and not pd.isna(profile.get("form_5")) else stats["form_5"]
        ppda = float(profile.get("ppda")) if profile is not None and "ppda" in profile and not pd.isna(profile.get("ppda")) else stats["ppda"]
        wc_history = compute_world_cup_history_score(team)
        mv = MARKET_VALUE_EUR_M.get(team, 100.0)

        offensive_power = compute_offensive_power(xg_pg)
        defensive_stability = compute_defensive_stability(xga_pg, ppda)
        squad_depth = compute_squad_depth_from_market_value(team)
        if profile is not None and "consistency" in profile and not pd.isna(profile.get("consistency")):
            consistency = float(profile.get("consistency"))
        else:
            consistency = compute_consistency_score([
                3 if form_5 > 0.75 else (1 if form_5 > 0.45 else 0)
            ] * 10)

        elo_momentum = float(profile.get("elo_momentum")) if profile is not None and "elo_momentum" in profile and not pd.isna(profile.get("elo_momentum")) else self._get_elo_momentum(team)
        days_since = float(profile.get("days_since_last_match")) if profile is not None and "days_since_last_match" in profile and not pd.isna(profile.get("days_since_last_match")) else self._get_days_since_last_match(team)
        fifa_rank = float(profile.get("fifa_rank")) if profile is not None and "fifa_rank" in profile and not pd.isna(profile.get("fifa_rank")) else 75.0

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
            "consistency_score": consistency,
            "wc_history_score": wc_history,
            "market_value_norm": min(mv / 1500, 1.0),
            "elo_momentum": elo_momentum,
            "days_since_last_match": days_since,
            "fifa_rank": fifa_rank,
        }
        self._team_features[team] = feats
        return feats

    def _get_elo_momentum(self, team: str) -> float:
        if self.elo_df is None or self.elo_df.empty:
            return 0.0
        return 0.0

    def _get_days_since_last_match(self, team: str) -> float:
        return 30.0

    def build_match_features(
        self,
        team_a: str,
        team_b: str,
        h2h_df: pd.DataFrame | None = None,
        context: dict[str, float] | None = None,
    ) -> np.ndarray:
        fa = self.build_team_features(team_a)
        fb = self.build_team_features(team_b)

        elo_diff = fa["elo_rating"] - fb["elo_rating"]
        elo_win_prob = compute_elo_win_probability(fa["elo_rating"], fb["elo_rating"])

        win_rate_a = self._h2h.get((team_a, team_b))
        if win_rate_a is not None:
            h2h_score = float(2.0 * win_rate_a - 1.0)
        elif h2h_df is not None and not h2h_df.empty:
            from src.features.historical_features import compute_h2h_record
            h2h = compute_h2h_record(h2h_df, team_a, team_b)
            h2h_score = h2h_advantage_score(h2h, team_a)
        else:
            h2h_score = 0.0

        tactical_adv = compute_tactical_advantage(
            elo_diff,
            fa["offensive_power"], fb["defensive_stability"],
            fb["offensive_power"], fa["defensive_stability"],
        )

        ctx = context or INFERENCE_CONTEXT_FEATURES
        features = np.array([
            elo_diff,
            elo_win_prob,
            fa["xg_per_game"] - fb["xg_per_game"],
            fa["xga_per_game"] - fb["xga_per_game"],
            fa["form_5"] - fb["form_5"],
            fa["offensive_power"] - fb["offensive_power"],
            fa["defensive_stability"] - fb["defensive_stability"],
            fa["squad_depth_score"] - fb["squad_depth_score"],
            fa["consistency_score"] - fb["consistency_score"],
            fa["wc_history_score"] - fb["wc_history_score"],
            fa["market_value_norm"] - fb["market_value_norm"],
            tactical_adv,
            h2h_score,
            ctx.get("is_tournament", 1.0),
            ctx.get("is_wc", 1.0),
            ctx.get("is_qualifier", 0.0),
            ctx.get("is_home", 0.0),
            fb["fifa_rank"] - fa["fifa_rank"],
            fa["elo_momentum"] - fb["elo_momentum"],
            fa["days_since_last_match"] - fb["days_since_last_match"],
            fb["days_since_last_match"] - fa["days_since_last_match"],
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
