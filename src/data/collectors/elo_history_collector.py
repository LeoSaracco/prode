"""Collector for historical Elo ratings computed from international match results.

Computes Elo ratings over time from match history rather than relying on
external datasets that may not be available locally.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from config.settings import ELO_HISTORY_DIR, CACHE_TTL_HOURS
from src.data.cache_manager import CacheManager

logger = logging.getLogger(__name__)

K_FACTOR = 32
HOME_ADVANTAGE = 65
ELO_INITIAL = 1500


class EloHistoryCollector:
    """Computes Elo rating history from match results.

    Uses a time-series of international matches to compute rolling Elo
    ratings for each team. This is the most reliable approach when
    external Elo datasets are not available.
    """

    def __init__(self, directory: Path | None = None):
        self.directory = directory or ELO_HISTORY_DIR
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache = CacheManager()

    def compute_from_matches(self, match_df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        """Compute Elo history from a match dataframe.

        match_df must have: date, team_a, team_b, goals_a, goals_b
        """
        cache_name = "elo_history_computed"
        if not force and self.cache.is_valid(cache_name, CACHE_TTL_HOURS * 248):
            cached = self.cache.load(cache_name)
            if cached is not None and not cached.empty:
                return cached

        if match_df.empty:
            return pd.DataFrame()

        df = match_df[["date", "team_a", "team_b", "goals_a", "goals_b"]].copy()
        df = df.sort_values("date").reset_index(drop=True)

        current_elo: dict[str, float] = {}
        history_rows: list[dict] = []

        for _, row in df.iterrows():
            ta, tb = row["team_a"], row["team_b"]
            ga, gb = int(row["goals_a"]), int(row["goals_b"])
            date = row["date"]

            elo_a = current_elo.get(ta, ELO_INITIAL)
            elo_b = current_elo.get(tb, ELO_INITIAL)

            history_rows.append({"date": date, "team": ta, "elo_rating": elo_a})
            history_rows.append({"date": date, "team": tb, "elo_rating": elo_b})

            expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - (elo_a + HOME_ADVANTAGE)) / 400.0))
            expected_b = 1.0 - expected_a

            if ga > gb:
                score_a, score_b = 1.0, 0.0
            elif ga < gb:
                score_a, score_b = 0.0, 1.0
            else:
                score_a, score_b = 0.5, 0.5

            margin = abs(ga - gb)
            if margin > 1:
                margin_factor = math.log(margin + 1, 2) * 0.75 + 0.75
            else:
                margin_factor = 1.0

            k_a = K_FACTOR * margin_factor
            k_b = K_FACTOR * margin_factor

            current_elo[ta] = elo_a + k_a * (score_a - expected_a)
            current_elo[tb] = elo_b + k_b * (score_b - expected_b)

        result = pd.DataFrame(history_rows)
        self.cache.save(result, cache_name)
        logger.info("Elo history computed: %s rows, %s teams", len(result), result["team"].nunique())
        return result

    def get_current_elo(self, match_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Get latest Elo for each team."""
        history = self.compute_from_matches(match_df) if match_df is not None else pd.DataFrame()
        if history.empty:
            return pd.DataFrame()
        latest_idx = history.groupby("team")["date"].idxmax()
        return history.loc[latest_idx].rename(columns={"team": "team", "elo_rating": "elo"}).reset_index(drop=True)
