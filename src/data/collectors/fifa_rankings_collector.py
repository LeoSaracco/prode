"""Collector for historical FIFA Men's World Rankings.

Downloads from the official FIFA rankings archive (csv) or a GitHub mirror.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from config.settings import FIFA_RANKINGS_DIR

logger = logging.getLogger(__name__)

FIFA_RANKINGS_URL = (
    "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/refs/heads/master/"
    "ranking_fifa_historical.csv"
)

FIFA_RANKINGS_INTERNATIONAL_URL = (
    "https://raw.githubusercontent.com/cnc8/fifa-world-ranking/master/fifa_ranking.csv"
)


class FIFARankingsCollector:
    """Downloads and caches historical FIFA rankings data."""

    def __init__(self, directory: Path | None = None):
        self.directory = directory or FIFA_RANKINGS_DIR
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path = self.directory / "fifa_rankings.parquet"

    def collect(self, force: bool = False) -> pd.DataFrame:
        if not force and self._path.exists():
            df = pd.read_parquet(self._path)
            logger.info("FIFA rankings loaded from cache: %s rows", len(df))
            return df

        df = self._download()
        if df is not None and not df.empty:
            df.to_parquet(self._path, index=False)
            logger.info("FIFA rankings saved: %s rows", len(df))
        return df if df is not None else pd.DataFrame()

    def _download(self) -> pd.DataFrame | None:
        for url in [FIFA_RANKINGS_URL, FIFA_RANKINGS_INTERNATIONAL_URL]:
            try:
                logger.info("Downloading FIFA rankings from %s", url)
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                raw = pd.read_csv(
                    pd.io.common.BytesIO(resp.content)
                    if hasattr(pd.io.common, "BytesIO")
                    else __import__("io").BytesIO(resp.content)
                )
                return self._standardize(raw)
            except Exception as e:
                logger.warning("FIFA rankings source failed (%s): %s", url, e)
        return None

    def _standardize(self, raw: pd.DataFrame) -> pd.DataFrame:
        date_col = _first_col(raw, ["rank_date", "date", "ranking_date"])
        team_col = _first_col(raw, ["country_full", "team_name", "country_name", "team", "country"])
        rank_col = _first_col(raw, ["rank", "ranking", "position"])
        points_col = _first_col(raw, ["total_points", "points", "score"])

        if not team_col or (not rank_col and not points_col):
            logger.warning("FIFA rankings CSV has unexpected columns: %s", list(raw.columns))
            return pd.DataFrame()

        df = pd.DataFrame()
        if date_col:
            df["date"] = pd.to_datetime(raw[date_col], errors="coerce")
        else:
            df["date"] = pd.NaT
        df["team_raw"] = raw[team_col].astype(str)
        if points_col:
            df["fifa_points"] = pd.to_numeric(raw[points_col], errors="coerce")
        if rank_col:
            df["fifa_rank"] = pd.to_numeric(raw[rank_col], errors="coerce")
        elif "fifa_points" in df.columns:
            df["fifa_rank"] = df.groupby("date")["fifa_points"].rank(ascending=False, method="first")

        df = df.dropna(subset=["fifa_rank"])
        df["fifa_rank"] = df["fifa_rank"].astype(int)

        from config.team_aliases import resolve_team_name
        df["team"] = df["team_raw"].apply(lambda name: resolve_team_name(str(name)))
        df = df.dropna(subset=["team"])

        wc_teams = self._wc2026_teams()
        df = df[df["team"].isin(wc_teams)].copy()
        df = df.sort_values(["team", "date"]).reset_index(drop=True)
        logger.info("FIFA rankings standardized: %s rows, %s teams", len(df), df["team"].nunique())
        return df

    def get_latest_rankings(self) -> pd.DataFrame:
        df = self.collect()
        if df.empty:
            return pd.DataFrame()
        latest_idx = df.groupby("team")["date"].idxmax()
        return df.loc[latest_idx].reset_index(drop=True)

    def get_rank_diff(self, team_a: str, team_b: str) -> float:
        df = self.get_latest_rankings()
        if df.empty:
            return 0.0
        rank_a = df.loc[df["team"] == team_a, "fifa_rank"]
        rank_b = df.loc[df["team"] == team_b, "fifa_rank"]
        if rank_a.empty or rank_b.empty:
            return 0.0
        return float(rank_b.iloc[0] - rank_a.iloc[0])

    @staticmethod
    def _wc2026_teams() -> set[str]:
        from config.team_aliases import TEAM_ALIASES
        return set(TEAM_ALIASES.keys())


def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return None
