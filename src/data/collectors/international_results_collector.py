"""Collector for men's international football results."""

import logging
from pathlib import Path

import pandas as pd
import requests

from config.settings import INTERNATIONAL_RESULTS_PATH, INTERNATIONAL_RESULTS_URL
from config.team_aliases import resolve_team_name

logger = logging.getLogger(__name__)


class InternationalResultsCollector:
    """Loads national-team match results from a local CSV or GitHub mirror."""

    def __init__(self, path: Path | None = None, url: str | None = None):
        self.path = path or INTERNATIONAL_RESULTS_PATH
        self.url = url or INTERNATIONAL_RESULTS_URL

    def _ensure_csv(self) -> None:
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading international results from %s", self.url)
        response = requests.get(self.url, timeout=30)
        response.raise_for_status()
        self.path.write_bytes(response.content)

    def collect_match_history(self) -> pd.DataFrame:
        self._ensure_csv()
        df = pd.read_csv(self.path)
        required = {"date", "home_team", "away_team", "home_score", "away_score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"International results CSV missing columns: {sorted(missing)}")

        df = df.rename(columns={
            "home_score": "goals_a",
            "away_score": "goals_b",
        })
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "home_team", "away_team", "goals_a", "goals_b"])
        df["team_a"] = df["home_team"].map(lambda name: resolve_team_name(str(name)))
        df["team_b"] = df["away_team"].map(lambda name: resolve_team_name(str(name)))
        df = df.dropna(subset=["team_a", "team_b"]).copy()
        df["goals_a"] = df["goals_a"].astype(int)
        df["goals_b"] = df["goals_b"].astype(int)
        df["result"] = df.apply(
            lambda r: "W" if r.goals_a > r.goals_b else ("L" if r.goals_a < r.goals_b else "D"),
            axis=1,
        )
        df["neutral"] = df.get("neutral", True).fillna(True).astype(bool)
        df["tournament"] = df.get("tournament", "Unknown").fillna("Unknown").astype(str)
        df["source"] = "international_results"
        logger.info("International results loaded: %s matches", len(df))
        return df.sort_values("date").reset_index(drop=True)
