"""Collector for men's international football results."""

import logging
from pathlib import Path

import pandas as pd
import requests

from config.settings import INTERNATIONAL_RESULTS_PATH, INTERNATIONAL_RESULTS_URL
from config.team_aliases import resolve_team_name

logger = logging.getLogger(__name__)

WC_TOURNAMENT_KEYWORDS = [
    "FIFA World Cup", "World Cup qualification", "World Cup",
    "UEFA Euro", "Copa America", "Copa América",
    "African Cup of Nations", "AFC Asian Cup",
    "CONCACAF Gold Cup", "Nations League",
    "Confederations Cup",
]

TOURNAMENT_KEYWORDS = [
    "FIFA World Cup", "World Cup", "UEFA Euro", "Copa America", "Copa América",
    "African Cup of Nations", "AFC Asian Cup", "CONCACAF Gold Cup",
    "Nations League", "Confederations Cup",
    "OFC Nations Cup", "WAFF Championship", "AFF Championship",
    "EAFF E-1", "SAFF Championship", "Arabian Gulf Cup",
]


class InternationalResultsCollector:
    """Loads national-team match results from a local CSV or GitHub mirror.

    The data source (martj42/international_results) covers 1872-present
    with ~45,000 matches.
    """

    def __init__(self, path: Path | None = None, url: str | None = None):
        self.path = path or INTERNATIONAL_RESULTS_PATH
        self.url = url or INTERNATIONAL_RESULTS_URL

    def _ensure_csv(self) -> None:
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading international results from %s", self.url)
        response = requests.get(self.url, timeout=120)
        response.raise_for_status()
        self.path.write_bytes(response.content)

    def collect_match_history(self, start_year: int | None = None) -> pd.DataFrame:
        self._ensure_csv()
        df = pd.read_csv(self.path, low_memory=False)
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

        if start_year is not None:
            df = df[df["date"] >= pd.Timestamp(f"{start_year}-01-01")]

        df["team_a"] = df["home_team"].map(lambda name: resolve_team_name(str(name)))
        df["team_b"] = df["away_team"].map(lambda name: resolve_team_name(str(name)))
        df = df.dropna(subset=["team_a", "team_b"]).copy()
        df["goals_a"] = df["goals_a"].astype(int)
        df["goals_b"] = df["goals_b"].astype(int)
        df["result"] = df.apply(
            lambda r: "W" if r.goals_a > r.goals_b else ("L" if r.goals_a < r.goals_b else "D"),
            axis=1,
        )
        if "neutral" in df.columns:
            df["neutral"] = df["neutral"].fillna(True).astype(bool)
        else:
            df["neutral"] = True

        if "tournament" in df.columns:
            df["tournament"] = df["tournament"].fillna("Friendly").astype(str)
        else:
            df["tournament"] = "Unknown"

        df["is_tournament"] = df["tournament"].str.contains(
            "|".join(TOURNAMENT_KEYWORDS), case=False, na=False
        ).astype(int)
        df["is_wc"] = df["tournament"].str.contains(
            "World Cup", case=False, na=False
        ).astype(int)
        df["is_qualifier"] = df["tournament"].str.contains(
            "qualification|qualifying", case=False, na=False
        ).astype(int)

        if "city" in df.columns:
            df["city"] = df["city"].fillna("Unknown").astype(str)
        if "country" in df.columns:
            df["country"] = df["country"].fillna("Unknown").astype(str)

        df["source"] = "international_results"
        logger.info("International results loaded: %s matches (%s teams)",
                     len(df), df["team_a"].nunique())
        return df.sort_values("date").reset_index(drop=True)

    def collect_all_for_wc2026_teams(self, start_year: int = 2000) -> pd.DataFrame:
        """Collect all matches involving at least one WC2026 team.

        Includes matches against non-WC2026 opponents for stats building.
        Returns DataFrame with valid team_a (WC2026 team) and team_b (any opponent).
        """
        self._ensure_csv()
        df = pd.read_csv(self.path, low_memory=False)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
        df = df[df["date"] >= pd.Timestamp(f"{start_year}-01-01")]

        from config.team_aliases import TEAM_ALIASES
        wc_teams = set(TEAM_ALIASES.keys())

        home_resolved = df["home_team"].map(lambda name: resolve_team_name(str(name)))
        away_resolved = df["away_team"].map(lambda name: resolve_team_name(str(name)))

        mask = home_resolved.notna() | away_resolved.notna()
        df = df[mask].copy()
        df["team_a"] = home_resolved
        df["team_b"] = away_resolved

        df["goals_a"] = pd.to_numeric(df["home_score"], errors="coerce").astype(int)
        df["goals_b"] = pd.to_numeric(df["away_score"], errors="coerce").astype(int)

        if "neutral" in df.columns:
            df["neutral"] = df["neutral"].fillna(True).astype(bool)
        else:
            df["neutral"] = True

        if "tournament" in df.columns:
            df["tournament"] = df["tournament"].fillna("Friendly").astype(str)
        else:
            df["tournament"] = "Unknown"

        df["source"] = "international_results"
        logger.info("All matches (>=1 WC2026 team): %s rows", len(df))
        return df.sort_values("date").reset_index(drop=True)
