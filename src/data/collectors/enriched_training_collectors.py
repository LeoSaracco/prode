"""Collectors for optional enriched football training datasets.

Kaggle datasets are read from local files because Kaggle access usually
requires credentials. The Hugging Face dataset is downloaded through the
optional `datasets` package when local files are absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from config.settings import (
    KAGGLE_ELO_DIR,
    KAGGLE_MATCH_FEATURES_DIR,
    STATSBOMB_SHOTS_DATASET,
    STATSBOMB_SHOTS_DIR,
)
from config.team_aliases import resolve_team_name

logger = logging.getLogger(__name__)


def _first_present(columns: Iterable[str], candidates: list[str]) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return None


def _read_tabular_files(directory: Path) -> pd.DataFrame:
    if not directory.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() == ".csv":
            frames.append(pd.read_csv(path))
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frames.append(pd.read_parquet(path))
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            frames.append(pd.read_excel(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


class KaggleInternationalEloCollector:
    """Reads Saif Alnimri's international Elo dataset from local files."""

    def __init__(self, directory: Path | None = None):
        self.directory = directory or KAGGLE_ELO_DIR

    def collect(self) -> pd.DataFrame:
        raw = _read_tabular_files(self.directory)
        if raw.empty:
            logger.warning("Kaggle Elo dataset not found in %s", self.directory)
            return pd.DataFrame()

        team_col = _first_present(raw.columns, ["team", "country", "name", "nation"])
        date_col = _first_present(raw.columns, ["date", "rank_date", "rating_date"])
        rating_col = _first_present(raw.columns, ["elo", "rating", "elo_rating", "current_elo"])
        if not team_col or not rating_col:
            logger.warning("Kaggle Elo dataset could not be standardized: %s", list(raw.columns))
            return pd.DataFrame()

        df = pd.DataFrame({
            "team": raw[team_col].map(lambda value: resolve_team_name(str(value))),
            "elo_rating": pd.to_numeric(raw[rating_col], errors="coerce"),
        })
        if date_col:
            df["date"] = pd.to_datetime(raw[date_col], errors="coerce")
        else:
            df["date"] = pd.NaT
        df = df.dropna(subset=["team", "elo_rating"]).copy()
        logger.info("Kaggle Elo standardized: %s rows", len(df))
        return df


class KaggleMatchFeaturesCollector:
    """Reads L. Chikry's match-features dataset from local files."""

    def __init__(self, directory: Path | None = None):
        self.directory = directory or KAGGLE_MATCH_FEATURES_DIR

    def collect(self) -> pd.DataFrame:
        raw = _read_tabular_files(self.directory)
        if raw.empty:
            logger.warning("Kaggle match-features dataset not found in %s", self.directory)
            return pd.DataFrame()

        home_col = _first_present(raw.columns, ["_home_team", "home_team", "home", "team_a", "team1"])
        away_col = _first_present(raw.columns, ["_away_team", "away_team", "away", "team_b", "team2"])
        home_goals = _first_present(raw.columns, ["home_goals", "home_score", "goals_home", "home_team_goal", "score_home"])
        away_goals = _first_present(raw.columns, ["away_goals", "away_score", "goals_away", "away_team_goal", "score_away"])
        date_col = _first_present(raw.columns, ["_date", "date", "match_date"])
        tournament_col = _first_present(raw.columns, ["_tournament", "tournament", "competition", "competition_name"])
        neutral_col = _first_present(raw.columns, ["is_neutral", "neutral"])
        xg_home_col = _first_present(raw.columns, ["home_xg", "xg_home", "home_team_xg"])
        xg_away_col = _first_present(raw.columns, ["away_xg", "xg_away", "away_team_xg"])
        if not all([home_col, away_col, home_goals, away_goals]):
            logger.warning("Kaggle match-features dataset could not be standardized: %s", list(raw.columns))
            return pd.DataFrame()

        df = pd.DataFrame({
            "date": pd.to_datetime(raw[date_col], errors="coerce") if date_col else pd.NaT,
            "team_a": raw[home_col].map(lambda value: resolve_team_name(str(value))),
            "team_b": raw[away_col].map(lambda value: resolve_team_name(str(value))),
            "goals_a": pd.to_numeric(raw[home_goals], errors="coerce"),
            "goals_b": pd.to_numeric(raw[away_goals], errors="coerce"),
            "tournament": raw[tournament_col].astype(str) if tournament_col else "Unknown",
            "neutral": raw[neutral_col].fillna(True).astype(bool) if neutral_col else True,
            "source": "kaggle_match_features",
        })
        if xg_home_col and xg_away_col:
            df["xg_a"] = pd.to_numeric(raw[xg_home_col], errors="coerce")
            df["xg_b"] = pd.to_numeric(raw[xg_away_col], errors="coerce")
        df = df.dropna(subset=["team_a", "team_b", "goals_a", "goals_b"]).copy()
        df["goals_a"] = df["goals_a"].astype(int)
        df["goals_b"] = df["goals_b"].astype(int)
        df["result"] = df.apply(
            lambda r: "W" if r.goals_a > r.goals_b else ("L" if r.goals_a < r.goals_b else "D"),
            axis=1,
        )
        logger.info("Kaggle match features standardized: %s matches", len(df))
        return df.sort_values("date").reset_index(drop=True)


class StatsBombShotsCollector:
    """Loads StatsBomb open-data shots and aggregates xG by national team."""

    def __init__(self, directory: Path | None = None, dataset_name: str | None = None):
        self.directory = directory or STATSBOMB_SHOTS_DIR
        self.dataset_name = dataset_name or STATSBOMB_SHOTS_DATASET

    def collect(self) -> pd.DataFrame:
        raw = _read_tabular_files(self.directory)
        if raw.empty:
            try:
                from datasets import load_dataset
            except ImportError:
                logger.warning("Install `datasets` or place StatsBomb shots files in %s", self.directory)
                return pd.DataFrame()
            try:
                dataset = load_dataset(self.dataset_name)
                raw = pd.concat(
                    [dataset[split].to_pandas() for split in dataset.keys()],
                    ignore_index=True,
                )
            except Exception as e:
                logger.warning("StatsBomb HF dataset error: %s", e)
                return pd.DataFrame()
        team_col = _first_present(raw.columns, ["team"])
        date_col = _first_present(raw.columns, ["match_date", "date"])
        xg_col = _first_present(raw.columns, ["xg_statsbomb", "xg"])
        competition_col = _first_present(raw.columns, ["competition_name", "competition"])
        if not team_col or not xg_col:
            logger.warning("StatsBomb shots could not be standardized: %s", list(raw.columns))
            return pd.DataFrame()

        df = pd.DataFrame({
            "date": pd.to_datetime(raw[date_col], errors="coerce") if date_col else pd.NaT,
            "team": raw[team_col].map(lambda value: resolve_team_name(str(value))),
            "xg": pd.to_numeric(raw[xg_col], errors="coerce"),
            "competition": raw[competition_col].astype(str) if competition_col else "Unknown",
        })
        df = df.dropna(subset=["team", "xg"]).copy()
        logger.info("StatsBomb shots standardized: %s shots", len(df))
        return df

    def collect_team_xg_profiles(self) -> pd.DataFrame:
        shots = self.collect()
        if shots.empty:
            return pd.DataFrame()
        grouped = shots.groupby("team").agg(
            statsbomb_shots=("xg", "size"),
            statsbomb_xg_total=("xg", "sum"),
            statsbomb_xg_per_shot=("xg", "mean"),
        ).reset_index()
        return grouped
