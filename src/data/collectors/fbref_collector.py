"""Recolector de estadísticas de FBref via soccerdata."""
import logging
import time
from pathlib import Path

import pandas as pd

from config.settings import DATA_RAW, FBREF_LEAGUES, FBREF_SEASONS, CACHE_TTL_HOURS

logger = logging.getLogger(__name__)

CACHE_SCHEDULE = DATA_RAW / "fbref" / "schedule.parquet"
CACHE_SHOOTING = DATA_RAW / "fbref" / "shooting.parquet"
CACHE_SEASON = DATA_RAW / "fbref" / "season_stats.parquet"


class FBrefCollector:
    """Wraps soccerdata.FBref para obtener stats de partidos y temporadas."""

    def __init__(self):
        self._fbref = None

    def _get_client(self, league: str, season: str):
        try:
            import soccerdata as sd
            return sd.FBref(leagues=league, seasons=season)
        except ImportError:
            logger.error("soccerdata no instalado. Instalar con: pip install soccerdata")
            return None
        except Exception as e:
            logger.warning(f"Error inicializando FBref para {league} {season}: {e}")
            return None

    def collect_match_schedule(
        self,
        leagues: list[str] | None = None,
        seasons: list[str] | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if not force_refresh and self._cache_valid(CACHE_SCHEDULE):
            logger.info("Usando cache de FBref schedule")
            return pd.read_parquet(CACHE_SCHEDULE)

        leagues = leagues or FBREF_LEAGUES
        seasons = seasons or FBREF_SEASONS
        frames = []
        for league in leagues:
            for season in seasons:
                client = self._get_client(league, season)
                if client is None:
                    continue
                try:
                    df = client.read_schedule()
                    df["league"] = league
                    df["season"] = season
                    frames.append(df)
                    logger.info(f"FBref schedule: {league} {season} -> {len(df)} partidos")
                    time.sleep(3)  # rate limiting
                except Exception as e:
                    logger.warning(f"Error FBref schedule {league} {season}: {e}")

        if not frames:
            logger.warning("FBref schedule: sin datos obtenidos")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        CACHE_SCHEDULE.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(CACHE_SCHEDULE, index=False)
        logger.info(f"FBref schedule guardado: {len(result)} registros")
        return result

    def collect_team_shooting_stats(
        self,
        leagues: list[str] | None = None,
        seasons: list[str] | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if not force_refresh and self._cache_valid(CACHE_SHOOTING):
            logger.info("Usando cache de FBref shooting")
            return pd.read_parquet(CACHE_SHOOTING)

        leagues = leagues or FBREF_LEAGUES
        seasons = seasons or FBREF_SEASONS
        frames = []
        for league in leagues:
            for season in seasons:
                client = self._get_client(league, season)
                if client is None:
                    continue
                try:
                    df = client.read_team_season_stats(stat_type="shooting")
                    df["league"] = league
                    df["season"] = season
                    frames.append(df)
                    time.sleep(3)
                except Exception as e:
                    logger.warning(f"Error FBref shooting {league} {season}: {e}")

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result.to_parquet(CACHE_SHOOTING, index=False)
        return result

    def collect_season_stats(
        self,
        leagues: list[str] | None = None,
        seasons: list[str] | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if not force_refresh and self._cache_valid(CACHE_SEASON):
            return pd.read_parquet(CACHE_SEASON)

        leagues = leagues or FBREF_LEAGUES
        seasons = seasons or FBREF_SEASONS
        frames = []
        for stat_type in ["standard", "shooting", "passing", "defense"]:
            for league in leagues:
                for season in seasons:
                    client = self._get_client(league, season)
                    if client is None:
                        continue
                    try:
                        df = client.read_team_season_stats(stat_type=stat_type)
                        df["league"] = league
                        df["season"] = season
                        df["stat_type"] = stat_type
                        frames.append(df)
                        time.sleep(3)
                    except Exception as e:
                        logger.warning(f"Error FBref {stat_type} {league} {season}: {e}")

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result.to_parquet(CACHE_SEASON, index=False)
        return result

    def _cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        return age_hours < CACHE_TTL_HOURS
