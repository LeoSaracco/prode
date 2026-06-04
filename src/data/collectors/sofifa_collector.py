"""Recolector de ratings de planteles desde SoFIFA via soccerdata."""
import logging
import time
from pathlib import Path

import pandas as pd

from config.settings import DATA_RAW, FBREF_LEAGUES, CACHE_TTL_HOURS

logger = logging.getLogger(__name__)

CACHE_TEAM = DATA_RAW / "sofifa" / "team_ratings.parquet"
CACHE_PLAYER = DATA_RAW / "sofifa" / "player_ratings.parquet"


class SoFIFACollector:
    """
    Wraps soccerdata.SoFIFA para ratings de equipos y jugadores.
    Requiere Chromium/Chrome instalado (browser automation).
    Completamente opcional — si falla, el sistema usa fallbacks.
    """

    def collect_team_ratings(
        self,
        leagues: list[str] | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if not force_refresh and self._cache_valid(CACHE_TEAM):
            logger.info("Usando cache de SoFIFA team ratings")
            return pd.read_parquet(CACHE_TEAM)

        leagues = leagues or FBREF_LEAGUES
        frames = []
        for league in leagues:
            try:
                import soccerdata as sd
                client = sd.SoFIFA(leagues=league)
                df = client.read_team_ratings()
                df["league"] = league
                frames.append(df)
                logger.info(f"SoFIFA team ratings: {league} -> {len(df)} equipos")
                time.sleep(2)
            except ImportError:
                logger.error("soccerdata no instalado")
                break
            except Exception as e:
                logger.warning(f"SoFIFA team ratings {league}: {e}. (¿Chromium disponible?)")

        if not frames:
            logger.warning("SoFIFA team ratings: sin datos. Usar fallback de ratings FIFA.")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        CACHE_TEAM.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(CACHE_TEAM, index=False)
        return result

    def collect_player_ratings(
        self,
        leagues: list[str] | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if not force_refresh and self._cache_valid(CACHE_PLAYER):
            return pd.read_parquet(CACHE_PLAYER)

        leagues = leagues or FBREF_LEAGUES
        frames = []
        for league in leagues:
            try:
                import soccerdata as sd
                client = sd.SoFIFA(leagues=league)
                df = client.read_player_ratings()
                df["league"] = league
                frames.append(df)
                time.sleep(2)
            except Exception as e:
                logger.warning(f"SoFIFA player ratings {league}: {e}")

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result.to_parquet(CACHE_PLAYER, index=False)
        return result

    def _cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        return age_hours < CACHE_TTL_HOURS
