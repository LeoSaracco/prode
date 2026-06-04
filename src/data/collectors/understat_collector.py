"""Recolector de datos xG de Understat via soccerdata."""
import logging
import time
from pathlib import Path

import pandas as pd

from config.settings import DATA_RAW, UNDERSTAT_LEAGUES, UNDERSTAT_SEASONS, CACHE_TTL_HOURS

logger = logging.getLogger(__name__)

CACHE_MATCH = DATA_RAW / "understat" / "match_stats.parquet"


class UnderstatCollector:
    """Obtiene xG a nivel de partido desde Understat (cubre Big 5 europeas)."""

    def collect_team_match_stats(
        self,
        leagues: list[str] | None = None,
        seasons: list[int] | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if not force_refresh and self._cache_valid(CACHE_MATCH):
            logger.info("Usando cache de Understat match stats")
            return pd.read_parquet(CACHE_MATCH)

        leagues = leagues or UNDERSTAT_LEAGUES
        seasons = seasons or UNDERSTAT_SEASONS
        frames = []

        for league in leagues:
            for season in seasons:
                try:
                    import soccerdata as sd
                    client = sd.Understat(leagues=league, seasons=season)
                    df = client.read_team_match_stats()
                    df["league"] = league
                    df["season"] = season
                    frames.append(df)
                    logger.info(f"Understat: {league} {season} -> {len(df)} registros")
                    time.sleep(2)
                except ImportError:
                    logger.error("soccerdata no instalado")
                    break
                except Exception as e:
                    logger.warning(f"Error Understat {league} {season}: {e}")

        if not frames:
            logger.warning("Understat: sin datos obtenidos")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        CACHE_MATCH.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(CACHE_MATCH, index=False)
        logger.info(f"Understat stats guardados: {len(result)} registros")
        return result

    def _cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        return age_hours < CACHE_TTL_HOURS
