"""Gestión del cache de datos procesados."""
import logging
import time
from pathlib import Path

import pandas as pd

from config.settings import DATA_PROCESSED, CACHE_TTL_HOURS

logger = logging.getLogger(__name__)


class CacheManager:
    def save(self, df: pd.DataFrame, name: str) -> Path:
        path = DATA_PROCESSED / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Cache guardado: {path} ({len(df)} filas)")
        return path

    def load(self, name: str) -> pd.DataFrame | None:
        path = DATA_PROCESSED / f"{name}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def is_valid(self, name: str, ttl_hours: float | None = None) -> bool:
        path = DATA_PROCESSED / f"{name}.parquet"
        if not path.exists():
            return False
        ttl = ttl_hours or CACHE_TTL_HOURS
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        return age_hours < ttl
