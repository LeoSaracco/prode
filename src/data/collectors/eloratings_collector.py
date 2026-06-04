"""Recolector de ratings Elo nacionales desde eloratings.net."""
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from config.settings import DATA_RAW, CACHE_TTL_HOURS

logger = logging.getLogger(__name__)

ELO_URL = "https://www.eloratings.net/World.tsv"
CACHE_PATH = DATA_RAW / "eloratings" / "world_elo.parquet"

# Formato real del TSV (31 cols):
# [0]=rank [1]=peak_rank [2]=iso2_code [3]=elo_current ...
ISO2_TO_TEAM: dict[str, str] = {
    "ES": "Spain", "AR": "Argentina", "FR": "France", "EN": "England",
    "BR": "Brazil", "PT": "Portugal", "NL": "Netherlands", "BE": "Belgium",
    "DE": "Germany", "HR": "Croatia", "IT": "Italy", "DK": "Denmark",
    "CH": "Switzerland", "AT": "Austria", "PL": "Poland", "CZ": "Czechia",
    "RS": "Serbia", "UA": "Ukraine", "TR": "Turkey", "SC": "Scotland",
    "HU": "Hungary", "SK": "Slovakia", "RO": "Romania", "GR": "Greece",
    "BA": "Bosnia and Herzegovina", "AL": "Albania", "SI": "Slovenia",
    "ME": "Montenegro", "GE": "Georgia", "IS": "Iceland",
    "IE": "Republic of Ireland", "NO": "Norway", "SE": "Sweden",
    "FI": "Finland", "RU": "Russia", "WA": "Wales",
    # CONMEBOL
    "UY": "Uruguay", "CL": "Chile", "CO": "Colombia", "PE": "Peru",
    "PY": "Paraguay", "EC": "Ecuador", "BO": "Bolivia", "VE": "Venezuela",
    # CONCACAF
    "MX": "Mexico", "US": "United States", "CA": "Canada",
    "CR": "Costa Rica", "PA": "Panama", "HN": "Honduras",
    "JM": "Jamaica", "TT": "Trinidad and Tobago", "CU": "Cuba",
    "GT": "Guatemala", "SV": "El Salvador", "HT": "Haiti",
    # CAF
    "MA": "Morocco", "SN": "Senegal", "NG": "Nigeria", "CM": "Cameroon",
    "EG": "Egypt", "TN": "Tunisia", "ML": "Mali", "DZ": "Algeria",
    "CI": "Cote d'Ivoire", "CD": "DR Congo", "GH": "Ghana",
    "ZA": "South Africa", "GA": "Gabon", "BF": "Burkina Faso",
    "GN": "Guinea", "MR": "Mauritania", "BJ": "Benin",
    # AFC
    "JP": "Japan", "KR": "South Korea", "IR": "Iran", "SA": "Saudi Arabia",
    "AU": "Australia", "UZ": "Uzbekistan", "QA": "Qatar", "BH": "Bahrain",
    "JO": "Jordan", "SY": "Syria", "IQ": "Iraq", "OM": "Oman",
    "AE": "United Arab Emirates", "KW": "Kuwait", "KG": "Kyrgyzstan",
    "TJ": "Tajikistan", "CN": "China PR", "IN": "India", "TH": "Thailand",
    "VN": "Vietnam", "MY": "Malaysia", "ID": "Indonesia",
    # OFC
    "NZ": "New Zealand", "FJ": "Fiji", "PG": "Papua New Guinea",
}


class EloRatingsCollector:
    """Obtiene ratings Elo de selecciones nacionales desde eloratings.net."""

    def collect_current_ratings(self, force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and self._cache_valid():
            logger.info("Usando cache de Elo ratings")
            return pd.read_parquet(CACHE_PATH)

        logger.info("Descargando Elo ratings desde eloratings.net...")
        df = self._fetch_with_retry()
        if df is not None and not df.empty:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(CACHE_PATH, index=False)
            logger.info(f"Elo ratings guardados: {len(df)} selecciones")
        return df if df is not None else pd.DataFrame()

    def _fetch_with_retry(self, max_retries: int = 3) -> pd.DataFrame | None:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        for attempt in range(max_retries):
            try:
                resp = requests.get(ELO_URL, headers=headers, timeout=30)
                resp.raise_for_status()
                content = resp.content.decode("utf-8", errors="replace")
                return self._parse_tsv(content)
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"Intento {attempt+1} fallido: {e}. Reintentando en {wait}s...")
                time.sleep(wait)
        logger.error("No se pudo obtener Elo ratings")
        return None

    def _parse_tsv(self, content: str) -> pd.DataFrame:
        lines = [line for line in content.strip().split("\n") if line.strip()]
        records = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 4:
                try:
                    iso2 = parts[2].strip()
                    elo = float(parts[3])
                    team_name = ISO2_TO_TEAM.get(iso2, iso2)
                    records.append({
                        "rank": int(parts[0]),
                        "iso2": iso2,
                        "team": team_name,
                        "elo_rating": elo,
                    })
                except (ValueError, IndexError):
                    continue
        return pd.DataFrame(records)

    def _cache_valid(self) -> bool:
        if not CACHE_PATH.exists():
            return False
        age_hours = (time.time() - CACHE_PATH.stat().st_mtime) / 3600
        return age_hours < CACHE_TTL_HOURS
