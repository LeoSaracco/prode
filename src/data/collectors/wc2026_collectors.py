"""Collectors for FIFA World Cup 2026 schedule, squads, and team values."""

from __future__ import annotations

import logging
import re
from io import BytesIO

import pandas as pd
import requests

from config.settings import (
    FIFA_WC2026_SCHEDULE_URL,
    FIFA_WC2026_SQUAD_PDF_URL,
    TRANSFERMARKT_WC2026_URL,
)
from config.team_aliases import resolve_team_name
from config.wc2026_groups import GROUPS

logger = logging.getLogger(__name__)

WC2026_TEAMS = {team for teams in GROUPS.values() for team in teams}


def _headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        ),
        "Accept-Language": "es-AR,es;q=0.9,en;q=0.7",
    }


def _canonical_team(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return resolve_team_name(text) or text


def _parse_percent(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace("%", "").replace(",", ".").strip()
    try:
        return round(float(text) / 100.0, 6)
    except ValueError:
        return None


def _parse_eur_millions(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).lower().replace("\xa0", " ").replace("€", "").strip()
    multiplier = 1.0
    if "mil mill" in text:
        multiplier = 1000.0
    elif "mill" in text:
        multiplier = 1.0
    elif "mil" in text:
        multiplier = 0.001
    text = re.sub(r"[^0-9,.\-]", "", text)
    if not text:
        return None
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", ".")
    try:
        return float(text) * multiplier
    except ValueError:
        return None


class FIFAWorldCupScheduleCollector:
    """Scrape a minimal, normalized WC2026 schedule from FIFA's public page."""

    def __init__(self, url: str | None = None):
        self.url = url or FIFA_WC2026_SCHEDULE_URL

    def collect(self) -> pd.DataFrame:
        from bs4 import BeautifulSoup

        response = requests.get(self.url, headers=_headers(), timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        rows: list[dict] = []
        for table in soup.find_all("table"):
            try:
                for _, row in pd.read_html(str(table))[0].iterrows():
                    rows.append({str(k): v for k, v in row.to_dict().items()})
            except Exception:
                continue

        if not rows:
            text = " ".join(soup.get_text(" ").split())
            for match_no in sorted(set(re.findall(r"\bM?(\d{1,3})\b", text))):
                rows.append({"match_no": int(match_no), "source_url": self.url})

        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("FIFA schedule page could not be parsed.")
            return df

        df = df.rename(columns={c: str(c).strip().lower().replace(" ", "_") for c in df.columns})
        if "match_no" not in df.columns:
            for col in df.columns:
                extracted = df[col].astype(str).str.extract(r"\bM?(\d{1,3})\b", expand=False)
                if extracted.notna().any():
                    df["match_no"] = pd.to_numeric(extracted, errors="coerce")
                    break
        df["source_url"] = self.url
        return df.drop_duplicates().reset_index(drop=True)


class FIFASquadListCollector:
    """Parse FIFA's official squad-list PDF into player-level rows."""

    def __init__(self, url: str | None = None):
        self.url = url or FIFA_WC2026_SQUAD_PDF_URL

    def collect(self) -> pd.DataFrame:
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning("Install pypdf to parse FIFA squad PDF.")
            return pd.DataFrame()

        response = requests.get(self.url, headers=_headers(), timeout=60)
        response.raise_for_status()
        reader = PdfReader(BytesIO(response.content))

        rows: list[dict] = []
        current_team: str | None = None
        player_re = re.compile(
            r"^(PO|DF|MC|DC)\s+(.+?)\s+(\d{1,2}/\d{1,2}/\d{4})\s+(.+?)\s+(\d{2,3})$"
        )
        team_re = re.compile(r"(?:CONVOCATORIA)?\s*([A-Za-zÀ-ÿ' .-]+)\s+\(([A-Z]{3})\)")

        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = " ".join(raw_line.split())
                if not line:
                    continue
                team_match = team_re.search(line)
                if team_match:
                    resolved = _canonical_team(team_match.group(1))
                    current_team = resolved if resolved in WC2026_TEAMS else resolved
                    continue
                if line.startswith("Entrenador") and current_team:
                    rows.append({
                        "team": current_team,
                        "row_type": "coach",
                        "raw_name": line.replace("Entrenador", "", 1).strip(),
                        "source_page": page_no,
                    })
                    continue
                match = player_re.match(line)
                if not match or not current_team:
                    continue
                pos, name_blob, birth_date, club, height = match.groups()
                rows.append({
                    "team": current_team,
                    "row_type": "player",
                    "position": pos,
                    "raw_name": name_blob.strip(),
                    "birth_date": pd.to_datetime(birth_date, format="%d/%m/%Y", errors="coerce"),
                    "club": club.strip(),
                    "height_cm": int(height),
                    "source_page": page_no,
                })

        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("FIFA squad PDF produced no parseable rows.")
            return df
        df = df[df["team"].isin(WC2026_TEAMS)].drop_duplicates().reset_index(drop=True)
        df["source_url"] = self.url
        return df


class TransfermarktWC2026Collector:
    """Collect team-level Transfermarkt aggregates for qualified WC2026 teams."""

    def __init__(self, url: str | None = None):
        self.url = url or TRANSFERMARKT_WC2026_URL

    def collect(self) -> pd.DataFrame:
        response = requests.get(self.url, headers=_headers(), timeout=30)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        candidates = [t for t in tables if any("Valor" in str(c) for c in t.columns)]
        if not candidates:
            logger.warning("Transfermarkt team-values table not found.")
            return pd.DataFrame()
        raw = candidates[0]
        raw.columns = [str(c).strip() for c in raw.columns]

        rows = []
        for _, row in raw.iterrows():
            values = [v for v in row.tolist() if not pd.isna(v)]
            if len(values) < 6:
                continue
            team = _canonical_team(values[0])
            if team not in WC2026_TEAMS:
                continue
            rows.append({
                "team": team,
                "squad_size": pd.to_numeric(values[1], errors="coerce"),
                "avg_age": pd.to_numeric(str(values[2]).replace(",", "."), errors="coerce"),
                "world_cup_appearances": pd.to_numeric(values[3], errors="coerce"),
                "foreign_players_pct": _parse_percent(values[4]),
                "market_value_eur_m": _parse_eur_millions(values[5]),
                "avg_market_value_eur_m": _parse_eur_millions(values[6] if len(values) > 6 else None),
                "source_url": self.url,
            })

        return pd.DataFrame(rows).drop_duplicates(subset=["team"]).reset_index(drop=True)
