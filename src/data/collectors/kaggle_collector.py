"""Recolector del dataset histórico de Kaggle (European Soccer Database)."""
import logging
from pathlib import Path

import pandas as pd

from config.settings import KAGGLE_DB_PATH

logger = logging.getLogger(__name__)


class KaggleCollector:
    """
    Lee el dataset 'European Soccer Database' de Hugo Mathien (SQLite, 2008-2016).
    Usado para entrenamiento histórico y benchmarking, no para features actuales.
    """

    def __init__(self):
        self.db_path = KAGGLE_DB_PATH

    def _check_db(self) -> bool:
        if not Path(self.db_path).exists():
            logger.warning(
                f"Dataset Kaggle no encontrado en {self.db_path}. "
                "Descargalo desde https://www.kaggle.com/datasets/hugomathien/soccer "
                "y coloca el archivo database.sqlite en data/raw/kaggle/"
            )
            return False
        return True

    def collect_match_history(self) -> pd.DataFrame:
        if not self._check_db():
            return pd.DataFrame()
        try:
            from sqlalchemy import create_engine
            engine = create_engine(f"sqlite:///{self.db_path}")
            query = """
                SELECT
                    m.id, m.season, m.date,
                    m.home_team_api_id, m.away_team_api_id,
                    m.home_team_goal, m.away_team_goal,
                    t1.team_long_name AS home_team,
                    t2.team_long_name AS away_team,
                    l.name AS league_name,
                    c.name AS country_name
                FROM Match m
                JOIN Team t1 ON m.home_team_api_id = t1.team_api_id
                JOIN Team t2 ON m.away_team_api_id = t2.team_api_id
                JOIN League l ON m.league_id = l.id
                JOIN Country c ON l.country_id = c.id
                WHERE m.home_team_goal IS NOT NULL
                  AND m.away_team_goal IS NOT NULL
                ORDER BY m.date
            """
            df = pd.read_sql(query, engine)
            df["date"] = pd.to_datetime(df["date"])
            df["result"] = df.apply(
                lambda r: "H" if r.home_team_goal > r.away_team_goal
                else ("A" if r.home_team_goal < r.away_team_goal else "D"),
                axis=1,
            )
            logger.info(f"Kaggle match history: {len(df)} partidos cargados")
            return df
        except Exception as e:
            logger.error(f"Error leyendo Kaggle DB: {e}")
            return pd.DataFrame()

    def collect_player_attributes(self) -> pd.DataFrame:
        if not self._check_db():
            return pd.DataFrame()
        try:
            from sqlalchemy import create_engine
            engine = create_engine(f"sqlite:///{self.db_path}")
            query = """
                SELECT pa.*, p.player_name, p.birthday, p.height, p.weight
                FROM Player_Attributes pa
                JOIN Player p ON pa.player_api_id = p.player_api_id
                ORDER BY pa.date DESC
            """
            df = pd.read_sql(query, engine)
            logger.info(f"Kaggle player attributes: {len(df)} registros cargados")
            return df
        except Exception as e:
            logger.error(f"Error leyendo player attributes: {e}")
            return pd.DataFrame()
