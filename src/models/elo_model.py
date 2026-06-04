"""Modelo de probabilidades basado en Elo (Bradley-Terry). Sin entrenamiento."""
from src.features.elo_features import compute_elo_probabilities, get_elo_rating
import pandas as pd


class EloModel:
    """Modelo sin entrenamiento. Usa fórmula Elo directamente."""

    def __init__(self, elo_df: pd.DataFrame | None = None):
        self.elo_df = elo_df if elo_df is not None else pd.DataFrame()

    def predict_proba(self, team_a: str, team_b: str) -> tuple[float, float, float]:
        """Retorna (p_win_a, p_draw, p_win_b)."""
        elo_a = get_elo_rating(team_a, self.elo_df)
        elo_b = get_elo_rating(team_b, self.elo_df)
        return compute_elo_probabilities(elo_a, elo_b)
