"""Features basadas en ratings Elo."""
import numpy as np
import pandas as pd

from config.feature_weights import ELO_WIN_PROB_DIVISOR
from src.data.national_team_proxy import FALLBACK_ELO


def get_elo_rating(team: str, elo_source: pd.DataFrame | dict | None) -> float:
    if isinstance(elo_source, dict) and team in elo_source:
        return float(elo_source[team])
    if elo_source is not None and hasattr(elo_source, "__getitem__") and not isinstance(elo_source, pd.DataFrame):
        return float(elo_source.get(team, FALLBACK_ELO.get(team, 1600.0)))
    if elo_source is not None and not isinstance(elo_source, dict) and not elo_source.empty:
        match = elo_source[elo_source["team"] == team]
        if not match.empty:
            return float(match.iloc[0]["elo_rating"])
    return FALLBACK_ELO.get(team, 1600.0)


def compute_elo_win_probability(elo_a: float, elo_b: float) -> float:
    """P(A wins) según fórmula Elo estándar."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / ELO_WIN_PROB_DIVISOR))


def compute_elo_probabilities(elo_a: float, elo_b: float) -> tuple[float, float, float]:
    """
    Retorna (p_win_a, p_draw, p_win_b) usando ajuste de Elo para empates.
    El modelo de empates usa una banda central alrededor de p=0.5.
    """
    p_win = compute_elo_win_probability(elo_a, elo_b)
    elo_diff = abs(elo_a - elo_b)
    # Mayor diferencia Elo → menor probabilidad de empate
    draw_base = 0.25
    draw_adj = max(0.08, draw_base - (elo_diff / 4000))
    if p_win > 0.5:
        p_win_adj = p_win - draw_adj / 2
        p_loss = 1.0 - p_win_adj - draw_adj
    else:
        p_loss = (1 - p_win) - draw_adj / 2
        p_win_adj = 1.0 - p_loss - draw_adj
    # Normalizar
    total = p_win_adj + draw_adj + p_loss
    return p_win_adj / total, draw_adj / total, p_loss / total
