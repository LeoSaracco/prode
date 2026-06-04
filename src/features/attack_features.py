"""Features ofensivas: offensive_power, big_match_rating."""
import numpy as np
import pandas as pd

from config.feature_weights import OFFENSIVE_POWER_WEIGHTS


def compute_offensive_power(
    xg_per_game: float,
    shots_on_target_pg: float = 4.5,
    conversion_rate: float = 0.30,
    big_chances_created: float = 1.5,
) -> float:
    """
    Score ofensivo compuesto, normalizado a [0, 1].
    Valores típicos de referencia: Argentina/Spain ~ 0.75-0.85
    """
    # Normalización por rangos típicos internacionales
    xg_norm = min(xg_per_game / 2.5, 1.0)
    sot_norm = min(shots_on_target_pg / 8.0, 1.0)
    conv_norm = min(conversion_rate / 0.45, 1.0)
    bc_norm = min(big_chances_created / 3.5, 1.0)

    w = OFFENSIVE_POWER_WEIGHTS
    return (
        w["xg_per_game"] * xg_norm +
        w["shots_on_target_per_game"] * sot_norm +
        w["conversion_rate"] * conv_norm +
        w["big_chances_created"] * bc_norm
    )


def compute_big_match_rating(
    form_5: float,
    wc_history_score: float,
    elo_rating: float,
) -> float:
    """
    Rendimiento en partidos grandes: combina forma reciente con historial
    de torneos de presión alta.
    """
    from config.feature_weights import WC_HISTORY_WEIGHT, RECENT_FORM_WEIGHT
    elo_norm = min((elo_rating - 1400) / 800, 1.0)
    score = (
        RECENT_FORM_WEIGHT * form_5 +
        WC_HISTORY_WEIGHT * wc_history_score +
        (1 - RECENT_FORM_WEIGHT - WC_HISTORY_WEIGHT) * elo_norm
    )
    return min(max(score, 0.0), 1.0)
