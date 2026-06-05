"""Features de calidad y profundidad del plantel."""
import numpy as np

from config.feature_weights import SQUAD_DEPTH_WEIGHTS
from src.data.national_team_proxy import MARKET_VALUE_EUR_M

MAX_MARKET_VALUE = 1500.0  # millones EUR (England/France ~top)
MAX_OVERALL_RATING = 88.0  # SoFIFA max típico para selecciones top


def compute_squad_depth_score(
    avg_overall_rating: float = 78.0,
    depth_falloff: float = 0.85,
    experience_factor: float = 0.70,
    market_value_eur_m: float = 500.0,
) -> float:
    """
    Score de profundidad del plantel normalizado a [0, 1].
    - avg_overall_rating: promedio SoFIFA de los 23 convocados
    - depth_falloff: rating 12vo / rating 1ero (cuánto cae la calidad)
    - experience_factor: % de jugadores con >30 partidos internacionales
    - market_value_eur_m: valor total del plantel en millones EUR
    """
    rating_norm = min(avg_overall_rating / MAX_OVERALL_RATING, 1.0)
    depth_norm = min(max(depth_falloff, 0.0), 1.0)
    exp_norm = min(max(experience_factor, 0.0), 1.0)
    mv_norm = min(market_value_eur_m / MAX_MARKET_VALUE, 1.0)

    w = SQUAD_DEPTH_WEIGHTS
    return (
        w["avg_overall_rating"] * rating_norm +
        w["depth_falloff"] * depth_norm +
        w["experience_factor"] * exp_norm +
        w["market_value_normalized"] * mv_norm
    )


def compute_fatigue_index(matches_last_30d: int = 4, tournament_round: str = "group") -> float:
    """
    Índice de fatiga [0, 1]. Mayor = más fatiga.
    Aumenta con más partidos recientes y a medida que avanza el torneo.
    """
    round_multiplier = {"group": 1.0, "r32": 1.1, "r16": 1.2, "qf": 1.3, "sf": 1.4, "final": 1.5}
    base = min(matches_last_30d / 12, 1.0)
    mult = round_multiplier.get(tournament_round, 1.0)
    return min(base * mult, 1.0)


def compute_squad_depth_from_market_value(team: str) -> float:
    """Fallback: calcula squad_depth_score a partir del valor de mercado."""
    mv = MARKET_VALUE_EUR_M.get(team, 100.0)
    return compute_squad_depth_score(
        avg_overall_rating=70.0 + (mv / MAX_MARKET_VALUE) * 18.0,
        depth_falloff=0.75 + (mv / MAX_MARKET_VALUE) * 0.20,
        experience_factor=0.60 + (mv / MAX_MARKET_VALUE) * 0.30,
        market_value_eur_m=mv,
    )


def compute_squad_quality_from_ratings(
    avg_overall: float,
    max_overall: float,
    avg_shooting: float = 65.0,
    avg_defending: float = 63.0,
) -> float:
    """
    Score de calidad del plantel [0, 1] basado en ratings reales de jugadores.

    - avg_overall: calidad media del plantel (FIFA ratings, rango ~60-88)
    - max_overall: rating del mejor jugador (rango ~65-92)
    - avg_shooting / avg_defending: calidad específica ofensiva/defensiva
    - depth_ratio = avg/max: cuán equilibrado es el plantel (1=uniforme, <0.85=estrella-dependiente)
    """
    avg_norm  = min(max((avg_overall - 60.0) / 30.0, 0.0), 1.0)   # [60,90] → [0,1]
    max_norm  = min(max((max_overall - 65.0) / 27.0, 0.0), 1.0)   # [65,92] → [0,1]
    depth_ratio = avg_overall / max(max_overall, 1.0)               # robustez sin la estrella
    sh_norm   = min(max((avg_shooting - 50.0) / 30.0, 0.0), 1.0)
    df_norm   = min(max((avg_defending - 50.0) / 30.0, 0.0), 1.0)

    return (
        0.35 * avg_norm +
        0.20 * max_norm +
        0.25 * depth_ratio +
        0.10 * sh_norm +
        0.10 * df_norm
    )


def compute_squad_quality(team: str) -> float:
    """
    Score principal de calidad del plantel para un equipo.
    Usa ratings FIFA si están disponibles; fallback a valor de mercado.
    """
    from src.data.national_team_proxy import load_player_ratings_from_kaggle
    ratings = load_player_ratings_from_kaggle()
    if team in ratings:
        r = ratings[team]
        return compute_squad_quality_from_ratings(
            avg_overall=r["avg_overall"],
            max_overall=r["max_overall"],
            avg_shooting=r.get("avg_shooting", 65.0),
            avg_defending=r.get("avg_defending", 63.0),
        )
    return compute_squad_depth_from_market_value(team)
