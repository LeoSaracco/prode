"""Features de riesgo: upset_probability, consistency_score, pressure_performance."""
import numpy as np


def compute_upset_probability(
    elo_favorite: float,
    elo_underdog: float,
    underdog_form_5: float = 0.60,
) -> float:
    """
    Probabilidad de sorpresa del underdog.
    Aumenta cuando el underdog tiene buena forma reciente.
    """
    elo_diff = max(elo_favorite - elo_underdog, 0)
    base_upset = 1.0 / (1.0 + 10 ** (elo_diff / 400))
    form_bonus = (underdog_form_5 - 0.5) * 0.10
    return min(max(base_upset + form_bonus, 0.02), 0.48)


def compute_consistency_score(results_last10: list[int]) -> float:
    """
    Consistencia de resultados en los últimos 10 partidos.
    Lista de puntos (3=W, 1=D, 0=L). Mide varianza: baja varianza = alta consistencia.
    """
    if not results_last10:
        return 0.5
    arr = np.array(results_last10, dtype=float)
    std = np.std(arr)
    return max(0.0, 1.0 - std / 1.5)


def compute_pressure_performance(
    form_in_knockout: float = 0.65,
    wc_history_score: float = 0.50,
    elo_rating: float = 1800.0,
) -> float:
    """
    Rendimiento bajo presión (eliminatorias, cuartos, semifinales).
    Equipos con mucha experiencia en rondas finales tienen mejor presión.
    """
    elo_norm = min((elo_rating - 1400) / 800, 1.0)
    return 0.40 * form_in_knockout + 0.35 * wc_history_score + 0.25 * elo_norm


def compute_tactical_advantage(
    elo_diff: float,
    offensive_power_a: float,
    defensive_stability_b: float,
    offensive_power_b: float,
    defensive_stability_a: float,
) -> float:
    """
    Ventaja táctica neta de A sobre B.
    Considera el clash ataque de A vs defensa de B y viceversa.
    Retorna valor en [-1, 1].
    """
    attack_clash_a = offensive_power_a - defensive_stability_b
    attack_clash_b = offensive_power_b - defensive_stability_a
    net_tactical = attack_clash_a - attack_clash_b
    elo_component = np.tanh(elo_diff / 300) * 0.3
    return np.tanh(net_tactical * 2) * 0.7 + elo_component
