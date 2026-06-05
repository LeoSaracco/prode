"""Monte Carlo match simulator. 100k iteraciones con numpy vectorizado."""
import numpy as np
from config.settings import MC_ITERATIONS, POISSON_MAX_GOALS


def simulate_match(
    lambda_a: float,
    lambda_b: float,
    n_sims: int = MC_ITERATIONS,
) -> dict:
    """
    Simula un partido usando distribución Poisson vectorizada.
    Más rápido que loop con numba para uso interactivo.
    Retorna dict con probabilidades y estadísticas.
    """
    rng = np.random.default_rng()
    goals_a = rng.poisson(lambda_a, n_sims)
    goals_b = rng.poisson(lambda_b, n_sims)

    wins_a = int(np.sum(goals_a > goals_b))
    draws = int(np.sum(goals_a == goals_b))
    wins_b = int(np.sum(goals_b > goals_a))

    p_win = wins_a / n_sims
    p_draw = draws / n_sims
    p_loss = wins_b / n_sims

    # Distribución de marcadores
    max_g = POISSON_MAX_GOALS
    scoreline_counts: dict[tuple[int, int], int] = {}
    for ga, gb in zip(goals_a, goals_b):
        ga_c = min(int(ga), max_g)
        gb_c = min(int(gb), max_g)
        key = (ga_c, gb_c)
        scoreline_counts[key] = scoreline_counts.get(key, 0) + 1

    top_scorelines = sorted(
        [(k[0], k[1], v / n_sims) for k, v in scoreline_counts.items()],
        key=lambda x: x[2],
        reverse=True,
    )[:5]

    return {
        "p_win_a": round(p_win, 4),
        "p_draw": round(p_draw, 4),
        "p_win_b": round(p_loss, 4),
        "xg_a": round(float(goals_a.mean()), 2),
        "xg_b": round(float(goals_b.mean()), 2),
        "top_scorelines": top_scorelines,
        "n_sims": n_sims,
    }


def simulate_match_from_models(
    team_a: str,
    team_b: str,
    poisson_model=None,
    ensemble_result: dict | None = None,
    n_sims: int = MC_ITERATIONS,
) -> dict:
    """Simulación Monte Carlo usando lambdas del modelo Poisson."""
    if poisson_model is not None:
        lambda_a, lambda_b = poisson_model.predict_goals(team_a, team_b)
    elif ensemble_result is not None:
        lambda_a = ensemble_result.get("xg_a", 1.2)
        lambda_b = ensemble_result.get("xg_b", 1.0)
    else:
        from src.data.national_team_proxy import FALLBACK_STATS
        lambda_a = FALLBACK_STATS.get(team_a, {"xg_pg": 1.2})["xg_pg"]
        lambda_b = FALLBACK_STATS.get(team_b, {"xg_pg": 1.0})["xg_pg"]

    return simulate_match(lambda_a, lambda_b, n_sims)
