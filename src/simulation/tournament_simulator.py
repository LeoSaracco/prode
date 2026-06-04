"""
Simulador completo del torneo FIFA 2026 (48 equipos).
Genera probabilidades de campeón para todas las selecciones.
"""
import logging
from itertools import combinations

import numpy as np
import pandas as pd

from config.settings import MC_ITERATIONS, WC2026_BEST_THIRDS
from config.wc2026_groups import GROUPS

logger = logging.getLogger(__name__)


def _simulate_match_result(
    team_a: str,
    team_b: str,
    poisson_model,
    rng: np.random.Generator,
) -> str:
    """Simula un partido. En eliminatorias: sin empates (penales si empatan en 90')."""
    from src.data.national_team_proxy import FALLBACK_STATS
    if poisson_model is not None:
        lam_a, lam_b = poisson_model.predict_goals(team_a, team_b)
    else:
        lam_a = FALLBACK_STATS.get(team_a, {"xg_pg": 1.2})["xg_pg"]
        lam_b = FALLBACK_STATS.get(team_b, {"xg_pg": 1.0})["xg_pg"]
    ga = int(rng.poisson(lam_a))
    gb = int(rng.poisson(lam_b))
    if ga > gb:
        return team_a
    elif gb > ga:
        return team_b
    else:
        # Penales: 50/50 con ajuste leve por Elo
        from src.features.elo_features import compute_elo_win_probability, get_elo_rating
        elo_a = get_elo_rating(team_a, None)
        elo_b = get_elo_rating(team_b, None)
        p = compute_elo_win_probability(elo_a, elo_b)
        return team_a if rng.random() < p else team_b


def _simulate_group(teams: list[str], poisson_model, rng: np.random.Generator) -> list[str]:
    """Simula grupo y retorna equipos ordenados [1ro, 2do, 3ro, 4to]."""
    table = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
    for ta, tb in combinations(teams, 2):
        from src.data.national_team_proxy import FALLBACK_STATS
        if poisson_model is not None:
            lam_a, lam_b = poisson_model.predict_goals(ta, tb)
        else:
            lam_a = FALLBACK_STATS.get(ta, {"xg_pg": 1.2})["xg_pg"]
            lam_b = FALLBACK_STATS.get(tb, {"xg_pg": 1.0})["xg_pg"]
        ga = int(rng.poisson(lam_a))
        gb = int(rng.poisson(lam_b))
        table[ta]["gf"] += ga
        table[ta]["gd"] += ga - gb
        table[tb]["gf"] += gb
        table[tb]["gd"] += gb - ga
        if ga > gb:
            table[ta]["pts"] += 3
        elif ga == gb:
            table[ta]["pts"] += 1
            table[tb]["pts"] += 1
        else:
            table[tb]["pts"] += 3
    return sorted(teams, key=lambda t: (table[t]["pts"], table[t]["gd"], table[t]["gf"]), reverse=True)


class TournamentSimulator:
    def __init__(self, poisson_model=None):
        self.poisson = poisson_model

    def simulate(self, n_sims: int = MC_ITERATIONS) -> pd.DataFrame:
        """
        Simula el torneo completo n_sims veces.
        Retorna DataFrame con probabilidades por ronda por equipo.
        """
        all_teams = [t for teams in GROUPS.values() for t in teams]
        group_stage_count = {t: 0 for t in all_teams}
        r32_count = {t: 0 for t in all_teams}
        r16_count = {t: 0 for t in all_teams}
        qf_count = {t: 0 for t in all_teams}
        sf_count = {t: 0 for t in all_teams}
        finalist_count = {t: 0 for t in all_teams}
        champion_count = {t: 0 for t in all_teams}

        rng = np.random.default_rng(seed=2026)

        for _ in range(n_sims):
            # Fase de grupos
            qualified = []
            third_place_teams = []

            for group_name, teams in GROUPS.items():
                ranked = _simulate_group(teams, self.poisson, rng)
                qualified.append(ranked[0])  # 1ro
                qualified.append(ranked[1])  # 2do
                third_place_teams.append(ranked[2])  # 3ro

            # Seleccionar los 8 mejores terceros (simplificado: tomar aleatoriamente con pesos Elo)
            from src.features.elo_features import get_elo_rating
            third_elos = [get_elo_rating(t, None) for t in third_place_teams]
            probs = np.array(third_elos) / sum(third_elos)
            best_thirds_idx = rng.choice(
                len(third_place_teams),
                size=min(WC2026_BEST_THIRDS, len(third_place_teams)),
                replace=False,
                p=probs,
            )
            best_thirds = [third_place_teams[i] for i in best_thirds_idx]
            qualified += best_thirds

            for t in qualified:
                if t in group_stage_count:
                    group_stage_count[t] += 1

            # Fase eliminatoria (se empareja por posición de grupo)
            # R32 (32 equipos)
            rng.shuffle(qualified)
            for i in range(0, len(qualified) - 1, 2):
                if i + 1 < len(qualified):
                    w = _simulate_match_result(qualified[i], qualified[i+1], self.poisson, rng)
                    r32_count[w] += 1

            # Continuar con rondas subsiguientes (simplificado: bracket aleatorio balanceado)
            round_teams = qualified.copy()
            round_counts = [r16_count, qf_count, sf_count, finalist_count]
            for rc in round_counts:
                winners = []
                rng.shuffle(round_teams)
                for i in range(0, len(round_teams) - 1, 2):
                    w = _simulate_match_result(round_teams[i], round_teams[i+1], self.poisson, rng)
                    winners.append(w)
                    rc[w] += 1
                round_teams = winners
                if len(round_teams) <= 1:
                    break

            if round_teams:
                champion_count[round_teams[0]] += 1

        # Construir resultados
        records = []
        for team in all_teams:
            records.append({
                "team": team,
                "p_group_stage": round(group_stage_count[team] / n_sims, 4),
                "p_round_32": round(r32_count[team] / n_sims, 4),
                "p_round_16": round(r16_count[team] / n_sims, 4),
                "p_quarterfinal": round(qf_count[team] / n_sims, 4),
                "p_semifinal": round(sf_count[team] / n_sims, 4),
                "p_finalist": round(finalist_count[team] / n_sims, 4),
                "p_champion": round(champion_count[team] / n_sims, 4),
            })

        df = pd.DataFrame(records).sort_values("p_champion", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        logger.info(f"Simulación completada: {n_sims:,} iteraciones")
        return df
