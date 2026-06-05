"""
Simulador completo del torneo FIFA WC2026 (48 equipos, bracket oficial).

Estructura:
  - 12 grupos de 4 equipos → top 2 + 8 mejores terceros = 32 equipos
  - Round of 32: 12 partidos de pod (ganador vs runner-up del grupo par)
                + 4 partidos de mejores terceros entre sí
  - R16, QF, SF, Final: árbol de bracket fijo (sin barajar entre rondas)
  - Mejores terceros seleccionados por pts > gd > gf, no por Elo
"""
import logging
from itertools import combinations

import numpy as np
import pandas as pd

from config.settings import MC_ITERATIONS, WC2026_BEST_THIRDS
from config.wc2026_groups import GROUPS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bracket oficial WC2026 - Round of 32
# Pods: (A,B), (C,D), (E,F), (G,H), (I,J), (K,L)
# Índices 0-15 corresponden a los 16 partidos del R32.
# ---------------------------------------------------------------------------

# Cada partido R32 como (slot_a, slot_b). "w_X" = ganador grupo X,
# "r_X" = runner-up grupo X, "t_N" = N-ésimo mejor tercero (1-indexed).
R32_SLOTS = [
    ("w_A", "r_B"),   # M00 — pod A-B
    ("w_B", "r_A"),   # M01
    ("w_C", "r_D"),   # M02 — pod C-D
    ("w_D", "r_C"),   # M03
    ("w_E", "r_F"),   # M04 — pod E-F
    ("w_F", "r_E"),   # M05
    ("w_G", "r_H"),   # M06 — pod G-H
    ("w_H", "r_G"),   # M07
    ("w_I", "r_J"),   # M08 — pod I-J
    ("w_J", "r_I"),   # M09
    ("w_K", "r_L"),   # M10 — pod K-L
    ("w_L", "r_K"),   # M11
    ("t_1", "t_2"),   # M12 — mejores terceros
    ("t_3", "t_4"),   # M13
    ("t_5", "t_6"),   # M14
    ("t_7", "t_8"),   # M15
]

# Árbol del bracket: cada ronda recibe los índices de partidos anteriores
# cuyos ganadores se cruzan.
# R16[i] = (idx_r32_a, idx_r32_b) → ganador de R32[idx_a] vs ganador de R32[idx_b]
R16_PAIRS = [
    (0, 12),   # WinA-vs-RupB vs 3ro1-vs-3ro2
    (1, 13),   # WinB-vs-RupA vs 3ro3-vs-3ro4
    (2,  3),   # C-D pod
    (4,  5),   # E-F pod
    (6, 14),   # WinG-vs-RupH vs 3ro5-vs-3ro6
    (7, 15),   # WinH-vs-RupG vs 3ro7-vs-3ro8
    (8,  9),   # I-J pod
    (10, 11),  # K-L pod
]

QF_PAIRS  = [(0, 1), (2, 3), (4, 5), (6, 7)]
SF_PAIRS  = [(0, 1), (2, 3)]
FIN_PAIRS = [(0, 1)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_ko_match(
    team_a: str,
    team_b: str,
    poisson_model,
    predictor,
    rng: np.random.Generator,
) -> str:
    """Partido eliminatorio sin empate (penales si hay igualdad en 90')."""
    from src.data.national_team_proxy import FALLBACK_STATS
    lam_a, lam_b = 1.2, 1.0
    if poisson_model is not None:
        try:
            lam_a, lam_b = poisson_model.predict_goals(team_a, team_b)
        except Exception:
            pass
    elif predictor is not None:
        try:
            r = predictor(team_a, team_b)
            lam_a = float(r.get("xg_a", 1.2))
            lam_b = float(r.get("xg_b", 1.0))
        except Exception:
            pass
    else:
        lam_a = FALLBACK_STATS.get(team_a, {"xg_pg": 1.2})["xg_pg"]
        lam_b = FALLBACK_STATS.get(team_b, {"xg_pg": 1.0})["xg_pg"]

    ga = int(rng.poisson(lam_a))
    gb = int(rng.poisson(lam_b))
    if ga > gb:
        return team_a
    if gb > ga:
        return team_b
    # Penales: Elo-weighted
    from src.features.elo_features import compute_elo_win_probability, get_elo_rating
    elo_a = get_elo_rating(team_a, None)
    elo_b = get_elo_rating(team_b, None)
    p = compute_elo_win_probability(elo_a, elo_b)
    return team_a if rng.random() < p else team_b


def _simulate_group(
    teams: list[str],
    poisson_model,
    predictor,
    rng: np.random.Generator,
) -> tuple[list[str], dict[str, dict]]:
    """Simula grupo. Retorna (ranking [1ro..4to], tabla con pts/gd/gf/ga)."""
    from src.data.national_team_proxy import FALLBACK_STATS

    table = {t: {"pts": 0, "gd": 0, "gf": 0, "ga": 0} for t in teams}

    for ta, tb in combinations(teams, 2):
        lam_a, lam_b = 1.2, 1.0
        if predictor is not None:
            try:
                r = predictor(ta, tb)
                probs = [r.get("p_win_b", 0.35), r.get("p_draw", 0.25), r.get("p_win_a", 0.4)]
                total = sum(probs)
                probs = [p / total for p in probs]
                outcome = int(rng.choice(3, p=probs))  # 0=B wins, 1=draw, 2=A wins
                lam_a = float(r.get("xg_a", 1.2))
                lam_b = float(r.get("xg_b", 1.0))
                ga = int(rng.poisson(lam_a))
                gb = int(rng.poisson(lam_b))
                # Forzar resultado consistente con outcome predicho
                if outcome == 2 and ga <= gb:
                    ga, gb = max(1, ga), max(0, ga - 1)
                elif outcome == 0 and gb <= ga:
                    ga, gb = max(0, gb - 1), max(1, gb)
                elif outcome == 1:
                    s = max(0, (ga + gb) // 2)
                    ga, gb = s, s
            except Exception:
                ga = int(rng.poisson(lam_a))
                gb = int(rng.poisson(lam_b))
        elif poisson_model is not None:
            try:
                lam_a, lam_b = poisson_model.predict_goals(ta, tb)
            except Exception:
                pass
            ga = int(rng.poisson(lam_a))
            gb = int(rng.poisson(lam_b))
        else:
            lam_a = FALLBACK_STATS.get(ta, {"xg_pg": 1.2})["xg_pg"]
            lam_b = FALLBACK_STATS.get(tb, {"xg_pg": 1.0})["xg_pg"]
            ga = int(rng.poisson(lam_a))
            gb = int(rng.poisson(lam_b))

        table[ta]["gf"] += ga; table[ta]["ga"] += gb; table[ta]["gd"] += ga - gb
        table[tb]["gf"] += gb; table[tb]["ga"] += ga; table[tb]["gd"] += gb - ga
        if ga > gb:
            table[ta]["pts"] += 3
        elif ga == gb:
            table[ta]["pts"] += 1; table[tb]["pts"] += 1
        else:
            table[tb]["pts"] += 3

    ranked = sorted(teams, key=lambda t: (table[t]["pts"], table[t]["gd"], table[t]["gf"]), reverse=True)
    return ranked, table


def _select_best_thirds(
    thirds: list[tuple[str, int, int, int]]  # (team, pts, gd, gf)
) -> list[str]:
    """Ordena 12 terceros por pts > gd > gf y devuelve los 8 mejores."""
    sorted_thirds = sorted(thirds, key=lambda x: (x[1], x[2], x[3]), reverse=True)
    return [t[0] for t in sorted_thirds[:WC2026_BEST_THIRDS]]


def _resolve_slot(slot: str, group_results: dict[str, tuple]) -> str:
    """Resuelve un slot de bracket a nombre de equipo.
    group_results: {'A': (winner, runner_up, third), ...}
    thirds_ranked: lista de equipos terceros ordenados por performance
    """
    raise NotImplementedError("Use _fill_r32 instead")


def _simulate_round(
    matches: list[tuple[str, str]],
    poisson_model,
    predictor,
    rng: np.random.Generator,
    count_dict: dict[str, int],
) -> list[str]:
    """Simula una ronda eliminatoria. Retorna los ganadores."""
    winners = []
    for ta, tb in matches:
        w = _simulate_ko_match(ta, tb, poisson_model, predictor, rng)
        count_dict[w] = count_dict.get(w, 0) + 1
        winners.append(w)
    return winners


# ---------------------------------------------------------------------------
# Simulador principal
# ---------------------------------------------------------------------------

class TournamentSimulator:
    def __init__(self, poisson_model=None, group_simulator=None, predictor=None):
        self.poisson = poisson_model
        self.group_simulator = group_simulator
        self.predictor = predictor

    def simulate(self, n_sims: int = MC_ITERATIONS) -> pd.DataFrame:
        """
        Simula el torneo completo n_sims veces con el bracket oficial WC2026.
        Retorna DataFrame con probabilidades por ronda por equipo.
        """
        all_teams = [t for teams in GROUPS.values() for t in teams]
        counts: dict[str, dict[str, int]] = {
            "gs":  {t: 0 for t in all_teams},
            "r32": {t: 0 for t in all_teams},
            "r16": {t: 0 for t in all_teams},
            "qf":  {t: 0 for t in all_teams},
            "sf":  {t: 0 for t in all_teams},
            "fin": {t: 0 for t in all_teams},
            "champ": {t: 0 for t in all_teams},
        }

        rng = np.random.default_rng()
        group_keys = list(GROUPS.keys())  # A, B, ..., L

        for _ in range(n_sims):
            # ── Fase de grupos ──────────────────────────────────────────────
            winners:  dict[str, str] = {}
            runners:  dict[str, str] = {}
            thirds_data: list[tuple[str, int, int, int]] = []  # (team, pts, gd, gf)

            for g in group_keys:
                teams = GROUPS[g]
                ranked, table = _simulate_group(teams, self.poisson, self.predictor, rng)
                winners[g] = ranked[0]
                runners[g] = ranked[1]
                third = ranked[2]
                thirds_data.append((third, table[third]["pts"], table[third]["gd"], table[third]["gf"]))

            # 8 mejores terceros por rendimiento (pts > gd > gf)
            best_thirds = _select_best_thirds(thirds_data)

            qualified = list(winners.values()) + list(runners.values()) + best_thirds
            for t in qualified:
                counts["gs"][t] = counts["gs"].get(t, 0) + 1

            # ── Resolver slots R32 ─────────────────────────────────────────
            slot_map: dict[str, str] = {}
            for g in group_keys:
                slot_map[f"w_{g}"] = winners[g]
                slot_map[f"r_{g}"] = runners[g]
            for i, t in enumerate(best_thirds, start=1):
                slot_map[f"t_{i}"] = t

            r32_matches = [(slot_map[a], slot_map[b]) for a, b in R32_SLOTS]

            # Convención: counts[X] = "llegó a la ronda X" (participantes, no ganadores)
            # R32: los 32 clasificados ya están en counts["gs"]

            # ── Round of 32 ───────────────────────────────────────────────
            r32_winners = _simulate_round(r32_matches, self.poisson, self.predictor, rng, {})
            # "reached R16" = ganadores del R32 (16 equipos)
            for t in r32_winners:
                counts["r16"][t] = counts["r16"].get(t, 0) + 1

            # ── Round of 16 ───────────────────────────────────────────────
            r16_matches = [(r32_winners[a], r32_winners[b]) for a, b in R16_PAIRS]
            r16_winners = _simulate_round(r16_matches, self.poisson, self.predictor, rng, {})
            # "reached QF" = ganadores del R16 (8 equipos)
            for t in r16_winners:
                counts["qf"][t] = counts["qf"].get(t, 0) + 1

            # ── Cuartos ───────────────────────────────────────────────────
            qf_matches = [(r16_winners[a], r16_winners[b]) for a, b in QF_PAIRS]
            qf_winners = _simulate_round(qf_matches, self.poisson, self.predictor, rng, {})
            # "reached SF" = ganadores de QF (4 equipos)
            for t in qf_winners:
                counts["sf"][t] = counts["sf"].get(t, 0) + 1

            # ── Semifinales ───────────────────────────────────────────────
            sf_matches = [(qf_winners[a], qf_winners[b]) for a, b in SF_PAIRS]
            sf_winners = _simulate_round(sf_matches, self.poisson, self.predictor, rng, {})
            # "reached Final" = ganadores de SF (2 equipos)
            for t in sf_winners:
                counts["fin"][t] = counts["fin"].get(t, 0) + 1

            # ── Final ─────────────────────────────────────────────────────
            fin_matches = [(sf_winners[a], sf_winners[b]) for a, b in FIN_PAIRS]
            fin_winners = _simulate_round(fin_matches, self.poisson, self.predictor, rng, {})

            if fin_winners:
                champion = fin_winners[0]
                counts["champ"][champion] = counts["champ"].get(champion, 0) + 1

        # ── Construir DataFrame ───────────────────────────────────────────
        records = []
        for team in all_teams:
            records.append({
                "team": team,
                "p_group_stage":  round(counts["gs"].get(team, 0)    / n_sims, 4),
                "p_round_32":     round(counts["r32"].get(team, 0)   / n_sims, 4),
                "p_round_16":     round(counts["r16"].get(team, 0)   / n_sims, 4),
                "p_quarterfinal": round(counts["qf"].get(team, 0)    / n_sims, 4),
                "p_semifinal":    round(counts["sf"].get(team, 0)    / n_sims, 4),
                "p_finalist":     round(counts["fin"].get(team, 0)   / n_sims, 4),
                "p_champion":     round(counts["champ"].get(team, 0) / n_sims, 4),
            })

        df = (
            pd.DataFrame(records)
            .sort_values("p_champion", ascending=False)
            .reset_index(drop=True)
        )
        df["rank"] = df.index + 1
        logger.info("Simulación completada: %d iteraciones", n_sims)
        return df
