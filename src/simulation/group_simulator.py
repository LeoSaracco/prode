"""
Simulador de fase de grupos del Mundial 2026.
12 grupos de 4 equipos. 2 primeros + 8 mejores terceros clasifican.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from math import factorial
from config.settings import MC_ITERATIONS
from config.wc2026_groups import GROUPS
from src.simulation.match_simulator import simulate_match


def _simulate_single_group(
    teams: list[str],
    match_models: dict[tuple[str, str], dict],
    rng: np.random.Generator,
) -> dict[str, dict]:
    """Simula los 6 partidos de un grupo. Retorna tabla con pts, gd, gf."""
    table = {t: {"pts": 0, "gd": 0, "gf": 0, "ga": 0} for t in teams}
    for team_a, team_b in combinations(teams, 2):
        model = match_models.get((team_a, team_b), {})
        lam_a, lam_b = model.get("lambdas", (1.2, 1.0))
        outcome_probs = model.get("outcome_probs")
        if outcome_probs is None:
            ga = int(rng.poisson(lam_a))
            gb = int(rng.poisson(lam_b))
        else:
            outcome = int(rng.choice(3, p=outcome_probs))
            ga, gb = _sample_scoreline_for_outcome(lam_a, lam_b, outcome, rng)
        table[team_a]["gf"] += ga
        table[team_a]["ga"] += gb
        table[team_b]["gf"] += gb
        table[team_b]["ga"] += ga
        table[team_a]["gd"] += ga - gb
        table[team_b]["gd"] += gb - ga
        if ga > gb:
            table[team_a]["pts"] += 3
        elif ga == gb:
            table[team_a]["pts"] += 1
            table[team_b]["pts"] += 1
        else:
            table[team_b]["pts"] += 3
    return table


def _sample_scoreline_for_outcome(
    lam_a: float,
    lam_b: float,
    outcome: int,
    rng: np.random.Generator,
    max_goals: int = 8,
) -> tuple[int, int]:
    """Samples a plausible scoreline conditional on A/D/B outcome."""
    candidates = []
    weights = []
    for ga in range(max_goals + 1):
        for gb in range(max_goals + 1):
            if outcome == 0 and ga <= gb:
                continue
            if outcome == 1 and ga != gb:
                continue
            if outcome == 2 and ga >= gb:
                continue
            # Recursive import avoidance; poisson pmf is cheap enough through numpy formula.
            weight = np.exp(-lam_a) * (lam_a ** ga) / max(factorial(ga), 1)
            weight *= np.exp(-lam_b) * (lam_b ** gb) / max(factorial(gb), 1)
            candidates.append((ga, gb))
            weights.append(weight)
    if not candidates or sum(weights) <= 0:
        ga = int(rng.poisson(lam_a))
        gb = int(rng.poisson(lam_b))
        return ga, gb
    probs = np.array(weights, dtype=float)
    probs /= probs.sum()
    return candidates[int(rng.choice(len(candidates), p=probs))]


def _rank_group(table: dict[str, dict]) -> list[str]:
    """Ordena equipos por pts > gd > gf (desempate FIFA simplificado)."""
    return sorted(
        table.keys(),
        key=lambda t: (table[t]["pts"], table[t]["gd"], table[t]["gf"]),
        reverse=True,
    )


class GroupSimulator:
    def __init__(self, poisson_model=None, elo_df: pd.DataFrame | None = None, predictor=None):
        self.poisson = poisson_model
        self.elo_df = elo_df
        self.predictor = predictor

    def _get_lambdas(self, team_a: str, team_b: str) -> tuple[float, float]:
        if self.poisson is not None:
            return self.poisson.predict_goals(team_a, team_b)
        from src.data.national_team_proxy import FALLBACK_STATS
        return (
            FALLBACK_STATS.get(team_a, {"xg_pg": 1.2})["xg_pg"],
            FALLBACK_STATS.get(team_b, {"xg_pg": 1.0})["xg_pg"],
        )

    def _get_outcome_probs(self, team_a: str, team_b: str) -> tuple[float, float, float] | None:
        if self.predictor is None:
            return None
        try:
            result = self.predictor(team_a, team_b)
            probs = np.array([
                float(result["p_win_a"]),
                float(result["p_draw"]),
                float(result["p_win_b"]),
            ])
            total = probs.sum()
            if total <= 0:
                return None
            probs = probs / total
            return float(probs[0]), float(probs[1]), float(probs[2])
        except Exception:
            return None

    def simulate_group(
        self,
        group_name: str,
        teams: list[str] | None = None,
        n_sims: int = MC_ITERATIONS,
    ) -> pd.DataFrame:
        """
        Simula un grupo n_sims veces.
        Retorna DataFrame con probabilidades de cada posición por equipo.
        """
        if teams is None:
            teams = GROUPS.get(group_name, [])
        if not teams:
            return pd.DataFrame()

        # Pre-calcular xG y probabilidades W/D/L para todos los emparejamientos.
        match_models = {}
        for ta, tb in combinations(teams, 2):
            lambdas = self._get_lambdas(ta, tb)
            outcome_probs = self._get_outcome_probs(ta, tb)
            match_models[(ta, tb)] = {"lambdas": lambdas, "outcome_probs": outcome_probs}
            reverse_probs = None if outcome_probs is None else (outcome_probs[2], outcome_probs[1], outcome_probs[0])
            match_models[(tb, ta)] = {"lambdas": (lambdas[1], lambdas[0]), "outcome_probs": reverse_probs}

        positions = {t: [0, 0, 0, 0] for t in teams}  # [1st, 2nd, 3rd, 4th]
        pts_accum = {t: 0 for t in teams}
        gd_accum = {t: 0 for t in teams}
        rng = np.random.default_rng(seed=2026)

        for _ in range(n_sims):
            table = _simulate_single_group(teams, match_models, rng)
            ranked = _rank_group(table)
            for pos, team in enumerate(ranked):
                positions[team][pos] += 1
                pts_accum[team] += table[team]["pts"]
                gd_accum[team] += table[team]["gd"]

        records = []
        for team in teams:
            pos = positions[team]
            qualify_prob = (pos[0] + pos[1]) / n_sims  # 1ro o 2do
            # 3ros: se calculará en TournamentSimulator con cross-group
            records.append({
                "team": team,
                "group": group_name,
                "prob_1st": round(pos[0] / n_sims, 4),
                "prob_2nd": round(pos[1] / n_sims, 4),
                "prob_3rd": round(pos[2] / n_sims, 4),
                "prob_4th": round(pos[3] / n_sims, 4),
                "qualify_direct_prob": round(qualify_prob, 4),
                "avg_pts": round(pts_accum[team] / n_sims, 2),
                "avg_gd": round(gd_accum[team] / n_sims, 2),
            })

        return pd.DataFrame(records).sort_values("qualify_direct_prob", ascending=False)

    def simulate_group_fixtures(
        self,
        group_name: str,
        teams: list[str] | None = None,
        n_sims: int = MC_ITERATIONS,
    ) -> list[dict]:
        """Retorna el marcador exacto mas probable para cada partido del grupo."""
        if teams is None:
            teams = GROUPS.get(group_name, [])
        if not teams:
            return []

        fixtures = []
        for team_a, team_b in combinations(teams, 2):
            lambda_a, lambda_b = self._get_lambdas(team_a, team_b)
            simulation = simulate_match(lambda_a, lambda_b, n_sims=n_sims)
            top_scoreline = simulation["top_scorelines"][0] if simulation["top_scorelines"] else None
            fixtures.append({
                "team_a": team_a,
                "team_b": team_b,
                "expected_goals": {
                    "team_a": simulation["xg_a"],
                    "team_b": simulation["xg_b"],
                },
                "most_likely_scoreline": {
                    "goals_a": int(top_scoreline[0]),
                    "goals_b": int(top_scoreline[1]),
                    "probability": round(float(top_scoreline[2]), 4),
                } if top_scoreline else None,
            })

        return fixtures
