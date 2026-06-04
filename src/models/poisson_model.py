"""
Modelo Dixon-Coles Bivariate Poisson para predicción de marcadores.
Retorna distribución de probabilidad sobre todos los posibles marcadores.
"""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize

from config.settings import MODELS_DIR, POISSON_MAX_GOALS
from src.data.national_team_proxy import FALLBACK_STATS

logger = logging.getLogger(__name__)


def _dixon_coles_tau(goals_h: int, goals_a: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    """Corrección Dixon-Coles para marcadores bajos (0-0, 1-0, 0-1, 1-1)."""
    if goals_h == 0 and goals_a == 0:
        return 1 - lambda_h * lambda_a * rho
    elif goals_h == 0 and goals_a == 1:
        return 1 + lambda_h * rho
    elif goals_h == 1 and goals_a == 0:
        return 1 + lambda_a * rho
    elif goals_h == 1 and goals_a == 1:
        return 1 - rho
    return 1.0


class PoissonGoalModel:
    """
    Modelo Poisson con corrección Dixon-Coles.
    Parámetros: attack_a, defense_a, attack_b, defense_b, rho (correlación).
    """

    def __init__(self):
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.rho_: float = -0.1
        self.home_advantage_: float = 0.25
        self.is_fitted_ = False

    def fit(self, match_df: pd.DataFrame) -> "PoissonGoalModel":
        """
        Ajusta el modelo a los partidos históricos.
        match_df: columnas team_a, team_b, goals_a, goals_b
        """
        if match_df.empty or len(match_df) < 10:
            logger.warning("Poisson: datos insuficientes, usando parámetros por defecto")
            self._set_default_params(match_df)
            return self

        if os.getenv("POISSON_OPTIMIZE", "0") != "1":
            return self._fit_empirical_rates(match_df)

        teams = list(set(match_df["team_a"].tolist() + match_df["team_b"].tolist()))
        team_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        # Inicializar: attack=0, defense=0 para todos
        x0 = np.zeros(2 * n + 2)  # attack, defense, rho, home_adv

        def neg_log_likelihood(params):
            attack = params[:n]
            defense = params[n:2*n]
            rho = params[-2]
            home_adv = params[-1]
            ll = 0.0
            for _, row in match_df.iterrows():
                ia = team_idx.get(row["team_a"])
                ib = team_idx.get(row["team_b"])
                if ia is None or ib is None:
                    continue
                neutral = bool(row.get("neutral", False))
                match_home_adv = 0.0 if neutral else home_adv
                lam_a = np.exp(attack[ia] - defense[ib] + match_home_adv)
                lam_b = np.exp(attack[ib] - defense[ia])
                ga = int(row["goals_a"])
                gb = int(row["goals_b"])
                tau = _dixon_coles_tau(ga, gb, lam_a, lam_b, rho)
                if tau <= 0:
                    continue
                ll += (np.log(tau) +
                       poisson.logpmf(ga, lam_a) +
                       poisson.logpmf(gb, lam_b))
            return -ll

        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                           options={"maxiter": 200, "ftol": 1e-4})

        if not result.success:
            logger.warning("Poisson fit did not converge (%s). Using default params.", result.message)
            self._set_default_params(pd.DataFrame())
            return self

        params = result.x
        self.attack_ = {t: params[team_idx[t]] for t in teams}
        self.defense_ = {t: params[n + team_idx[t]] for t in teams}
        self.rho_ = params[-2]
        self.home_advantage_ = params[-1]
        self.is_fitted_ = True
        logger.info(f"Poisson ajustado: {n} equipos, rho={self.rho_:.3f}")
        return self

    def _fit_empirical_rates(self, match_df: pd.DataFrame) -> "PoissonGoalModel":
        """Fast attack/defense fit from team scoring and conceding rates."""
        rows = []
        for _, row in match_df.iterrows():
            rows.append({
                "team": row["team_a"],
                "goals_for": float(row["goals_a"]),
                "goals_against": float(row["goals_b"]),
            })
            rows.append({
                "team": row["team_b"],
                "goals_for": float(row["goals_b"]),
                "goals_against": float(row["goals_a"]),
            })

        team_df = pd.DataFrame(rows)
        global_gf = max(float(team_df["goals_for"].mean()), 0.25)
        grouped = team_df.groupby("team").agg(
            goals_for=("goals_for", "mean"),
            goals_against=("goals_against", "mean"),
            matches=("goals_for", "size"),
        )

        prior_weight = 8.0
        for team, row in grouped.iterrows():
            n_matches = float(row["matches"])
            gf = (float(row["goals_for"]) * n_matches + global_gf * prior_weight) / (n_matches + prior_weight)
            ga = (float(row["goals_against"]) * n_matches + global_gf * prior_weight) / (n_matches + prior_weight)
            self.attack_[team] = float(np.log(max(gf / global_gf, 0.25)))
            self.defense_[team] = float(np.log(max(global_gf / ga, 0.25)))

        low_score = match_df[(match_df["goals_a"] <= 1) & (match_df["goals_b"] <= 1)]
        draw_rate = float((low_score["goals_a"] == low_score["goals_b"]).mean()) if len(low_score) else 0.35
        self.rho_ = float(np.clip(0.35 - draw_rate, -0.2, 0.2))
        self.home_advantage_ = 0.20
        self.is_fitted_ = True
        logger.info("Poisson empirico ajustado: %d equipos, rho=%.3f", len(grouped), self.rho_)
        return self

    def _set_default_params(self, match_df: pd.DataFrame) -> None:
        """Parámetros por defecto basados en FALLBACK_STATS."""
        for team, stats in FALLBACK_STATS.items():
            self.attack_[team] = np.log(stats["xg_pg"])
            self.defense_[team] = np.log(stats["xga_pg"])
        self.rho_ = -0.1
        self.home_advantage_ = 0.20
        self.is_fitted_ = True

    def _get_lambdas(
        self,
        team_a: str,
        team_b: str,
        neutral_venue: bool = True,
    ) -> tuple[float, float]:
        """Calcula tasas de gol esperadas para A y B."""
        att_a = self.attack_.get(team_a, np.log(FALLBACK_STATS.get(team_a, {"xg_pg": 1.2})["xg_pg"]))
        def_a = self.defense_.get(team_a, np.log(FALLBACK_STATS.get(team_a, {"xga_pg": 1.1})["xga_pg"]))
        att_b = self.attack_.get(team_b, np.log(FALLBACK_STATS.get(team_b, {"xg_pg": 1.2})["xg_pg"]))
        def_b = self.defense_.get(team_b, np.log(FALLBACK_STATS.get(team_b, {"xga_pg": 1.1})["xga_pg"]))
        home_adv = 0.0 if neutral_venue else self.home_advantage_
        lambda_a = np.exp(att_a - def_b + home_adv)
        lambda_b = np.exp(att_b - def_a)
        return float(lambda_a), float(lambda_b)

    def predict_score_matrix(
        self, team_a: str, team_b: str, neutral_venue: bool = True
    ) -> np.ndarray:
        """
        Retorna matriz (POISSON_MAX_GOALS+1) x (POISSON_MAX_GOALS+1) de probabilidades
        donde matrix[i][j] = P(A scores i, B scores j).
        """
        lam_a, lam_b = self._get_lambdas(team_a, team_b, neutral_venue)
        max_g = POISSON_MAX_GOALS
        matrix = np.zeros((max_g + 1, max_g + 1))
        for i in range(max_g + 1):
            for j in range(max_g + 1):
                tau = _dixon_coles_tau(i, j, lam_a, lam_b, self.rho_)
                matrix[i][j] = tau * poisson.pmf(i, lam_a) * poisson.pmf(j, lam_b)
        # Normalizar
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix

    def predict_goals(self, team_a: str, team_b: str, neutral_venue: bool = True) -> tuple[float, float]:
        """Retorna (xG_A, xG_B) esperados."""
        return self._get_lambdas(team_a, team_b, neutral_venue)

    def predict_outcome_probs(
        self, team_a: str, team_b: str, neutral_venue: bool = True
    ) -> tuple[float, float, float]:
        """Retorna (p_win_a, p_draw, p_win_b) a partir de la matriz de marcadores."""
        matrix = self.predict_score_matrix(team_a, team_b, neutral_venue)
        p_win_a = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_win_b = np.triu(matrix, 1).sum()
        return float(p_win_a), float(p_draw), float(p_win_b)

    def get_top_scorelines(self, team_a: str, team_b: str, n: int = 5) -> list[tuple[int, int, float]]:
        """Retorna los n marcadores más probables como lista de (goles_a, goles_b, prob)."""
        matrix = self.predict_score_matrix(team_a, team_b)
        candidates = []
        for i in range(POISSON_MAX_GOALS + 1):
            for j in range(POISSON_MAX_GOALS + 1):
                candidates.append((i, j, float(matrix[i][j])))
        return sorted(candidates, key=lambda x: x[2], reverse=True)[:n]

    def save(self, path: Path | None = None) -> None:
        import joblib
        p = path or MODELS_DIR / "poisson_model.pkl"
        joblib.dump({"attack": self.attack_, "defense": self.defense_,
                     "rho": self.rho_, "home_adv": self.home_advantage_}, p)
        logger.info(f"Poisson model guardado en {p}")

    def load(self, path: Path | None = None) -> "PoissonGoalModel":
        import joblib
        p = path or MODELS_DIR / "poisson_model.pkl"
        if not Path(p).exists():
            logger.warning("Modelo Poisson no encontrado, usando defaults")
            self._set_default_params(pd.DataFrame())
            return self
        data = joblib.load(p)
        self.attack_ = data["attack"]
        self.defense_ = data["defense"]
        self.rho_ = data["rho"]
        self.home_advantage_ = data["home_adv"]
        self.is_fitted_ = True
        return self
