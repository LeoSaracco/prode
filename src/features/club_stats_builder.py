"""
Construye estadísticas de equipos directamente desde el dataset Kaggle.
Usado para entrenar XGBoost/LightGBM con features reales de clubes.
"""
import numpy as np
import pandas as pd
from collections import defaultdict


def build_club_stats_from_history(match_df: pd.DataFrame) -> dict[str, dict]:
    """
    Calcula stats por equipo desde el historial de partidos.
    Retorna dict: team_name -> {xg_pg, xga_pg, form_5, elo_approx, ...}
    """
    stats: dict[str, dict] = defaultdict(lambda: {
        "goals_for": [], "goals_against": [], "results": [], "elo": 1600.0
    })

    for _, row in match_df.sort_values("date").iterrows():
        ht = row.get("home_team") or row.get("team_a")
        at = row.get("away_team") or row.get("team_b")
        hg = row.get("home_team_goal") or row.get("goals_a", np.nan)
        ag = row.get("away_team_goal") or row.get("goals_b", np.nan)
        if pd.isna(hg) or pd.isna(ag):
            continue
        hg, ag = float(hg), float(ag)

        stats[ht]["goals_for"].append(hg)
        stats[ht]["goals_against"].append(ag)
        stats[at]["goals_for"].append(ag)
        stats[at]["goals_against"].append(hg)

        if hg > ag:
            stats[ht]["results"].append(3)
            stats[at]["results"].append(0)
        elif hg == ag:
            stats[ht]["results"].append(1)
            stats[at]["results"].append(1)
        else:
            stats[ht]["results"].append(0)
            stats[at]["results"].append(3)

    # Calcular Elo aproximado a partir del récord de victorias
    for team, s in stats.items():
        n = len(s["results"])
        if n > 0:
            win_rate = s["results"].count(3) / n
            # Mapear win_rate a rango Elo [1500, 2100]
            s["elo"] = 1500 + win_rate * 600

    # Construir perfil final
    profiles: dict[str, dict] = {}
    for team, s in stats.items():
        n_goals = len(s["goals_for"])
        if n_goals == 0:
            continue
        gf = s["goals_for"]
        ga = s["goals_against"]
        results = s["results"]

        form_5 = sum(results[-5:]) / 15 if len(results) >= 5 else sum(results) / max(len(results) * 3, 1)
        momentum_weights = [0.85 ** i for i in range(min(10, len(results)))]
        recent = list(reversed(results[-10:])) if len(results) >= 1 else [1]
        momentum = sum(r * w for r, w in zip(recent, momentum_weights))
        momentum_max = sum(3 * w for w in momentum_weights)
        momentum_norm = momentum / momentum_max if momentum_max > 0 else 0.5

        profiles[team] = {
            "xg_pg": float(np.mean(gf)),
            "xga_pg": float(np.mean(ga)),
            "form_5": float(form_5),
            "ppda": float(max(8.0, 14.0 - (float(np.mean(gf)) - 1.5) * 2)),
            "momentum": float(momentum_norm),
            "elo": float(s["elo"]),
            "games": n_goals,
        }

    return profiles


class KaggleFeatureBuilder:
    """
    Feature builder adaptado para usar estadísticas de clubes del Kaggle.
    Reemplaza FALLBACK_STATS para el entrenamiento.
    """

    def __init__(self, club_stats: dict[str, dict], elo_df: pd.DataFrame | None = None):
        self.club_stats = club_stats
        self.elo_df = elo_df

    def get_team_stats(self, team: str) -> dict:
        if team in self.club_stats:
            return self.club_stats[team]
        return {"xg_pg": 1.2, "xga_pg": 1.1, "form_5": 0.50, "ppda": 13.0, "elo": 1600.0}

    def build_match_features(self, team_a: str, team_b: str) -> np.ndarray:
        from src.features.attack_features import compute_offensive_power
        from src.features.defense_features import compute_defensive_stability
        from src.features.elo_features import compute_elo_win_probability

        sa = self.get_team_stats(team_a)
        sb = self.get_team_stats(team_b)

        elo_a = sa["elo"]
        elo_b = sb["elo"]
        elo_diff = elo_a - elo_b
        elo_win_prob = compute_elo_win_probability(elo_a, elo_b)

        op_a = compute_offensive_power(sa["xg_pg"])
        op_b = compute_offensive_power(sb["xg_pg"])
        ds_a = compute_defensive_stability(sa["xga_pg"], sa["ppda"])
        ds_b = compute_defensive_stability(sb["xga_pg"], sb["ppda"])

        form_diff = sa["form_5"] - sb["form_5"]
        momentum_a = sa.get("momentum", sa["form_5"])
        momentum_b = sb.get("momentum", sb["form_5"])

        tactical_adv = (op_a - ds_b) - (op_b - ds_a)
        form_times_elo = form_diff * (elo_diff / 400)
        attack_vs_def = op_a - ds_b

        return np.array([
            elo_diff,
            elo_win_prob,
            sa["xg_pg"] - sb["xg_pg"],
            sa["xga_pg"] - sb["xga_pg"],
            form_diff,
            op_a - op_b,
            ds_a - ds_b,
            0.0,  # squad_depth_diff (no disponible para clubes)
            0.0,  # big_match_rating_diff
            0.0,  # pressure_performance_diff
            0.0,  # consistency_diff
            0.0,  # wc_history_diff
            0.0,  # market_value_diff
            tactical_adv,
            0.0,  # h2h_advantage
            form_times_elo,
            attack_vs_def,
        ], dtype=np.float32)

    def build_training_matrix(self, match_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        label_map = {"W": 2, "D": 1, "L": 0, "H": 2, "A": 0}
        X_rows, y_rows = [], []
        for _, row in match_df.iterrows():
            team_a = row.get("team_a") or row.get("home_team")
            team_b = row.get("team_b") or row.get("away_team")
            result = row.get("result", "D")
            if team_a is None or team_b is None:
                continue
            try:
                x = self.build_match_features(str(team_a), str(team_b))
                X_rows.append(x)
                y_rows.append(label_map.get(str(result), 1))
            except Exception:
                continue
        return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)
