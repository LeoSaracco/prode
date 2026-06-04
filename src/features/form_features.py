"""Features de forma reciente y momentum."""
import numpy as np
import pandas as pd

from config.feature_weights import MOMENTUM_DECAY


def compute_form_index(match_results: pd.DataFrame, team: str, n: int = 5) -> float:
    """
    Calcula el índice de forma de los últimos n partidos.
    Retorna valor normalizado en [0, 1]. Win=3pts, Draw=1pt, Loss=0pts.
    """
    if match_results.empty:
        return 0.5  # valor neutro cuando no hay datos
    df = match_results[
        (match_results["team"] == team)
    ].sort_values("date", ascending=False).head(n)
    if df.empty:
        return 0.5
    points = df["points"].sum() if "points" in df.columns else 0
    max_points = n * 3
    return min(points / max_points, 1.0)


def compute_momentum_score(match_results: pd.DataFrame, team: str, n: int = 10) -> float:
    """
    Momentum ponderado exponencialmente. Partidos recientes pesan más.
    decay_factor^0 (más reciente) ... decay_factor^(n-1) (más antiguo).
    """
    if match_results.empty:
        return 0.5
    df = match_results[
        match_results["team"] == team
    ].sort_values("date", ascending=False).head(n)
    if df.empty:
        return 0.5

    total_weight = 0.0
    weighted_points = 0.0
    for i, (_, row) in enumerate(df.iterrows()):
        weight = MOMENTUM_DECAY ** i
        pts = row.get("points", 1.0)
        weighted_points += weight * pts
        total_weight += weight * 3  # máximo posible

    return min(weighted_points / total_weight, 1.0) if total_weight > 0 else 0.5


def build_match_results_from_schedule(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma el schedule de FBref en un DataFrame de resultados por equipo.
    Columnas requeridas: date, home_team, away_team, home_goals, away_goals.
    """
    if schedule_df.empty:
        return pd.DataFrame()
    records = []
    col_map = {
        "home": ("home_team", "away_team", "home_goals", "away_goals"),
        "away": ("away_team", "home_team", "away_goals", "home_goals"),
    }
    for _, row in schedule_df.iterrows():
        for side, (team_col, opp_col, gf_col, ga_col) in col_map.items():
            team = row.get(team_col) or row.get("home") or row.get("away")
            if team is None:
                continue
            gf = row.get(gf_col, np.nan)
            ga = row.get(ga_col, np.nan)
            if pd.isna(gf) or pd.isna(ga):
                continue
            if gf > ga:
                result, points = "W", 3
            elif gf == ga:
                result, points = "D", 1
            else:
                result, points = "L", 0
            records.append({
                "team": team,
                "date": row.get("date"),
                "opponent": row.get(opp_col),
                "goals_for": gf,
                "goals_against": ga,
                "result": result,
                "points": points,
                "is_home": side == "home",
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
