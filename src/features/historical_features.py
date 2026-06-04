"""Features históricas: H2H y rendimiento en Mundiales."""
import pandas as pd

from config.wc2026_groups import WC_HISTORY_SCORE


def compute_world_cup_history_score(team: str) -> float:
    return WC_HISTORY_SCORE.get(team, 0.10)


def compute_h2h_record(
    h2h_data: pd.DataFrame,
    team_a: str,
    team_b: str,
) -> dict:
    """
    Retorna dict con wins_a, draws, wins_b, goals_a, goals_b.
    Si hay menos de 3 encuentros, retorna valores neutros.
    """
    default = {"wins_a": 0, "draws": 0, "wins_b": 0,
               "goals_a": 0.0, "goals_b": 0.0, "matches": 0}
    if h2h_data.empty:
        return default

    mask = (
        ((h2h_data["team_a"] == team_a) & (h2h_data["team_b"] == team_b)) |
        ((h2h_data["team_a"] == team_b) & (h2h_data["team_b"] == team_a))
    )
    df = h2h_data[mask]
    if len(df) < 3:
        return default

    wins_a = wins_b = draws = 0
    goals_a = goals_b = 0.0
    for _, row in df.iterrows():
        if row["team_a"] == team_a:
            ga, gb = row["goals_a"], row["goals_b"]
        else:
            ga, gb = row["goals_b"], row["goals_a"]
        goals_a += ga
        goals_b += gb
        if ga > gb:
            wins_a += 1
        elif ga < gb:
            wins_b += 1
        else:
            draws += 1

    return {
        "wins_a": wins_a, "draws": draws, "wins_b": wins_b,
        "goals_a": goals_a / len(df), "goals_b": goals_b / len(df),
        "matches": len(df),
    }


def h2h_advantage_score(h2h: dict, team_a: str) -> float:
    """Score H2H normalizado [-1, 1]. Positivo = ventaja team_a."""
    total = h2h["wins_a"] + h2h["draws"] + h2h["wins_b"]
    if total == 0:
        return 0.0
    return (h2h["wins_a"] - h2h["wins_b"]) / total
