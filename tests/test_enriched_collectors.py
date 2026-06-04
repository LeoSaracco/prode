import pandas as pd

from src.data.collectors.wc2026_collectors import (
    _canonical_team,
    _parse_eur_millions,
    _parse_percent,
)
from src.models.trainer import RollingNationalTeamFeatureBuilder


def test_transfermarkt_numeric_parsers():
    assert _parse_eur_millions("1,55 mil mill. EUR") == 1550.0
    assert _parse_eur_millions("799,50 mill. EUR") == 799.5
    assert _parse_eur_millions("991 mil EUR") == 0.991
    assert _parse_percent("92,3 %") == 0.923


def test_source_team_names_normalize_to_canonical_wc2026_names():
    assert _canonical_team("Países Bajos") == "Netherlands"
    assert _canonical_team("Costa de Marfil") == "Cote d'Ivoire"
    assert _canonical_team("Estados Unidos") == "United States"
    assert _canonical_team("Turquía") == "Turkiye"


def test_rolling_builder_accepts_initial_elo_without_future_match_results():
    initial_elo = pd.DataFrame([
        {"team": "Alpha", "elo_rating": 1800.0, "date": pd.Timestamp("2023-01-01")},
        {"team": "Beta", "elo_rating": 1500.0, "date": pd.Timestamp("2023-01-01")},
    ])
    row = pd.Series({
        "date": pd.Timestamp("2024-01-01"),
        "team_a": "Alpha",
        "team_b": "Beta",
        "goals_a": 0,
        "goals_b": 9,
        "result": "L",
        "neutral": True,
        "tournament": "Friendly",
    })

    builder = RollingNationalTeamFeatureBuilder(initial_elo_df=initial_elo)
    features_before_update = builder.build_match_features(row)

    assert features_before_update[0] == 300.0
