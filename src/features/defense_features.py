"""Features defensivas: defensive_stability."""
import numpy as np

from config.feature_weights import DEFENSIVE_STABILITY_WEIGHTS


def compute_defensive_stability(
    xga_per_game: float,
    ppda: float,
    big_chances_conceded: float = 1.5,
    clean_sheet_pct: float = 0.30,
) -> float:
    """
    Score defensivo compuesto, normalizado a [0, 1].
    Valores típicos: Argentina/Morocco ~ 0.70-0.85
    xga y ppda son mejores cuanto más bajos (se invierten).
    """
    # Invertir: menor xGA y PPDA → mejor defensa
    xga_norm = max(0.0, 1.0 - (xga_per_game / 2.5))
    ppda_norm = max(0.0, 1.0 - (ppda / 20.0))
    bc_norm = max(0.0, 1.0 - (big_chances_conceded / 4.0))
    cs_norm = min(clean_sheet_pct / 0.5, 1.0)

    w = DEFENSIVE_STABILITY_WEIGHTS
    return (
        w["xga_per_game"] * xga_norm +
        w["ppda"] * ppda_norm +
        w["big_chances_conceded"] * bc_norm +
        w["clean_sheet_pct"] * cs_norm
    )
