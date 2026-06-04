# Pesos para la composición de features compuestas
# Todos los pesos dentro de un grupo deben sumar 1.0

OFFENSIVE_POWER_WEIGHTS = {
    "xg_per_game": 0.40,
    "shots_on_target_per_game": 0.25,
    "conversion_rate": 0.20,
    "big_chances_created": 0.15,
}

DEFENSIVE_STABILITY_WEIGHTS = {
    "xga_per_game": 0.40,        # invertido: menor = mejor
    "ppda": 0.25,                 # invertido: menor = más presión
    "big_chances_conceded": 0.20, # invertido
    "clean_sheet_pct": 0.15,
}

SQUAD_DEPTH_WEIGHTS = {
    "avg_overall_rating": 0.40,
    "depth_falloff": 0.30,        # rating 12vo vs 1ero
    "experience_factor": 0.20,
    "market_value_normalized": 0.10,
}

# Peso de la diferencia de Elo en la probabilidad final del ensemble
ELO_WIN_PROB_DIVISOR = 400  # Factor K estándar de Elo

# Factor de decaimiento para momentum (matches más recientes pesan más)
MOMENTUM_DECAY = 0.85

# Peso del historial de Mundiales en big_match_rating
WC_HISTORY_WEIGHT = 0.30
RECENT_FORM_WEIGHT = 0.70
