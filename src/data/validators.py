"""Validación y reporte de cobertura de datos."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def report_coverage(profiles: pd.DataFrame) -> None:
    """Imprime un reporte de calidad de datos por selección."""
    if profiles.empty:
        logger.warning("No hay perfiles de selecciones para reportar")
        return
    total = len(profiles)
    high = (profiles["data_quality"] == "HIGH").sum()
    medium = (profiles["data_quality"] == "MEDIUM").sum()
    low = (profiles["data_quality"] == "LOW").sum()
    logger.info(
        f"Cobertura de datos: {total} equipos | "
        f"HIGH={high} ({high/total:.0%}) | "
        f"MEDIUM={medium} ({medium/total:.0%}) | "
        f"LOW={low} ({low/total:.0%})"
    )
    if low > 0:
        low_teams = profiles.loc[profiles["data_quality"] == "LOW", "team"].tolist()
        logger.warning(f"Equipos con datos limitados: {low_teams}")


def validate_features(features: pd.DataFrame) -> list[str]:
    """Retorna lista de columnas con NaN excesivos (>30%)."""
    issues = []
    for col in features.columns:
        if col == "team":
            continue
        pct_null = features[col].isna().mean()
        if pct_null > 0.30:
            issues.append(f"{col}: {pct_null:.0%} NaN")
    return issues
