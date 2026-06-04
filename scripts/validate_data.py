"""Script de validación y reporte de cobertura de datos."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cache_manager import CacheManager
from src.data.validators import report_coverage, validate_features

cache = CacheManager()

print("=== Reporte de cobertura de datos ===\n")

profiles = cache.load("team_profiles")
if profiles is not None:
    report_coverage(profiles)
    issues = validate_features(profiles)
    if issues:
        print("\nVariables con datos faltantes (>30%):")
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("✓ Sin problemas de datos faltantes")
else:
    print("⚠ team_profiles no encontrado. Ejecutar: python scripts/run_pipeline.py --fast")

elo = cache.load("elo_ratings")
if elo is not None:
    print(f"\n✓ Elo ratings: {len(elo)} selecciones")
else:
    print("\n⚠ Elo ratings no encontrados")
