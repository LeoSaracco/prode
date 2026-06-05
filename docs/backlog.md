# Backlog - prode-ML FIFA World Cup 2026

## Resumen De Fases

| Fase | Estado | Resultado | Cambio clave |
|---|---|---:|---|
| Inicial | Completado | 42.0% blend | XGB+LGBM+Elo, split con leakage. |
| A - Pipeline | Completado | - | Split cronologico, Elo computacional, 3,199 partidos. |
| B - Ensemble | Completado | CatBoost ~49% | RF+CatBoost y features refinadas. |
| C - Tuning | Completado | - | Optuna y `best_params.json`. |
| D - Auxiliares | Completado | - | TwoStage y ConfederationModels. |
| E - Performance | Completado | Voting ~49% | Weighted voting, entrenamiento paralelo, Windows mas rapido. |
| F - Validacion ML | Completado | Voting 50.21% | Features rolling sin leakage, rankings FIFA, confianza calibrada. |
| G - Calibracion y Recencia | Completado | Voting 50.62% | Temperature scaling, sample_weight por recencia, TwoStage fuera del reporte, Optuna con objetivo compuesto, Kaggle credentials. |
| H - Features, Datos y Simulacion | Completado | Voting 51.25%, ALTO 82.5% | Poisson al ensemble, squad quality desde ratings FIFA, bracket oficial WC2026, seeds variables, FALLBACK_STATS diversificados. |

## Estado Actual

- CLI interactivo: implementado.
- API REST FastAPI: implementada.
- Frontend React/Vite: implementado.
- Grupos oficiales WC2026: implementados.
- Reporte PDF: implementado.
- Automatizacion local: `run_all.bat` (con `npm install` automatico para frontend).
- Entrenamiento rolling sin leakage: implementado.
- FIFA rankings historicos: collector reparado y cacheado.
- Confidence thresholds: implementados.
- Poisson integrado al ensemble como modelo 6 (peso fijo ~8%).
- Squad quality desde ratings FIFA de jugadores (player_aggregates.csv).
- Bracket oficial WC2026 con pods predefinidos y seleccion de terceros por rendimiento.
- Marcadores Poisson-sampled (ya no deterministicos con round(xG)).
- Seeds variables en simulaciones Monte Carlo.

Artefactos relevantes:

- `models/model_metadata.json`
- `models/voting_weights.json`
- `models/confidence_thresholds.json`
- `models/inference_team_profiles.json`
- `models/inference_h2h_stats.json`
- `data/raw/kaggle/match_features/player_aggregates.csv`

## Metricas Actuales

Ultimo entrenamiento: `2026-06-05T10:46:10`.

| Metrica | Valor | Que significa |
|---|---:|---|
| `accuracy_voting` | 51.25% | Acierto general del ensemble de 6 modelos. |
| `accuracy_catboost` | 50.93% | Acierto de CatBoost solo. |
| `accuracy_lgbm` | 50.36% | Acierto de LightGBM. |
| `accuracy_rf` | 48.99% | Acierto de RandomForest. |
| `accuracy_xgb` | 47.30% | Acierto de XGBoost. |
| `accuracy_high_confidence` | **82.50%** | Acierto cuando el sistema marca `ALTO`. Target >80% alcanzado. |
| `n_high_confidence_matches` | 40 | Partidos test con confianza `ALTO`. |
| `accuracy_high_elo_diff_200` | 65.52% | Acierto cuando la diferencia Elo es >= 200 (n=290). |
| `log_loss_voting` | 1.0003 | Calidad de probabilidades; menor es mejor. |
| Partidos de entrenamiento | 8,273 | Incluye datos base + Kaggle match features. |

Notas:

- `accuracy_voting` es la metrica principal global.
- El target de >80% en alta confianza fue alcanzado por primera vez en Fase H.
- Las metricas son honestas: features rolling sin leakage, evaluacion unica en test cronologico.

## Completado En Fase H (2026-06-05)

### Mejoras Sin Reentrenamiento

- **Marcadores variables**: `_scoreline()` en `generate_report.py` usa Poisson sampling condicionado al outcome predicho. Reemplaza `round(xG)` deterministico que siempre daba 1-0/1-1.
- **Seeds variables**: eliminados `seed=42` en `match_simulator.py` y `seed=2026` en `group_simulator.py`. Cada simulacion produce resultados distintos.
- **FALLBACK_STATS diversificados**: todos los valores de xG/xGA de los 48 equipos WC2026 diferenciados. Elimina los 15+ equipos con `xg_pg=1.20` identico.
- **top_scorelines ampliado**: de n=5 a n=15 para mayor variedad en seleccion de marcador por outcome.

### Mejoras Con Reentrenamiento

- **Poisson al ensemble**: `_get_all_base_probs()` en `ensemble.py` incluye Poisson si esta entrenado. Peso fijo 0.07 (similar al Elo con 0.10). El trainer computa `poisson_val_probs` y `poisson_test_probs` desde nombres de equipos y los incluye en el blend.
- **Squad quality desde ratings FIFA**: `compute_squad_quality()` en `squad_features.py` reemplaza `compute_squad_depth_from_market_value()`. Usa avg_overall, max_overall, avg_shooting, avg_defending del CSV de Kaggle. `load_player_ratings_from_kaggle()` en `national_team_proxy.py` carga y cachea el CSV con fallback para equipos faltantes.
- **Bracket oficial WC2026**: `tournament_simulator.py` reescrito con 12 partidos de pod fijos (Ganador A vs 2do B, etc.), 4 partidos de bridge para mejores terceros, arbol de bracket fijo para R16/QF/SF/Final, y conteos correctos por ronda (32/16/8/4/2/1 equipos).
- **Mejores terceros por rendimiento**: `_select_best_thirds()` ordena por pts > gd > gf en lugar de Elo random.
- **Pesos Elo y Poisson fijos**: en `trainer.py` los pesos de Elo (0.10) y Poisson (0.07) son fijos antes de normalizacion, independientes de val accuracy.

## Pendiente (Fase I)

### Calibracion De xG Por Calidad De Rival

Prioridad: alta. Causa principal del sesgo en favor de Inglaterra.

Los xG rolling no filtran por calidad del rival. Equipos con muchas goleadas en clasificatorias faciles (ej: England 7-0 vs San Marino) inflan su xG_per_game. Propuesta: ponderar cada partido por rating Elo del rival al calcular el rolling xG.

### Optuna 50 Trials

Prioridad: media. Ya implementado con objetivo compuesto. Solo ejecutar:

```powershell
python scripts/tune_hyperparams.py --trials 50
```

### Bracket WC2026 Exacto (Tabla De Terceros FIFA)

Prioridad: media.

La tabla completa de cruces para los 8 mejores terceros segun su grupo de origen tiene 495 combinaciones posibles. La implementacion actual es una aproximacion (terceros 1-2 vs 3-4 vs 5-6 vs 7-8 entre si). Implementar la tabla oficial de la FIFA para cruces exactos de terceros.

### Datos Y Robustez

Prioridad: media.

- Valor de mercado dinamico por fecha de partido (no dato estatico).
- Pruebas de no-leakage automatizadas.
- Tests que validen que probabilidades suman 1.
- Fix `test_report_generation.py` que busca funciones renombradas (`_match_row`, `_pdf_text`, `_tournament_section`).

### Docker Y Deploy

Prioridad: baja.

- `Dockerfile.api`, `Dockerfile.frontend`, `docker-compose.yml`.

## Notas Operativas

- Consultar siempre `models/model_metadata.json` para metricas reales.
- No comparar metricas antiguas con leakage contra metricas nuevas rolling como si fueran equivalentes.
- Reentrenar despues de cambios en features, collector de datos o pesos.
- Mantener `docs/README.md`, `docs/plan.md` y `docs/backlog.md` alineados con metadata real.
