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
| H - Features y Datos | En curso | Target >52% | Arreglar h2h/consistency, fix Kaggle match features collector, Poisson al ensemble. |

## Estado Actual

- CLI interactivo: implementado.
- API REST FastAPI: implementada.
- Frontend React/Vite: implementado.
- Grupos oficiales WC2026: implementados.
- Reporte PDF: implementado.
- Automatizacion local: `run_all.bat`.
- Entrenamiento rolling sin leakage: implementado.
- FIFA rankings historicos: collector reparado y cacheado.
- Confidence thresholds: implementados.

Artefactos relevantes:

- `models/model_metadata.json`
- `models/voting_weights.json`
- `models/confidence_thresholds.json`
- `models/inference_team_profiles.json`

## Metricas Actuales

| Metrica | Valor | Que significa |
|---|---:|---|
| `accuracy_voting` | 50.21% | Acierto general del ensemble principal. |
| `accuracy_catboost` | 51.04% | Acierto del mejor modelo individual actual. |
| `accuracy_lgbm` | 49.58% | Acierto de LightGBM. |
| `accuracy_rf` | 48.96% | Acierto de RandomForest. |
| `accuracy_xgb` | 46.67% | Acierto de XGBoost. |
| `accuracy_confederation` | 48.33% | Acierto del modelo por confederaciones. |
| `accuracy_two_stage` | 35.42% | Acierto del modelo auxiliar de dos etapas. |
| `accuracy_high_confidence` | 69.70% | Acierto cuando el sistema marca `ALTO`. |
| `accuracy_high_elo_diff_200` | 57.75% | Acierto cuando la diferencia Elo es >= 200. |
| `log_loss_voting` | 1.0396 | Calidad de probabilidades; menor es mejor. |

Notas:

- `accuracy_voting` es la metrica principal global.
- `accuracy_high_confidence` depende de `n_high_confidence_matches`; hoy son 33 casos.
- Las metricas actuales son mas honestas que versiones anteriores porque usan features rolling sin mirar el futuro.

## Completado Reciente

### Fase F - Validacion ML Y Confianza

Cambios realizados:

- **Rolling features sin leakage**: cada partido usa solo informacion anterior a su fecha.
- **FIFA rankings reproducibles**: collector actualizado a una fuente GitHub activa y ranking derivado por puntos cuando no viene columna `rank`.
- **Features activas**: `fifa_rank_diff`, `elo_momentum_diff`, `days_since_last_match_diff`, `rest_days_diff`.
- **Perfiles de inferencia**: se guarda `models/inference_team_profiles.json` para que CLI/API usen el mismo snapshot del entrenamiento.
- **Confianza calibrada**: se guarda `models/confidence_thresholds.json`.
- **Umbral ALTO conservador**: minimo `high_prob = 0.70`.
- **Tuning alineado**: `scripts/tune_hyperparams.py` ya optimiza weighted voting, no meta-learner viejo.

### Fase E - Performance Local

Cambios realizados:

- Entrenamiento base paralelo: RF, XGB, LGBM, CatBoost.
- Threads parametrizables con `TRAIN_MODEL_JOBS`.
- ConfederationModels parametrizable con `CONFED_N_ESTIMATORS` y `CONFED_MODEL_JOBS`.
- Poisson empirico rapido por defecto; optimizacion pesada disponible con `POISSON_OPTIMIZE=1`.
- `run_all.bat` mas descriptivo y con salida sin buffer.

## Explicacion De Accuracy

El problema predice tres resultados posibles:

1. gana equipo A
2. empate
3. gana equipo B

Por eso, acertar al azar seria aproximadamente 33.3%.

Ejemplo:

```text
accuracy_voting = 50.21%
```

Significa que el ensemble acerto 50.21 de cada 100 partidos del test historico.

No significa que vaya a acertar exactamente 50.21% en el Mundial real. El Mundial
tiene menos partidos, contexto distinto, planteles concretos y eliminacion directa.

## Completado Reciente

### Fase G - Calibracion, Recencia y Calidad (2026-06-04)

- **Temperature scaling**: reemplaza isotonic regression (que inflaba log_loss). T=1.0447 ajustado en val. Mejora log_loss calibrado a 1.0337 vs 1.0364 sin calibrar.
- **Sample weight por recencia**: entrenamiento con decaimiento exponencial (half-life 3 anos). Partidos recientes pesan mas. CatBoost sube de 49.59% a 51.65%.
- **TwoStage fuera del reporte**: marcado como "diagnostic only" en el PDF. Sigue entrenandose pero no contamina las predicciones mostradas.
- **Optuna objetivo compuesto**: `log_loss - 0.3 * high_conf_acc`, guarda tambien `high_conf_acc` y `n_high_conf` como user attrs.
- **Kaggle credentials**: `~/.kaggle/kaggle.json` configurado. Dos datasets descargados (Elo: 1991 filas, match features: descargado pero collector pendiente de fix).

## Pendiente Prioritario (Fase H)

### Bugs Con Impacto Directo En Features

Prioridad: alta.

- **`h2h_advantage` siempre es 0.0**: hardcodeado en `trainer.py`. Computar historial directo desde datos de entrenamiento rolling (sin leakage).
- **`consistency_diff` duplica `form_5_diff`**: ambos calculan `sa["form_5"] - sb["form_5"]`. Reemplazar `consistency_diff` con desviacion estandar de ultimos 5 resultados.

### Fix Collector Kaggle Match Features

Prioridad: alta.

El dataset `lchikry/international-football-match-features-and-statistics` se descargo (3.67MB) pero el collector no puede parsearlo. Las columnas reales son `_home_team`, `_away_team`, `_date`, `home_goals`, `away_goals`. Arreglarlo sumaria ~8k partidos con features de Elo, forma y ratings de planteles.

### Optuna 50 Trials

Prioridad: media. Ya implementado con objetivo compuesto. Solo ejecutar: `python scripts/tune_hyperparams.py --trials 50`.

### Poisson Al Ensemble

Prioridad: media.

Las probabilidades implicitas W/D/L derivadas del xG Poisson son una senal ortogonal al ML. Incorporar como modelo 6 del voting con peso inicial bajo (0.05-0.10).

### Datos Y Robustez

Prioridad: media.

- Valor de mercado dinamico por fecha de partido (no dato estatico).
- Pruebas de no-leakage automatizadas.
- Tests que validen que probabilidades suman 1.

### Docker Y Deploy

Prioridad: baja.

- `Dockerfile.api`, `Dockerfile.frontend`, `docker-compose.yml`.

## Notas Operativas

- Consultar siempre `models/model_metadata.json` para metricas reales.
- No comparar metricas antiguas con leakage contra metricas nuevas rolling como si fueran equivalentes.
- Reentrenar despues de cambios en features, collector de datos o pesos.
- Mantener `docs/README.md`, `docs/plan.md` y `docs/backlog.md` alineados con metadata real.
