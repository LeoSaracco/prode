# Plan Tecnico - prode-ML FIFA World Cup 2026

## Contexto

El objetivo del proyecto es predecir partidos del Mundial FIFA 2026 y simular grupos/torneo usando modelos entrenados y fuentes de datos futboleras.

Estado actual:

- CLI principal en `cli.py`.
- API REST en `api/`.
- Frontend React/Vite en `frontend/`.
- Modelos entrenados en `models/`.
- Grupos oficiales en `config/wc2026_groups.py`.
- Simulacion de grupos con tabla de clasificacion probable y marcador mas probable por fixture.
- Fase A completada: pipeline robusto, split temporal cronologico, Elo computacional, features contextuales.

## Arquitectura

Componentes principales:

- `config/`: settings, aliases, grupos oficiales y pesos de features.
- `src/data/`: collectors, cache, pipeline, validadores, Elo historico y FIFA rankings.
- `src/features/`: construccion de features para prediccion (21 features).
- `src/models/`: Poisson, XGBoost, LightGBM, Elo y ensemble con split temporal.
- `src/simulation/`: simulacion de partidos, grupos y torneo.
- `src/output/`: formateadores de consola.
- `api/`: FastAPI, schemas, routers y carga de runtime.
- `frontend/`: UI web para consumir la API.

## Features del Modelo (21 features)

Features base (17):

- `elo_diff`
- `elo_win_prob_a`
- `xg_diff`
- `xga_diff`
- `form_5_diff`
- `offensive_power_diff`
- `defensive_stability_diff`
- `squad_depth_diff`
- `big_match_rating_diff`
- `pressure_diff`
- `consistency_diff`
- `wc_history_diff`
- `market_value_diff`
- `tactical_advantage`
- `h2h_advantage`
- `form_times_elo_diff`
- `attack_vs_defense_clash`

Features contextuales (4, agregadas en Fase A):

- `is_tournament`: partido de torneo oficial (1) vs amistoso (0)
- `is_wc`: partido de Mundial (1) o no (0)
- `is_qualifier`: clasificatorio (1) o no (0)
- `is_home`: localia (1) o neutral/visitante (0)

## Split Temporal (Fase A)

El split train/val/test es estrictamente cronologico (70/15/15):

- Train: partidos mas antiguos (70%)
- Val: partidos intermedios (15%)
- Test: partidos mas recientes (15%)

`_add_reverse_perspective` se aplica SOLO al train set para evitar data leakage.
El metadata ahora incluye `split_type: time_series_chronological` y `accuracy_high_elo_diff_200`.

## Ensemble (Fase B)

Arquitectura de stacking con meta-learner:

```text
Layer 1 — Base models:
├── RandomForest (500 trees, OOB)
├── XGBoost (600 iter, max_depth=5)
├── LightGBM (600 iter, num_leaves=31)
├── CatBoost (600 iter, depth=6)
└── Elo (baseline estadistico)

Layer 2 — Meta-learner:
└── LogisticRegressionCV (Cs=8, multinomial)
    Entrenado sobre predicciones base en val set.

Layer 3 — Calibracion:
└── IsotonicRegression (3-fold CV)
```

El peso de cada modelo base es aprendido por el meta-learner,
no hardcodeado. El modelo Poisson solo se usa para xG y scorelines.

## Hyperparameter Tuning (Fase C)

Se usa Optuna con objetivo de minimizar log_loss en test set cronologico.

Espacio de busqueda:

- **RF**: n_estimators (200-800), max_depth (4-20), min_samples_split (5-30),
  min_samples_leaf (2-15), max_features (sqrt/log2/None)
- **XGBoost**: n_estimators (300-900), max_depth (3-9), lr (0.01-0.15 log),
  subsample/colsample (0.6-1.0), min_child_weight (1-10),
  gamma (0-2.0), reg_alpha/lambda (log)
- **LightGBM**: n_estimators (300-900), num_leaves (15-63), lr (0.01-0.15 log),
  subsample/colsample (0.6-1.0), min_child_samples (10-50), reg_alpha/lambda (log)
- **CatBoost**: iterations (300-900), depth (3-9), lr (0.01-0.15 log),
  l2_leaf_reg (0.5-10.0), border_count (32-255)
- **Meta**: Cs (5-15)

Resultados guardados en `models/best_params.json` y cargados automaticamente
por `trainer.py` en el proximo entrenamiento.

## Simulacion de Grupos

Para cada grupo A-L:

- Se simulan los 6 partidos del grupo.
- Se acumulan puntos, diferencia de gol y goles a favor.
- Se ordena por puntos, diferencia de gol y goles a favor.
- Se reporta probabilidad de terminar 1ro, 2do, 3ro y 4to.
- Se reporta probabilidad de clasificacion directa como 1ro o 2do.
- Se reportan puntos y diferencia de gol promedio.
- Se reporta el marcador exacto mas probable para cada fixture del grupo.

Ejemplo conceptual:

```text
RESULTADOS MAS PROBABLES - GRUPO J
Argentina vs Austria      1-0  (12.8%)
Argentina vs Algeria      2-0  (11.4%)
Argentina vs Jordan       2-0  (13.1%)
Austria vs Algeria        1-1  (10.6%)
Austria vs Jordan         1-0  (11.2%)
Algeria vs Jordan         1-1  (11.0%)
```

La API expone estos datos en `fixtures` dentro de `GET /api/v1/simulate/group/{group_name}`.

## Accuracy

Targets aspiracionales originales:

- Global W/D/L: mayor a 55%.
- Alta confianza: mayor a 80%.
- Delta Elo mayor a 200: mayor a 85%.

Estado real: el entrenamiento actual genera modelos funcionales, pero las metricas reales en `models/model_metadata.json` no alcanzan todavia esos targets. La mejora de accuracy queda como backlog activo.

## Operacion

Comandos principales:

```powershell
python cli.py predict "Argentina" "Portugal"
python cli.py simulate-group J
python cli.py simulate-tournament
python scripts/run_pipeline.py --fast
python scripts/train_models.py
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Flujo automatizado recomendado con Bash:

```bash
bash scripts/run_full_stack.sh --fast
```

Este script ejecuta pipeline, validacion, entrenamiento, API y frontend en ese orden. Si falla una etapa de datos o entrenamiento, no levanta servicios.

En Windows puede ser necesario:

```powershell
$env:PYTHONIOENCODING='utf-8'
```
