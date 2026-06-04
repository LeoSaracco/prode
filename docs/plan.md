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

## Ensemble (Fase E)

Arquitectura de weighted voting basada en accuracy en val set:

```text
Base models (paralelizados con ThreadPoolExecutor):
├── RandomForest (300 trees, OOB)
├── XGBoost (300 iter, max_depth=5)
├── LightGBM (300 iter, num_leaves=31)
├── CatBoost (200 iter, depth=6, opcional)
└── Elo (baseline, peso fijo 0.10)

Voting:
  peso_modelo = max(0.05, accuracy_val - 0.33)
  Se normalizan para sumar 1.0.

TwoStage (8 features para draw, 21 para win/loss)
Confederation (14 pares, RF 300 trees cada uno)
```

El entrenamiento es totalmente paralelo. Si CatBoost falla o tarda
mas de 120s, se skipea y el ensemble vota con los 3 modelos restantes.
Los pesos se guardan en `models/voting_weights.json`.

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

## Two-Stage Prediction (Fase D)

En lugar de clasificar W/D/L directamente (3 clases), se descompone en:

```
Stage 1 — DrawClassifier (RF binario):
  P(draw) vs P(not-draw)

Stage 2 — WinClassifier (RF binario, entrenado solo en no-draws):
  P(win | not-draw) vs P(loss | not-draw)

Combinacion:
  P_win  = (1 - P_draw) * P(win | not_draw)
  P_draw = P_draw
  P_loss = (1 - P_draw) * (1 - P(win | not_draw))
```

Ventaja: los clasificadores binarios logran mayor accuracy que uno
multiclase, especialmente para draws que son eventos mas raros (~25%).

## Confederation Models (Fase D)

14 modelos RandomForest especificos por par de confederaciones:

- UEFA-UEFA, UEFA-CONMEBOL, UEFA-CAF, UEFA-AFC, UEFA-CONCACAF
- CONMEBOL-CONMEBOL, CONMEBOL-CAF, CONMEBOL-AFC, CONMEBOL-CONCACAF
- CAF-CAF, CAF-AFC, AFC-AFC, AFC-CONCACAF, CONCACAF-CONCACAF

Cada modelo se entrena solo con partidos de ese par de confederaciones.
Rare matchups (ej: OFC vs cualquiera) usan el modelo global como fallback.

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
