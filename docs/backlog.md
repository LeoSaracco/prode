# Backlog - prode-ML FIFA World Cup 2026

Este archivo resume el estado actual y lo que queda pendiente.

## Estado Actual

- CLI interactivo: implementado.
- Modelos entrenados: implementado, con artefactos en `models/`.
- API REST FastAPI: implementada.
- Frontend React/Vite: implementado.
- Grupos oficiales WC2026: implementados en `config/wc2026_groups.py`.
- Simulacion de grupos: implementada con tabla de clasificacion probable y marcador mas probable por fixture.
- Automatizacion local Bash: implementada en `scripts/run_full_stack.sh`.
- Fase A completada: pipeline de datos robusto, split temporal corregido, features contextuales.

## Completado

### Fase A — Pipeline de Datos y Split Temporal

Estado: completado.

Cambios realizados:

- **Fix data leakage**: el split train/val/test ahora es estrictamente temporal (70/15/15 cronologico). Antes `_add_reverse_perspective` contaminaba el test con datos de train.
- **Elo historico computacional**: nuevo `EloHistoryCollector` que calcula Elo rolling desde resultados historicos sin depender de datasets externos.
- **FIFA Rankings collector**: `FIFARankingsCollector` descarga rankings historicos FIFA desde GitHub.
- **Features contextuales**: `is_tournament`, `is_wc`, `is_qualifier`, `is_home` agregados a la matriz de features (21 features total, antes 17).
- **Rango de datos ampliado**: `MATCH_HISTORY_START_YEAR=2000` en settings. El colector de resultados internacionales ahora preserva info de torneo y neutral.
- **Metricas segmentadas**: `accuracy_high_elo_diff_200` medida en test set para partidos con diferencia Elo >= 200.
- **Pesos del ensemble normalizados**: se normalizan automaticamente si no suman 1.0.

### Fase B — Re-ingenieria de Ensemble y Features

Estado: completado.

Cambios realizados:

- **RandomForest**: nuevo `RFOutcomeClassifier` con 500 arboles, OOB scoring, class_weight balanced.
- **CatBoost**: nuevo `CatBoostOutcomeClassifier` con ordered boosting y early stopping.
- **Meta-learner**: `LogisticRegressionCV` entrenado sobre predicciones de los 4 modelos base + Elo en val set. Reemplaza pesos fijos.
- **Feature selection**: removidos `big_match_rating_diff`, `pressure_diff`, `form_times_elo_diff`, `attack_vs_defense_clash`.
- **Nuevos features**: `fifa_rank_diff`, `elo_momentum_diff`, `days_since_last_match_diff`, `rest_days_diff`.
- **Architectura ensemble**: RF + XGBoost + LightGBM + CatBoost + Elo -> LogisticRegressionCV meta-learner.
- **Runtime**: carga RF, CatBoost y meta-learner automaticamente.

### Fase C — Hyperparameter Tuning con Optuna

Estado: completado.

Cambios realizados:

- **Optuna integrado**: `scripts/tune_hyperparams.py` con 30+ parametros tuneables en 5 modelos.
- **Modelos parametrizables**: RF, XGB, LGBM, CatBoost aceptan `params dict` para override de defaults.
- **TimeSeriesSplit implicito**: split cronologico estricto dentro del objective de Optuna.
- **best_params.json**: guarda los mejores parametros encontrados en `models/best_params.json`.
- **Trainer auto-tuned**: `_fit_models` carga `best_params.json` automaticamente si existe.
- **Study persistente**: almacenamiento SQLite en `models/optuna_study.db` para retomar sesiones.
- **Meta_Cs tuneado**: el numero de C values del LogisticRegressionCV tambien se optimiza.

### Fase E — Fixes y Optimizaciones

Estado: completado.

Cambios realizados:

- **Weighted voting reemplaza meta-learner LR**: los pesos se calculan como `accuracy_val - 0.33` por modelo, normalizados. Elo siempre pesa base 0.10. Resultado: ensemble supera al mejor modelo individual.
- **Poisson con timeout**: `maxiter=200`, `ftol=1e-4`. Si no converge usa default params.
- **TwoStage con feature mask**: stage1 (draw) usa solo 8 features: elo_diff, elo_win_prob, xg_diff, xga_diff, form_5_diff, defensive_stability_diff, consistency_diff, is_tournament.
- **Entrenamiento paralelo**: ThreadPoolExecutor(4) entrena RF, XGB, LGBM, CatBoost simultaneamente.
- **CatBoost opcional**: si falla o timeout (120s), se skipea y el ensemble sigue con los otros 3.
- **Pipeline rapido simplificado**: run_fast solo baja international results + Elo (sin StatsBomb/Kaggle/FIFA rotos).

### API REST con FastAPI

Estado: completado.

Endpoints disponibles:

- `GET /health`
- `GET /api/v1/teams`
- `GET /api/v1/groups`
- `POST /api/v1/predict`
- `GET /api/v1/simulate/group/{group_name}`
- `GET /api/v1/simulate/tournament`

`GET /api/v1/simulate/group/{group_name}` devuelve `results` para la tabla de grupo y `fixtures` con el marcador mas probable de cada partido.

### Frontend Web

Estado: completado.

Incluye:

- Vista de prediccion de partido.
- Vista de simulacion de grupos.
- Vista de simulacion de torneo.
- Render de resultados mas probables por partido en la vista de grupos.

### Grupos Oficiales

Estado: completado.

Los grupos oficiales del Mundial 2026 estan en `config/wc2026_groups.py`.

### Automatizacion Local Bash

Estado: completado.

`scripts/run_full_stack.sh` ejecuta pipeline, validacion, entrenamiento, API y frontend en orden. Incluye flags para modo rapido, refresh forzado, puertos custom, saltar etapas y evitar instalacion de dependencias.

## Pendiente

### Mejorar Accuracy del Modelo

Prioridad: alta.

El sistema entrena y predice, pero las metricas actuales no alcanzan los targets aspiracionales originales. Acciones recomendadas:

- Integrar mas partidos recientes de selecciones nacionales.
- Mejorar features de localia, lesiones y forma reciente.
- Calibrar probabilidades del ensemble.
- Evaluar pesos adaptativos o meta-learner.
- Medir por segmento: global, alta confianza y delta Elo alto.

### Docker y Deploy

Prioridad: baja.

Pendiente crear:

- `Dockerfile.api`
- `Dockerfile.frontend`
- `docker-compose.yml`

### Robustez Windows

Prioridad: media.

Pendiente evitar errores de encoding en Windows sin depender de `PYTHONIOENCODING=utf-8`.

## Notas Operativas

- No reentrenar modelos salvo que haya nuevos datos o cambios en features.
- Consultar siempre `models/model_metadata.json` para metricas reales.
- Mantener `docs/README.md` alineado con endpoints y comandos reales.
