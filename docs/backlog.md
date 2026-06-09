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
| I - UX y Performance | Completado | Frontend 4 vistas, API cacheada | Calendario, bracket mirror, comparar partidos, spinner, pair_cache, elo_dict, score matrix unificada. |

## Estado Actual

- CLI interactivo: implementado.
- API REST FastAPI: implementada (6 endpoints + `output=bracket`, `pair_cache`).
- Frontend React/Vite: implementado (4 vistas: Prediccion, Calendario, Grupos, Torneo).
- Grupos oficiales WC2026: implementados.
- Reporte PDF: implementado.
- Automatizacion local: `run_all.bat` y `start_solution.bat`.
- Entrenamiento rolling sin leakage: implementado.
- FIFA rankings historicos: collector reparado y cacheado.
- Confidence thresholds: implementados.
- Poisson integrado al ensemble como modelo 6 (peso fijo ~8%).
- Squad quality desde ratings FIFA de jugadores (player_aggregates.csv).
- Bracket oficial WC2026 con pods predefinidos y seleccion de terceros por rendimiento.
- Marcadores Poisson-sampled (ya no deterministicos con round(xG)).
- Seeds variables en simulaciones Monte Carlo.
- **Calendario**: `GET /api/v1/fixtures` con fechas reales del fixture FIFA (72 partidos, resultados.csv).
- **Bracket mirror**: diagrama simetrico de eliminacion directa (R32→SF←R32) con conectores diamante y badge de campeon.
- **Comparar partidos**: prediccion dual lado a lado con auto-predict debounce.
- **Spinner**: loading overlay con dual-ring CSS en todas las llamadas API (incluye carga inicial).
- **Performance**: `pair_cache` cross-request en PredictionRuntime, `elo_dict` O(1), score matrix unica, skip SHAP/aux en simulacion, scoreline counting vectorizado con numpy, ThreadPoolExecutor en simulacion de torneo.
- **Responsive**: 4 breakpoints (1200/1023/820/639/480), scroll horizontal en bracket, touch-action.

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

## Completado En Fase I — UX y Performance (2026-06-09)

### Backend — Nuevos Endpoints

- **`GET /api/v1/fixtures`**: devuelve los 72 partidos de fase de grupos con fecha, grupo, venue, prediccion (outcome, scoreline, xG, confianza). Lee `data/raw/international/results.csv`, normaliza nombres con `resolve_team_name()`, calcula matchday por rango de fechas. Cachea predicciones en `runtime.pair_cache`.
- **`GET /api/v1/simulate/tournament?output=bracket`**: nuevo parametro `output` que bifurca entre lista (`TournamentSimulationResponse`) y arbol (`BracketResponse` con 5 rondas: r32/r16/qf/sf/final, cada slot con equipos + probabilidades + `feeds_from`). Nuevo metodo `TournamentSimulator.simulate_bracket()` con slot occupancy tracking.

### Backend — Optimizaciones de Performance

- **`PredictionRuntime.pair_cache`**: cache cross-request que persiste en memoria del servidor. La primera llamada a un endpoint de simulacion o fixtures llena el cache con 72 predicciones (~5s). Requests subsiguientes son O(1) (~0s). Ubicado en `src/runtime.py:35-44`.
- **`elo_dict`**: `load_prediction_runtime()` convierte `elo_df` a `dict[str, float]` al iniciar. `get_elo_rating()` acepta dict o DataFrame. Elimina scan lineal de pandas por cada predict. Ubicado en `src/runtime.py:51-53` y `src/features/elo_features.py:9-16`.
- **`include_shap=False, include_aux=False`**: nuevos parametros en `predict_match()`. La simulacion y fixtures pasan `False` para evitar SHAP TreeExplainer y modelos auxiliares (two_stage, confederation). Ahorra ~40% por llamada. Ubicado en `src/runtime.py:99-118`.
- **Score matrix unificada**: `EnsemblePredictor.predict()` computa `predict_score_matrix()` una sola vez y la reusa para `_scorelines_from_matrix()`, `top_scorelines`, y `get_representative_scoreline()`. Antes eran 3 llamadas independientes (300 evaluaciones PMF+Tau cada una). Helper `_scorelines_from_matrix()` en `src/models/ensemble.py:28-33`.
- **Scoreline counting vectorizado**: `simulate_match()` reemplaza el for-loop Python de 100k iteraciones con `np.unique(encoded, return_counts=True)`. ~50x mas rapido. Ubicado en `src/simulation/match_simulator.py:28-36`.
- **`ThreadPoolExecutor` en simulacion de torneo**: `TournamentSimulator.simulate()` divide `n_sims` en chunks y corre con `ThreadPoolExecutor` (max 4 workers). Cada worker tiene su propia instancia de `np.random.default_rng()`. Metodo refactorizado en `_sim_one_iteration()` + `_sim_chunk()`. Ubicado en `src/simulation/tournament_simulator.py`.

### Frontend — Nuevas Vistas

- **Calendario** (`view="calendar"`): timeline de fase de grupos con toggle "Por jornada" / "Por fecha". Agrupa fixtures en secciones colapsables con header sticky. Cada `FixtureCard` muestra fecha, grupo, equipos, marcador predicho, barra de probabilidades compacta, badge de confianza, xG. Consume `GET /api/v1/fixtures`. Ubicado en `main.tsx` (componentes `CalendarTimeline`, `FixtureCard`).
- **Comparar partidos**: toggle `[Comparar]` en vista Prediccion. Duplica `TeamCombobox` + `PredictionPanel` lado a lado. Auto-predict con debounce de 600ms al cambiar cualquier equipo. `PredictionPanel` en modo `compact` (oculta SHAP features, reduce scorelines a top 3). Ubicado en `main.tsx`.
- **Bracket mirror**: diagrama simetrico de eliminacion directa. Layout:
  ```
  Rama izquierda: R32(8) → R16(4) → QF(2) → SF(1)
  Centro: 🏆 CAMPEON
  Rama derecha: SF(1) → QF(2) → R16(4) → R32(8)
  ```
  Componentes: `BracketTree` → `BracketRoundColumn` → `BracketMatch` (2 `BracketTeamRow` + conector) → `ChampionBadge`. Conectores estilo grapa industrial con linea + diamante `◆` rotado 45°. Colores: gris `#444` (R32→QF), granate `#6B1030` (SF→Final, campeon). Consume `GET /api/v1/simulate/tournament?output=bracket`. Ubicado en `main.tsx:671-758`.

### Frontend — UX Transversal

- **Spinner global**: `LoadingOverlay` con dual-ring CSS (anillo exterior `#d6f36c`, interior `#8cc7ff`, rotacion opuesta), fondo `rgba(16,20,22,0.85)` + `backdrop-filter: blur(8px)`, texto contextual. Cubre **todas** las llamadas API: carga inicial (`initialLoading`), fixtures, prediccion, grupo, torneo. `role="alert" aria-busy="true"` para a11y. Ubicado en `main.tsx:736-748`.
- **Responsive**: 4 breakpoints — 1023px (tablet), 820px (small tablet), 639px (mobile), 480px (mobile small). Bracket con `overflow-x: auto` + scroll horizontal. Cards de calendario 3→2→1 columna. `touch-action: manipulation` en todos los interactivos. Dropdowns con `max-height: 35-40vh`. Ubicado en `styles.css`.

### Archivos Modificados en Fase I

| Archivo | Cambio |
|---------|--------|
| `api/schemas.py` | + `FixturePrediction`, `FixturesResponse`, `BracketTeam`, `BracketSlot`, `BracketRound`, `BracketResponse` |
| `api/routers/predictions.py` | + `GET /fixtures`, + `_load_wc_fixtures()`, `_compute_matchday()` |
| `api/routers/simulation.py` | + query param `output=list\|bracket`, `_cached_predictor` usa `runtime.cached_predict()` |
| `src/runtime.py` | + `elo_dict`, + `pair_cache`, + `cached_predict()`, + `include_shap/include_aux` |
| `src/features/elo_features.py` | `get_elo_rating()` acepta `dict` ademas de `DataFrame` |
| `src/models/ensemble.py` | + `_scorelines_from_matrix()`, score matrix unificada, `elo_source` param |
| `src/simulation/tournament_simulator.py` | + `_prewarm_cache()`, `_sim_one_iteration()`, `_sim_chunk()`, `_build_df()`, `ThreadPoolExecutor` |
| `src/simulation/match_simulator.py` | Scoreline counting con `np.unique` vectorizado |
| `cli.py` | + `_predict_cache`, + `_elo_dict` |
| `frontend/src/api.ts` | + `FixturePrediction`, `BracketResponse` y rama de tipos, + `fetchFixtures()`, `simulateTournamentBracket()` |
| `frontend/src/main.tsx` | + vista calendar, `CalendarTimeline`, `FixtureCard`, `BracketTree` mirror, `BracketRoundColumn`, `BracketMatch`, `BracketTeamRow`, `ChampionBadge`, `_FLAGS`, compareMode, auto-predict, `initialLoading`, `LoadingOverlay` |
| `frontend/src/styles.css` | + ~300 lineas: calendar, fixture cards, toggle groups, compare mode, bracket mirror, champion badge, connectors, spinner overlay, 4 media queries, mobile universal rules |

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

## Pendiente (Fase J+)

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

### Frontend — Polish Visual

Prioridad: media.

- **Conectores SVG**: los conectores del bracket usan CSS (`::before`/`::after` con diamantes rotados). Reemplazar por SVG inline para lineas curvas estilo "grapa industrial" completa (como la imagen de referencia de Infobae), con trazados curvos en lugar de lineas rectas con diamantes.
- **Animacion de entrada**: stagger fade-in de los slots del bracket (animation-delay escalonado por indice).
- **Hover path highlight**: al hacer hover sobre un equipo en el bracket, resaltar su camino completo hasta la final (opacidad 1 en el camino, 0.3 en el resto).
- **Flag sprites o PNG**: los emojis de bandera se renderizan distinto segun SO. Evaluar usar un sprite sheet o PNG de banderas para consistencia visual.

### Docker Y Deploy

Prioridad: baja.

- `Dockerfile.api`, `Dockerfile.frontend`, `docker-compose.yml`.

### Performance Adicional

Prioridad: baja.

- **Precalentar pair_cache en lifespan**: mover el warm-up de predicciones al `lifespan` de FastAPI para que el primer request ya sea instantaneo (actualmente el primer request a `/fixtures` o `/simulate/tournament` tarda ~5s en llenar el cache).
- **Predict match cache por feature vector**: `predict_match()` tarda ~1s por llamada aun sin SHAP. Cachear el resultado por feature vector (ya que `build_match_features` es determinista para un par de equipos) reduciria las 72 predicciones iniciales a ~0s si los features ya se calcularon.
- **Paralelizar predict_match con multiprocessing**: `ThreadPoolExecutor` no ayuda con CPU-bound en Python por el GIL. Usar `ProcessPoolExecutor` para las 72 predicciones iniciales (cada proceso carga sus propios modelos, ~4-8 workers).

## Notas Operativas

- Consultar siempre `models/model_metadata.json` para metricas reales.
- No comparar metricas antiguas con leakage contra metricas nuevas rolling como si fueran equivalentes.
- Reentrenar despues de cambios en features, collector de datos o pesos.
- Mantener `docs/README.md`, `docs/plan.md` y `docs/backlog.md` alineados con metadata real.
