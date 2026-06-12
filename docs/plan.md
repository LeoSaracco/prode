# Plan Tecnico - prode-ML FIFA World Cup 2026

## Contexto

El objetivo del proyecto es predecir partidos del Mundial 2026 y simular grupos
/ torneo con modelos entrenados sobre resultados internacionales historicos.

El sistema tiene:

- CLI en `cli.py`.
- API REST en `api/`.
- Frontend React/Vite en `frontend/`.
- Modelos y artefactos en `models/`.
- Grupos oficiales en `config/wc2026_groups.py`.
- Pipeline automatizado en `run_all.bat`.

## Arquitectura Actual

Flujo principal:

1. Recolectar resultados internacionales, Elo y rankings FIFA.
2. Ordenar partidos cronologicamente.
3. Construir features rolling sin mirar partidos futuros.
4. Entrenar modelos base en paralelo (RF, XGB, LGBM, CatBoost).
5. Entrenar Poisson sobre datos de train; evaluar W/D/L implicito en validation.
6. Calcular pesos del ensemble en validation.
7. Evaluar una sola vez en test cronologico.
8. Guardar modelos, pesos, umbrales de confianza y metadata.

Componentes:

- `src/data/`: collectors, cache y pipeline.
- `src/features/`: features de partido e inferencia.
- `src/models/`: modelos, trainer, ensemble, Poisson y auxiliares.
- `src/runtime.py`: carga modelos para CLI/API.
- `src/simulation/`: simulacion de grupos/torneo.

## Features Del Modelo

El modelo usa 21 features por partido:

| Feature | Que representa |
|---|---|
| `elo_diff` | Diferencia de rating Elo entre equipo A y equipo B. |
| `elo_win_prob_a` | Probabilidad Elo teorica de que gane A. |
| `xg_diff` | Diferencia estimada de goles esperados a favor. |
| `xga_diff` | Diferencia estimada de goles esperados en contra. |
| `form_5_diff` | Diferencia de forma reciente basada en ultimos partidos previos. |
| `offensive_power_diff` | Diferencia de potencia ofensiva derivada de xG. |
| `defensive_stability_diff` | Diferencia de estabilidad defensiva. |
| `squad_depth_diff` | Calidad del plantel desde ratings FIFA reales (avg_overall, max_overall, depth_ratio, shooting, defending). Reemplaza el valor de mercado. |
| `consistency_diff` | Diferencia de regularidad reciente (std de resultados). |
| `wc_history_diff` | Diferencia de historia/rendimiento mundialista. |
| `market_value_diff` | Diferencia de valor de mercado relativo. |
| `tactical_advantage` | Feature combinada de ventaja tactica. |
| `h2h_advantage` | Historial directo rolling (min 3 partidos; 0 si no hay dato). |
| `is_tournament` | 1 si es partido competitivo/torneo, 0 si amistoso. |
| `is_wc` | 1 si es Mundial. |
| `is_qualifier` | 1 si es clasificatorio. |
| `is_home` | 1 si A tiene localia, 0 si neutral/visitante. |
| `fifa_rank_diff` | Diferencia de ranking FIFA historico disponible antes del partido. |
| `elo_momentum_diff` | Diferencia de momentum Elo reciente. |
| `days_since_last_match_diff` | Diferencia de dias desde el ultimo partido. |
| `rest_days_diff` | Diferencia inversa de descanso entre ambos equipos. |

## Control De Leakage

El entrenamiento usa `feature_generation: rolling_no_future_leakage`.

Esto significa:

- Para un partido en fecha `D`, las features se calculan solo con partidos anteriores a `D`.
- Train, validation y test estan separados por fecha.
- La perspectiva inversa se agrega solo en train.
- Validation y test no se duplican ni contaminan con datos futuros.

Este punto es clave: una metrica sin leakage puede verse mas baja que una metrica
inflada, pero representa mejor como se comportara el modelo ante partidos reales.

## Modelos

Modelos base (6):

- RandomForest
- XGBoost
- LightGBM
- CatBoost
- Elo baseline (Bradley-Terry puro)
- Poisson (W/D/L implicito desde distribucion bivariate de goles)

Ensemble principal:

```text
RF + XGB + LGBM + CatBoost + Elo + Poisson -> accuracy-weighted voting
```

Los pesos se calculan con accuracy en validation:

```text
peso_RF/XGB/LGBM/CatBoost = max(0.05, accuracy_val - 0.33)
peso_elo    = 0.10  (fijo, baseline estable)
peso_poisson = 0.07 (fijo, senal ortogonal de distribucion de goles)
```

Luego se normalizan para sumar 1.0.

Modelos auxiliares (diagnostico, no integrados al ensemble):

- `TwoStageClassifier`: separa empate/no empate y luego win/loss. Corregido el
  2026-06-12 (se quito `class_weight="balanced"` del clasificador de empates);
  accuracy subio de 34.65% (por debajo del azar) a 48.83%, en linea con los
  modelos base.
- `ConfederationModels`: RandomForest por pares de confederaciones.
- `PoissonGoalModel`: tambien usado para xG, marcadores probables y simulacion del torneo.

## Metricas Explicadas

Todas las `accuracy_*` son porcentajes de acierto sobre partidos historicos del
conjunto de test. El problema tiene tres clases:

- gana equipo A
- empate
- gana equipo B

Por azar puro, la referencia aproximada es 33.3%.

| Metrica | Significado |
|---|---|
| `accuracy_rf` | Acierto usando solo RandomForest. |
| `accuracy_xgb` | Acierto usando solo XGBoost. |
| `accuracy_lgbm` | Acierto usando solo LightGBM. |
| `accuracy_catboost` | Acierto usando solo CatBoost. |
| `accuracy_voting` | Acierto del ensemble de 6 modelos. Es la metrica global mas importante. |
| `accuracy_two_stage` | Acierto del modelo auxiliar de dos etapas. Diagnostico (48.83% desde el fix del 2026-06-12). |
| `accuracy_confederation` | Acierto del modelo especializado por confederacion. |
| `accuracy_high_confidence` | Acierto solo donde la prediccion fue marcada `ALTO`. |
| `accuracy_high_elo_diff_200` | Acierto en partidos con diferencia Elo de al menos 200 puntos. |

Otras metricas:

- `n_high_confidence_matches`: cantidad de casos usados para medir `accuracy_high_confidence`.
- `n_high_elo_matches`: cantidad de casos usados para medir `accuracy_high_elo_diff_200`.
- `log_loss_voting`: calidad de las probabilidades; mas bajo es mejor.
- `val_accuracies`: aciertos por modelo en validation.
- `voting_weights`: peso final de cada modelo en el ensemble.
- `temperature_scaling`: factor T aplicado post-ensemble. T>1 suaviza, T<1 sharpens. T~1 indica probs bien calibradas.

## Confianza

La etiqueta de confianza se basa en la probabilidad maxima del ensemble:

- `ALTO`: probabilidad maxima >= `high_prob`.
- `MEDIO`: probabilidad maxima >= `medium_prob` o senal Elo suficiente.
- `BAJO`: el modelo ve el partido como incierto.

Los umbrales actuales se guardan en `models/confidence_thresholds.json`.
En el ultimo entrenamiento, `ALTO` acerto **82.5%** en test sobre 40 partidos.
Target aspiracional de >80% alcanzado en Fase H.

## Estado Real Actual

Ultimo entrenamiento: `2026-06-12T09:20:41` (Fase J).

| Campo | Valor |
|---|---:|
| Partidos fuente | 8,273 |
| Train samples | 11,582 |
| Validation samples | 1,241 |
| Test samples | 1,241 |
| `accuracy_voting` | 50.36% |
| `accuracy_lgbm` | 50.77% (mejor individual) |
| `accuracy_catboost` | 50.44% |
| `accuracy_two_stage` | 48.83% (corregido, antes 34.65%) |
| `accuracy_high_confidence` | **81.93%** (n=83) |
| `accuracy_high_elo_diff_200` | 65.17% (n=290) |
| `log_loss_voting` (calibrado) | 1.0002 |
| `log_loss_uncalibrated` | 1.0002 |
| `temperature_scaling` | T=0.9625 |
| `recency_weighted_training` | True |
| Ensemble | RF+XGB+LGBM+CatBoost+Elo+Poisson |

Targets aspiracionales:

- Global W/D/L: >55% (actual: 50.36%).
- Alta confianza: >80% (alcanzado: 82.5%).
- Delta Elo >200: >85% (actual: 65.17%).

## Operacion

Comandos principales:

```powershell
python scripts/run_pipeline.py --fast
python scripts/download_enriched_data.py
python scripts/train_models.py
python scripts/generate_report.py
python cli.py predict "Argentina" "Portugal"
python cli.py simulate-tournament
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Flujo completo:

```powershell
.\run_all.bat
```

Tuning de hiperparametros (opcional, ~3-4 min):

```powershell
python scripts/tune_hyperparams.py --trials 50
```

## Proximas Mejoras Recomendadas (Fase I)

- **Alta prioridad**: calibrar xG rolling por calidad del rival (partidos contra equipos debiles no deben inflar el xG del modelo tanto como partidos contra fuertes).
- **Media prioridad**: correr Optuna 50 trials con objetivo compuesto ya implementado.
- **Media prioridad**: implementar tabla exacta FIFA para cruces de 8 mejores terceros (495 combinaciones posibles).
- **Baja prioridad**: Docker, deploy, tests automatizados de no-leakage, fix `test_report_generation.py`.
