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
4. Entrenar modelos base.
5. Calcular pesos del ensemble en validation.
6. Evaluar una sola vez en test cronologico.
7. Guardar modelos, pesos, umbrales de confianza y metadata.

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
| `squad_depth_diff` | Diferencia aproximada de profundidad/calidad de plantel. |
| `consistency_diff` | Diferencia de regularidad reciente. |
| `wc_history_diff` | Diferencia de historia/rendimiento mundialista. |
| `market_value_diff` | Diferencia de valor de mercado relativo. |
| `tactical_advantage` | Feature combinada de ventaja tactica. |
| `h2h_advantage` | Historial directo; hoy queda neutral si no hay dato confiable. |
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

Modelos base:

- RandomForest
- XGBoost
- LightGBM
- CatBoost
- Elo baseline

Ensemble principal:

```text
RF + XGB + LGBM + CatBoost + Elo -> accuracy-weighted voting
```

Los pesos se calculan con accuracy en validation:

```text
peso_modelo = max(0.05, accuracy_val - 0.33)
```

Luego se normalizan para sumar 1.0.

Modelos auxiliares:

- `TwoStageClassifier`: separa empate/no empate y luego win/loss.
- `ConfederationModels`: RandomForest por pares de confederaciones.
- `PoissonGoalModel`: estima xG y marcadores probables.

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
| `accuracy_voting` | Acierto del ensemble principal. Es la metrica global mas importante. |
| `accuracy_two_stage` | Acierto del modelo auxiliar de dos etapas. |
| `accuracy_confederation` | Acierto del modelo especializado por confederacion. |
| `accuracy_high_confidence` | Acierto solo donde la prediccion fue marcada `ALTO`. |
| `accuracy_high_elo_diff_200` | Acierto en partidos con diferencia Elo de al menos 200 puntos. |

Otras metricas:

- `n_high_confidence_matches`: cantidad de casos usados para medir `accuracy_high_confidence`.
- `n_high_elo_matches`: cantidad de casos usados para medir `accuracy_high_elo_diff_200`.
- `log_loss_voting`: calidad de las probabilidades; mas bajo es mejor.
- `val_accuracies`: aciertos por modelo en validation.
- `voting_weights`: peso final de cada modelo en el ensemble.

## Confianza

La etiqueta de confianza se basa en la probabilidad maxima del ensemble:

- `ALTO`: probabilidad maxima >= `high_prob`.
- `MEDIO`: probabilidad maxima >= `medium_prob` o senal Elo suficiente.
- `BAJO`: el modelo ve el partido como incierto.

Los umbrales actuales se guardan en `models/confidence_thresholds.json`:

```json
{
  "high_prob": 0.70,
  "medium_prob": 0.65,
  "target_high_accuracy": 0.75
}
```

Importante: `ALTO` no significa seguro. En la corrida actual, `ALTO` acerto
69.70% en test sobre 33 partidos.

## Estado Real Actual

Ultimo entrenamiento: `2026-06-04T12:14:51` (Fase G).

| Campo | Valor |
|---|---:|
| Partidos fuente | 3,221 |
| Train samples | 4,508 |
| Validation samples | 483 |
| Test samples | 484 |
| FIFA rankings rows | 14,853 |
| Kaggle Elo rows | 1,991 |
| `accuracy_voting` | 50.62% |
| `accuracy_catboost` | 51.65% |
| `accuracy_high_confidence` | 70.73% (n=41) |
| `accuracy_high_elo_diff_200` | 61.95% (n=113) |
| `log_loss_voting` (calibrado) | 1.0337 |
| `log_loss_uncalibrated` | 1.0364 |
| `temperature_scaling` | T=1.0447 |
| `recency_weighted_training` | True |

Targets aspiracionales:

- Global W/D/L: >55%.
- Alta confianza: >80%.
- Delta Elo >200: >85%.

Estos targets todavia no estan alcanzados. Proximo objetivo: >52% con fix de h2h y datos Kaggle.

## Operacion

Comandos principales:

```powershell
python scripts/run_pipeline.py --fast
python scripts/download_enriched_data.py
python scripts/train_models.py
python scripts/generate_report.py
python cli.py predict "Argentina" "Portugal"
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

## Proximas Mejoras Recomendadas (Fase H)

- **Alta prioridad**: arreglar `h2h_advantage` (siempre 0) y `consistency_diff` (duplicado de `form_5_diff`).
- **Alta prioridad**: fix collector Kaggle match features para parsear columnas `_home_team`/`_away_team`/`_date`.
- **Media prioridad**: correr Optuna 50 trials con objetivo compuesto ya implementado.
- **Media prioridad**: incorporar probabilidades Poisson al ensemble como modelo 6.
- **Baja prioridad**: Docker, deploy, tests automatizados de no-leakage.
