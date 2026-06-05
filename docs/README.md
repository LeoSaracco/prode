# prode-ML - FIFA World Cup 2026 Predictor

Sistema de prediccion para el Mundial FIFA 2026. Combina resultados
historicos de selecciones, ratings Elo, rankings FIFA, modelos de ML,
Poisson para marcadores y simulaciones Monte Carlo.

## Estado Actual

- **Datos de entrenamiento**: 8,273 partidos internacionales (dataset ampliado con Kaggle match features).
- **Split temporal**: 70% train, 15% validation, 15% test, en orden cronologico.
- **Features sin mirar el futuro**: `rolling_no_future_leakage`.
- **21 features**: Elo, forma reciente, goles a favor/en contra, ranking FIFA, descanso, torneo/localia, historia mundialista y calidad de plantel desde ratings FIFA de jugadores.
- **Modelos base**: RandomForest, XGBoost, LightGBM, CatBoost, Elo y Poisson (6 modelos).
- **Ensemble principal**: weighted voting con pesos aprendidos en validation. Poisson con peso fijo 0.08 como senal ortogonal.
- **Calibracion probabilistica**: temperature scaling post-ensemble, ajustado en validation.
- **Pesos de recencia**: partidos recientes pesan mas en el entrenamiento (half-life 3 anos).
- **Modelos auxiliares**: TwoStage (diagnostico, no integrado al ensemble) y ConfederationModels.
- **Squad quality**: feature de calidad del plantel basada en ratings FIFA reales (avg_overall, max_overall, depth_ratio) desde `player_aggregates.csv`. Reemplaza el valor de mercado como proxy.
- **Simulacion de torneo**: bracket oficial WC2026 con cruces de pod predefinidos. Mejores terceros seleccionados por rendimiento (pts > gd > gf), no por Elo.
- **Marcadores**: fallback con Poisson sampling condicionado al resultado predicho; sin marcadores deterministicos.
- **Runtime**: CLI, API FastAPI, frontend React/Vite y PDF.

Metricas reales: `models/model_metadata.json`.

## Resultado Actual Del Modelo

Ultimo entrenamiento: `2026-06-05T10:46:10`.

| Metrica | Valor | Lectura rapida |
|---|---:|---|
| `accuracy_voting` | **51.25%** | Acierto general del ensemble de 6 modelos. |
| `accuracy_catboost` | 50.93% | Mejor modelo individual en esta corrida. |
| `accuracy_lgbm` | 50.36% | Acierto de LightGBM solo. |
| `accuracy_rf` | 48.99% | Acierto de RandomForest solo. |
| `accuracy_xgb` | 47.30% | Acierto de XGBoost solo. |
| `accuracy_high_confidence` | **82.50%** | Acierto cuando el sistema marca confianza `ALTO`. Target >80% alcanzado. |
| `n_high_confidence_matches` | 40 | Cantidad de partidos test que entraron en confianza `ALTO`. |
| `accuracy_high_elo_diff_200` | **65.52%** | Acierto en partidos con diferencia Elo >= 200 (n=290). |
| `log_loss_voting` | **1.0003** | Calidad de probabilidades calibradas; mas bajo es mejor. |
| `log_loss_uncalibrated` | 1.0002 | Log-loss sin calibrar, para comparacion. |
| `temperature_scaling` | T=0.9935 | Factor de calibracion (proximo a 1.0: probs bien calibradas). |

Interpretacion: el modelo supera el azar de 33.3% para un problema de tres clases
(gana A / empate / gana B). El target de >80% en alta confianza fue alcanzado por
primera vez (82.5%). El target global de >55% accuracy sigue siendo aspiracional.

## Que Significa Cada Metrica

- `accuracy_rf`: porcentaje de partidos del test donde RandomForest acerto la clase final.
- `accuracy_xgb`: porcentaje de aciertos de XGBoost solo.
- `accuracy_lgbm`: porcentaje de aciertos de LightGBM solo.
- `accuracy_catboost`: porcentaje de aciertos de CatBoost solo.
- `accuracy_voting`: porcentaje de aciertos del ensemble principal. Combina RF, XGB, LGBM, CatBoost, Elo y Poisson con pesos aprendidos en validation.
- `accuracy_two_stage`: porcentaje de aciertos del modelo auxiliar de dos etapas. Diagnostico solamente.
- `accuracy_confederation`: porcentaje de aciertos de los modelos entrenados por pares de confederaciones.
- `accuracy_high_confidence`: porcentaje de aciertos solo en predicciones donde el sistema dijo `ALTO`.
- `n_high_confidence_matches`: cuantas predicciones del test fueron consideradas `ALTO`. Si este numero es bajo, la metrica puede variar mucho.
- `accuracy_high_elo_diff_200`: porcentaje de aciertos en partidos donde una seleccion tenia al menos 200 puntos Elo de diferencia contra la otra.
- `log_loss_voting`: mide que tan buenas son las probabilidades, no solo la clase ganadora. Penaliza fuerte cuando el modelo esta muy seguro y se equivoca.
- `val_accuracies`: accuracy de cada modelo en validation. Se usa para calcular los pesos del voting.
- `voting_weights`: pesos finales del ensemble. Si un modelo valida mejor, pesa mas. Elo fijo en 0.10, Poisson fijo en 0.07.
- `confidence_thresholds`: umbrales usados para decir `BAJO`, `MEDIO` o `ALTO`.

## Confianza BAJO / MEDIO / ALTO

La confianza no significa "certeza absoluta". Es una etiqueta operacional:

- `ALTO`: la probabilidad maxima del modelo supera el umbral calibrado actual.
- `MEDIO`: supera el umbral medio o hay senal fuerte de Elo.
- `BAJO`: el modelo ve el partido como mas parejo o incierto.

Los umbrales se guardan en `models/confidence_thresholds.json`.
En el ultimo entrenamiento, `ALTO` acerto 82.5% en test sobre 40 partidos.

## Arquitectura Del Entrenamiento

1. Se ordenan partidos por fecha.
2. Para cada partido se calculan features usando solo informacion anterior a ese partido.
3. Solo el train se duplica con perspectiva inversa (A vs B y B vs A).
4. Se entrenan RF, XGB, LGBM y CatBoost en paralelo.
5. Se entrena Poisson sobre datos de train y se evalua W/D/L implicito en validation.
6. Se calcula la accuracy de cada modelo en validation.
7. Se crean pesos del ensemble:

```text
peso_modelo = max(0.05, accuracy_val - 0.33)
peso_elo    = 0.10  (fijo, baseline estable)
peso_poisson = 0.07 (fijo, senal ortogonal de distribucion de goles)
```

8. Los pesos se normalizan para sumar 1.0.
9. Se evalua una sola vez en test cronologico.

Artefactos importantes:

- `models/model_metadata.json`: metricas y configuracion de entrenamiento.
- `models/voting_weights.json`: pesos del ensemble.
- `models/confidence_thresholds.json`: umbrales de confianza.
- `models/inference_team_profiles.json`: perfiles rolling usados en inferencia.
- `models/inference_h2h_stats.json`: historial directo entre pares de equipos.

## Calidad Del Plantel (Squad Quality)

El feature `squad_depth_diff` ya no usa valor de mercado como proxy lineal.
Ahora se computa desde ratings FIFA reales de jugadores:

| Sub-feature | Que captura |
|---|---|
| `avg_overall` | Calidad media del plantel (FIFA ratings, rango ~60-88) |
| `max_overall` | Rating del mejor jugador (Messi, Mbappe, etc.) |
| `depth_ratio` | avg / max: robustez sin la estrella. <0.90 = muy dependiente de una figura |
| `avg_shooting` | Capacidad ofensiva del plantel |
| `avg_defending` | Solidez defensiva del plantel |

Fuente: `data/raw/kaggle/match_features/player_aggregates.csv` (FIFA 15-24).
Se usa la version mas reciente disponible por pais. Fallback en `PLAYER_RATINGS_FALLBACK`
para los 4 paises WC2026 que no estan en el CSV (Czechia, Turkiye, Cote d'Ivoire, Cape Verde).

## Simulacion Del Torneo

El simulador usa el bracket oficial WC2026:

- **12 partidos de pod fijos** en Round of 32: Ganador A vs 2do B, Ganador B vs 2do A, etc. Los 12 grupos se organizan en 6 pods: (A,B), (C,D), (E,F), (G,H), (I,J), (K,L).
- **4 partidos de bridge**: los 8 mejores terceros se cruzan entre si.
- **Mejores terceros por rendimiento**: pts > gd > gf, no por Elo.
- **Arbol fijo**: R16, QF, SF y Final siguen posiciones predefinidas. No se baraja entre rondas.
- **Conteos por ronda**: cada columna de probabilidad representa "llego a esa ronda" (32/16/8/4/2/1 equipos).

## Uso Rapido

```powershell
.\run_all.bat
```

El script:

1. Activa el virtualenv.
2. Verifica dependencias (instala si faltan).
3. Corre pipeline rapido de datos.
4. Descarga datos enriquecidos.
5. Valida datos.
6. Entrena modelos.
7. Genera PDF.
8. Levanta API (puerto 8000) y frontend (puerto 5173).

Solo reentrenar:

```powershell
python scripts/train_models.py
```

Prediccion puntual:

```powershell
python cli.py predict "Argentina" "Portugal"
python cli.py predict "Brazil" "France"
```

Simulacion del torneo:

```powershell
python cli.py simulate-tournament
```

API:

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## Grupos Oficiales 2026

- A: Mexico, South Africa, South Korea, Czechia
- B: Canada, Switzerland, Qatar, Bosnia and Herzegovina
- C: Brazil, Morocco, Scotland, Haiti
- D: United States, Paraguay, Australia, Turkiye
- E: Germany, Ecuador, Cote d'Ivoire, Curacao
- F: Netherlands, Sweden, Japan, Tunisia
- G: Belgium, Egypt, Iran, New Zealand
- H: Spain, Uruguay, Saudi Arabia, Cape Verde
- I: France, Senegal, Norway, Iraq
- J: Argentina, Austria, Algeria, Jordan
- K: Portugal, Colombia, Uzbekistan, DR Congo
- L: England, Croatia, Ghana, Panama

## Limitaciones Actuales

- El target de >55% global todavia no esta alcanzado (actual: 51.25%).
- Los xG rolling de algunos equipos estan inflados por goleadas en clasificatorias faciles (ej: England con 2.2 xG/partido en datos de entrenamiento). El modelo no filtra por calidad del rival.
- No hay datos de lesiones, convocatoria final o minutos recientes por jugador.
- Las metricas se basan en partidos historicos; el Mundial real tendra condiciones distintas.
- El bracket WC2026 implementado es una aproximacion de pod; la tabla exacta de cruces de terceros de la FIFA requiere investigacion adicional.
