# prode-ML - FIFA World Cup 2026 Predictor

Sistema de prediccion para el Mundial FIFA 2026. Combina resultados
historicos de selecciones, ratings Elo, rankings FIFA, modelos de ML,
Poisson para marcadores y simulaciones Monte Carlo.

## Estado Actual

- **Datos de entrenamiento**: 8,273 partidos internacionales (dataset ampliado con Kaggle match features).
- **Split temporal**: 70% train, 15% validation, 15% test, en orden cronologico.
- **Features sin mirar el futuro**: `rolling_no_future_leakage`.
- **27 features**: Elo, forma reciente, xG/xGA, ranking FIFA, descanso, torneo/localia, historia mundialista, perfiles WC recientes y calidad de plantel desde ratings FIFA de jugadores.
- **Perfiles WC recientes reales**: `wc_recent_goal_balance`, `wc_recent_win_rate` y `wc_knockout_depth` se calculan desde el historial real de partidos mundialistas de cada selección (ventana de 8 anos), ya no son placeholders neutrales.
- **Modelos base**: RandomForest, XGBoost, LightGBM, CatBoost, Elo y Poisson (6 modelos).
- **Ensemble principal**: weighted voting con pesos aprendidos en validation. Poisson con peso fijo 0.08 como senal ortogonal.
- **Calibracion probabilistica**: temperature scaling post-ensemble, ajustado en validation.
- **Pesos de recencia**: partidos recientes pesan mas en el entrenamiento (half-life 3 anos).
- **Modelos auxiliares**: TwoStage (diagnostico, no integrado al ensemble; corregido el 2026-06-12 al quitar `class_weight="balanced"` del clasificador de empates, ahora rinde ~48.8%, en linea con los modelos base) y ConfederationModels.
- **Squad quality**: feature de calidad del plantel basada en ratings FIFA reales (avg_overall, max_overall, depth_ratio) desde `player_aggregates.csv`. Reemplaza el valor de mercado como proxy.
- **Simulacion de torneo**: bracket oficial WC2026 con cruces de pod predefinidos. Mejores terceros seleccionados por rendimiento (pts > gd > gf), no por Elo.
- **Marcadores**: se conserva el marcador exacto modal (`exact_most_likely_scoreline`) y se comunica un marcador recomendado (`outcome_scoreline`) condicionado al resultado predicho y al volumen xG. Esto evita reportes tipo `GANA 0-0` y reduce el sesgo conservador a `1-0`.
- **PDF analitico**: el reporte incluye graficos de candidatos al titulo, distribucion de marcadores, xG vs confianza, ataques con mayor xG, defensas mas solidas/vulnerables, pesos del ensemble y accuracy por umbral de confianza.
- **Runtime**: CLI, API FastAPI, frontend React/Vite y PDF.

Metricas reales: `models/model_metadata.json`.

## Resultado Actual Del Modelo

Ultimo entrenamiento: `2026-06-12T09:20:41`.

| Metrica | Valor | Lectura rapida |
|---|---:|---|
| `accuracy_voting` | **50.36%** | Acierto general del ensemble de 6 modelos. |
| `accuracy_lgbm` | 50.77% | Mejor modelo individual en esta corrida. |
| `accuracy_catboost` | 50.44% | Acierto de CatBoost solo. |
| `accuracy_rf` | 50.04% | Acierto de RandomForest solo. |
| `accuracy_xgb` | 48.03% | Acierto de XGBoost solo. |
| `accuracy_two_stage` | 48.83% | Acierto del modelo auxiliar de dos etapas (corregido el 2026-06-12, antes 34.65%). |
| `accuracy_high_confidence` | **81.93%** | Acierto cuando el sistema marca confianza `ALTO`. Target >80% alcanzado. |
| `n_high_confidence_matches` | 83 | Cantidad de partidos test que entraron en confianza `ALTO`. |
| `accuracy_high_elo_diff_200` | **65.17%** | Acierto en partidos con diferencia Elo >= 200 (n=290). |
| `log_loss_voting` | **1.0002** | Calidad de probabilidades calibradas; mas bajo es mejor. |
| `log_loss_uncalibrated` | 1.0002 | Log-loss sin calibrar, para comparacion. |
| `temperature_scaling` | T=0.9625 | Factor de calibracion (proximo a 1.0: probs bien calibradas). |

Interpretacion: el modelo supera el azar de 33.3% para un problema de tres clases
(gana A / empate / gana B). El target de >80% en alta confianza se mantiene logrado
(81.93%). El target global de >55% accuracy sigue siendo aspiracional.

## Que Significa Cada Metrica

- `accuracy_rf`: porcentaje de partidos del test donde RandomForest acerto la clase final.
- `accuracy_xgb`: porcentaje de aciertos de XGBoost solo.
- `accuracy_lgbm`: porcentaje de aciertos de LightGBM solo.
- `accuracy_catboost`: porcentaje de aciertos de CatBoost solo.
- `accuracy_voting`: porcentaje de aciertos del ensemble principal. Combina RF, XGB, LGBM, CatBoost, Elo y Poisson con pesos aprendidos en validation.
- `accuracy_two_stage`: porcentaje de aciertos del modelo auxiliar de dos etapas. Diagnostico solamente (no integrado al ensemble), pero desde el fix del 2026-06-12 rinde en linea con los modelos base (~48.8%).
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
En el ultimo entrenamiento, `ALTO` acerto 81.93% en test sobre 83 partidos.

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

## Reporte PDF Y Marcadores

El PDF se genera con datos actuales de `models/model_metadata.json` y `MATCH_FEATURE_COLUMNS`, por lo que el conteo de features se muestra dinamicamente. En la corrida actual el reporte dice **27 variables**.

Campos de marcador:

- `exact_most_likely_scoreline`: modo puro de la matriz Poisson/Dixon-Coles, sin forzar compatibilidad con el outcome del clasificador.
- `outcome_scoreline` / `most_likely_scoreline`: marcador recomendado para comunicar. Debe ser compatible con el resultado predicho y representativo del xG total, del xG de cada equipo y de la debilidad defensiva del rival.

Validacion manual actual sobre los 72 partidos de fase de grupos:

- Marcadores recomendados observados: `1-0`, `2-0`, `0-1`, `2-1`, `1-1`, `0-2`, `1-2`, `0-0`, `3-0`.
- Partidos con mas de 2 goles recomendados: 10/72.
- Promedio de goles recomendados: 1.60; promedio de xG total: 1.87.
- Ataques con mayor xG promedio: England, Germany, Spain, Brazil, Argentina.
- Defensas mas vulnerables por xGA permitido: Panama, New Zealand, Ecuador, Cape Verde, Jordan.

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

- El target de >55% global todavia no esta alcanzado (actual: 50.36%).
- Los xG rolling de algunos equipos estan inflados por goleadas en clasificatorias faciles (ej: England con 2.2 xG/partido en datos de entrenamiento). El modelo no filtra por calidad del rival.
- No hay datos de lesiones, convocatoria final o minutos recientes por jugador.
- Las metricas se basan en partidos historicos; el Mundial real tendra condiciones distintas.
- El bracket WC2026 implementado es una aproximacion de pod; la tabla exacta de cruces de terceros de la FIFA requiere investigacion adicional.
