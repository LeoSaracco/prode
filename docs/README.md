# prode-ML - FIFA World Cup 2026 Predictor

Sistema de prediccion para el Mundial FIFA 2026. Combina resultados
historicos de selecciones, ratings Elo, rankings FIFA, modelos de ML,
Poisson para marcadores y simulaciones Monte Carlo.

## Estado Actual

- **Datos de entrenamiento**: 3,199 partidos internacionales desde 2000 hasta 2026.
- **Split temporal**: 70% train, 15% validation, 15% test, en orden cronologico.
- **Features sin mirar el futuro**: `rolling_no_future_leakage`.
- **21 features**: Elo, forma reciente, goles a favor/en contra, ranking FIFA, descanso, torneo/localia, historia mundialista y valor de mercado.
- **Modelos base**: RandomForest, XGBoost, LightGBM, CatBoost y Elo.
- **Ensemble principal**: weighted voting, con pesos aprendidos en validation.
- **Calibracion probabilistica**: isotonic regression post-ensemble, ajustada en validation.
- **Pesos de recencia**: partidos recientes pesan mas en el entrenamiento (half-life 3 anos).
- **Modelos auxiliares**: TwoStage (diagnostico, no integrado al ensemble) y ConfederationModels.
- **Poisson**: estima xG y marcadores probables.
- **Runtime**: CLI, API FastAPI, frontend React/Vite y PDF.

Metricas reales: `models/model_metadata.json`.

## Resultado Actual Del Modelo

Ultimo entrenamiento: `2026-06-04T12:14:51`.

| Metrica | Valor | Lectura rapida |
|---|---:|---|
| `accuracy_voting` | **50.62%** | Acierto general del ensemble (probs calibradas con temperature scaling). |
| `accuracy_catboost` | **51.65%** | Mejor modelo individual en esta corrida. |
| `accuracy_lgbm` | 49.38% | Acierto de LightGBM solo. |
| `accuracy_rf` | 47.52% | Acierto de RandomForest solo. |
| `accuracy_xgb` | 48.14% | Acierto de XGBoost solo. |
| `accuracy_confederation` | 48.35% | Acierto del modelo por confederaciones. |
| `accuracy_two_stage` | 33.06% | Diagnostico solamente; por debajo del azar, no integrado al ensemble. |
| `accuracy_high_confidence` | **70.73%** | Acierto cuando el sistema marca confianza `ALTO`. |
| `n_high_confidence_matches` | 41 | Cantidad de partidos test que entraron en confianza `ALTO`. |
| `accuracy_high_elo_diff_200` | 61.95% | Acierto en partidos con diferencia Elo >= 200 (n=113). |
| `log_loss_voting` | **1.0337** | Calidad de probabilidades calibradas; mas bajo es mejor. |
| `log_loss_uncalibrated` | 1.0364 | Log-loss sin calibrar, para comparacion. |
| `temperature_scaling` | T=1.0447 | Factor de calibracion (>1 suaviza probabilidades). |

Interpretacion: el modelo ya supera bastante el azar de 33.3% para un problema
de tres clases (gana A / empate / gana B), pero todavia no alcanza los targets
aspiracionales originales (>55% global y >80% en alta confianza).

## Que Significa Cada Metrica

- `accuracy_rf`: porcentaje de partidos del test donde RandomForest acerto la clase final.
- `accuracy_xgb`: porcentaje de aciertos de XGBoost solo.
- `accuracy_lgbm`: porcentaje de aciertos de LightGBM solo.
- `accuracy_catboost`: porcentaje de aciertos de CatBoost solo.
- `accuracy_voting`: porcentaje de aciertos del ensemble principal. Combina RF, XGB, LGBM, CatBoost y Elo con pesos aprendidos en validation.
- `accuracy_two_stage`: porcentaje de aciertos del modelo auxiliar que primero decide si hay empate y luego decide ganador/perdedor.
- `accuracy_confederation`: porcentaje de aciertos de los modelos entrenados por pares de confederaciones, por ejemplo UEFA-CONMEBOL o AFC-AFC.
- `accuracy_high_confidence`: porcentaje de aciertos solo en predicciones donde el sistema dijo `ALTO`.
- `n_high_confidence_matches`: cuantas predicciones del test fueron consideradas `ALTO`. Si este numero es bajo, la metrica puede variar mucho.
- `accuracy_high_elo_diff_200`: porcentaje de aciertos en partidos donde una seleccion tenia al menos 200 puntos Elo de diferencia contra la otra.
- `log_loss_voting`: mide que tan buenas son las probabilidades, no solo la clase ganadora. Penaliza fuerte cuando el modelo esta muy seguro y se equivoca.
- `val_accuracies`: accuracy de cada modelo en validation. Se usa para calcular los pesos del voting.
- `voting_weights`: pesos finales del ensemble. Si un modelo valida mejor, pesa mas.
- `confidence_thresholds`: umbrales usados para decir `BAJO`, `MEDIO` o `ALTO`.

Ejemplo simple: si `accuracy_voting = 50.21%`, significa que de cada 100 partidos
del conjunto de test, el ensemble acerto aproximadamente 50 resultados W/D/L.

## Confianza BAJO / MEDIO / ALTO

La confianza no significa "certeza absoluta". Es una etiqueta operacional:

- `ALTO`: la probabilidad maxima del modelo supera el umbral calibrado actual (`0.70`).
- `MEDIO`: supera el umbral medio (`0.65`) o hay senal fuerte de Elo.
- `BAJO`: el modelo ve el partido como mas parejo o incierto.

Los umbrales se guardan en `models/confidence_thresholds.json`. En el ultimo
entrenamiento, validation sugeria que `ALTO` podia acertar cerca de 72.73%, y en
test obtuvo 69.70% sobre 33 partidos. Por eso todavia no debe interpretarse como
un pronostico garantizado.

## Arquitectura Del Entrenamiento

1. Se ordenan partidos por fecha.
2. Para cada partido se calculan features usando solo informacion anterior a ese partido.
3. Solo el train se duplica con perspectiva inversa (A vs B y B vs A).
4. Se entrenan RF, XGB, LGBM y CatBoost en paralelo.
5. Se calcula la accuracy de cada modelo en validation.
6. Se crean pesos del ensemble:

```text
peso_modelo = max(0.05, accuracy_val - 0.33)
```

7. Los pesos se normalizan para sumar 1.0.
8. Se evalua una sola vez en test cronologico.

Artefactos importantes:

- `models/model_metadata.json`: metricas y configuracion de entrenamiento.
- `models/voting_weights.json`: pesos del ensemble.
- `models/confidence_thresholds.json`: umbrales de confianza.
- `models/inference_team_profiles.json`: perfiles rolling usados en inferencia.

## Uso Rapido

```powershell
.\run_all.bat
```

El script:

1. Activa el virtualenv.
2. Verifica dependencias.
3. Corre pipeline rapido de datos.
4. Valida datos.
5. Entrena modelos.
6. Genera PDF.
7. Levanta API y frontend.

Solo reentrenar:

```powershell
python scripts/train_models.py
```

Prediccion puntual:

```powershell
python cli.py predict "Argentina" "Portugal"
python cli.py predict "Brazil" "France"
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

- El target de >55% global todavia no esta alcanzado.
- `accuracy_high_confidence` esta debajo del objetivo de 80%.
- FIFA rankings ayudan, pero no resuelven calibracion por si solos.
- No hay datos robustos de lesiones, convocatoria final o minutos recientes de jugadores.
- Las metricas se basan en partidos historicos; el Mundial real tendra condiciones distintas.
