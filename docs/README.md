# prode-ML - FIFA World Cup 2026 Predictor

Sistema de predicciones para el Mundial FIFA 2026. Combina Elo ratings,
modelos Poisson, RandomForest, XGBoost, LightGBM, CatBoost, TwoStage,
Confederation y simulaciones Monte Carlo.

## Estado Actual

- **5 modelos base**: RandomForest, XGBoost, LightGBM, CatBoost, Elo
- **Ensemble**: accuracy-weighted voting (aprende pesos del val set)
- **TwoStage**: prediccion draw/no-draw binaria (8 features especificas)
- **Confederation**: 14 modelos RF por par de confederaciones
- **21 features** con features contextuales (torneo, WC, localia)
- **3,198 partidos** de entrenamiento (2000-2026, 47 selecciones)
- **Split cronologico** 70/15/15 sin data leakage
- **Pipeline rapido**: `run_all.bat` (~2-3 min total)
- **Reportes PDF**: predicciones, simulaciones, ranking de campeones
- **API REST** FastAPI en `api/`
- **Frontend** React/Vite en `frontend/`
- **CLI** interactivo en `cli.py`

Metricas reales en `models/model_metadata.json`.

## Histórico de Versiones

| Fase | Commit | Qué cambió | Métrica clave |
|------|--------|-----------|---------------|
| **Inicial** | `83da947` | XGB+LGBM+Elo, 17 features, pesos fijos, 1,413 matches, split con leakage | Blend: **42.0%** |
| **A** | `47a14ed` | Split cronologico sin leakage, Elo computacional (6,396 filas), FIFA rankings, features contextuales (21), 3,198 matches, metricas segmentadas | — |
| **B** | `fba0c5e` | RandomForest + CatBoost, meta-learner LogisticRegressionCV, feature selection, 21 features refinados | Mejor modelo: **CatBoost 49.0%** |
| **C** | `7259fd4` | Optuna 30+ params, modelos parametrizables, best_params.json, study SQLite | — |
| **D** | `b621c05` | TwoStageClassifier binario, 14 ConfederationModels por par de confederacion | — |
| **E** | `b9d12b8` | Weighted voting reemplaza LR, entrenamiento paralelo (ThreadPool), Poisson maxiter=200, TwoStage 8-feature mask, pipeline sin StatsBomb/Kaggle rotos | Voting: **>=49%** |

### Comparativa inicial vs actual

| | Inicial | Actual |
|---|---|---|
| Partidos training | 1,413 | **3,198** |
| Split | Aleatorio con leakage | **Cronologico sin leakage** |
| Features | 17 | **21** |
| Modelos base | 2 (XGB, LGBM) | **4 (RF, XGB, LGBM, CatBoost)** |
| Ensemble | Pesos fijos (42%) | **Weighted voting (>=49%)** |
| Entrenamiento | Secuencial, ~4 min (colgado) | **Paralelo, ~2 min** |
| Pipeline | ~3 min (StatsBomb colgado) | **~20s** |
| Reportes | No | **PDF automatico** |

## Instalacion

```powershell
cd "C:\Users\Leandro\Documents\prode-ML"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

En Windows, si aparecen errores de encoding:

```powershell
$env:PYTHONIOENCODING='utf-8'
```

## Uso Rapido

### Pipeline completo (recomendado)

```powershell
.\run_all.bat
```

El script ejecuta 7 pasos:
1. Activar venv + instalar dependencias
2. Pipeline de datos (~20s)
3. Validar datos
4. Entrenar modelos (paralelo, ~90s)
5. Generar PDF de predicciones en `reports/`
6. Levantar API en `http://127.0.0.1:8000`
7. Levantar frontend en `http://127.0.0.1:5173`

Si CatBoost se cuelga, el script tiene fallback automatico sin el.

### Solo reentrenar modelos

```powershell
python scripts/train_models.py
```

### Solo generar reporte PDF

```powershell
python scripts/generate_report.py
```

El PDF incluye: predicciones de los 72 partidos, simulacion de grupos, ranking de campeones.

### Prediccion de un partido

```powershell
python cli.py predict "Argentina" "Portugal"
python cli.py predict "Brazil" "France"
```

### Modo interactivo

```powershell
python cli.py
```

Comandos: `predict A B`, `group J`, `tournament`, `top 10`, `list`, `quit`

## Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                21 FEATURES                          │
│  elo_diff, xg_diff, form_5_diff, wc_history_diff,  │
│  tactical_advantage, is_tournament, is_wc, ...      │
└──────────┬──────────────────┬──────────────────────┘
           │                  │
    ┌──────▼──────┐    ┌──────▼──────┐    ┌──────────┐
    │  TWO-STAGE  │    │  ENSEMBLE   │    │  CONFED  │
    │  Draw?→Win? │    │  VOTING     │    │  14 RF   │
    │  (8+21 feat)│    │  RF    30%  │    │  pairs   │
    └──────┬──────┘    │  XGB   20%  │    └────┬─────┘
           │           │  LGBM  20%  │         │
           │           │  CB    20%  │         │
           │           │  Elo   10%  │         │
           │           └──────┬──────┘         │
           └──────────────────┼────────────────┘
                              ▼
                   ┌──────────────────┐
                   │  PREDICCION      │
                   │  + xG + confianza│
                   │  + upset risk    │
                   └──────────────────┘
```

### Como funciona el weighted voting

Cada modelo se evalua en el validation set. El peso se calcula como:

```
peso = max(0.05, accuracy_val - 0.33)
```

Luego se normalizan para que sumen 1.0. Elo siempre pesa 0.10 fijo como baseline.
Los pesos se guardan en `models/voting_weights.json` y se recalculan en cada entrenamiento.

## API REST

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Endpoints: `GET /health`, `GET /api/v1/teams`, `GET /api/v1/groups`,
`POST /api/v1/predict`, `GET /api/v1/simulate/group/{group_name}`,
`GET /api/v1/simulate/tournament`

## Accuracy

Targets del proyecto:

| Metrica | Target |
|---------|--------|
| Global W/D/L | >55% |
| Alta confianza (>65% prob) | >80% |
| Delta Elo >200 | >85% |

Metricas actuales en `models/model_metadata.json`:

```json
{
  "accuracy_rf": 0.xxx,
  "accuracy_xgb": 0.xxx,
  "accuracy_lgbm": 0.xxx,
  "accuracy_catboost": 0.xxx,
  "accuracy_voting": 0.xxx,
  "accuracy_two_stage": 0.xxx,
  "accuracy_confederation": 0.xxx,
  "accuracy_high_elo_diff_200": 0.xxx,
  "log_loss_voting": 0.xxx,
  "split_type": "time_series_chronological",
  "ensemble_architecture": "RF+XGB+LGBM+CB+Elo -> accuracy-weighted voting"
}
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
