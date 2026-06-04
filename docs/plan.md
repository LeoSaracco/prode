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

## Arquitectura

Componentes principales:

- `config/`: settings, aliases, grupos oficiales y pesos de features.
- `src/data/`: collectors, cache, pipeline y validadores.
- `src/features/`: construccion de features para prediccion.
- `src/models/`: Poisson, XGBoost, LightGBM, Elo y ensemble.
- `src/simulation/`: simulacion de partidos, grupos y torneo.
- `src/output/`: formateadores de consola.
- `api/`: FastAPI, schemas, routers y carga de runtime.
- `frontend/`: UI web para consumir la API.

## Features del Modelo

El modelo usa 17 variables diferenciales por partido:

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

## Ensemble

Blend actual:

```text
XGBoost: 35%
LightGBM: 30%
Elo: 20%
Poisson: 15%
```

El modelo Poisson tambien se usa para goles esperados, marcadores probables y simulaciones Monte Carlo.

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
