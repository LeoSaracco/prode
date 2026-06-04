# prode-ML - FIFA World Cup 2026 Predictor

Sistema de predicciones para el Mundial FIFA 2026. Combina Elo ratings,
modelos Poisson, XGBoost, LightGBM y simulaciones Monte Carlo.

## Estado Actual

- Modelos entrenados y guardados en `models/`.
- Ultimo entrenamiento registrado en `models/model_metadata.json`.
- CLI interactivo disponible en `cli.py`.
- API REST FastAPI implementada en `api/`.
- Frontend React/Vite implementado en `frontend/`.
- Grupos oficiales configurados en `config/wc2026_groups.py`.
- La simulacion de grupos muestra tabla de clasificacion probable y el marcador exacto mas probable de cada partido del grupo.

Nota: las metricas actuales del entrenamiento no alcanzan todavia los targets aspiracionales originales. Ver `models/model_metadata.json` para las metricas reales.

## Instalacion

```powershell
cd "C:\Users\Leandro\Documents\prode-ML"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

En Windows, si aparecen errores de encoding al imprimir tablas, ejecutar los comandos con:

```powershell
$env:PYTHONIOENCODING='utf-8'
```

## Uso Rapido

### Ejecutar pipeline completo, entrenar y levantar app

Desde Git Bash, WSL o una terminal con Bash:

```bash
bash scripts/run_full_stack.sh --fast
```

Ese comando ejecuta el flujo completo:

1. Instala dependencias si faltan.
2. Corre `scripts/run_pipeline.py` para actualizar datos.
3. Corre `scripts/validate_data.py`.
4. Corre `scripts/train_models.py` para regenerar los modelos en `models/`.
5. Levanta la API en `http://127.0.0.1:8000`.
6. Levanta el frontend en `http://127.0.0.1:5173`.

Opciones utiles:

```bash
bash scripts/run_full_stack.sh --fast --force
bash scripts/run_full_stack.sh --skip-pipeline --skip-train
bash scripts/run_full_stack.sh --api-port 8001 --front-port 5174
bash scripts/run_full_stack.sh --no-install
```

El script falla antes de levantar servicios si falla el pipeline, la validacion o el entrenamiento. Asi se evita que el backend consuma modelos viejos por accidente.

### Prediccion de un partido

```powershell
python cli.py predict "Argentina" "Portugal"
python cli.py predict "Brazil" "France"
python cli.py predict "Espana" "Alemania"
```

### Simular fase de grupos

```powershell
python cli.py simulate-group J
```

Ejemplo de salida esperada para el Grupo J:

```text
GRUPO J | 100,000 simulaciones
Argentina   1ro 38.3%  2do 26.0%  Clasifica 64.3%
Austria     1ro 29.3%  2do 30.2%  Clasifica 59.5%
Algeria     1ro 16.6%  2do 23.0%  Clasifica 39.6%
Jordan      1ro 15.8%  2do 20.8%  Clasifica 36.6%

RESULTADOS MAS PROBABLES - GRUPO J
Argentina vs Austria      1-0  (12.8%)
Argentina vs Algeria      2-0  (11.4%)
Argentina vs Jordan       2-0  (13.1%)
Austria vs Algeria        1-1  (10.6%)
Austria vs Jordan         1-0  (11.2%)
Algeria vs Jordan         1-1  (11.0%)
```

Los porcentajes exactos pueden variar segun los modelos entrenados y `n_sims`.

### Simular torneo completo

```powershell
python cli.py simulate-tournament
python cli.py simulate-tournament --top 10
```

### Ver equipos disponibles

```powershell
python cli.py list-teams
```

### Modo interactivo

```powershell
python cli.py
```

Comandos disponibles:

```text
predict Argentina Portugal
group J
tournament
top 10
list
quit
```

## API REST

Iniciar la API:

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Endpoints principales:

- `GET /health`
- `GET /api/v1/teams`
- `GET /api/v1/groups`
- `POST /api/v1/predict`
- `GET /api/v1/simulate/group/{group_name}?n_sims=10000`
- `GET /api/v1/simulate/tournament?n_sims=5000&top_n=20`

La respuesta de `GET /api/v1/simulate/group/J` incluye:

```json
{
  "group": "J",
  "n_sims": 10000,
  "results": [
    {
      "team": "Argentina",
      "group": "J",
      "prob_1st": 0.383,
      "prob_2nd": 0.260,
      "prob_3rd": 0.198,
      "prob_4th": 0.159,
      "qualify_direct_prob": 0.643,
      "avg_pts": 4.89,
      "avg_gd": 1.18
    }
  ],
  "fixtures": [
    {
      "team_a": "Argentina",
      "team_b": "Austria",
      "expected_goals": {"team_a": 1.42, "team_b": 1.08},
      "most_likely_scoreline": {"goals_a": 1, "goals_b": 0, "probability": 0.128}
    }
  ]
}
```

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

Build de produccion:

```powershell
npm run build
```

La vista de grupos muestra los equipos del grupo, la tabla de simulacion y los 6 partidos con su marcador mas probable.

## Datos y Entrenamiento

Pipeline rapido:

```powershell
python scripts/run_pipeline.py --fast
```

Pipeline completo:

```powershell
python scripts/run_pipeline.py
```

Entrenamiento:

```powershell
python scripts/train_models.py
```

Validacion de datos:

```powershell
python scripts/validate_data.py
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

## Accuracy

Los targets originales del proyecto eran aspiracionales:

- Global W/D/L: mayor a 55%
- Alta confianza: mayor a 80%
- Delta Elo mayor a 200: mayor a 85%

El entrenamiento actual funciona y genera modelos, pero las metricas reales deben consultarse en `models/model_metadata.json` antes de tomar decisiones.
