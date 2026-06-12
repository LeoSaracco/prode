# Contexto de despliegue: GitHub Pages sin backend (Fase K)

> Este documento es un **handoff** para que otro agente implemente el
> despliegue del frontend de prode-ML a GitHub Pages vía GitHub Actions.
> Está escrito para ser autocontenido: no asume acceso a conversaciones
> previas. Si algo referenciado aquí (rutas, funciones, nombres) ya no
> existe o cambió, verificalo en el código antes de asumir que sigue así.

## 1. Objetivo y restricciones

El usuario quiere publicar el frontend (React + Vite + TS, en `frontend/`) en
GitHub Pages, desplegado mediante un workflow de GitHub Actions. Restricción
explícita y no negociable:

> El sitio publicado **no debe depender de un backend Python en vivo**. Los
> datos que hoy vienen de la API FastAPI deben generarse **localmente**
> (de antemano, antes de hacer push) llamando **in-process** a la lógica de
> predicción/simulación de Python — sin levantar un servidor HTTP — y
> guardarse como **JSON estático**. El frontend, en su build para GitHub
> Pages, debe leer esos JSON estáticos en lugar de hacer `fetch` a un backend.

El workflow de GitHub Actions en sí **solo construye y despliega el
frontend** (Node/Vite). No necesita Python, no entrena modelos, no levanta
`uvicorn`.

## 2. Arquitectura actual (resumen para no tener que explorar)

### 2.1 Frontend → API

`frontend/src/api.ts` expone 7 funciones, todas contra
`${import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1"}`:

| Función | Endpoint | Parámetros | Naturaleza |
|---|---|---|---|
| `fetchTeams()` | `GET /teams` | ninguno | estático (48 equipos) |
| `fetchGroups()` | `GET /groups` | ninguno | estático (12 grupos A-L) |
| `predictMatch(teamA, teamB)` | `POST /predict` | `{team_a, team_b}` | **dinámico**: el usuario elige cualquier par de los 48 equipos vía `TeamCombobox` → 48×47 = 2256 combinaciones ordenadas posibles |
| `fetchFixtures()` | `GET /fixtures` | ninguno | estático (72 partidos de fase de grupos con fecha real) |
| `simulateGroup(group, nSims?)` | `GET /simulate/group/{group}?n_sims=...` | `group` ∈ {A..L}, `n_sims` (default 5000) | 12 grupos enumerables |
| `simulateTournament(nSims?, topN?)` | `GET /simulate/tournament?n_sims=...&top_n=...` | n_sims=2500, top_n=20 por defecto | sin más variantes de parámetros en el frontend |
| `simulateTournamentBracket(nSims?)` | `GET /simulate/tournament?output=bracket&n_sims=...` | n_sims=5000 por defecto | variante "bracket" del mismo endpoint |

Cada función hace `fetch(...)` y devuelve JSON tipado (los tipos TS ya están
definidos en `api.ts` — reusalos para los archivos estáticos).

### 2.2 Lógica Python in-process (sin HTTP)

El patrón de referencia es **`scripts/generate_report.py`** (genera el PDF
analítico sin levantar un servidor). Usa:

```python
from src.runtime import load_prediction_runtime, predict_match
from src.simulation.group_simulator import GroupSimulator
from src.simulation.tournament_simulator import TournamentSimulator

runtime = load_prediction_runtime()

result, shap_features = predict_match(runtime, team_a, team_b)
# o, sin SHAP/auxiliares (mucho más rápido, usado por fixtures/simulación):
result, _ = predict_match(runtime, team_a, team_b, include_shap=False, include_aux=False)

simulator = GroupSimulator(poisson_model=runtime.poisson, predictor=lambda a, b: runtime.cached_predict(a, b))
df = simulator.simulate_group(group_name, n_sims=10000)

tournament_sim = TournamentSimulator(...)  # ver firma exacta en src/simulation/tournament_simulator.py
```

- `load_prediction_runtime()` (en `src/runtime.py`) carga todos los modelos
  (RF/XGB/LGBM/CatBoost/Elo/Poisson/TwoStage/Confederation), el scaler, los
  perfiles de equipo (`models/inference_team_profiles.json`) y el historial
  H2H (`models/inference_h2h_stats.json`).
- `runtime.pair_cache` / `runtime.cached_predict(a, b)`: cache en memoria de
  predicciones por par de equipos — usalo para no recomputar si vas a llamar
  `predict_match` para los mismos pares más de una vez.
- Para reproducir exactamente la forma de respuesta de cada endpoint, mirá
  cómo la arman los routers en `api/routers/predictions.py` (función
  `_format_prediction` u homóloga) y `api/routers/simulation.py` — el JSON
  estático debe tener la **misma forma** que hoy devuelve la API, para que
  `api.ts` no tenga que cambiar los tipos.

### 2.3 Equipos y grupos

`config/wc2026_groups.py` define:
- `GROUPS`: dict de 12 grupos (A-L), 4 equipos cada uno = 48 equipos totales.
- `CONFEDERATION`, `WC_HISTORY_SCORE`, `get_group_for_team()`,
  `get_group_opponents()`.

### 2.4 Modelos y `.gitignore`

`models/*.pkl`, `models/*.json`, `models/*.txt`, `models/*.ubj` están en
`.gitignore` (solo `models/.gitkeep` está versionado). Esto significa:

- **No asumas que CI tiene los modelos entrenados.** Re-entrenar en CI tomaría
  ~20-50 minutos (tiempo observado localmente para `scripts/train_models.py`)
  y requeriría todo `requirements-api.txt` + datasets.
- **Recomendación**: el script de export estático (sección 3) se corre
  **localmente**, donde sí existen los `.pkl`/`.json` entrenados. Su output
  (los JSON estáticos para el frontend) se commitea al repo. El workflow de
  GitHub Actions (sección 5) entonces solo necesita Node/npm — nunca toca
  Python ni `models/`.

### 2.5 Build del frontend

- `frontend/package.json`: `build` = `tsc && vite build` → genera
  `frontend/dist/`.
- `frontend/vite.config.ts`: **no tiene `base` configurado** (default `/`).
  Para GitHub Pages bajo `https://<usuario>.github.io/<repo>/`, Vite necesita
  `base: '/<repo>/'` — si no, todos los assets (`/assets/...`, `/data/...`)
  van a 404 porque el navegador los busca en la raíz del dominio.

## 3. Script de export estático (a crear)

**Ruta sugerida**: `scripts/export_static_data.py`.

**Salida sugerida**: `frontend/public/data/` (Vite copia todo el contenido de
`public/` a `dist/` tal cual, preservando la ruta — en runtime queda accesible
como `${BASE_URL}data/<archivo>.json`).

Archivos a generar:

| Archivo | Contenido | Fuente |
|---|---|---|
| `teams.json` | `{teams: [...]}` — igual forma que `GET /teams` | estático, desde `config/wc2026_groups.py` o el mismo código que usa el router `/teams` |
| `groups.json` | `{groups: {...}}` — igual forma que `GET /groups` | `config/wc2026_groups.py` (`GROUPS`) |
| `fixtures.json` | igual forma que `GET /fixtures` (72 partidos con fecha, grupo, predicción) | misma lógica que `api/routers/predictions.py` (`_load_wc_fixtures`, `_compute_matchday`) + `predict_match(..., include_shap=False, include_aux=False)` vía `cached_predict` |
| `predictions.json` | dict consolidado `"TeamA|TeamB": MatchResult` para **las 2256 combinaciones ordenadas** (48×47, A≠B) | loop sobre `itertools.permutations(GROUPS teams, 2)`, `predict_match(runtime, a, b, include_shap=False, include_aux=False)` |
| `groups/A.json` ... `groups/L.json` (12 archivos) | resultado de `GroupSimulator.simulate_group(group, n_sims=...)` para cada grupo | `src/simulation/group_simulator.py` |
| `tournament.json` | igual forma que `GET /simulate/tournament` (lista, n_sims=2500, top_n=20) | `TournamentSimulator` |
| `tournament_bracket.json` | igual forma que `GET /simulate/tournament?output=bracket` (n_sims=5000) | `TournamentSimulator.simulate_bracket()` (ver Fase I en `docs/backlog.md`) |

### Notas sobre `predictions.json` (el archivo más grande/costoso)

- **2256 llamadas a `predict_match`** con `include_shap=False,
  include_aux=False` (la Fase I dice que esto ahorra ~40% por llamada
  respecto de incluir SHAP/auxiliares). Aun así, 2256 llamadas secuenciales
  pueden tardar bastante (orden de minutos) — para un script que se corre
  localmente y ocasionalmente, esto es aceptable. Si resulta muy lento,
  considerar `ProcessPoolExecutor` (ya mencionado como pendiente de
  performance en `docs/backlog.md`).
- **Trade-off explícito**: con `include_shap=False`, el campo `top_features`
  (SHAP) queda vacío `[]` en el JSON estático. El panel de predicción del
  frontend debe tolerar esto (ya lo hace en modo "compact" para "Comparar
  partidos" — reusar ese fallback visual).
- Clave del dict: `f"{team_a}|{team_b}"` (mismo orden que el usuario eligió en
  el combobox — A es "local"/primer seleccionado, B el rival).

### Progreso y modo rápido

El script debe imprimir progreso (2256 pares es la parte larga) y aceptar un
flag `--quick` que reduzca `n_sims` de las simulaciones de grupo/torneo para
iteración rápida durante desarrollo.

## 4. Adaptación de `frontend/src/api.ts`

Agregar un flag de build, p.ej. `VITE_STATIC_MODE` (string `"true"`/`"false"`
o ausente). **No romper el modo dev/live actual** — la rama estática es
adicional.

Patrón sugerido (pseudocódigo):

```ts
const STATIC_MODE = import.meta.env.VITE_STATIC_MODE === "true";

async function readStatic<T>(file: string): Promise<T> {
  const res = await fetch(`${import.meta.env.BASE_URL}data/${file}.json`);
  if (!res.ok) throw new Error(`static data not found: ${file}`);
  return res.json();
}

export async function fetchTeams(): Promise<TeamsResponse> {
  if (STATIC_MODE) return readStatic("teams");
  return request("/teams");
}

export async function predictMatch(teamA: string, teamB: string): Promise<MatchResult> {
  if (STATIC_MODE) {
    const all = await readStatic<Record<string, MatchResult>>("predictions");
    const result = all[`${teamA}|${teamB}`];
    if (!result) throw new Error(`no static prediction for ${teamA} vs ${teamB}`);
    return result;
  }
  return request("/predict", { method: "POST", body: JSON.stringify({ team_a: teamA, team_b: teamB }) });
}

export async function simulateGroup(group: string, nSims?: number) {
  if (STATIC_MODE) return readStatic(`groups/${group}`);
  return request(`/simulate/group/${group}?n_sims=${nSims ?? 5000}`);
}

// simulateTournament -> readStatic("tournament")
// simulateTournamentBracket -> readStatic("tournament_bracket")
// fetchFixtures -> readStatic("fixtures")
// fetchGroups -> readStatic("groups")
```

Considerar cachear en memoria `predictions.json` (es el más pesado) la primera
vez que se pide, para no volver a descargarlo en cada `predictMatch`.

Las firmas y tipos exportados **no cambian** — los componentes de
`frontend/src/main.tsx` no necesitan modificarse.

## 5. Workflow de GitHub Actions

**Ruta**: `.github/workflows/deploy-pages.yml` (el directorio `.github/`
no existe todavía, hay que crearlo).

Esquema:

```yaml
name: Deploy frontend to GitHub Pages

on:
  push:
    branches: [main]
    paths:
      - "frontend/**"
      - ".github/workflows/deploy-pages.yml"
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
          cache-dependency-path: frontend/package-lock.json
      - run: npm ci
        working-directory: frontend
      - run: npm run build
        working-directory: frontend
        env:
          VITE_STATIC_MODE: "true"
          VITE_BASE_PATH: /${{ github.event.repository.name }}/
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: frontend/dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

### `vite.config.ts`

Agregar:

```ts
export default defineConfig({
  base: process.env.VITE_BASE_PATH ?? "/",
  // ...resto de la config existente
});
```

Esto **no afecta** `npm run dev` ni builds locales (sin la env var, `base`
sigue siendo `/`).

### Pre-requisito crítico

`frontend/public/data/*.json` (generados por `scripts/export_static_data.py`,
sección 3) deben estar **commiteados en el repo** antes de que corra el
workflow — el workflow NO los genera. Verificar que `.gitignore` no excluya
`frontend/public/data/` ni `*.json` dentro de `frontend/`.

## 6. Decisiones abiertas para quien implemente

1. **Nombre del repo / base path**: el workflow de ejemplo usa
   `${{ github.event.repository.name }}` para derivar `VITE_BASE_PATH`. Si el
   repo se publica como Pages de usuario/org en la raíz (`<usuario>.github.io`),
   `base` debe ser `/` en cambio — confirmar cuál es el caso real antes de
   fijar esto.
2. **Tamaño de `predictions.json`**: 2256 entradas de `MatchResult` (sin SHAP)
   probablemente sean unos pocos MB. Si termina siendo demasiado grande para
   commitear cómodamente, alternativas: dividir por equipo
   (`predictions/<TeamA>.json` con las 47 entradas de ese equipo como "local"),
   o usar Git LFS / un release asset descargado en el paso de build.
3. **Cadencia de regeneración**: cada vez que se reentrenen los modelos
   (`scripts/train_models.py`), hay que volver a correr
   `scripts/export_static_data.py` y commitear los JSON actualizados — esto es
   manual, no automático. Documentar este paso en `docs/README.md` o
   `docs/backlog.md` una vez implementado.
4. **Endpoints faltantes en `api.ts`**: si en el futuro se agregan endpoints
   nuevos, el script de export y la rama `STATIC_MODE` de `api.ts` deben
   actualizarse en paralelo.
