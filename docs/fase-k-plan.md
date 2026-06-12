# Fase K — Export Estatico + GitHub Pages

> Plan de implementacion con fases y checkpoints.
> Fecha inicio: 2026-06-12 | Fecha fin: 2026-06-12
> Rama: `main`

## Objetivo

Publicar el frontend en GitHub Pages via GitHub Actions, sin backend Python en vivo.
Los datos de prediccion/simulacion se generan **localmente** in-process (sin HTTP) y se
commitean como JSON estaticos. El workflow de CI solo compila Vite y despliega.

---

## Checklist de Progreso

### K.1 — Shared formatters

- [x] `api/formatters.py`: extraer `_scoreline_tuple_to_schema()` y `_format_prediction()` de `api/routers/predictions.py`
- [x] `api/routers/predictions.py`: importar desde `api.formatters`
- [x] Verificar que `from api.formatters import ...` no genera import circular con `api.schemas`

### K.2 — Script de export `scripts/export_static_data.py`

- [x] `load_prediction_runtime()` carga modelos in-process (sin HTTP)
- [x] `ProcessPoolExecutor` para las 2256 predicciones (6 workers, chunking)
- [x] Split de `predictions/` en 48 archivos por equipo (no un solo JSON gigante)
- [x] `teams.json` — desde `config.wc2026_groups.py` + `get_elo_rating()`
- [x] `groups.json` — dict `GROUPS`
- [x] `fixtures.json` — `_load_wc_fixtures()` + `cached_predict()` (72 pares)
- [x] `groups/A.json` … `groups/L.json` — `GroupSimulator.simulate_group()` x 12
- [x] `tournament.json` — `TournamentSimulator.simulate(n_sims=2500)`
- [x] `tournament_bracket.json` — `TournamentSimulator.simulate_bracket(n_sims=5000)`
- [x] Flag `--quick` para n_sims reducidos en desarrollo
- [x] `tqdm` progress bar para las 2256 predicciones
- [x] `include_shap=False, include_aux=False` en todas las predicciones
- [x] Output en `frontend/public/data/`
- [x] Probado localmente con `python scripts/export_static_data.py --quick`

### K.3 — Frontend `api.ts` static mode

- [x] Variable `STATIC_MODE` via `import.meta.env.VITE_STATIC_MODE === "true"`
- [x] `BASE` via `import.meta.env.BASE_URL`
- [x] Helper `readStatic<T>(file)` con fetch + error handling
- [x] `fetchTeams()` → `readStatic("teams")`
- [x] `fetchGroups()` → `readStatic("groups")`
- [x] `predictMatch(a, b)` → fetch `predictions/${a}.json`, lookup `b`
- [x] Cache en memoria (`Map<string, Record<string, MatchResult>>`) para `predictions`
- [x] `fetchFixtures()` → `readStatic("fixtures")`
- [x] `simulateGroup(g)` → `readStatic("groups/" + g)`
- [x] `simulateTournament()` → `readStatic("tournament")`
- [x] `simulateTournamentBracket()` → `readStatic("tournament_bracket")`
- [x] Tipos TS exportados sin cambios
- [x] Modo dev/live (no STATIC_MODE) sigue funcionando igual

### K.4 — Vite config

- [x] `base: process.env.VITE_BASE_PATH ?? "/"`
- [x] `npm run dev` local sin cambios

### K.5 — GitHub Actions workflow

- [x] `.github/workflows/deploy-pages.yml` creado
- [x] Trigger: `push` a `main` en paths `frontend/**` + workflow
- [x] Trigger: `workflow_dispatch` (manual)
- [x] `permissions: contents: read, pages: write, id-token: write`
- [x] `concurrency: group: pages, cancel-in-progress: true`
- [x] Job `build`: checkout@v4, setup-node@v4 (node 20), npm ci, npm run build
- [x] Env vars: `VITE_STATIC_MODE: "true"`, `VITE_BASE_PATH: /${{ github.event.repository.name }}/`
- [x] `configure-pages@v5` + `upload-pages-artifact@v3` (path: `frontend/dist`)
- [x] Job `deploy`: `deploy-pages@v4`, environment `github-pages`
- [x] YAML sintaxis valida

### K.6 — Verificacion

- [x] `scripts/export_static_data.py --quick` corre sin errores (3.2 min total)
- [x] `frontend/public/data/` contiene todos los JSON esperados (68 archivos)
- [x] `npm run build` con `VITE_STATIC_MODE=true` compila OK (4.3s)
- [x] `dist/data/` copia todos los JSON correctamente
- [x] 28/28 tests pasan sin regresiones
- [ ] Push a `main` → workflow corre y despliega (pendiente push)
- [ ] URL de GitHub Pages funciona (pendiente tras deploy)

---

## Resultados de performance

| Metrica | Valor |
|---|---|
| Predictions (2256 pares, 6 workers) | 2.2 min |
| Export total (quick mode) | 3.2 min |
| Predictions total (full mode, estimado) | ~2.2 min (mismos 2256) |
| Full export (estimado, n_sims 10000/2500/5000) | ~5-8 min |
| Build Vite | 4.3s |
| Tamano total `public/data/` | ~6.5 MB |
| Archivo individual mas grande | predictions/Bosnia and Herzegovina.json (130 KB) |

## Archivos modificados / creados

| Archivo | Accion |
|---|---|
| `api/formatters.py` | Creado — helpers compartidos |
| `api/routers/predictions.py` | Editado — importa de formatters |
| `scripts/export_static_data.py` | Creado — script de export |
| `frontend/src/api.ts` | Editado — rama STATIC_MODE + cache |
| `frontend/vite.config.ts` | Editado — base dinamico |
| `.github/workflows/deploy-pages.yml` | Creado — CI/CD |
| `docs/fase-k-plan.md` | Creado — este documento |

## Decisiones de Performance

| Decision | Razon |
|---|---|
| `ProcessPoolExecutor` en vez de `ThreadPoolExecutor` | Python GIL: threads no ayudan con CPU-bound. Cada worker tiene su propio runtime. |
| Split `predictions/` en 48 archivos por equipo | Frontend carga solo ~130 KB por equipo seleccionado, no 6.5 MB de una vez. |
| `include_shap=False, include_aux=False` | Ahorra ~40% por llamada. SHAP vacio es aceptable en static (frontend ya lo tolera en modo compact). |
| `tqdm` para progreso | Feedback visual para el usuario local mientras corre el script de export. |
| Cache `Map<string, ...>` en api.ts | Evita re-fetchear el mismo archivo de predicciones por equipo en subsiguientes `predictMatch`. |

## Riesgos

| Riesgo | Mitigacion |
|---|---|
| `VITE_BASE_PATH` incorrecto si el repo no se llama `prode-ML` | Usa `${{ github.event.repository.name }}` en el workflow. Si es user page (`<usuario>.github.io`), cambiar a `/`. |
| Regeneracion manual de JSON tras reentrenamiento | Documentado: correr `python scripts/export_static_data.py` y commitear. |
| `top_features` vacio en static | Frontend ya tolera array vacio (modo compact en Comparar partidos). |
| `frontend/public/` no existia | Creado en este PR. Vite copia `public/` a `dist/` automaticamente. |

## Pasos post-implementacion

1. Commitear todos los cambios + los JSON estaticos
2. Push a `main`
3. Verificar que el workflow corre en GitHub Actions
4. Configurar GitHub Pages en Settings del repo (source: GitHub Actions)
5. Visitar la URL publicada y verificar las 4 vistas
