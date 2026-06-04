# Backlog - prode-ML FIFA World Cup 2026

Este archivo resume el estado actual y lo que queda pendiente.

## Estado Actual

- CLI interactivo: implementado.
- Modelos entrenados: implementado, con artefactos en `models/`.
- API REST FastAPI: implementada.
- Frontend React/Vite: implementado.
- Grupos oficiales WC2026: implementados en `config/wc2026_groups.py`.
- Simulacion de grupos: implementada con tabla de clasificacion probable y marcador mas probable por fixture.
- Automatizacion local Bash: implementada en `scripts/run_full_stack.sh`.
- Fase A completada: pipeline de datos robusto, split temporal corregido, features contextuales.

## Completado

### Fase A — Pipeline de Datos y Split Temporal

Estado: completado.

Cambios realizados:

- **Fix data leakage**: el split train/val/test ahora es estrictamente temporal (70/15/15 cronologico). Antes `_add_reverse_perspective` contaminaba el test con datos de train.
- **Elo historico computacional**: nuevo `EloHistoryCollector` que calcula Elo rolling desde resultados historicos sin depender de datasets externos.
- **FIFA Rankings collector**: `FIFARankingsCollector` descarga rankings historicos FIFA desde GitHub.
- **Features contextuales**: `is_tournament`, `is_wc`, `is_qualifier`, `is_home` agregados a la matriz de features (21 features total, antes 17).
- **Rango de datos ampliado**: `MATCH_HISTORY_START_YEAR=2000` en settings. El colector de resultados internacionales ahora preserva info de torneo y neutral.
- **Metricas segmentadas**: `accuracy_high_elo_diff_200` medida en test set para partidos con diferencia Elo >= 200.
- **Pesos del ensemble normalizados**: se normalizan automaticamente si no suman 1.0.

### API REST con FastAPI

Estado: completado.

Endpoints disponibles:

- `GET /health`
- `GET /api/v1/teams`
- `GET /api/v1/groups`
- `POST /api/v1/predict`
- `GET /api/v1/simulate/group/{group_name}`
- `GET /api/v1/simulate/tournament`

`GET /api/v1/simulate/group/{group_name}` devuelve `results` para la tabla de grupo y `fixtures` con el marcador mas probable de cada partido.

### Frontend Web

Estado: completado.

Incluye:

- Vista de prediccion de partido.
- Vista de simulacion de grupos.
- Vista de simulacion de torneo.
- Render de resultados mas probables por partido en la vista de grupos.

### Grupos Oficiales

Estado: completado.

Los grupos oficiales del Mundial 2026 estan en `config/wc2026_groups.py`.

### Automatizacion Local Bash

Estado: completado.

`scripts/run_full_stack.sh` ejecuta pipeline, validacion, entrenamiento, API y frontend en orden. Incluye flags para modo rapido, refresh forzado, puertos custom, saltar etapas y evitar instalacion de dependencias.

## Pendiente

### Mejorar Accuracy del Modelo

Prioridad: alta.

El sistema entrena y predice, pero las metricas actuales no alcanzan los targets aspiracionales originales. Acciones recomendadas:

- Integrar mas partidos recientes de selecciones nacionales.
- Mejorar features de localia, lesiones y forma reciente.
- Calibrar probabilidades del ensemble.
- Evaluar pesos adaptativos o meta-learner.
- Medir por segmento: global, alta confianza y delta Elo alto.

### Docker y Deploy

Prioridad: baja.

Pendiente crear:

- `Dockerfile.api`
- `Dockerfile.frontend`
- `docker-compose.yml`

### Robustez Windows

Prioridad: media.

Pendiente evitar errores de encoding en Windows sin depender de `PYTHONIOENCODING=utf-8`.

## Notas Operativas

- No reentrenar modelos salvo que haya nuevos datos o cambios en features.
- Consultar siempre `models/model_metadata.json` para metricas reales.
- Mantener `docs/README.md` alineado con endpoints y comandos reales.
