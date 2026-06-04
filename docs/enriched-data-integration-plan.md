# Plan de Integracion de Datasets Enriquecidos y Scraping WC2026

Fecha: 2026-06-04 10:29:11 -03:00

## Resumen

Se incorporan datasets externos y fuentes oficiales/web al flujo de datos para mejorar el modelo sin ensuciarlo con informacion ruidosa o leakage.

El objetivo es descargar, limpiar, integrar al entrenamiento y comparar contra el modelo actual. El modelo actual usa principalmente resultados internacionales, ranking FIFA y Elo. Los nuevos datos se usan solo si pasan validaciones de calidad.

## Fuentes

- Kaggle: International Football Elo Ratings de Saif Alnimri.
- Kaggle: International Football Match Features & Statistics de L. Chikry.
- HuggingFace: StatsBomb open-data shots.
- FIFA: calendario/fixture oficial Mundial 2026.
- FIFA PDF: convocatorias oficiales.
- Transfermarkt: datos agregados por seleccion, con scraping minimo.

## Implementacion

- `scripts/download_enriched_data.py` descarga y cachea fuentes enriquecidas.
- Kaggle usa credenciales locales en `C:\Users\Leandro\.kaggle\kaggle.json`.
- Los datos crudos se guardan en `data/raw`.
- Los datos limpios se guardan en `data/processed`.
- `scripts/train_models.py` carga caches enriquecidos si existen y registra en metadata cuantas filas entraron.

## Limpieza

- Normalizar nombres de paises con aliases del repo.
- Convertir fechas, monedas, porcentajes y alturas a tipos correctos.
- Eliminar duplicados.
- Descartar filas incompletas o equipos fuera del Mundial 2026 cuando corresponda.
- Evitar columnas que puedan filtrar informacion futura.
- Usar features rolling: cada partido solo puede ver datos anteriores a su fecha.

## Validacion

- Tests de descarga/parsing.
- Tests de limpieza de nombres, fechas y monedas.
- Tests anti-leakage.
- Tests de metadata de entrenamiento enriquecido.
- Comparacion de metricas antes/despues.

## Seguridad

- No guardar credenciales Kaggle en el repo.
- No pegar tokens de Kaggle en chats o documentos.
- Si un token se expone, revocarlo y crear uno nuevo.

