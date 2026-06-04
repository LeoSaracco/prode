@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8

echo ============================================================
echo  prode-ML - Pipeline Completo (Fases A-B-C-D)
echo ============================================================
echo.

cd /d "%~dp0"

:: [1/7] Activar venv
echo [1/7] Activando entorno virtual...
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo ERROR: No se pudo activar el venv. Ejecuta: python -m venv venv
    pause
    exit /b 1
)

:: [2/7] Instalar dependencias si faltan
echo [2/7] Verificando dependencias...
python -c "import catboost" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Instalando catboost + optuna...
    pip install catboost optuna --quiet
)
python -c "import fpdf" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Instalando fpdf2...
    pip install fpdf2 --quiet
)
echo OK

:: [3/7] Pipeline de datos
echo.
echo [3/7] Pipeline de datos (modo rapido)...
echo ============================================================
python scripts/run_pipeline.py --fast --force 2>&1
if %ERRORLEVEL% neq 0 (
    echo ADVERTENCIA: Pipeline tuvo errores pero continuamos...
)
echo OK

:: [4/7] Validacion
echo.
echo [4/7] Validando datos...
echo ============================================================
python scripts/validate_data.py 2>&1
echo OK

:: [5/7] Entrenamiento
echo.
echo [5/7] Entrenando modelos (RF + XGB + LGBM + CatBoost + TwoStage + Confederation)...
echo ============================================================
python scripts/train_models.py 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR en entrenamiento. Intentando con datos reducidos...
    python -c "import logging,sys;sys.path.insert(0,'.');logging.basicConfig(level=logging.INFO,format='%%(levelname)s: %%(message)s');from src.data.collectors.international_results_collector import InternationalResultsCollector;from src.data.collectors.elo_history_collector import EloHistoryCollector;from src.data.cache_manager import CacheManager;from src.models.trainer import ModelTrainer;from config.settings import MATCH_HISTORY_START_YEAR;df=InternationalResultsCollector().collect_match_history(start_year=MATCH_HISTORY_START_YEAR);elo=CacheManager().load('elo_ratings');eh=EloHistoryCollector().compute_from_matches(df);m=ModelTrainer(elo_df=elo).train_all(kaggle_df=None,international_df=df,enriched_match_df=None,elo_history_df=eh,statsbomb_xg_df=None);print();print('=== METRICS ===');[print(f'  {k}: {v}') for k,v in m.items()]"
    if %ERRORLEVEL% neq 0 (
        echo ERROR FATAL: No se pudo entrenar. Revisa logs arriba.
        pause
        exit /b 1
    )
)
echo OK

:: [6/7] Generar reporte PDF
echo.
echo [6/7] Generando reporte PDF de predicciones...
echo ============================================================
python scripts/generate_report.py 2>&1
if %ERRORLEVEL% neq 0 (
    echo ADVERTENCIA: No se pudo generar el PDF.
) else (
    echo PDF generado en: reports\
)
echo OK

:: [7/7] Levantar API + Frontend
echo.
echo [7/7] Levantando API (puerto 8000) + Frontend (puerto 5173)...
echo ============================================================
echo.
echo API:       http://127.0.0.1:8000
echo Frontend:  http://127.0.0.1:5173
echo Docs:      http://127.0.0.1:8000/docs
echo Reportes:  reports\
echo.
echo Presiona Ctrl+C para detener todo.
echo ============================================================

:: Iniciar API en background
start "prode-ML API" cmd /c "cd /d %CD% && venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload"

:: Esperar a que levante
timeout /t 3 /nobreak >nul

:: Iniciar Frontend en background
start "prode-ML Frontend" cmd /c "cd /d %CD%\frontend && npm run dev -- --host 127.0.0.1 --port 5173"

echo.
echo Todo listo. Abriendo navegador...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:5173

pause
