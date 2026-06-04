@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8

echo ============================================================
echo  prode-ML - Pipeline Completo (Fases A-B-C-D)
echo ============================================================
echo.

cd /d "%~dp0"

:: Activar venv
echo [1/6] Activando entorno virtual...
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo ERROR: No se pudo activar el venv. Ejecuta: python -m venv venv
    pause
    exit /b 1
)

:: Instalar dependencias si faltan
echo [2/6] Verificando dependencias...
python -c "import catboost" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Instalando catboost + optuna...
    pip install catboost optuna --quiet
)
echo OK

:: Pipeline de datos
echo.
echo [3/6] Pipeline de datos (modo rapido)...
echo ============================================================
python scripts/run_pipeline.py --fast --force 2>&1
if %ERRORLEVEL% neq 0 (
    echo ADVERTENCIA: Pipeline tuvo errores pero continuamos...
)
echo OK

:: Validacion
echo.
echo [4/6] Validando datos...
echo ============================================================
python scripts/validate_data.py 2>&1
echo OK

:: Entrenamiento
echo.
echo [5/6] Entrenando modelos (RF + XGB + LGBM + CatBoost + TwoStage + Confederation)...
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

:: Levantar API
echo.
echo [6/6] Levantando API (puerto 8000) + Frontend (puerto 5173)...
echo ============================================================
echo.
echo API:     http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
echo Docs:    http://127.0.0.1:8000/docs
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
