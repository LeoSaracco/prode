@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not defined TRAIN_BASE_WORKERS set TRAIN_BASE_WORKERS=4
if not defined TRAIN_MODEL_JOBS set TRAIN_MODEL_JOBS=2
if not defined CONFED_MODEL_JOBS set CONFED_MODEL_JOBS=1
if not defined CONFED_N_ESTIMATORS set CONFED_N_ESTIMATORS=150
if not defined POISSON_OPTIMIZE set POISSON_OPTIMIZE=0

echo ============================================================
echo  prode-ML - Pipeline Completo (Fases A-B-C-D-E)
echo ============================================================
echo  Workers base: %TRAIN_BASE_WORKERS% ^| Threads por modelo: %TRAIN_MODEL_JOBS% ^| Poisson optimize: %POISSON_OPTIMIZE%
echo  Confederation: %CONFED_N_ESTIMATORS% arboles/modelo ^| Threads: %CONFED_MODEL_JOBS%
echo  Nota: despues de CatBoost siguen Poisson, TwoStage y Confederation.
echo.

cd /d "%~dp0"

:: [1/8] Activar venv
echo [1/8] Activando entorno virtual...
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo ERROR: No se pudo activar el venv. Ejecuta: python -m venv venv
    pause
    exit /b 1
)

:: [2/8] Instalar dependencias si faltan
echo [2/8] Verificando dependencias...
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
python -c "import kaggle" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Instalando kaggle...
    pip install kaggle --quiet
)
python -c "import pypdf" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Instalando pypdf...
    pip install pypdf --quiet
)
python -c "import datasets" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Instalando datasets...
    pip install datasets --quiet
)
python -c "import bs4" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Instalando beautifulsoup4...
    pip install beautifulsoup4 --quiet
)
python -c "import lxml" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Instalando lxml...
    pip install lxml --quiet
)
echo OK

:: [3/8] Pipeline de datos
echo.
echo [3/8] Pipeline de datos (modo rapido)...
echo ============================================================
python -u scripts/run_pipeline.py --fast --force 2>&1
if %ERRORLEVEL% neq 0 (
    echo ADVERTENCIA: Pipeline tuvo errores pero continuamos...
)
echo OK

:: [4/8] Datos enriquecidos
echo.
echo [4/8] Descargando/limpiando datasets enriquecidos...
echo ============================================================
python -u scripts/download_enriched_data.py 2>&1
if %ERRORLEVEL% neq 0 (
    echo ADVERTENCIA: Datos enriquecidos no disponibles; continuamos con datos base...
)
echo OK

:: [5/8] Validacion
echo.
echo [5/8] Validando datos...
echo ============================================================
python -u scripts/validate_data.py 2>&1
echo OK

:: [6/8] Entrenamiento
echo.
echo [6/8] Entrenando modelos (RF + XGB + LGBM + CatBoost + TwoStage + Confederation)...
echo ============================================================
echo Esto puede tardar unos minutos. El script mostrara progreso por subetapa.
echo Base models: RF/XGB/LGBM/CatBoost en paralelo.
echo Luego: Poisson, voting weights, TwoStage y ConfederationModels.
echo.
python -u scripts/train_models.py 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR en entrenamiento. Intentando con datos reducidos...
    python -u -c "import logging,sys;sys.path.insert(0,'.');logging.basicConfig(level=logging.INFO,format='%%(levelname)s: %%(message)s');from src.data.collectors.international_results_collector import InternationalResultsCollector;from src.data.collectors.elo_history_collector import EloHistoryCollector;from src.data.cache_manager import CacheManager;from src.models.trainer import ModelTrainer;from config.settings import MATCH_HISTORY_START_YEAR;cache=CacheManager();df=InternationalResultsCollector().collect_match_history(start_year=MATCH_HISTORY_START_YEAR);elo=cache.load('elo_ratings');ranks=cache.load('fifa_rankings');eh=EloHistoryCollector().compute_from_matches(df);m=ModelTrainer(elo_df=elo).train_all(kaggle_df=None,international_df=df,enriched_match_df=None,elo_history_df=eh,statsbomb_xg_df=None,fifa_rankings_df=ranks);print();print('=== METRICS ===');[print(f'  {k}: {v}') for k,v in m.items()]"
    if %ERRORLEVEL% neq 0 (
        echo ERROR FATAL: No se pudo entrenar. Revisa logs arriba.
        pause
        exit /b 1
    )
)
echo OK

:: [7/8] Generar reporte PDF
echo.
echo [7/8] Generando reporte PDF de predicciones...
echo ============================================================
python -u scripts/generate_report.py 2>&1
if %ERRORLEVEL% neq 0 (
    echo ADVERTENCIA: No se pudo generar el PDF.
) else (
    echo PDF generado en: reports\
)
echo OK

:: [8/8] Levantar API + Frontend
echo.
echo [8/8] Levantando API (puerto 8000) + Frontend (puerto 5173)...
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
