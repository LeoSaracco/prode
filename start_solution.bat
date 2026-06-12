@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo ============================================================
echo  prode-ML - Solucion (sin entrenar modelo)
echo ============================================================
echo  Pipeline de datos + API + Frontend. No entrena modelos.
echo  No genera reporte PDF. Usa los modelos ya persistidos en models\.
echo.

cd /d "%~dp0"

:: [1/6] Activar venv
echo [1/6] Activando entorno virtual...
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: No se encontro el venv. Ejecuta: python -m venv venv
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo ERROR: No se pudo activar el venv. Ejecuta: python -m venv venv
    pause
    exit /b 1
)
echo OK

:: [2/6] Instalar dependencias si faltan
echo.
echo [2/6] Verificando dependencias...
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
python -c "import shap" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Instalando shap...
    pip install shap --quiet
)
echo OK

:: [3/6] Pipeline de datos
echo.
echo [3/6] Pipeline de datos (modo rapido)...
echo ============================================================
python -u scripts/run_pipeline.py --fast --force 2>&1
if %ERRORLEVEL% neq 0 (
    echo ADVERTENCIA: Pipeline tuvo errores pero continuamos...
)
echo OK

:: [4/6] Datos enriquecidos
echo.
echo [4/6] Descargando/limpiando datasets enriquecidos...
echo ============================================================
python -u scripts/download_enriched_data.py 2>&1
if %ERRORLEVEL% neq 0 (
    echo ADVERTENCIA: Datos enriquecidos no disponibles; continuamos con datos base...
)
echo OK

:: [5/6] Verificar que existen modelos entrenados
echo.
echo [5/6] Verificando modelos entrenados...
if not exist "models\model_metadata.json" (
    echo ALERTA: No se encontraron modelos entrenados en models\.
    echo El script no entrena modelos. Ejecuta run_all.bat si necesitas entrenar.
    echo Continuamos sin modelos -- la prediccion no funcionara.
) else (
    for /f "tokens=*" %%i in ('python -c "import json; d=json.load(open('models/model_metadata.json')); print(d.get('trained_at','?'))"') do set MODEL_DATE=%%i
    echo Modelos encontrados ^(entrenados: !MODEL_DATE!^)
)
echo OK

:: [6/6] Levantar API + Frontend
echo.
echo [6/6] Levantando API (puerto 8000) + Frontend (puerto 5173)...
echo ============================================================
echo.
echo API:       http://127.0.0.1:8000
echo Frontend:  http://127.0.0.1:5173
echo Docs:      http://127.0.0.1:8000/docs
echo.
echo Presiona Ctrl+C para detener todo.
echo ============================================================

:: Iniciar API en background
start "prode-ML API" cmd /c "cd /d %CD% && venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload"

:: Esperar a que levante y verificar health
echo Verificando API...
set HEALTH_OK=0
for /L %%i in (1,1,10) do (
    timeout /t 2 /nobreak >nul
    python -c "import urllib.request; import json; r=urllib.request.urlopen('http://127.0.0.1:8000/health'); d=json.loads(r.read()); print(d['status'])" 2>nul | find "ok" >nul
    if !ERRORLEVEL! equ 0 (
        set HEALTH_OK=1
        echo API lista.
        goto :api_ready
    )
)
:api_ready
if !HEALTH_OK! equ 0 (
    echo ADVERTENCIA: La API tardo en responder. El frontend puede fallar.
)

:: Iniciar Frontend en background
start "prode-ML Frontend" cmd /c "cd /d %CD%\frontend && npm install && npm run dev -- --host 127.0.0.1 --port 5173"

echo.
echo Todo listo. Abriendo navegador...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:5173

pause
