#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST="127.0.0.1"
API_PORT="8000"
FRONTEND_PORT="5173"
FAST_PIPELINE=false
FORCE_REFRESH=false
SKIP_PIPELINE=false
SKIP_TRAIN=false
INSTALL_DEPS=true
API_PID=""
FRONTEND_PID=""

usage() {
    cat <<'EOF'
Usage: scripts/run_full_stack.sh [OPTIONS]

Runs the full prode-ML local workflow:
  1. Install/check dependencies
  2. Run data pipeline
  3. Validate processed data
  4. Retrain models
  5. Start FastAPI backend
  6. Start Vite frontend

Options:
  --fast              Run the fast data pipeline path.
  --force             Ignore data cache and refresh sources.
  --skip-pipeline     Do not run scripts/run_pipeline.py.
  --skip-train        Do not run scripts/train_models.py.
  --no-install        Do not install Python/npm dependencies.
  --host HOST         Host for API and frontend. Default: 127.0.0.1.
  --api-port PORT     FastAPI port. Default: 8000.
  --front-port PORT   Frontend port. Default: 5173.
  -h, --help          Show this help.

Examples:
  scripts/run_full_stack.sh --fast
  scripts/run_full_stack.sh --fast --force
  scripts/run_full_stack.sh --skip-pipeline --skip-train
EOF
}

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
    log "ERROR: $*"
    exit 1
}

cleanup() {
    local exit_code=$?
    if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
        log "Stopping frontend (PID ${FRONTEND_PID})"
        kill "${FRONTEND_PID}" 2>/dev/null || true
    fi
    if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
        log "Stopping API (PID ${API_PID})"
        kill "${API_PID}" 2>/dev/null || true
    fi
    exit "${exit_code}"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --fast)
                FAST_PIPELINE=true
                shift
                ;;
            --force)
                FORCE_REFRESH=true
                shift
                ;;
            --skip-pipeline)
                SKIP_PIPELINE=true
                shift
                ;;
            --skip-train)
                SKIP_TRAIN=true
                shift
                ;;
            --no-install)
                INSTALL_DEPS=false
                shift
                ;;
            --host)
                [[ $# -ge 2 ]] || die "--host requires a value"
                HOST="$2"
                shift 2
                ;;
            --api-port)
                [[ $# -ge 2 ]] || die "--api-port requires a value"
                API_PORT="$2"
                shift 2
                ;;
            --front-port)
                [[ $# -ge 2 ]] || die "--front-port requires a value"
                FRONTEND_PORT="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "Unknown option: $1"
                ;;
        esac
    done
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

select_python() {
    if [[ -x "${PROJECT_ROOT}/venv/Scripts/python.exe" ]]; then
        printf '%s\n' "${PROJECT_ROOT}/venv/Scripts/python.exe"
    elif [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
        printf '%s\n' "${PROJECT_ROOT}/venv/bin/python"
    elif command -v python >/dev/null 2>&1; then
        printf '%s\n' "python"
    elif command -v python3 >/dev/null 2>&1; then
        printf '%s\n' "python3"
    else
        die "Python not found"
    fi
}

wait_for_api() {
    local api_url="http://${HOST}:${API_PORT}/health"
    local python_bin="$1"
    local attempts=40

    log "Waiting for API at ${api_url}"
    for _ in $(seq 1 "${attempts}"); do
        if API_URL="${api_url}" "${python_bin}" - <<'PY' >/dev/null 2>&1
import json
import os
import urllib.request

with urllib.request.urlopen(os.environ["API_URL"], timeout=2) as response:
    payload = json.loads(response.read().decode("utf-8"))
    raise SystemExit(0 if payload.get("status") == "ok" else 1)
PY
        then
            log "API is ready"
            return 0
        fi
        sleep 1
    done

    die "API did not become ready at ${api_url}"
}

install_dependencies() {
    local python_bin="$1"

    if [[ "${INSTALL_DEPS}" != true ]]; then
        log "Skipping dependency install"
        return
    fi

    log "Installing Python dependencies"
    "${python_bin}" -m pip install -r "${PROJECT_ROOT}/requirements.txt"

    if [[ ! -d "${PROJECT_ROOT}/frontend/node_modules" ]]; then
        log "Installing frontend dependencies"
        (cd "${PROJECT_ROOT}/frontend" && npm install)
    else
        log "Frontend dependencies already installed"
    fi
}

run_pipeline() {
    local python_bin="$1"
    local args=()

    if [[ "${SKIP_PIPELINE}" == true ]]; then
        log "Skipping data pipeline"
        return
    fi

    [[ "${FAST_PIPELINE}" == true ]] && args+=(--fast)
    [[ "${FORCE_REFRESH}" == true ]] && args+=(--force)

    log "Running data pipeline: scripts/run_pipeline.py ${args[*]-}"
    "${python_bin}" "${PROJECT_ROOT}/scripts/run_pipeline.py" "${args[@]}"

    log "Validating data"
    PYTHONIOENCODING=utf-8 "${python_bin}" "${PROJECT_ROOT}/scripts/validate_data.py"
}

train_models() {
    local python_bin="$1"

    if [[ "${SKIP_TRAIN}" == true ]]; then
        log "Skipping model training"
        return
    fi

    log "Training models"
    PYTHONIOENCODING=utf-8 "${python_bin}" "${PROJECT_ROOT}/scripts/train_models.py"
}

start_api() {
    local python_bin="$1"

    log "Starting API on http://${HOST}:${API_PORT}"
    (
        cd "${PROJECT_ROOT}"
        exec "${python_bin}" -m uvicorn api.main:app --host "${HOST}" --port "${API_PORT}"
    ) &
    API_PID=$!
    wait_for_api "${python_bin}"
}

start_frontend() {
    log "Starting frontend on http://${HOST}:${FRONTEND_PORT}"
    (
        cd "${PROJECT_ROOT}/frontend"
        exec env VITE_API_URL="http://${HOST}:${API_PORT}/api/v1" npm run dev -- --host "${HOST}" --port "${FRONTEND_PORT}"
    ) &
    FRONTEND_PID=$!
}

main() {
    parse_args "$@"
    trap cleanup EXIT INT TERM

    require_command npm
    local python_bin
    python_bin="$(select_python)"

    log "Project root: ${PROJECT_ROOT}"
    log "Python: ${python_bin}"

    install_dependencies "${python_bin}"
    run_pipeline "${python_bin}"
    train_models "${python_bin}"
    start_api "${python_bin}"
    start_frontend

    log "Ready"
    log "API:      http://${HOST}:${API_PORT}"
    log "Frontend: http://${HOST}:${FRONTEND_PORT}"
    log "Press Ctrl+C to stop both services."

    wait
}

main "$@"
