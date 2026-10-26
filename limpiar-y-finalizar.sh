#!/usr/bin/env bash
# Limpieza y cierre de sesión de trabajo.
#   bash limpiar-y-finalizar.sh          -> limpia cachés, apaga contenedores, revisa secretos y estado git
#   bash limpiar-y-finalizar.sh --full   -> además borra .venv, mlruns/ y models/ (NO borra data/ ni mlflow.db)
set -euo pipefail
cd "$(dirname "$0")"

FULL=false
[[ "${1:-}" == "--full" ]] && FULL=true

echo "==> Limpiando cachés de Python, pruebas y linters"
find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" \
    -o -name ".ipynb_checkpoints" \) -not -path "./.venv/*" -prune -exec rm -rf {} +
rm -rf htmlcov .coverage coverage.xml build dist ./*.egg-info

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo "==> Apagando contenedores del proyecto"
    docker compose down --remove-orphans || true
fi

if $FULL; then
    echo "==> Limpieza completa: .venv, mlruns/, modelos exportados"
    rm -rf .venv mlruns mlartifacts
    find models -type f ! -name ".gitkeep" -delete 2>/dev/null || true
fi

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "==> Escaneo de secretos (gitleaks vía pre-commit)"
    if command -v uv >/dev/null 2>&1; then
        uv run pre-commit run gitleaks --all-files || echo "!! Revisa los hallazgos de gitleaks antes de hacer push"
    fi

    echo "==> Archivos de datos que Git estaría rastreando (debe estar vacío)"
    git ls-files | grep -Ei '\.(csv|xlsx|xls|parquet|feather|joblib|pkl|db|bundle)$' || echo "   OK: ninguno"

    echo "==> Estado del repositorio"
    git status --short --branch
fi

echo "==> Listo."
