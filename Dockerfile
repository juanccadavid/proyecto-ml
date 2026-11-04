# syntax=docker/dockerfile:1
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Dependencias (capa cacheada mientras no cambien pyproject/uv.lock)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2) Código del proyecto, el contrato de datos (el código lo lee para validar y para la huella) y los scripts
#    (el servicio "campeon" de docker-compose corre scripts/elegir_campeon.py; el simulador no los necesita).
COPY README.md ./
COPY src ./src
COPY contratos ./contratos
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Usuario sin privilegios (uid fijo 1000: en Linux, models/ y reports/ montados deben ser suyos para escribir).
# /mlflow es del mismo usuario: el volumen de MLflow hereda ese dueño y el servidor no necesita root.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/models /app/data /app/reports /mlflow \
    && chown -R appuser /app /mlflow
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Por defecto, el simulador del abandono (el mismo de docker-compose y de Streamlit Cloud).
CMD ["streamlit", "run", "src/proyecto_ml/app_abandono.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
