# Makefile del Proyecto I. Requiere uv y, en Windows, Git for Windows (aporta bash).
# Sin make: cada objetivo muestra abajo el comando `uv run ...` equivalente.

# En Windows, make usa cmd.exe por defecto aunque se llame desde PowerShell;
# se fuerza el bash de Git (ruta corta, sin espacios) para que las recetas funcionen igual.
ifeq ($(OS),Windows_NT)
  GIT_BASH := C:/PROGRA~1/Git/bin/bash.exe
  ifneq ($(wildcard $(GIT_BASH)),)
    SHELL := $(GIT_BASH)
  endif
endif

UV  := uv
PY  := $(UV) run python
RUN := $(UV) run

MLFLOW_PORT := 5000
APP_PORT    := 8501
DEMO_DATA   := data/processed/sintetico.parquet
BUNDLE      := rediseno-v3.bundle
EJEMPLO     := data/ejemplo/asociados_ejemplo.parquet

# Variables opcionales: make train DATA=ruta.parquet ARGS="--pca 0.95 --smote"
DATA ?=
ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help setup setup-hf demo-data demo train train-hf abandono mlflow-ui app \
        ejemplo contrato contrato-sql contrato-web supabase-subir supabase-bajar formulario \
        test test-fast cov lint format check hooks secrets up down logs salt bundle clean clean-all

# ==============================================================================
help: ## Muestra los objetivos disponibles
	@echo ""
	@echo "  Proyecto I - objetivos disponibles"
	@echo "  ----------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ==============================================================================
# Entorno
# ==============================================================================
setup: ## Instala dependencias (incluye dev) y los hooks de pre-commit
	$(UV) sync
	@test -f .env || (cp .env.example .env && echo "Creado .env a partir de .env.example: revisa sus valores")
	@if [ -f .pre-commit-config.yaml ]; then $(RUN) pre-commit install; \
	  else echo "!! Falta .pre-commit-config.yaml: no se instalaron los hooks"; fi

setup-hf: ## Instala además los extras de Hugging Face (TabPFN, torch ~2 GB)
	$(UV) sync --extra hf

# ==============================================================================
# Datos y entrenamiento
# ==============================================================================
demo-data: ## Genera el dataset sintético en data/processed/sintetico.parquet
	$(PY) scripts/generar_datos_sinteticos.py --output $(DEMO_DATA)

demo: demo-data ## Genera datos sintéticos y entrena con ellos
	$(PY) -m proyecto_ml.train --data $(DEMO_DATA) $(ARGS)

train: ## Entrena y compara modelos (DATA del .env; acepta DATA=... ARGS=...)
	$(PY) -m proyecto_ml.train $(if $(DATA),--data $(DATA)) $(ARGS)

train-hf: ## Entrena incluyendo TabPFN (requiere make setup-hf)
	$(UV) run --extra hf python -m proyecto_ml.train --tabpfn $(if $(DATA),--data $(DATA)) $(ARGS)

abandono: ## Ridge + KMeans de abandono, validados con asociados nuevos (acepta DATA=... ARGS=...)
	$(PY) scripts/modelo_abandono.py $(if $(DATA),--datos $(DATA)) $(ARGS)

# ==============================================================================
# Servicios locales
# ==============================================================================
mlflow-ui: ## Abre la interfaz de MLflow en http://localhost:5000
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

app: ## Lanza la demo de Streamlit en http://localhost:8501
	$(RUN) streamlit run src/proyecto_ml/app.py --server.port $(APP_PORT)

# ==============================================================================
# Contrato de datos y Supabase
# ==============================================================================
ejemplo: ## Regenera los 50 asociados ficticios de data/ejemplo/
	$(PY) scripts/generar_asociados_ejemplo.py

contrato: ## Valida un dataset contra el contrato (DATA=..., por defecto el de ejemplo)
	$(PY) scripts/contrato_datos.py validar $(or $(DATA),$(EJEMPLO))

contrato-sql: ## Regenera supabase/asociados.sql desde contratos/asociados.yaml
	$(PY) scripts/contrato_datos.py sql

contrato-web: ## Regenera web/contrato.js (campos del formulario) desde el contrato
	$(PY) scripts/contrato_datos.py web

supabase-subir: ## Sube un dataset a Supabase (DATA=..., por defecto el de ejemplo)
	$(PY) scripts/contrato_datos.py subir $(or $(DATA),$(EJEMPLO))

supabase-bajar: ## Descarga la tabla de Supabase a data/raw/ (valida con el contrato)
	$(PY) scripts/contrato_datos.py bajar $(ARGS)

formulario: ## Abre el formulario web en el navegador
	@test -f web/config.js || echo "!! Falta web/config.js (copia web/config.example.js): podrás generar pero no enviar"
	@start web/formulario.html 2>/dev/null || open web/formulario.html 2>/dev/null || xdg-open web/formulario.html

# ==============================================================================
# Calidad
# ==============================================================================
test: ## Corre todas las pruebas
	$(RUN) pytest

test-fast: ## Corre las pruebas sin las de integración
	$(RUN) pytest -m "not integration"

cov: ## Pruebas con reporte de cobertura
	$(RUN) pytest --cov --cov-report=term-missing

lint: ## Revisa el estilo con ruff (código y notebooks)
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format: ## Formatea y corrige automáticamente con ruff
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

check: lint test ## Lint + pruebas (lo mínimo antes de abrir un PR)

hooks: ## Corre todos los hooks de pre-commit sobre el repo completo
	$(RUN) pre-commit run --all-files

secrets: ## Escanea secretos con gitleaks en todo el repo
	$(RUN) pre-commit run gitleaks --all-files

# ==============================================================================
# Docker
# ==============================================================================
up: ## Levanta app (:8501) y MLflow (:5000) con Docker
	docker compose up --build -d

down: ## Apaga los contenedores
	docker compose down --remove-orphans

logs: ## Muestra los logs de los contenedores
	docker compose logs -f

# ==============================================================================
# Utilidades
# ==============================================================================
salt: ## Genera un valor aleatorio para ANON_SALT (copiarlo al .env, no compartir)
	@$(PY) -c "import secrets; print(secrets.token_urlsafe(32))"

bundle: ## Respaldo offline del historial completo de Git en un archivo
	git bundle create $(BUNDLE) --all
	@echo "Restaurar con: git clone $(BUNDLE) carpeta"

clean: ## Limpia cachés, apaga contenedores y revisa secretos/estado git
	bash limpiar-y-finalizar.sh

clean-all: ## Como clean, pero además borra .venv, mlruns/ y models/
	bash limpiar-y-finalizar.sh --full
