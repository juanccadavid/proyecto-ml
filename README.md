# Proyecto I — Especialización en Ciencia de Datos

Proyecto de clasificación supervisada de punta a punta: EDA con Plotly, preprocesamiento con
`Pipeline` + `ColumnTransformer`, comparación de al menos 2 modelos, seguimiento de experimentos con
MLflow y demo en Streamlit.

> **Estado:** estructura base lista y probada con un **dataset sintético**. El dataset definitivo está por
> definir (ver `sesiones/bitacora.md`). Cambiarlo solo requiere ajustar el `.env`.

## Estructura

```
├── src/proyecto_ml/      Código fuente (paquete instalable)
│   ├── config.py         Rutas y variables de entorno
│   ├── data.py           Carga, X/y y particiones (estratificada, temporal, por grupos)
│   ├── features.py       Imputación, escalado, OneHot, PCA (ColumnTransformer)
│   ├── models.py         Logística, Random Forest, HistGB, TabPFN (opcional)
│   ├── evaluate.py       PR-AUC, ROC-AUC, recall@10 %, F1
│   ├── train.py          Entrenamiento + MLflow + exportación del mejor modelo
│   ├── anonymize.py      Hash con sal, eliminación de PII, agrupación en rangos
│   ├── synthetic.py      Generador de datos sintéticos
│   └── app.py            Demo Streamlit
├── scripts/              Ingesta, anonimización, datos sintéticos
├── sesiones/             Notebooks (EDA) y bitácora del equipo
├── tests/                Pruebas unitarias e integración (pytest)
├── reports/              Métricas y figuras versionables
├── data/                 Datos locales — NO van a Git
└── models/               Modelos exportados — NO van a Git
```

## Requisitos

- [uv](https://docs.astral.sh/uv/) (gestiona Python 3.12 y dependencias)
- Git. Opcionales: Docker Desktop, `make` (en Windows: Git Bash + `winget install ezwinports.make`)

## Inicio rápido

```bash
git clone <url-del-repo> && cd <repo>
make setup                   # uv sync + .env desde .env.example + hooks de pre-commit
# (ajustar valores del .env)

make demo                    # genera el dataset sintético y entrena con él (MLflow)
make train                   # entrena con DATA_PATH del .env (o: make train DATA=... ARGS="--pca 0.95")
make mlflow-ui               # http://localhost:5000
make app                     # http://localhost:8501
make check                   # lint + pruebas, antes de abrir un PR
```

`make` (sin argumentos) lista todos los objetivos. Sin `make`, cada objetivo del `Makefile` muestra
el comando `uv run ...` equivalente.

### Componentes Hugging Face (opcional)

```bash
uv sync --extra hf           # TabPFN + transformers (descarga torch, ~2 GB)
uv run python -m proyecto_ml.train --tabpfn
```

TabPFN: recomendado hasta ~50k filas; la licencia de la versión 2.5 **no permite uso comercial**;
en CPU es lento con miles de filas (usar Colab con GPU).

### Docker

```bash
docker compose up --build    # app :8501 + servidor MLflow :5000
```

La imagen no contiene datos ni modelos: `data/` y `models/` se montan como volúmenes de solo lectura.

## Seguridad de datos

- `data/`, `models/`, `mlflow.db`, `.env` y `*.bundle` están en `.gitignore`.
- Pre-commit bloquea archivos de datos, borra salidas de notebooks (nbstripout) y escanea secretos (Gitleaks).
- Los datos reales se anonimizan con `scripts/anonimizar.py` **por quien los custodia, antes de compartirlos**.
- Cualquier demo pública se hace con datos sintéticos.

## Experimentos (MLflow)

Cada `make train` registra parámetros, métricas y modelo en `mlflow.db` (local, no versionado).
El JSON de métricas en `reports/metrics/` sí se versiona para que el equipo compare resultados.

## Respaldo offline

`make bundle` genera `rediseno-v3.bundle` (historial completo de Git en un archivo). Se restaura con
`git clone rediseno-v3.bundle carpeta`. No se versiona.

## Equipo

Ver [CONTRIBUTING.md](CONTRIBUTING.md) para el flujo de ramas y Pull Requests.
