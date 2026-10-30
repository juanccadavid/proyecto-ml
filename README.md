# Proyecto I — Especialización en Ciencia de Datos

Proyecto de clasificación supervisada de punta a punta: EDA con Plotly, preprocesamiento con
`Pipeline` + `ColumnTransformer`, comparación de al menos 2 modelos, seguimiento de experimentos con
MLflow y demo en Streamlit.

> **Estado:** estructura base lista y probada con un **dataset sintético**. El dataset definitivo está por
> definir (ver `sesiones/bitacora.md`). Cambiarlo solo requiere ajustar el `.env`.

> **¿Nuevo en el proyecto?** Empieza por [`docs/guia_comandos.txt`](docs/guia_comandos.txt): Git, uv, Make,
> imports, contratos, pipeline y generador de datos, con los comandos de este proyecto.

## Estructura

```
├── src/proyecto_ml/      Código fuente (paquete instalable)
│   ├── config.py         Rutas y variables de entorno
│   ├── data.py           Carga, X/y y particiones (estratificada, temporal, por grupos)
│   ├── features.py       Imputación, escalado, OneHot, PCA (ColumnTransformer)
│   ├── models.py         Logística, Random Forest, HistGB, TabPFN (opcional)
│   ├── evaluate.py       PR-AUC, ROC-AUC, recall@10 %, F1
│   ├── train.py          Entrenamiento + MLflow + exportación del mejor modelo
│   ├── abandono.py       Modelos de abandono: Ridge (supervisado) y KMeans (sin objetivo)
│   ├── contrato.py       Contrato de datos (pandera, 3 niveles) + SQL de Supabase + campos del formulario
│   ├── generador.py      GeneradorAsociados: asociados ficticios con el esquema real
│   ├── anonymize.py      Hash con sal, eliminación de PII, agrupación en rangos
│   ├── synthetic.py      Generador sintético genérico (make demo)
│   └── app.py            Demo Streamlit
├── scripts/              Ingesta, anonimización, generadores, contrato, modelo de abandono
├── contratos/            Contratos de datos (YAML)
├── supabase/             SQL de la tabla y migraciones
├── web/                  Formulario para enviar asociados a Supabase
├── docs/                 Guía de comandos y conceptos (docs/guia_comandos.txt)
├── sesiones/             Notebooks (EDA) y bitácora del equipo
├── tests/                Pruebas unitarias e integración (pytest); tests/data: reglas del contrato
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

## Modelos de abandono (Ridge y KMeans)

`make abandono` responde "¿este asociado abandona el fondo?" con dos enfoques
([`src/proyecto_ml/abandono.py`](src/proyecto_ml/abandono.py)):

- **Ridge** (supervisado, aprende de `#RETIROS`): métricas en entrenamiento, validación cruzada y datos nuevos.
- **KMeans** (sin objetivo): agrupa por comportamiento (solicitudes, ahorros, intención de retiro) en 2 grupos;
  el de "riesgo" se elige por su perfil, no por el objetivo. Después se compara con el abandono real.

Para validar, se generan asociados **que los modelos nunca vieron** (mismo generador, otra semilla):

```bash
make abandono                                        # entrena con data/ejemplo, valida con 1000 nuevos
make abandono ARGS="--nuevos 5000 --semilla-nuevos 7 --alpha 10"
make abandono ARGS="--validar mi_archivo.parquet"    # validar además con otro archivo
```

Se excluyen las columnas que solo existen porque el asociado ya se fue (fecha/motivo de retiro, `ESTADO`,
fechas de actualización...): con ellas cualquier modelo acierta el 100 % pero no anticipa nada.

## Contrato de datos y Supabase

[`contratos/asociados.yaml`](contratos/asociados.yaml) se valida con **pandera** en tres niveles:

| Nivel | En el contrato | Qué atrapa | Ejemplos | ¿También en la BD? |
|---|---|---|---|---|
| 1. Por fila | `grupos` → `pa.Column` + `Check.ge/le/isin/str_matches` | registros corruptos | `ESTRATO` en 1–6, `SEXO` en F/M | Sí (CHECK) |
| 2. Distribución | `distribucion` → check sobre el lote | problemas sistemáticos | `salario_en_pesos`, `volumen_minimo` | No: la BD ve fila a fila |
| 3. Entre columnas | `reglas` → check con varias columnas | incoherencias | `retiro_coherente`, `nombreintegrado_coherente` | Sí (CHECK) |

De ahí salen la validación en Python, la tabla de Supabase y el formulario web:

```bash
make contrato                              # valida data/ejemplo (o DATA=...)
uv run python scripts/ingesta.py --input ... --target "#RETIROS" --contrato contratos/asociados.yaml
make contrato-sql contrato-web             # regenerar tras cambiar el contrato
```

**Configurar Supabase (una vez):**

1. Crear un proyecto gratuito en [supabase.com](https://supabase.com).
2. *SQL Editor* → pegar [`supabase/asociados.sql`](supabase/asociados.sql) → *Run*. Crea la tabla con todas
   las restricciones del contrato y RLS: la clave pública **solo puede insertar** (no leer, editar ni borrar).
   Si la tabla ya existía con una versión anterior del contrato, correr en orden los archivos de
   [`supabase/migraciones/`](supabase/migraciones/) que falten.
3. *Project Settings → API Keys*: copiar la URL y la clave **pública** (anon / publishable).
   La `service_role` / secret nunca va en el HTML, el `.env` compartido ni Git.
4. Ponerlas en `.env` (`SUPABASE_URL`, `SUPABASE_KEY`) y en `web/config.js` (copiar `web/config.example.js`).

```bash
make supabase-subir                        # sube los 50 asociados de ejemplo (omite cédulas repetidas)
make formulario                            # abre web/formulario.html: datos aleatorios + botón Enviar
```

**Formulario en internet (GitHub Pages, gratis):** `https://juanccadavid.github.io/proyecto-ml/`.
El workflow [`.github/workflows/formulario.yml`](.github/workflows/formulario.yml) lo publica en cada cambio de
`web/` en `main`, armando `config.js` con variables del repositorio (la clave no está en Git). Activación, una vez:

1. *Settings → Secrets and variables → Actions → Variables*: crear `SUPABASE_URL` y `SUPABASE_KEY`
   (la clave **publishable**; el workflow se niega a publicar una `sb_secret_` o `service_role`).
2. *Settings → Pages → Source*: **GitHub Actions**.
3. *Actions → Formulario (GitHub Pages) → Run workflow*.

Cualquiera con el enlace puede **insertar** asociados (no leer, editar ni borrar): la tabla los valida con el
contrato. Para cerrarlo: *Settings → Pages → Unpublish site*.

### Qué va en el contrato y qué va en el test

| | Contrato | Test |
|---|---|---|
| Qué es | La regla: describe el dato válido ("`ESTRATO` está entre 1 y 6") | La prueba de que la regla está encendida ("si llega estrato 7, la validación falla") |
| Dónde vive | `contratos/asociados.yaml` + `src/proyecto_ml/contrato.py` | `tests/data/test_contrato_datos.py` |
| Cuándo corre | En la frontera del pipeline: ingesta `--contrato`, `make supabase-subir`, CHECK de la BD | En el CI (`.github/workflows/ci.yml`) y con `make test`; nunca en producción |

Hacen falta los dos: un contrato sin tests se degrada (alguien relaja un rango para desbloquear algo y nadie
lo nota). Por eso los límites del test están **escritos a mano**, no leídos del YAML: relajar una regla rompe
un test con nombre propio (p. ej. `test_nivel_1_por_fila[ESTRATO=7]`), y el PR queda en rojo. Si cambias una
regla a conciencia, actualiza su caso en el mismo PR.

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
