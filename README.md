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
│   ├── abandono.py       Modelos de abandono: supervisados (Ridge, logística, bagging, boosting) y KMeans
│   ├── busqueda_optuna.py  Optuna para afinar Random Forest y HistGB (make optuna)
│   ├── contrato.py       Contrato de datos (pandera, 3 niveles) + SQL de Supabase + campos del formulario
│   ├── generador.py      GeneradorAsociados: asociados ficticios con el esquema real
│   ├── anonymize.py      Hash con sal, eliminación de PII, agrupación en rangos
│   ├── synthetic.py      Generador sintético genérico (make demo)
│   ├── app.py            Demo Streamlit
│   └── app_abandono.py   Simulador Streamlit del abandono (make simulador)
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
make train ARGS="--clip-outliers iqr --clip-columns salario"  # además recorta atípicos de salario
make train ARGS="--sin-limpieza"  # sin la limpieza de cleaning.py, para comparar en MLflow
make mlflow-ui               # http://localhost:5000
make app                     # http://localhost:8501
make simulador               # http://localhost:8501: mueve las variables y mira si el asociado se retira
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

## Modelos de abandono (supervisados y KMeans)

`make abandono` responde "¿este asociado abandona el fondo?" con dos enfoques
([`src/proyecto_ml/abandono.py`](src/proyecto_ml/abandono.py)):

- **Supervisados** (aprenden de `#RETIROS`), con el mismo preprocesador: métricas en entrenamiento,
  validación cruzada y datos nuevos, más una tabla que los compara sobre los nuevos.

  | Modelo | Tipo | Idea |
  |---|---|---|
  | Ridge | lineal | regresión con penalización sobre la clase -1/+1; coeficientes interpretables |
  | Regresión logística | lineal | probabilidad de abandono; también interpretable |
  | Random Forest | **bagging** | muchos árboles independientes (muestras bootstrap) que votan |
  | HistGradientBoosting | **boosting** | árboles en secuencia, cada uno corrige al anterior (estilo LightGBM) |
- **KMeans** (sin objetivo): agrupa por comportamiento (solicitudes, ahorros, intención de retiro) en 2 grupos;
  el de "riesgo" se elige por su perfil, no por el objetivo. Después se compara con el abandono real.

Para validar, se generan asociados **que los modelos nunca vieron** (mismo generador, otra semilla):

```bash
make abandono                                        # entrena con data/ejemplo, valida con 1000 nuevos
make abandono ARGS="--nuevos 5000 --semilla-nuevos 7 --alpha 10"
make abandono ARGS="--validar mi_archivo.parquet"    # validar además con otro archivo
make abandono ARGS="--sin-limpieza"                  # sin la limpieza de cleaning.py, para comparar
```

**Limpieza activa por defecto** ([`cleaning.py`](src/proyecto_ml/cleaning.py), de Juan Sebastián):

| Paso | Dónde | Qué hace |
|---|---|---|
| `contrato.limpiar_extracto` | al cargar (`a.cargar`), antes de partir | espacios, textos tipo 'N/A' → vacío, números como texto, filas repetidas |
| `ColumnDropper(max_missing=0.5)` | dentro del pipeline (aprende en train) | descarta variables con más de 50 % vacías o con un solo valor |
| `OutlierClipper("iqr")` | rama `clip` del pipeline | recorta atípicos de `SALARIO`, `EDAD` y `ANTIGUEDAD` (no los conteos) |

**Qué variables aportan a cada modelo.** `make abandono` calcula la *permutation importance* sobre los
asociados nuevos: desordena una variable a la vez y mide cuánto cae el AUC. Sirve igual para los cuatro
modelos (HistGB no trae `feature_importances_`) y mide la variable completa, no cada columna del OneHot.
Imprime las 5 principales de cada modelo y guarda `reports/figures/abandono_importancia.png`.

Además, para el Random Forest, `reports/figures/abandono_importancia_rf.html` es un gráfico **interactivo**
(Plotly: al pasar el mouse muestra variable e importancia) con su `feature_importances_` por columna ya
transformada, con el prefijo de su rama (`num__`, `clip__`, `cat__SEXO_F`...). Es la importancia propia del
bosque (cuánto reduce la impureza): favorece a las continuas como `SALARIO` o `EDAD`, que en la de
permutación casi no pesan. En código: `figura_importancia_interactiva(importancia_arbol(rf), "titulo")`.

Al final imprime una **tabla comparativa** al estilo de `compare_models` de PyCaret: Accuracy, AUC, Recall,
Prec., F1, Kappa, MCC y **TT (Sec)** (tiempo promedio de entrenamiento por partición), con validación cruzada
y las mismas particiones para todos. Se guarda en `reports/figures/abandono_comparativa.html` con el mejor
de cada columna en amarillo. Se ordena con `make abandono ARGS="--ordenar Recall"` (o `"--ordenar 'TT (Sec)'"`).
En un notebook: `resaltar(tabla_comparativa(modelos, X, y))` de `proyecto_ml.evaluate`.

**Diagrama de los pipelines.** `make abandono` guarda `reports/figures/abandono_pipelines.html` con el diagrama
interactivo de cada modelo (el mismo de Jupyter): el `ColumnTransformer` con su rama numérica (SimpleImputer →
StandardScaler) y su rama categórica (SimpleImputer → OneHotEncoder), y luego el modelo. Cada caja se despliega con
un clic para ver parámetros y columnas. `make afinar` hace lo mismo con los afinados en
`afinamiento_pipelines.html`: ahí se ve que PyCaret usa OrdinalEncoder en la rama categórica. En un notebook basta
con dejar el pipeline al final de una celda; desde código: `pagina_pipelines({"ridge": modelo})` de
`proyecto_ml.features`.

Se excluyen las columnas que solo existen porque el asociado ya se fue (fecha/motivo de retiro, `ESTADO`,
fechas de actualización...): con ellas cualquier modelo acierta el 100 % pero no anticipa nada.

### Comparación automática con PyCaret (opcional)

`make pycaret` ([`scripts/comparar_pycaret.py`](scripts/comparar_pycaret.py)) le da las mismas variables a
**PyCaret**, que arma solo el preprocesamiento (imputación, codificación, escalado) y compara con validación
cruzada `lr`, `rf`, `et` (bagging), `ada`, `gbc` y `lightgbm` (boosting). La tabla comparativa junta esos 6
y nuestros 4 (con tiempo; PyCaret 4.0 ya no lo entrega) en `reports/figures/pycaret_comparativa.html`, y luego
todos se validan con los mismos asociados nuevos.

```bash
make pycaret                                   # instala el extra la primera vez (uv sync --extra pycaret)
make pycaret ARGS="--ordenar Recall"
```

Usa **PyCaret 4.0 alfa**: la estable (3.3) exige pandas < 2.2 y no es compatible con el proyecto. Por eso es un
extra opcional (no lo instala `make setup` ni el CI). A diferencia de nuestro preprocesador, PyCaret codifica
las categóricas con OrdinalEncoder (no OneHot) y no compensa el desbalance (`class_weight`).

### Afinar hiperparámetros: Ridge y `pycaret_lr`

`make afinar` ([`scripts/afinar_modelos.py`](scripts/afinar_modelos.py)) busca en grilla (`GridSearchCV`) los
mejores hiperparámetros de los dos mejores de la tabla comparativa, con las mismas particiones y métricas:

| Modelo | Hiperparámetros | Valores |
|---|---|---|
| Ridge | `alpha` (más alto = más regularización), `class_weight` | 0.001 … 1000 (13, escala log) × None/balanced |
| `pycaret_lr` | `C` (más alto = menos regularización), `l1_ratio` (0 = L2, 1 = L1), `class_weight` | 0.001 … 1000 × L2/L1 × None/balanced |
| los dos | preprocesamiento: vacíos numéricos, codificación, recorte de atípicos | mediana/media × OneHot/ordinal × sin/IQR |

El preprocesamiento entra a la búsqueda con `features.Preprocesador` (parámetros `numeric_strategy`,
`encoding`, `clip_outliers`). `pycaret_lr` usa la LogisticRegression de PyCaret con su preprocesamiento
(media + ordinal) reproducido con `build_preprocessor`: predice igual, y así también se puede cambiar.
El recorte solo toca `SALARIO`, `EDAD` y `ANTIGUEDAD` (`a.COLUMNAS_RECORTE`): en los conteos de
comportamiento el IQR vale [0, 0] y recortarlos borraría la señal.

**Tabla por partición** (`pipeline | fold | Accuracy | AUC`): para cada preprocesamiento de cada modelo
(`Ridge_median_onehot`, `LogisticRegression_mean_ordinal_iqr`...), su mejor combinación del resto en cada
una de las 5 particiones. Debajo, el resumen por preprocesamiento: promedio y desviación entre particiones
y, la prueba honesta, el resultado sobre asociados **nuevos**. Con `data/ejemplo` (50 filas) el orden se
invierte: en validación cruzada gana ordinal, con nuevos gana OneHot. Con tan pocas filas, buscar entre
más combinaciones sobreajusta la validación cruzada. El **gráfico de bigotes**
(`reports/figures/afinamiento_bigotes.png`) muestra lo mismo: una caja por pipeline con sus 5 particiones
y un rombo con el resultado sobre nuevos. `afinamiento_importancia.png` muestra las variables que más
aportan a los dos afinados.

Imprime las 5 mejores combinaciones de cada uno, la tabla "por defecto vs. afinado" y, lo que cuenta,
la comparación con asociados **nuevos** (el puntaje de validación cruzada del mejor es optimista: se
eligió mirando esos mismos datos). Guarda las curvas en `reports/figures/afinamiento_curvas.png` y cada
modelo afinado en tres `.joblib` (`a.guardar_modelo`):

| Archivo en `models/` | Contenido | Uso |
|---|---|---|
| `abandono_<m>_afinado.joblib` | pipeline completo | predecir: `joblib.load(...).predict(a.preparar(df))` |
| `abandono_<m>_afinado_preprocesamiento.joblib` | solo el preprocesamiento ajustado | transformar datos / ver medianas y categorías |
| `abandono_<m>_afinado_modelo.joblib` | solo el estimador | ver coeficientes; recibe datos ya transformados |

También arma una **tabla de predicciones** de los dos afinados (como `predict_model` de PyCaret) sobre los
asociados nuevos: variables, abandono real y, por modelo, `<modelo>_label` (1 = abandona) y `<modelo>_score`
(riesgo de abandono: probabilidad en la logística; en Ridge, distancia a la frontera, > 0 = abandona). Muestra
unas filas en la terminal (`ARGS="--muestra 10"`) y guarda todas en `data/processed/predicciones_afinados.csv`
(no va a Git). Con regularización fuerte las probabilidades se acercan a 0.5: sirven para **ordenar** el riesgo,
no como probabilidad literal. En código: `a.tabla_predicciones({"ridge": modelo}, df)`.

```bash
make afinar                         # elige por AUC
make afinar ARGS="--metrica F1"     # o Recall, MCC... (las columnas de la tabla comparativa)
```

### Afinar los árboles con Optuna: Random Forest y HistGB

`make optuna` ([`scripts/afinar_optuna.py`](scripts/afinar_optuna.py), lógica en
[`busqueda_optuna.py`](src/proyecto_ml/busqueda_optuna.py)) afina los dos modelos de árboles, que tienen
demasiados hiperparámetros para una grilla. Optuna busca en **rangos** y su muestreador TPE propone cada
intento cerca de los que mejor van. Cada intento usa las mismas 5 particiones y métricas que la tabla
comparativa, y la **poda** (`MedianPruner`) corta a mitad de la validación cruzada los intentos que van
peor que la mediana. No necesita PyCaret.

| Modelo | Hiperparámetros (rangos) |
|---|---|
| Random Forest | `n_estimators` 100–600, `max_depth` 2–20, `min_samples_leaf` 1–10, `max_features` 0.1–1 (fracción), `class_weight` None/balanced/balanced_subsample |
| HistGB | `learning_rate` 0.01–0.3 (log), `max_iter` 50–500, `max_leaf_nodes` 4–64 (log), `max_depth` 2–10, `min_samples_leaf` 2–20, `l2_regularization` 0.0001–10 (log), `class_weight` None/balanced |
| los dos | preprocesamiento: mediana/media × OneHot/ordinal × sin/IQR (como en `make afinar`) |

Imprime los 5 mejores intentos, la importancia de cada hiperparámetro y "por defecto vs. Optuna" en
validación cruzada y con asociados **nuevos**. Guarda los afinados como `models/abandono_<m>_optuna*.joblib`
(los mismos tres archivos que `make afinar`), un HTML interactivo por estudio en
`reports/figures/optuna_<m>.html` (historia de la búsqueda, importancia, cortes y poda) y el resumen en
`reports/metrics/optuna_*.json`.

Con `data/ejemplo` (50 filas), 60 intentos por modelo, unos 2 minutos:

- **HistGB es el que más gana.** Por defecto (`min_samples_leaf=20`) casi no puede cortar con ~40 filas
  por partición: AUC de validación cruzada 0.22. Optuna baja la hoja a 4 y sube `learning_rate`; con
  nuevos, AUC 0.853 → 0.938 y F1 0.674 → 0.782.
- **La métrica que se maximiza importa.** Por AUC, Random Forest elige `class_weight=None`: ordena igual
  de bien (AUC 0.967 con nuevos) pero casi nunca supera el umbral de 0.5 y el recall cae de 0.44 a 0.18.
  Con `--metrica F1` elige `balanced_subsample` y hojas de 10: recall 0.84 y F1 0.80 con nuevos (AUC 0.948).

```bash
make optuna                                         # 60 intentos por modelo, maximiza AUC
make optuna ARGS="--metrica F1"                     # o Recall, MCC... (columnas de la tabla comparativa)
make optuna ARGS="--modelos hist_gb --intentos 200" # un solo modelo, más intentos
make optuna ARGS="--timeout 60"                     # máximo 60 segundos por modelo
make optuna ARGS="--storage sqlite:///optuna.db"    # guarda el estudio; al repetir, continúa donde quedó
```

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
make supabase-bajar                        # baja la tabla a data/raw/ (requiere SUPABASE_SECRET_KEY)
make formulario                            # abre web/formulario.html: datos aleatorios + botón Enviar
```

**Entrenar con lo que llega a Supabase:**

```
formulario / supabase-subir ──► Supabase ──► make supabase-bajar ──► data/raw/supabase_asociados.parquet ──► make abandono
                                              (clave secreta,          (+ copia fechada; valida el contrato)
                                               solo en tu .env)
```

1. En Supabase, *Project Settings → API Keys → Secret keys*: crear una y ponerla en el `.env` como
   `SUPABASE_SECRET_KEY`. Nunca en Git, en la web ni en un chat: salta el RLS.
2. `make supabase-bajar`: descarga, valida con el contrato (si no cumple, no guarda) y deja la última copia.
3. En el `.env`, el interruptor `DATOS_ASOCIADOS` elige con qué entrenan los modelos (una línea sin comentar):
   `data/ejemplo/asociados_ejemplo.parquet` o `data/raw/supabase_asociados.parquet`. Luego `make abandono`.

Se entrena con la copia descargada, no con la BD en vivo: así un entrenamiento se puede repetir con los mismos datos.

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
| Cuándo corre | En la frontera del pipeline: ingesta `--contrato`, `make supabase-subir`/`supabase-bajar`, CHECK de la BD | En el CI (`.github/workflows/ci.yml`) y con `make test`; nunca en producción |

Hacen falta los dos: un contrato sin tests se degrada (alguien relaja un rango para desbloquear algo y nadie
lo nota). Por eso los límites del test están **escritos a mano**, no leídos del YAML: relajar una regla rompe
un test con nombre propio (p. ej. `test_nivel_1_por_fila[ESTRATO=7]`), y el PR queda en rojo. Si cambias una
regla a conciencia, actualiza su caso en el mismo PR.

## Seguridad de datos

- `data/`, `models/`, `mlflow.db`, `.env` y `*.bundle` están en `.gitignore`.
- Pre-commit bloquea archivos de datos, borra salidas de notebooks (nbstripout) y escanea secretos (Gitleaks).
- Los datos reales se anonimizan con `scripts/anonimizar.py` **por quien los custodia, antes de compartirlos**.
  Antes aplica `contrato.limpiar_extracto` (espacios, textos tipo 'N/A', números como texto, filas
  repetidas; `--sin-limpiar` lo apaga) con una configuración que no rompe el contrato: `clean_dataset` con
  sus valores por defecto renombra `#RETIROS`, borra columnas opcionales vacías y vuelve números los teléfonos.
- Cualquier demo pública se hace con datos sintéticos.

## Experimentos (MLflow)

Cada `make train` registra parámetros, métricas y modelo en `mlflow.db` (local, no versionado), incluidas
las opciones de limpieza (`limpieza`, `clip_outliers`, `clip_columns`, `drop_columns`) para comparar con y
sin ellas. Por defecto corre `clean_dataset` (sin renombrar columnas) y un `ColumnDropper` (`--drop-columns`
0.5); `--sin-limpieza` apaga todo.
El JSON de métricas en `reports/metrics/` sí se versiona para que el equipo compare resultados.

## Respaldo offline

`make bundle` genera `rediseno-v3.bundle` (historial completo de Git en un archivo). Se restaura con
`git clone rediseno-v3.bundle carpeta`. No se versiona.

## Equipo

Ver [CONTRIBUTING.md](CONTRIBUTING.md) para el flujo de ramas y Pull Requests.
