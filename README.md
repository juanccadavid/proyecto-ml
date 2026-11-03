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
│   ├── features.py       Imputación, escalado, codificación, binning, PCA, RFE (ColumnTransformer)
│   ├── models.py         Logística, Random Forest, HistGB, TabPFN (opcional)
│   ├── evaluate.py       PR-AUC, ROC-AUC, MCC, Kappa, recall@10 %, F1, importancia de variables
│   ├── train.py          Entrenamiento + MLflow + exportación del mejor modelo
│   ├── abandono.py       Modelos de abandono: supervisados (Ridge, logística, bagging, boosting) y KMeans
│   ├── busqueda_optuna.py  Optuna para afinar Random Forest y HistGB (make optuna)
│   ├── tecnicas.py       Compara técnicas de preprocesamiento y feature engineering (make tecnicas)
│   ├── correlacion.py    Correlación entre variables y con el objetivo (Spearman, V de Cramér)
│   ├── contrato.py       Contrato de datos (pandera, 3 niveles) + SQL de Supabase + campos del formulario
│   ├── generador.py      GeneradorAsociados: asociados ficticios con el esquema real
│   ├── anonymize.py      Hash con sal, eliminación de PII, agrupación en rangos
│   ├── synthetic.py      Generador sintético genérico (make demo)
│   ├── app.py            Demo Streamlit
│   ├── campeon.py        Modelo campeón: PR-AUC en CV repetida, umbrales, zonas, ROC y confusión (make campeon)
│   ├── explicar.py       SHAP: por qué el modelo le dio a cada asociado su puntaje
│   ├── validacion.py     Escenarios, partición temporal y KFold / Stratified / TimeSeriesSplit
│   ├── regresion.py      Regresión de ejemplo: el salario con MAE, MSE, RMSE y R² (make regresion)
│   ├── segmentacion.py   No supervisado: KMeans y DBSCAN + mapa interactivo de segmentos (make segmentos)
│   ├── predict.py        Predicción con el campeón guardado: zona, alerta y grupo (make predecir)
│   ├── registro.py       Registro en MLflow (experimento abandono, alias campeon)
│   └── app_abandono.py   Simulador Streamlit del abandono con el campeón (make simulador)
├── scripts/              Ingesta, anonimización, generadores, contrato, modelo de abandono
├── streamlit_app.py      Entrada de Streamlit Community Cloud (abre el simulador)
├── .github/workflows/    ci.yml (lint + pruebas + make pipeline en cada push) y formulario.yml (Pages)
├── contratos/            Contratos de datos (YAML)
├── supabase/             SQL de la tabla y migraciones
├── web/                  Formulario para enviar asociados a Supabase
├── docs/                 Guía de comandos y conceptos (docs/guia_comandos.txt)
├── sesiones/             Notebooks (EDA) y bitácora del equipo
├── tests/                Pruebas unitarias e integración (pytest); tests/data: reglas del contrato
├── reports/              Métricas y figuras versionables
├── data/                 Datos locales — NO van a Git
└── models/               Solo el campeón (y best_model de make train) — NO van a Git
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
make campeon                 # elige el modelo campeón, su umbral y zonas (MLflow: abandono@campeon)
make simulador               # http://localhost:8501: mueve las variables y mira si el asociado se retira
make pipeline                # todo el flujo: campeón, predicción, segmentos, validación cruzada, regresión
make check                   # lint + pruebas, antes de abrir un PR
```

`make` (sin argumentos) lista todos los objetivos. Sin `make`, cada objetivo del `Makefile` muestra
el comando `uv run ...` equivalente.

### Estructura frente a la plantilla del curso

| Plantilla | Aquí | Nota |
|---|---|---|
| `README.md`, `Makefile`, `pyproject.toml` | ✅ los mismos | `make` sin argumentos lista los comandos; dependencias con uv (`uv.lock`) |
| `.github/workflows/ci.yml` | ✅ | en cada push: lint + pruebas y, si pasan, `make pipeline` (ver abajo) |
| `notebooks/` | `sesiones/` | los notebooks de EDA y la bitácora del equipo |
| `src/<paquete>/config.py`, `data.py`, `features.py`, `train.py` | ✅ `src/proyecto_ml/` | |
| `model.py` | `models.py` | el catálogo de modelos; el abandono los arma en `abandono.armar_pipeline` |
| `predict.py` | ✅ `predict.py` | `make predecir DATA=...`: filas crudas → puntaje, zona, alerta y grupo |
| `app/` (Streamlit o FastAPI) | `streamlit_app.py` + `src/proyecto_ml/app_abandono.py` | Streamlit Cloud abre `streamlit_app.py` en la raíz |
| `tests/` | ✅ | pytest, con las pruebas del contrato en `tests/data/` |
| `models/` | ✅ gitignored | solo el campeón; el CI lo deja como artefacto |

### Integración continua: cada push deja el código ejecutado

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) corre en **cada push, en cualquier rama** (y con el botón
*Run workflow* de la pestaña Actions):

1. **calidad**: `ruff` (lint y formato) y todas las pruebas, con las versiones exactas de `uv.lock`.
2. **ejecutar** (si la anterior pasa): `make pipeline`, que corre en orden `campeon`, `predecir`, `segmentos`,
   `validacion` y `regresion`, con un MLflow local del runner.
   - En la pestaña *Summary* de la corrida aparece el resumen del campeón: PR-AUC, recall, MCC y las zonas.
   - Las figuras, las métricas y el campeón quedan en el artefacto **resultados** (14 días).

Todo corre con datos generados (semilla fija): el extracto real nunca va a GitHub. Si haces otro push a la
misma rama mientras corre, se cancela la corrida vieja. Lo mismo se corre en local con `make pipeline`.

### Componentes Hugging Face (opcional)

```bash
uv sync --extra hf           # TabPFN + transformers (descarga torch, ~2 GB)
uv run python -m proyecto_ml.train --tabpfn
```

TabPFN: recomendado hasta ~50k filas; la licencia de la versión 2.5 **no permite uso comercial**;
en CPU es lento con miles de filas (usar Colab con GPU).

### Docker

```bash
docker compose up --build    # simulador :8501 + servidor MLflow :5000
```

La imagen no contiene datos ni modelos: `data/` y `models/` se montan como volúmenes de solo lectura.
El servicio `app` corre el simulador del abandono (`app_abandono.py`) con el campeón de `models/`; sin
él, lo elige al arrancar. `app.py` queda como la demo genérica de la plantilla (`make app`).

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

**Un solo pipeline para todos** (`abandono.armar_pipeline`). Supervisados, campeón, Optuna, `make afinar` y
`make tecnicas` arman el modelo con la misma fábrica:

```
preparar → [crear variables] → preprocess → [SMOTE/ADASYN] → model
```

El primer paso (`PrepararVariables`) calcula EDAD y ANTIGUEDAD y elige las columnas. Por eso el modelo recibe la
**fila tal como llega** (extracto, Supabase, formulario): `modelo.predict(df)`, sin llamar a `preparar()` antes.
Eso vale también para el campeón guardado y para `models:/abandono@campeon` en MLflow. Si la fila ya viene
preparada, la deja igual; así siguen funcionando el simulador y la importancia por permutación.

```python
X, y = a.separar(vista)  # quita #RETIROS; el pipeline prepara el resto
modelo = a.armar_pipeline(LogisticRegression(class_weight="balanced"), crear_variables=True, scaler="robust")
modelo.fit(X, y).predict_proba(fila_de_supabase)
a.armar_pipeline(modelo_rf, buscable=True)  # preprocess__<opción> para GridSearchCV u Optuna
```

**Diagrama de los pipelines.** `make abandono` guarda `reports/figures/abandono_pipelines.html` con el diagrama
interactivo de cada modelo (el mismo de Jupyter): `PrepararVariables`, el `ColumnTransformer` con su rama
numérica (SimpleImputer → StandardScaler) y su rama categórica (SimpleImputer → OneHotEncoder), y luego el modelo. Cada caja se despliega con
un clic para ver parámetros y columnas. `make afinar` hace lo mismo con los afinados en
`afinamiento_pipelines.html`: ahí se ve que PyCaret usa OrdinalEncoder en la rama categórica. En un notebook basta
con dejar el pipeline al final de una celda; desde código: `pagina_pipelines({"ridge": modelo})` de
`proyecto_ml.features`.

Se excluyen las columnas que solo existen porque el asociado ya se fue (fecha/motivo de retiro, `ESTADO`,
fechas de actualización...): con ellas cualquier modelo acierta el 100 % pero no anticipa nada.

### La foto del asociado: fecha de observación y horizonte

Sin opciones, el modelo mira a todos los asociados **al corte** (31-oct-2026): EDAD y ANTIGUEDAD se miden
después de que los retirados se fueron, y como el generador reparte los retiros en los 12 meses anteriores,
en la práctica predice "¿se retiró en el último año?". El diseño correcto es una **foto** en t0 = corte − H:

- **Población:** asociados activos en t0 (los que se retiraron antes ya no eran asociados en la foto).
- **Objetivo:** `#RETIROS` = 1 si se retira entre t0 y el corte (usa `FECHARETIRO`).
- **Variables:** EDAD y ANTIGUEDAD a t0 (`preparar` lee la columna `FECHA_OBSERVACION` que agrega `foto`).

```python
vista = a.foto(df, horizonte_meses=3)  # t0 = 2026-07-31
X, y = a.separar(vista)  # el pipeline calcula EDAD y ANTIGUEDAD a t0
a.resumen_foto(df)  # por horizonte: asociados, retiros y tasa en la foto
```

| Datos | Foto a 3 meses | Foto a 12 meses (= sin foto) |
|---|---|---|
| 50 de `data/ejemplo` | 39 asociados, **1 retiro** | 50, 12 retiros (24 %) |
| 2.000 generados (semilla 7) | 1.535 asociados, **135 retiros (8,8 %)** | 2.000, 600 retiros (30 %) |

Con 50 filas la foto no deja con qué entrenar, así que es opcional (`--horizonte`) y se usa con más filas:

```bash
make ejemplo ARGS="--rows 2000 --seed 7 --output-dir data/processed --nombre asociados_2000"
make abandono DATA=data/processed/asociados_2000.parquet ARGS="--horizonte 3"
```

Ojo: los conteos de solicitudes (`ULT3MESES`...) se toman como vienen. Si el extracto los mide a la fecha de
extracción y no a t0, para los retirados incluyen lo que pasó después de irse (fuga). Hay que confirmarlo con
el fondo. EDA de la foto: [`sesiones/02_juan_eda_foto_asociado.ipynb`](sesiones/02_juan_eda_foto_asociado.ipynb).

### El modelo campeón: `make campeon`

Los afinados, Optuna y PyCaret son **experimentos** (quedan en MLflow); el **campeón** es el único modelo
"oficial": lo usa el simulador ([`campeon.py`](src/proyecto_ml/campeon.py),
[`scripts/elegir_campeon.py`](scripts/elegir_campeon.py)).

- **Datos:** los de desarrollo, 2.000 asociados generados con semilla 7 y **10 % de retiro** (más cerca de un
  fondo real que el 30 % por defecto), con la foto a 3 meses: 1.845 asociados, 40 retiros (2,2 %). `make
  datos-desarrollo` los escribe en `data/processed/`; sin el archivo se generan en memoria, idénticos. Los 50
  de `data/ejemplo` quedan para el contrato y las pruebas.
- **Criterio:** PR-AUC promedio en validación cruzada **repetida** (5 particiones × 3 repeticiones), no en los
  "nuevos" del generador. PR-AUC y no AUC: con 2 % de retiros el AUC pasa de 0,98 en todos los modelos.
- **Umbrales**, con las predicciones fuera de partición (cada asociado puntuado por un modelo que no lo vio):
  - *alerta*: el que maximiza **F2** (el recall pesa el doble: perder a quien se va cuesta más que una llamada);
  - *alto*: el menor puntaje desde el cual **al menos 8 de cada 10 alertas son retiros reales**.
- **Tres zonas:** 😟 **se retira** (≥ alto) · 😐 **zona gris** (entre los dos) · 😀 **se queda** (< alerta).
- **KMeans** con los mismos datos y sin el objetivo: el simulador muestra en qué grupo cae el asociado y la
  tasa de retiro real del grupo.

| Candidato | PR-AUC (CV) | umbral alerta | precisión / recall | umbral alto | precisión / recall |
|---|---|---|---|---|---|
| **Regresión logística (campeón)** | **0,875 ± 0,005** | 0,29 | 0,56 / 0,93 | 0,88 | 0,82 / 0,78 |
| Ridge | 0,827 ± 0,012 | 0,26 | 0,78 / 0,80 | 0,28 | 0,82 / 0,78 |
| Random Forest | 0,816 ± 0,019 | 0,10 | 0,62 / 0,95 | 0,19 | 0,81 / 0,53 |
| HistGB | 0,777 ± 0,033 | 0,10 | 0,52 / 0,85 | 0,54 | 0,81 / 0,65 |

Zonas del campeón con 925 asociados nuevos (foto a 3 meses): **se retira** 19 (14 se retiran, 74 %),
**zona gris** 13 (ninguno), **se queda** 893 (1 se retira). KMeans: en el grupo "riesgo" se retira el 12,9 %;
en el "estable", el 1,0 %.

Guarda solo el campeón en `models/`:

- `abandono_campeon.joblib`, con el pipeline y sus dos umbrales juntos (`campeon.ModeloZonas`);
- `abandono_kmeans.joblib`;
- `campeon.json`, con el modelo, los umbrales, las métricas de todos los candidatos, los grupos, los datos y la
  fecha.

También guarda los gráficos de `reports/figures/` (umbrales, escenarios, ROC, matriz de confusión y SHAP: ver
abajo) y registra el campeón en MLflow con el alias `campeon`.

El modelo guardado recibe la fila cruda y **devuelve la zona**. Lo mismo hace el registrado en MLflow:

```python
campeon = mlflow.sklearn.load_model("models:/abandono@campeon")  # o: c.cargar(MODELS_DIR)[0]
campeon.predict(filas)  # ["se queda", "zona gris", "se retira", ...]
campeon.predict_proba(filas)  # el puntaje (decision_function si es Ridge)
campeon.alerta(filas)  # 1 desde el umbral de alerta
```

Un campeón guardado antes de este cambio (solo el pipeline) se carga igual: `c.cargar` lo envuelve con los
umbrales de `campeon.json`. KMeans también recibe la fila cruda (`agrupador.predict(filas)`).

```bash
make campeon                                     # foto a 3 meses, datos de desarrollo
make campeon ARGS="--horizonte 12 --precision-min 0.9"
make campeon DATA=data/raw/supabase_asociados.parquet   # con el extracto, cuando llegue
make simulador                                   # usa el campeón
```

#### Evaluación del campeón: ROC, matriz de confusión, MCC/Kappa y SHAP

`make campeon` también guarda, todo con las predicciones **fuera de partición**:

- `reports/figures/campeon_umbrales.html`: precisión y recall según el umbral (el *trade-off*), con las
  tres zonas.
- `campeon_roc.html`: la curva ROC de los cuatro candidatos, con los dos umbrales del campeón marcados.
- `campeon_confusion.html`: la matriz de confusión con cada umbral. Usa el umbral elegido, no el 0,5 de
  `predict`, así que cuadra con la precisión y el recall de la tabla. Los conteos también van en
  `campeon.json` (`confusion_cv`).
- `campeon_shap.html`: SHAP en 100 asociados nuevos y la explicación del de mayor riesgo.

| Umbral del campeón | real se queda: sin alerta / alerta | real se retira: sin alerta / alerta |
|---|---|---|
| alerta (0,29) | 1.776 / 29 | 3 / **37** (recall 92 %) |
| se retira (0,88) | 1.798 / 7 | 9 / **31** (precisión 82 %) |

**MCC y Kappa** (al umbral de alerta) van en la tabla de candidatos y en MLflow (`cv_mcc`, `cv_kappa`):

| | Logística (campeón) | Ridge | Random Forest | HistGB |
|---|---|---|---|---|
| MCC | 0,71 | **0,79** | 0,76 | 0,66 |
| Kappa | 0,69 | **0,79** | 0,75 | 0,64 |

Ridge gana en MCC porque su umbral F2 alerta a menos personas: más precisión (0,78 frente a 0,56) y menos
recall (0,80 frente a 0,93). El MCC pesa por igual las dos clases de error; nosotros no, como explica la
siguiente sección.

#### Qué métrica usamos y por qué

```
¿Clases balanceadas (50/50)?  NO: se retira el 2,2 % en la foto a 3 meses
   └─> Accuracy engaña: decir "nadie se va" acierta el 97,8 %
       └─> MCC / Kappa (corrigen el desbalance) para comparar sin fijarse en el costo
           └─> ¿Qué es más costoso?  Los FALSOS NEGATIVOS: un asociado que se va sin que nadie lo llame
               └─> RECALL: umbral de alerta con F2 (el recall pesa el doble que la precisión)
```

- **Para elegir el modelo**, la PR-AUC: resume la precisión y el recall en todos los umbrales y no se infla
  con el desbalance, como sí pasa con el AUC-ROC (todos pasan de 0,98).
- **Para el umbral de alerta**, F2. Una llamada de más cuesta poco; perder un asociado cuesta mucho.
- **Para la zona "se retira"**, una precisión mínima de 0,8: cuando el modelo está seguro, debe acertar.
- **MCC y Kappa** se reportan para comparar con PyCaret y con la literatura, pero no deciden.

#### SHAP: por qué el modelo le dio a este asociado su puntaje

[`explicar.py`](src/proyecto_ml/explicar.py) reparte el puntaje de **cada asociado** entre sus variables:

```
puntaje = promedio de 100 asociados de referencia + contribución de cada variable
```

Una contribución en rojo sube el riesgo frente al promedio; una en azul lo baja. La suma da exactamente el
puntaje del modelo. Las contribuciones se miden en puntos de probabilidad, o en distancia a la frontera si el
campeón es Ridge. Se explican las variables originales (EDAD, SEXO...), no las columnas del OneHot, con
`shap.PermutationExplainer`. Ese método solo necesita el puntaje, así que sirve para los cuatro candidatos.

- **En el simulador**, debajo del riesgo, aparece "¿Por qué este puntaje?": una cascada con las 8 variables
  que más pesan para ese asociado. Se recalcula al mover cualquier control.
- **En `make campeon`**, la importancia global, que es la |contribución| promedio en 100 asociados nuevos.
  Arriba quedan las señales de salida: cancelación de ahorros (2,2 pts), intención de retiro (1,7) y
  disminución de cuotas (1,2). Es la misma lectura de la importancia por permutación, pero también se puede
  explicar asociado por asociado.

```python
from proyecto_ml import explicar as e

explicador = e.Explicador(campeon, datos_de_referencia)  # crudos o preparados
explicador.explicar(fila)  # variable | valor | contribucion, de la que más pesa a la que menos
```

### Validación: qué miden las cifras (para la sustentación)

> Las métricas sobre asociados nuevos miden si el modelo **recupera la regla del generador**. La validación
> real será **temporal**, con el extracto del fondo: entrenar con una foto vieja y probar con la más reciente
> (`validacion.particion_temporal`, lista y probada con datos generados).

Mientras tanto, `make campeon` valida cada candidato, con su umbral, en escenarios menos favorables
([`validacion.py`](src/proyecto_ml/validacion.py)): otra tasa de retiro y una señal más débil
(`GeneradorAsociados(fuerza_senal=...)`: 1 = el generador de siempre, 0,5 = retirados la mitad de distintos).
Esa tabla reemplaza al "0,97 en nuevos" en la presentación:

| Escenario (foto a 3 meses) | Logística (campeón): AUC · precisión · recall | Random Forest | HistGB |
|---|---|---|---|
| tasa 30 %, señal 100 % (generador) | 0,98 · 0,78 · 0,89 | 0,97 · 0,77 · 0,84 | 0,98 · 0,86 · 0,78 |
| tasa 10 %, señal 100 % | 1,00 · 0,44 · 0,93 | 1,00 · 0,47 · 0,93 | 0,99 · 0,38 · 0,80 |
| tasa 10 %, señal 50 % | 0,99 · 0,52 · 0,80 | 0,97 · 0,38 · 0,60 | 0,96 · 0,50 · 0,47 |
| tasa 10 %, señal 25 % | 0,75 · 0,19 · 0,20 | 0,74 · 0,14 · 0,13 | 0,75 · 0,12 · 0,13 |

Si la señal real es la mitad de clara, el campeón aún encuentra a 8 de cada 10; con un cuarto, el AUC cae a
0,75. **Pendiente con el fondo** (datos agregados, sin anonimizar): tasa de retiro anual, % de asociados con al
menos una solicitud de cancelación o disminución de cuotas, y mediana de antigüedad y salario, para calibrar
el generador y elegir el umbral con la tasa real.

### KFold, StratifiedKFold y TimeSeriesSplit: `make validacion`

[`scripts/comparar_validacion.py`](scripts/comparar_validacion.py) parte los mismos datos del campeón de las
tres formas y entrena la logística en cada partición (`validacion.comparar_validaciones`):

| Estrategia | PR-AUC | ± entre particiones | retirados por prueba | filas de entrenamiento |
|---|---|---|---|---|
| KFold (al azar) | 0,869 | 0,095 | 5 a 12 | 1.476 |
| **StratifiedKFold** (la del campeón) | **0,885** | **0,067** | 8 en todas | 1.476 |
| TimeSeriesSplit (por fecha de ingreso) | 0,838 | 0,207 | 3 a 7 | 310 → 1.538 |

- **KFold**: con 40 retirados, una prueba tiene 5 y otra 12, y la cifra salta más.
- **StratifiedKFold**: deja la misma tasa de retiro en cada prueba. Por eso la usan el campeón, Optuna y la
  grilla.
- **TimeSeriesSplit**: solo prueba con filas posteriores a las de su entrenamiento. La foto no tiene una
  fecha por fila, así que se ordena por fecha de ingreso: entrena con los más antiguos y prueba con los más
  nuevos. Las primeras particiones entrenan con pocos datos y por eso varía tanto. La validación temporal de
  verdad, con el extracto, es `particion_temporal` (foto vieja → foto reciente).

```bash
make validacion                                  # logística, 5 particiones
make validacion ARGS="--modelo random_forest --folds 10"
```

### Regresión de ejemplo: el salario (`make regresion`)

El abandono es una clasificación. [`regresion.py`](src/proyecto_ml/regresion.py) es la contraparte de
regresión, con un uso real: **estimar el salario** de quien no lo reporta (el 5 % en el generador). Se entrena
con quienes sí lo reportan y se valida con **KFold**, porque en regresión no hay clases que estratificar.
TIPOPACTO no entra: el pacto integral se define por el salario y sería fuga.

| Métrica | Qué dice | Referencia (promedio) | Ridge | Random Forest | HistGB |
|---|---|---|---|---|---|
| **MAE** | error promedio, en pesos | $1.254.761 | $1.261.859 | $1.276.987 | $1.327.236 |
| **MSE** | castiga más los errores grandes (pesos²) | 2,84 × 10¹² | 2,88 × 10¹² | 2,92 × 10¹² | 3,15 × 10¹² |
| **RMSE** | raíz del MSE, otra vez en pesos | $1.684.354 | $1.697.895 | $1.710.078 | $1.775.802 |
| **R²** | fracción de la variación explicada (0 = el promedio) | 0,00 | −0,02 | −0,03 | −0,11 |

**Ningún modelo le gana a predecir el promedio, y eso es lo correcto:** el generador saca el salario al azar
(lognormal), sin relación con las demás variables. Un R² alto aquí sería fuga. Si el RMSE es mucho mayor que
el MAE, hay pocos errores muy grandes, que son los salarios altos. Con el extracto real, donde el salario
depende del cargo, la antigüedad y el estrato, el R² debería subir. Resultados en
`reports/figures/regresion_salario.html` y en MLflow (experimento `salario`).

### Supervisado vs. no supervisado: segmentación con KMeans y DBSCAN (`make segmentos`)

| | Supervisado | No supervisado |
|---|---|---|
| Aprende de | el retiro real (`#RETIROS`) | solo el comportamiento: nunca ve quién se retiró |
| Modelos | Ridge, logística, Random Forest, HistGB (el campeón) | KMeans y DBSCAN ([`segmentacion.py`](src/proyecto_ml/segmentacion.py)) |
| Responde | ¿este asociado se retira? (puntaje y zona) | ¿qué tipos de asociado hay? (segmentos) |
| Se valida con | PR-AUC, recall, precisión (validación cruzada) | silueta; y, solo al final, la tasa de retiro de cada segmento |

[`scripts/segmentar.py`](scripts/segmentar.py) agrupa a los asociados de la foto por sus 9 variables de
comportamiento, imputadas y estandarizadas:

- **KMeans** parte a todos en `k` grupos alrededor de `k` centros. `k` se elige con la **silueta**:
  - k = 2 da 0,295, k = 3 da 0,268, y de k = 4 en adelante queda por debajo de 0,13.
  - El segmento 1 es el de más señales de salida, ordenado por sus centros sin mirar el retiro.
- **DBSCAN** no pide `k`: forma grupos con las zonas densas (al menos 18 vecinos a menos de `eps` = 2,62, el
  cuantil 90 de la distancia al vecino 18) y deja aparte a los **atípicos**, que no caen en ninguna.

| Marcar "en riesgo" a… | asociados | se retira | recall | ARI con el retiro |
|---|---|---|---|---|
| KMeans: segmento 1 | 186 (10 %) | 12,9 % | 60 % | 0,16 |
| **DBSCAN: atípicos** | 84 (5 %) | **36,9 %** | **78 %** | **0,47** |
| (todos los asociados) | 1.845 | 2,2 % | | |

Sin ver el retiro, DBSCAN aparta a un 5 % de asociados con comportamiento raro: muchas cancelaciones,
disminuciones e intenciones de retiro. Ahí está el 78 % de los que se van. Aun así, el supervisado es mucho
mejor (el campeón encuentra al 92 % con una precisión del 56 %), porque aprende directamente del retiro. Los
segmentos sirven para **describir** a los asociados y para campañas, no para reemplazar al campeón.

**Mapa interactivo** (`reports/figures/segmentos.html` y, en el simulador, "Mapa de segmentos"):
- Cada asociado va en dos ejes que se leen directo: **señales de salida** (cancelaciones, disminuciones,
  intención de retiro) y **vínculo con el fondo** (ahorros, créditos, servicios).
- Cada segmento tiene su color y su contorno.
- Un botón cambia entre KMeans y DBSCAN; los atípicos van en gris con ✕.
- Al pasar el mouse sale el segmento, su tasa de retiro y el comportamiento del asociado.
- En el simulador, una estrella marca dónde cae el asociado que se está simulando.

Con PCA, en cambio, dos componentes solo explican el 29 % de la variación y los segmentos se enciman.

```bash
make segmentos                                   # k por silueta, eps por vecinos
make segmentos ARGS="--k 4"                      # 4 segmentos (los dos de riesgo: 12,8 % y 6,4 % de retiro)
```

### Mapa de evaluación (para la sustentación)

| Concepto | Dónde está |
|---|---|
| Probabilidad vs. etiqueta | `abandono.puntaje` (probabilidad) vs. `ModeloZonas.predict` (zona) y `.alerta` (0/1) |
| Umbral de decisión | `campeon.elegir_umbrales`: alerta (máximo F2) y alto (precisión ≥ 0,8) |
| Matriz de confusión | `campeon.matrices_confusion` / `campeon_confusion.html` (con los umbrales del campeón) |
| Métricas de evaluación | `evaluate.compute_metrics` y `SCORING`: accuracy, precisión, recall, F1, AUC, PR-AUC, MCC, Kappa |
| Trade-off precisión vs. recall | `campeon.figura_umbrales` / `campeon_umbrales.html` |
| AUC-ROC | `campeon.figura_roc` / `campeon_roc.html` (los cuatro candidatos) |
| Qué métrica elegir | sección "Qué métrica usamos y por qué" |
| MAE, MSE, RMSE, R² | `regresion.py` / `make regresion` |
| Hiperparámetros comunes | `busqueda_optuna.espacio_random_forest`: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features |
| KFold, StratifiedKFold, TimeSeriesSplit | `validacion.comparar_validaciones` / `make validacion` |
| Supervisado vs. no supervisado | campeón (`campeon.py`) vs. KMeans y DBSCAN (`segmentacion.py`, `make segmentos`) |
| Segmentación y su gráfico interactivo | `segmentacion.figura_mapa` / `segmentos.html` y el simulador |
| Predicción con el modelo guardado | `predict.py` / `make predecir` |
| Importancia de variables | permutación (`evaluate.importancia_variables`), MDI del bosque (`importancia_arbol`), coeficientes (`abandono.coeficientes`) y SHAP (`explicar.py`) |

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
modelo afinado en MLflow (corrida "afinar", una hija por afinado con sus hiperparámetros y el modelo). Para
guardar un pipeline por partes (completo, solo preprocesamiento y solo estimador) sigue `a.guardar_modelo`.

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
| Random Forest | `n_estimators` 50–500, `max_depth` 3/5/10/20/None, `min_samples_split` 2–10, `min_samples_leaf` 1–10, `max_features` sqrt/log2/None, `class_weight` None/balanced/balanced_subsample |
| HistGB | `learning_rate` 0.01–0.3 (log), `max_iter` 50–500, `max_leaf_nodes` 4–64 (log), `max_depth` 2–10, `min_samples_leaf` 2–20, `l2_regularization` 0.0001–10 (log), `class_weight` None/balanced |
| los dos | preprocesamiento: mediana/media × OneHot/ordinal × sin/IQR (como en `make afinar`) |

Imprime los 5 mejores intentos, la importancia de cada hiperparámetro y "por defecto vs. Optuna" en
validación cruzada y con asociados **nuevos**. Registra en MLflow el mejor intento de cada estudio (sus
hiperparámetros, `cv_`, `nuevos_` y el resumen del estudio, no una corrida por intento), un HTML interactivo por estudio en
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

### Técnicas de preprocesamiento y feature engineering: `make tecnicas`

`make tecnicas` compara las técnicas de preprocesamiento sobre la foto del asociado, con los mismos datos que
`make campeon`. Cada técnica cambia **una sola cosa** respecto a la base (el preprocesador de los supervisados
con una logística con `class_weight="balanced"`). Todas se miden con PR-AUC en las mismas particiones
(5 × 3), así que la diferencia con la base es lo que aporta esa técnica. Todo se aprende dentro de cada
partición de entrenamiento: SMOTE y ADASYN solo re-muestrean en `fit`, y TargetEncoder usa validación cruzada
interna. Así nada se filtra de la validación.

Los datos generados no traen vacíos, así que para comparar imputadores el script pone un 10 % de vacíos al azar
(`--vacios`) en SALARIO, ESTRATO, EDAD, ANTIGUEDAD y #CANTIDAD_SERVICIOS.

| Técnica | Dónde | PR-AUC | Dif. con la base |
|---|---|---|---|
| **Base**: mediana, OneHot, StandardScaler, recorte IQR, `class_weight` | `features.build_preprocessor` | 0.893 | — |
| Sin balancear | `tecnicas.clasificador` | 0.902 | +0.008 |
| SMOTE | `imblearn` (también `make train ARGS="--smote"`) | 0.888 | −0.006 |
| ADASYN | `imblearn` | 0.887 | −0.006 |
| SimpleImputer(mean) | `numeric_strategy="mean"` | 0.898 | +0.005 |
| KNNImputer | `numeric_strategy="knn"` | 0.891 | −0.003 |
| IterativeImputer | `numeric_strategy="iterative"` | 0.897 | +0.004 |
| SimpleImputer + MissingIndicator | `missing_indicator=True` | 0.914 | +0.021 |
| OrdinalEncoder | `encoding="ordinal"` | 0.883 | −0.010 |
| TargetEncoder | `encoding="target"` | 0.871 | −0.022 |
| MinMaxScaler | `scaler="minmax"` | 0.848 | −0.046 |
| MaxAbsScaler | `scaler="maxabs"` | 0.851 | −0.042 |
| RobustScaler | `scaler="robust"` | 0.888 | −0.005 |
| Normalizer | `scaler="normalizer"` | 0.821 | −0.073 |
| Variables nuevas (creación y transformación) | `abandono.CrearVariables` | **0.925** | **+0.032** |
| Binning (KBinsDiscretizer, 5 rangos) | `bins=5, bin_columns=...` | 0.894 | +0.001 |
| PCA (95 % de la varianza) | `pca_components=0.95` | 0.888 | −0.005 |
| RFE (15 columnas) | `rfe_features=15` | 0.876 | −0.018 |

*Datos de desarrollo, foto a 3 meses: 1.845 asociados, de los que 40 se retiran. Logística, 5 particiones × 3
repeticiones.*

**Cómo leerlo.** Con 40 retirados, la PR-AUC varía ±0.077 entre particiones, así que casi todas las diferencias
caben en el azar. El indicador de vacíos es un ejemplo: aquí los vacíos se pusieron al azar, así que no pueden
anunciar un retiro, y aun así sube 0.021. Eso es ruido. Lo que sí se sostiene:

- **Las variables nuevas son lo que más suma.** SENALES_SALIDA junta en una columna las cinco solicitudes que
  anuncian un retiro.
- **Algunos escaladores perjudican a la logística.** Min-max y max-abs son sensibles a los atípicos, y
  Normalizer escala cada fila y no cada columna. A los árboles el escalado no les cambia nada (pruébalo con
  `--clasificador hist_gb`).
- **SMOTE y ADASYN no le ganan a `class_weight`.** Con una logística, reponderar las clases y re-muestrear hacen
  casi lo mismo.

**Feature engineering** (`abandono.CrearVariables`, opcional; no cambia el campeón):

- **Creación por agregación:** `SENALES_SALIDA` (suma de las solicitudes de salida) y `PRODUCTOS` (ahorros a la
  vista + contractuales + servicios).
- **Creación por interacción:** `SALIDA_POR_PRODUCTO = SENALES_SALIDA / (1 + PRODUCTOS)`, es decir, pedir salir
  teniendo poco que perder.
- **Transformación logarítmica:** `SALARIO_LOG = log(1 + SALARIO)`. Antes estaban EDAD y ANTIGUEDAD, que se
  calculan desde las fechas.

**Correlación de variables** (`correlacion.py`):

- **Entre numéricas:** Spearman, guardada en `reports/figures/correlaciones.html`, y los pares redundantes con
  |r| ≥ 0,8 (por ejemplo, SALARIO y SALARIO_LOG: sobra una).
- **Con el retiro:** Spearman para las numéricas y V de Cramér para las categóricas. La más fuerte es
  #SOLICITUD_CANCELACIONAHORROS (0.29). Las categóricas casi no se relacionan, salvo TIPOCONTRATO.

**Lo que no se agregó, y por qué:**

- **LabelEncoder** es para el objetivo (y), no para las variables: scikit-learn lo advierte, porque le pondría
  un orden falso a las categorías. Nuestro objetivo ya es 0/1. Para las variables están OrdinalEncoder (si hay un
  orden real) y OneHotEncoder.
- **Embeddings** se aprenden con una red neuronal y sirven con miles de categorías (texto, productos). Aquí la
  categórica más grande es CARGO, con unas decenas de valores: TargetEncoder y OneHot cubren eso sin entrenar
  una red.

Salidas: `reports/figures/tecnicas.html` (un punto por técnica con su desviación; la línea punteada es la base),
`reports/figures/correlaciones.html`, `reports/metrics/tecnicas_<fecha>.json` y la corrida "tecnicas" en MLflow.

```bash
make tecnicas                                           # logística, 5 x 3, 10 % de vacíos
make tecnicas ARGS="--clasificador hist_gb"             # con árboles (el escalado ya no importa)
make tecnicas ARGS="--vacios 0 --repeticiones 1"        # sin vacíos, más rápido
make tecnicas ARGS="--datos data/raw/supabase_asociados.parquet --sin-mlflow"
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
2. *SQL Editor* → pegar [`supabase/asociados.sql`](supabase/asociados.sql) → *Run*. Crea `asociados` con todas
   las restricciones del contrato, cerrada con RLS (la clave pública sola no lee, inserta, edita ni borra), y
   la parte del formulario: `equipo` y la política que deja insertar al equipo (ver abajo). Si la tabla ya
   existía, correr en orden los archivos de [`supabase/migraciones/`](supabase/migraciones/) que falten (la
   **003** deja que el equipo inserte directo en `asociados`).
3. *Project Settings → API Keys*: copiar la URL, la clave **pública** (anon / publishable) y crear una
   **secreta** (*Secret keys*). La secreta va solo en tu `.env` (`SUPABASE_SECRET_KEY`): nunca en el HTML,
   Git ni un chat, porque salta el RLS.
4. `.env`: `SUPABASE_URL`, `SUPABASE_KEY` (pública) y `SUPABASE_SECRET_KEY`. `web/config.js` (copiar
   `web/config.example.js`): URL y clave **pública**.

```bash
make supabase-subir                        # sube los 50 de ejemplo con la clave secreta (omite repetidos)
make supabase-bajar                        # baja la tabla a data/raw/ (clave secreta; valida el contrato)
make formulario                            # abre web/formulario.html: sesión del equipo + Enviar
```

**Formulario solo para el equipo** ([migración 003](supabase/migraciones/003_formulario_directo.sql)):

```
formulario ──(correo y contraseña: Supabase Auth)──► asociados ──► make supabase-bajar
              solo correos de public.equipo          de una vez, con las restricciones del contrato
```

- **Una sola entrada con correo y contraseña.** Se entra una vez por navegador y se envían todos los registros
  que se quiera: la sesión se guarda y se renueva sola hasta *Salir*. Supabase no envía correos (sin SMTP
  propio solo permite unos pocos por hora). Sin registro abierto: solo cuentas creadas en el panel.
- **Lista de correos permitidos.** La tabla `public.equipo`; la política `equipo_inserta` de `asociados` exige
  `public.es_del_equipo()` (el correo del token está en `equipo`). La clave pública sola, sin sesión, no
  inserta; y con sesión el equipo solo inserta (no lee, edita ni borra).
- **Directo a la tabla.** Cada envío queda de una vez en `asociados`. La base aplica todas las restricciones del
  contrato por fila y no deja repetir cédula ni IDASOCIADOS: si algo no cumple, el formulario muestra el motivo
  y no se guarda.

Activarlo en Supabase (una vez, en el panel):

1. *SQL Editor*: correr la migración **003** (si falta, antes la 002; o `asociados.sql` completo en una base nueva).
2. *Authentication → Sign In / Providers → Email*: activo, y **desactivar "Allow new users to sign up"**.
3. *Authentication → Users → Add user → **Create new user***: correo y contraseña de cada integrante, con
   **Auto Confirm User** marcado (así no se envía ningún correo). La contraseña se le pasa a cada uno en persona.
4. *Table Editor → equipo → Insert row*: el mismo correo **en minúsculas** (y el nombre).

Para probar en tu PC: `make formulario` (o la dirección de Pages, `https://<usuario>.github.io/<repo>/formulario.html`).

**Entrenar con lo que llega a Supabase:**

```
formulario / supabase-subir ──► asociados ──► make supabase-bajar ──► data/raw/supabase_asociados.parquet ──► make abandono
                                              (clave secreta,          (+ copia fechada; valida el contrato)
                                               solo en tu .env)
```

1. En Supabase, *Project Settings → API Keys → Secret keys*: crear una y ponerla en el `.env` como
   `SUPABASE_SECRET_KEY`. Nunca en Git, en la web ni en un chat: salta el RLS.
2. `make supabase-bajar`: descarga, valida con el contrato (si no cumple, no guarda) y deja la última copia.
3. En el `.env`, el interruptor `DATOS_ASOCIADOS` elige con qué entrenan los modelos (una línea sin comentar):
   `data/ejemplo/asociados_ejemplo.parquet` o `data/raw/supabase_asociados.parquet`. Luego `make abandono`.

Se entrena con la copia descargada, no con la BD en vivo: así un entrenamiento se puede repetir con los mismos datos.

**Formulario en internet (GitHub Pages, gratis):** `https://juanccadavid.github.io/proyecto-ml/formulario.html`
(la raíz del sitio redirige ahí).
El workflow [`.github/workflows/formulario.yml`](.github/workflows/formulario.yml) lo publica en cada cambio de
`web/` en `main`, armando `config.js` con variables del repositorio (la clave no está en Git). Activación, una vez:

1. *Settings → Secrets and variables → Actions → Variables*: crear `SUPABASE_URL` y `SUPABASE_KEY`
   (la clave **publishable**; el workflow se niega a publicar una `sb_secret_` o `service_role`).
2. *Settings → Pages → Source*: **GitHub Actions**.
3. *Actions → Formulario (GitHub Pages) → Run workflow*.

Quien abra el enlace sin ser del equipo no puede enviar nada: no tiene cuenta (no hay registro abierto) y,
aunque la tuviera, su correo no está en `equipo`. Si las migraciones 002 y 003 aún no están aplicadas en la
base, lo mínimo es despublicar: *Settings → Pages → Unpublish site*, y publicarlo solo el día de la demo.

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

**Abandono:** experimento `abandono`. Cada script abre una corrida (`modelo_abandono`, `campeon`, `optuna`,
`afinar`, `pycaret`) con una hija por modelo: parámetros (datos, filas, limpieza, hiperparámetros
`model__...`), métricas con prefijo (`entrenamiento_`, `cv_`, `nuevos_`, `esc_t10_s50_...` para los
escenarios) y los gráficos y el JSON adjuntos. El campeón queda en el registro de modelos como `abandono`
con el alias `campeon`. `--sin-mlflow` lo apaga en cualquiera de los scripts. `make mlflow-ui` para verlos.

**Plantilla genérica (`make train`):**
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
