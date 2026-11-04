# Experimentos (MLflow) y versión de los datos

> Parte de la documentación de [proyecto-ml](../README.md). Todas las cifras son de **datos sintéticos**.

**Abandono:** experimento `abandono`. Cada script abre una corrida (`modelo_abandono`, `campeon`, `optuna`,
`afinar`, `pycaret`) con una hija por modelo: parámetros (datos, filas, limpieza, hiperparámetros
`model__...`), métricas con prefijo (`entrenamiento_`, `cv_`, `nuevos_`, `esc_t10_s50_...` para los
escenarios) y los gráficos y el JSON adjuntos. El campeón queda en el registro de modelos como `abandono`
con el alias `campeon`. `--sin-mlflow` lo apaga en cualquiera de los scripts. `make mlflow-ui` para verlos.

**Versión de los datos (liviana, sin copiarlos):** `campeon`, `tecnicas`, `segmentos`, `validacion_cruzada` y
`regresion_salario` registran sus datos como *dataset* de MLflow (`registro.registrar_datos`). Se ven en la
columna y la pestaña **Datasets** de cada corrida:

| Qué queda | Para qué |
|---|---|
| **Huella** (`registro.huella`, 12 caracteres) | Cambia si cambia una sola celda, un vacío o el orden de las filas. Dos corridas con la misma huella usaron los mismos datos. |
| Nombre, contexto (`eleccion`, `nuevos`, `entrenamiento`) y fuente | Qué datos y para qué en la corrida, y de dónde salieron (archivo o generador con su semilla). |
| Esquema y número de filas | Si cambian las columnas o el tamaño, se nota sin abrir el archivo. |
| `datos_huella` y `contrato_huella` como parámetros | Para filtrar en la tabla de corridas, p. ej. `params.datos_huella = "21d2bf025f00"`. También van en `campeon.json` y en el resumen del CI. |

- **No se guarda una copia de los datos.** El extracto real es sensible y no debe quedar en `mlruns/`. Los
  sintéticos se reconstruyen exactos con el generador, la semilla y el commit, que MLflow guarda solo.
- **El modelo registrado no lleva firma (`signature`), a propósito.** Recibe la fila cruda, con conteos enteros
  que a veces traen vacíos: una firma con enteros rechaza los vacíos y una con decimales rechaza los enteros
  (MLflow no convierte int64 a float64). Lo que entra lo valida el contrato de datos.
- **La huella es nuestra, no la de MLflow.** La de MLflow solo mira las primeras 10.000 filas y descarta las
  fechas y las categóricas con vacíos. La nuestra recorre todo el contenido y cuenta el tipo por familia
  (entero, texto, fecha…). Así los datos leídos del parquet en el PC y los generados en memoria en el CI dan la
  misma huella si son los mismos.
- **Si llegan extractos nuevos seguido** y hay que recuperar uno viejo exacto, el paso siguiente sería DVC con
  un almacenamiento privado. Hoy no hace falta.

**Plantilla genérica (`make train`):**
Cada `make train` registra parámetros, métricas y modelo en `mlflow.db` (local, no versionado), incluidas
las opciones de limpieza (`limpieza`, `clip_outliers`, `clip_columns`, `drop_columns`) para comparar con y
sin ellas. Por defecto corre `clean_dataset` (sin renombrar columnas) y un `ColumnDropper` (`--drop-columns`
0.5); `--sin-limpieza` apaga todo.
El JSON de métricas de `make train` en `reports/metrics/` sí se versiona para que el equipo compare resultados
(los que regenera `make pipeline` no: ver [`reports/README.md`](../reports/README.md)).
