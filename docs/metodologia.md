# Metodología: modelos de abandono, campeón y evaluación

> Parte de la documentación de [proyecto-ml](../README.md). Las métricas de abandono y segmentación sintética se
> identifican como tales. Las cifras del extracto real se limitan a dominios y a la regresión del salario, y no
> constituyen validación del abandono.
>
> **Vigencia de las cifras.** Las cifras sintéticas de esta página se recalcularon el 25-nov-2026 con el generador
> actual: 5.000 asociados, 503 etiquetas sintéticas (10,1 %), corte 31-mar-2026 y evento en
> (31-mar, 30-jun-2026]. Experimento principal: `reports/metrics/campeon_20261125_193634.json`; técnicas:
> `reports/metrics/tecnicas_20261125_180351.json`. Son experimentos reproducibles del generador, no validación
> del fondo. Los `campeon_*.json` no se versionan por defecto (los regenera cada `make pipeline`): el que se cita
> aquí se fija a propósito con `git add -f`, como indica [reports/README.md](../reports/README.md). Con el
> extracto real no hay métricas de abandono: no existe una etiqueta futura (ver [datos_reales.md](datos_reales.md)).

`make abandono` responde "¿este asociado abandona el fondo?" con dos enfoques
([`src/proyecto_ml/abandono.py`](../src/proyecto_ml/abandono.py)):

- **Supervisados** (aprenden de `EVENTO_RETIRO_3M_SINTETICO`), con el mismo preprocesador: métricas en entrenamiento,
  validación cruzada y datos nuevos, más una tabla que los compara sobre los nuevos.

  | Modelo | Tipo | Idea |
  |---|---|---|
  | Ridge | lineal | regresión con penalización sobre la clase -1/+1; coeficientes interpretables |
  | Regresión logística | lineal | probabilidad de abandono; también interpretable |
  | Random Forest | **bagging** | muchos árboles independientes (muestras bootstrap) que votan |
  | HistGradientBoosting | **boosting** | árboles en secuencia, cada uno corrige al anterior (estilo LightGBM) |
- **KMeans** (sin objetivo): agrupa por comportamiento (solicitudes, ahorros, intención de retiro) en 2 grupos;
  el de "riesgo" se elige por su perfil, no por el objetivo. Después se compara con el evento futuro sintético.

Para validar, se generan asociados **que los modelos nunca vieron** (`generador.asociados_nuevos`: mismo generador,
otra semilla y la **misma tasa de retiro** que los datos de desarrollo; con otra prevalencia, la precisión, la PR-AUC
y el umbral fijo de 0,5 no se podrían comparar con la validación cruzada ni con `make campeon`). Hasta el 25-nov-2026
`make abandono`, `afinar`, `optuna` y `pycaret` los generaban con 30 % de retiros: con los mismos modelos, la PR-AUC
sobre nuevos daba ~0,79 (Random Forest 0,788; HistGB 0,784) y con la tasa de desarrollo da ~0,58 (0,584; 0,586).
La diferencia la explica la prevalencia, no el modelo:

```bash
make abandono                                        # entrena con los datos de make campeon, valida con 1000 nuevos
make abandono ARGS="--nuevos 5000 --semilla-nuevos 7 --alpha 10"
make abandono ARGS="--validar mi_archivo.parquet"    # validar además con otro archivo
make abandono ARGS="--sin-limpieza"                  # sin la limpieza de cleaning.py, para comparar
```

**Limpieza activa por defecto** ([`cleaning.py`](../src/proyecto_ml/cleaning.py), de Juan Sebastián):

| Paso | Dónde | Qué hace |
|---|---|---|
| `contrato.limpiar_extracto` | al cargar (`a.cargar`), antes de partir | espacios, textos tipo 'N/A' → vacío, números como texto, filas repetidas |
| `ColumnDropper(max_missing=0.5)` | dentro del pipeline (aprende en train) | descarta variables con más de 50 % vacías o con un solo valor |
| `OutlierClipper("iqr")` | rama `clip` del pipeline | recorta atípicos de `SALARIO`, `EDAD` y las dos antigüedades en meses (no los conteos) |

**Qué variables aportan a cada modelo.** `make abandono` calcula la *permutation importance* sobre los
asociados nuevos: desordena una variable a la vez y mide cuánto cae el AUC. Sirve igual para los cuatro
modelos (HistGB no trae `feature_importances_`) y mide la variable completa, no cada columna del OneHot.
Imprime las 5 principales de cada modelo y guarda `reports/figures/abandono_importancia.png`.

Además, para el Random Forest, `reports/figures/abandono_importancia_rf.html` es un gráfico **interactivo**
(Plotly: al pasar el mouse muestra variable e importancia) con su `feature_importances_` por columna ya
transformada, con el prefijo de su rama (`num__`, `clip__`, `cat__SEXO_F`...). Es la importancia propia del
bosque (cuánto reduce la impureza): favorece a las continuas, que admiten muchos cortes. En
`reports/metrics/abandono_20261125_184901.json`, `EDAD` y las dos antigüedades tienen 0,066–0,071 de MDI cada una y no
aparecen entre las 5 primeras por permutación. `SALARIO` sí aporta en las dos (MDI 0,095; permutación 0,037 en el
bosque), porque el generador lo hace depender de la etiqueta sintética. En código:
`figura_importancia_interactiva(importancia_arbol(rf), "titulo")`.

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

El primer paso (`PrepararVariables`) tipa EDAD y las antigüedades en meses y elige las columnas. Por eso el modelo recibe la
**fila tal como llega** (extracto, Supabase, formulario): `modelo.predict(df)`, sin llamar a `preparar()` antes.
Eso vale también para el campeón guardado y para `models:/abandono@campeon` en MLflow. Si la fila ya viene
preparada, la deja igual; así siguen funcionando el simulador y la importancia por permutación.

```python
X, y = a.separar(vista)  # quita solo la etiqueta sintética; #RETIROS queda como predictor histórico
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

## Definición del problema: la foto del asociado y el horizonte

| Elemento | Definición (contrato v2) | Dónde |
|---|---|---|
| **Unidad y población** | Un asociado **activo** al corte C (31-03-2026 en el extracto real; todos tienen `ESTADO` = A) | `abandono.foto` |
| **Momento de la predicción** | t0 = C: ahí el fondo mide todas las variables | `abandono.fecha_observacion` |
| **Variables** | Disponibles en t0: edad y antigüedades a C; conteos `ULT3MES` de (C − 3 meses, C]; productos al corte; `#RETIROS` (retiros previos con reingreso) acumulado hasta C | `abandono.preparar` |
| **Evento (abandono confirmado)** | Retiro del fondo en **(C, C + 3 meses]**: posterior a toda variable | `abandono.fin_horizonte` |
| **Etiqueta disponible** | Solo sintética: `EVENTO_RETIRO_3M_SINTETICO`. El extracto real no trae el evento y el fondo no puede construirlo | `generador.py` |
| **Lo que NO es abandono** | Una solicitud de retiro parcial, de cancelación de ahorros o de disminución de cuotas: son **señales** medidas en t0, no el evento | — |

Como el evento es posterior a t0 y todas las variables son de t0 o anteriores, el diseño no tiene fuga temporal
por construcción. En el esquema anterior (v1, con fecha de retiro) la foto se tomaba en t0 = C − 3 meses y el
evento caía entre t0 y el corte; con variables medidas al corte, esa definición habría puesto el evento **antes**
de la medición, que es fuga. `abandono.foto` ya no filtra filas ni toca la etiqueta: anota t0 y el horizonte.

```python
vista = a.foto(df)  # t0 = 2026-03-31; el evento cae hasta 2026-06-30 (vista.attrs["fin_horizonte"])
X, y = a.separar(vista)  # el pipeline selecciona y tipa los predictores permitidos
a.resumen_foto(df)  # t0, fin del horizonte, asociados y cuántos tienen el evento sintético
```

**Población que se puede evaluar.** Con sintéticos, todos los asociados de la foto (la etiqueta existe). Con el
extracto real, ninguno: sin el evento no hay métrica de abandono, solo el puntaje (`predecir-real`) y su
parecido con los sintéticos de entrenamiento (`validacion.desplazamiento`). EDA de la foto:
[`sesiones/02_juan_eda_foto_asociado.ipynb`](../sesiones/02_juan_eda_foto_asociado.ipynb).

## El modelo campeón: `make campeon`

Los afinados, Optuna y PyCaret son **experimentos** (quedan en MLflow); el **campeón** es el único modelo
"oficial": lo usa el simulador ([`campeon.py`](../src/proyecto_ml/campeon.py),
[`scripts/elegir_campeon.py`](../scripts/elegir_campeon.py)).

- **Datos:** los de desarrollo, 5.000 asociados generados con semilla 7 y **10 % de etiquetas sintéticas**.
  La etiqueta se asigna a la foto completa: 503 eventos (10,1 %); no se filtran 398 filas como en la definición
  temporal anterior. `make datos-desarrollo` los escribe en `data/processed/`; sin el archivo se generan en
  memoria, idénticos. Los 50 de `data/ejemplo` quedan para el contrato y las pruebas.
- **Criterio:** PR-AUC promedio en validación cruzada **repetida** (5 particiones × 3 repeticiones), no en los
  asociados nuevos del generador. PR-AUC se informa junto a la prevalencia (10,1 %); no es evidencia de
  desempeño real.
- **Candidatos:** los cuatro supervisados y dos alternativas motivadas por codificación/desbalance
  (`campeon.lista_candidatos`). Se conservan como exploratorios; ni los resultados de `make tecnicas` ni los de
  distintas semillas justifican declarar una técnica general ganadora. Con `--optuna N` se agregan los árboles
  afinados, con la cautela de que compiten en los mismos datos.
- **Umbrales**, con las predicciones fuera de partición (cada asociado puntuado por un modelo que no lo vio):
  - *alerta*: el que maximiza **F2** (el recall pesa el doble: perder a quien se va cuesta más que una llamada);
  - *alto*: el menor puntaje desde el cual **al menos 8 de cada 10 alertas son etiquetas positivas sintéticas**.
- **Tres zonas:** 😟 **se retira** (≥ alto) · 😐 **zona gris** (entre los dos) · 😀 **se queda** (< alerta).
- **KMeans** con los mismos datos y sin el objetivo: el simulador muestra en qué grupo cae el asociado y la
  tasa del evento futuro sintético del grupo.

| Candidato | PR-AUC ± entre particiones | IC 95 % (bootstrap) | campeón − candidato (IC 95 %) | umbral alerta | precisión / recall |
|---|---|---|---|---|---|
| **Random Forest (campeón sintético)** | **0,568 ± 0,043** | 0,535–0,613 | — | 0,117 | 0,306 / 0,716 |
| HistGradientBoosting (boosting) | 0,559 ± 0,042 | 0,527–0,610 | 0,010 (−0,018 a 0,037) | 0,207 | 0,281 / 0,769 |
| Logística + TargetEncoder | 0,547 ± 0,042 | 0,506–0,591 | 0,018 (−0,006 a 0,042) | 0,528 | 0,281 / 0,761 |
| Logística sin balancear | 0,538 ± 0,039 | 0,499–0,585 | 0,028 (0,003 a 0,054) ✔ | 0,102 | 0,249 / 0,789 |
| Regresión logística | 0,531 ± 0,041 | 0,493–0,581 | 0,035 (0,006 a 0,063) ✔ | 0,430 | 0,232 / 0,827 |
| Ridge | 0,498 ± 0,039 | 0,467–0,551 | 0,072 (0,034 a 0,109) ✔ | 0,0003 | 0,259 / 0,757 |

✔ = la diferencia es concluyente según la t corregida de Nadeau-Bengio. La elección del Random Forest no es
concluyente frente a HistGradientBoosting ni TargetEncoder. El campeón designa únicamente el ganador de esta
comparación sintética concreta; no una recomendación para producción.

El umbral de alerta maximiza F2 (0,117); el umbral alto es 0,410. Fuera de partición, la alerta tiene
precisión 0,306 y recall 0,716; la zona alta, precisión 0,801 y recall 0,416. En validación anidada, la
precisión baja a 0,284 para alerta y 0,793 para zona alta (objetivo 0,80), con recall 0,724 y 0,396
respectivamente. En 5.000 asociados nuevos sintéticos, las zonas contienen 277 (tasa sintética 74,4 %),
912 (13,5 %) y 3.811 (3,7 %). Son proporciones condicionadas por el generador, no tasas reales.

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
make campeon                                     # t0 = corte, evento a 3 meses, datos de desarrollo
make campeon ARGS="--horizonte 12 --precision-min 0.9"
make campeon DATA=data/raw/supabase_asociados.parquet   # se detiene sin una etiqueta futura válida
make simulador                                   # usa el campeón
```

### Incertidumbre: qué tan seguros estamos de cada cifra

- **± es entre particiones, no entre repeticiones.** La PR-AUC del campeón varía ± 0,043 entre las 15
  particiones; entre las tres repeticiones, la desviación es 0,006. El IC bootstrap de la PR-AUC es
  0,535–0,613.
- **El campeón contra cada candidato, partición a partición** (todos usan las mismas particiones), con la prueba t
  corregida de Nadeau y Bengio, que tiene en cuenta que las particiones comparten casi todo el entrenamiento. Si
  el intervalo de la diferencia toca el 0, es un empate (`campeon.json`: `comparacion` y `empatados`). La
  diferencia se mide con la PR-AUC de cada partición; el criterio de elección, con la del puntaje fuera de
  partición completo, así que en un empate muy cerrado el orden puede no coincidir.
- **Los umbrales, en validación anidada** (`umbrales_honestos`): elegirlos y medirlos sobre las mismas
  predicciones es optimista. En cada partición externa se eligen solo con el entrenamiento y se miden en la
  prueba:

  | Umbral | fuera de partición (optimista) | validación anidada | asociados nuevos |
  |---|---|---|---|
  | alerta: precisión / recall | 0,306 / 0,716 | **0,284 / 0,724** | 0,277 / 0,701 |
  | se retira: precisión / recall (pedida 0,80) | 0,801 / 0,416 | **0,793 / 0,396** | 0,744 / 0,410 |

- **Sensibilidad a semillas y baseline.** Con los datos de desarrollo generados con las semillas 7/42/2026
  (`reports/metrics/tecnicas_20261125_180351.json`, `reports/metrics/tecnicas_20261125_195011.json` y
  `reports/metrics/tecnicas_20261125_195110.json`), la PR-AUC base fue 0,531/0,509/0,529;
  sin balancear, 0,541/0,516/0,542; TargetEncoder, 0,552/0,505/0,555; Normalizer, 0,544/0,554/0,547.
  El DummyClassifier de prior obtuvo AP igual a la prevalencia (0,094–0,101) y ROC-AUC 0,5. Sin balancear
  fue positivo en los tres seeds, aunque los intervalos pareados no lo respaldan en todos; TargetEncoder
  empeoró en uno y Normalizer no tiene una justificación de dominio suficiente para promoverlo solo por la
  métrica. No se declara ganador general ni se elige técnica para el fondo real.
- **Candidatos abiertos** (`campeon.lista_candidatos`): se conservan los cuatro supervisados, logística sin
  balancear y TargetEncoder como hipótesis comparables; ninguno se considera candidato validado en producción.
  `make campeon ARGS="--optuna 20"` añade árboles afinados sobre estos mismos datos y puede sobreajustar la
  búsqueda.

### Evaluación del campeón: ROC, matriz de confusión, MCC/Kappa y SHAP

`make campeon` también guarda, todo con las predicciones **fuera de partición**:

- `reports/figures/campeon_umbrales.html`: precisión y recall según el umbral (el *trade-off*), con las
  tres zonas.
- `campeon_roc.html`: la curva ROC de todos los candidatos, con los umbrales del campeón marcados.
- `campeon_confusion.html`: la matriz de confusión con cada umbral. Usa el umbral elegido, no el 0,5 de
  `predict`, así que cuadra con la precisión y el recall de la tabla. Los conteos también van en
  `campeon.json` (`confusion_cv`).
- `campeon_shap.html`: SHAP en 100 asociados nuevos y la explicación del de mayor riesgo.
- `campeon_calibracion.html`: si la probabilidad del modelo es creíble (el "30 %" es un 30 % de verdad), por
  grupos de probabilidad con cortes finos cerca de 0; el tamaño del punto es el número de asociados del grupo.
  Solo si el campeón da probabilidades (no con Ridge). El Brier va en las métricas, con el **Brier skill**
  (`brier_skill`: 1 = perfecto, 0 = igual que decir siempre la tasa de retiro). Las probabilidades y la
  calibración se refieren exclusivamente a la etiqueta sintética. Si el campeón queda por debajo de 0,5, el
  simulador muestra "Puntaje de riesgo (no calibrado)" en
  vez de "Probabilidad de retiro".

| Umbral del campeón | etiqueta sintética 0: sin alerta / alerta | etiqueta sintética 1: sin alerta / alerta |
|---|---|---|
| alerta = se retira (0,117) | 3.682 / 815 | 143 / **360** (recall 71,6 %, precisión 30,6 %) |

**MCC y Kappa** (al umbral de alerta) van en la tabla de candidatos y en MLflow (`cv_mcc`, `cv_kappa`):

| | Ridge | Regresión logística | Random Forest (bagging) | HistGradientBoosting (boosting) | Logística sin balancear | Logística + TargetEncoder |
|---|---|---|---|---|---|---|
| MCC | 0,250 | 0,327 | 0,379 | 0,370 | 0,338 | 0,367 |
| Kappa | 0,278 | 0,244 | 0,335 | 0,310 | 0,267 | 0,309 |

El MCC pesa por igual las dos clases de error; nosotros no (ver la siguiente sección), así que no decide.

### Qué métrica usamos y por qué

```
¿Clases balanceadas (50/50)?  NO: 10,1 % de etiquetas sintéticas
   └─> Accuracy engaña: decir "nadie se va" acierta el 89,9 %
       └─> MCC / Kappa (corrigen el desbalance) para comparar sin fijarse en el costo
           └─> ¿Qué es más costoso?  Los FALSOS NEGATIVOS: un asociado que se va sin que nadie lo llame
               └─> RECALL: umbral de alerta con F2 (el recall pesa el doble que la precisión)
```

- **Para elegir el modelo**, la PR-AUC: resume la precisión y el recall en todos los umbrales y su baseline
  es la prevalencia (10,1 %). ROC-AUC es complementaria, no evidencia de generalización real.
- **Para el umbral de alerta**, F2. Una llamada de más cuesta poco; perder un asociado cuesta mucho.
- **Para la zona "se retira"**, una precisión mínima de 0,8: cuando el modelo está seguro, debe acertar.
- **MCC y Kappa** se reportan para comparar con PyCaret y con la literatura, pero no deciden.

### SHAP: por qué el modelo le dio a este asociado su puntaje

[`explicar.py`](../src/proyecto_ml/explicar.py) reparte el puntaje de **cada asociado** entre sus variables:

```
puntaje = promedio de 100 asociados de referencia + contribución de cada variable
```

Una contribución en rojo sube el riesgo frente al promedio; una en azul lo baja. La suma da exactamente el
puntaje del modelo. Las contribuciones se miden en puntos de probabilidad, o en distancia a la frontera si el
campeón es Ridge. Se explican las variables originales (EDAD, SEXO...), no las columnas del OneHot, con
`shap.PermutationExplainer`. Ese método solo necesita el puntaje, así que sirve para todos los candidatos.

- **En el simulador**, debajo del riesgo, aparece "¿Por qué este puntaje?": una cascada con las 8 variables
  que más pesan para ese asociado. Se recalcula al mover cualquier control.
- **En `make campeon`**, la importancia global es la |contribución| promedio en 100 asociados nuevos.
  Arriba quedan retiro parcial, ahorros contractuales, servicios, disminución de cuotas y cancelación de ahorros.
  El generador asigna la etiqueta antes y condiciona directamente varias de estas señales a ella; esta importancia
  describe las reglas sintéticas, no relaciones verificadas en el fondo.

```python
from proyecto_ml import explicar as e

explicador = e.Explicador(campeon, datos_de_referencia)  # crudos o preparados
explicador.explicar(fila)  # variable | valor | contribucion, de la que más pesa a la que menos
```

## Auditoría de fuga y validación

Cada punto se verificó en el código y tiene una prueba que impide que vuelva a pasar:

| Qué se revisó | Resultado | Dónde / prueba |
|---|---|---|
| Imputación, codificación, escalado, recortes, PCA, RFE y TargetEncoder | dentro del Pipeline: se ajustan solo con el entrenamiento de cada partición | `armar_pipeline` · `test_el_preprocesamiento_aprende_solo_del_entrenamiento` |
| SMOTE / ADASYN | solo en el `fit` (pipeline de imblearn), una vez por partición y con sus filas de entrenamiento; predecir no re-muestrea | `test_smote_solo_en_el_entrenamiento_de_cada_particion` |
| Identificadores, contacto, fechas crudas, `ESTADO` y objetivo | no entran a los modelos | `preparar` · `test_identificadores_y_fuga_no_son_predictores` |
| Selección del campeón, umbrales e hiperparámetros | con validación cruzada sobre los datos de elección; los asociados nuevos no participan; los umbrales se miden además en validación anidada | `campeon.elegir`, `umbrales_honestos` |
| Tiempo | un solo corte; no se afirma validación fuera de tiempo | `docs/datos_reales.md` |
| Cachés del simulador | SHA-256 del contenido de modelo, KMeans y `campeon.json` (umbrales); marcas de contenido de datos, corte, horizonte y predictores. SHAP depende solo del modelo de puntaje para evitar invalidaciones innecesarias | `campeon.firma`, `datos_campeon.marca_datos`, pruebas AppTest de cambios con ruta conservada |
| Reproducibilidad | misma semilla, mismos candidatos y métricas | `test_reproducible` |

**Riesgos y su estado con el extracto real:**

- **Conteos medidos después de t0.** El fondo confirmó que los conteos se miden al corte (`ULT3MES`, los tres meses
  anteriores). Para entrenar con datos reales haría falta además una etiqueta posterior al corte, que no existe.
- **Optuna se afina con los mismos datos** con que compite (`--optuna`): su PR-AUC de validación cruzada es algo
  optimista.
- **Otras distribuciones.** El contrato detiene cambios de esquema y rangos, no cambios de distribución.
  `predecir-real` aplica a los asociados reales los umbrales (F2 y precisión ≥ 0,8) elegidos con sintéticos: sin
  etiqueta real no se pueden recalibrar ni medir.
- **TargetEncoder con varias fotos.** Su validación cruzada interna no agrupa por asociado; hoy no importa porque
  ningún flujo entrena con varias fotos del mismo asociado (detalle en [datos_reales.md](datos_reales.md)).

## Validación: qué miden las cifras (para la sustentación)

> Las métricas sobre asociados nuevos miden si el modelo **recupera la regla del generador**. Una validación real
> tendría que ser **temporal** (entrenar con un corte y medir con retiros posteriores), pero el fondo entrega un
> solo corte y no puede decir quién se retiró después: con los datos disponibles no hay validación real del
> abandono. `validacion.particion_temporal` solo prueba el mecanismo con la etiqueta del generador.

**Qué del campeón sale directamente de la lógica sintética.** El generador hace depender de la etiqueta solo
siete variables (`generador.PRODUCTOS`, `DISMINUCION_CUOTAS` y un factor sobre el salario); las demás se sortean
sin relación con ella. La importancia SHAP del campeón (`shap_nuevos` en
`reports/metrics/campeon_20261125_193634.json`, 100 asociados nuevos) reproduce esa construcción:

| Variable | ¿Depende de la etiqueta en el generador? | |contribución| SHAP promedio |
|---|---|---|
| `#SOLICITUD_RETIROPARCIALAHORROS` | sí: P(tener) 0,02 → 0,30 | 0,043 |
| `#CANTIDAD_AHORROSCONTRACTUALES` | sí: 0,43 → 0,10 | 0,039 |
| `#CANTIDAD_SERVICIOS` | sí: 0,35 → 0,10 | 0,024 |
| `#SOLICITUD_DISMINUCIONCUOTAS` | sí: P(SI) 0,005 → 0,10 | 0,021 |
| `#SOLICITUD_CANCELACIONAHORROS` | sí: 0,005 → 0,17 | 0,020 |
| `SALARIO` | sí: × e^0,08 en la etiqueta 1 | 0,014 |
| `#SOLICITUDCREDITOS_ULT3MES` | sí: 0,15 → 0,03 | 0,010 |
| `ESTRATO`, `EDAD`, `CODPROFESION`, `ESTADOCIVIL`, `ANTIGUEDAD_F_MESES`… | **no** | ≤ 0,005 (piso de ruido) |
| `#CANTIDAD_AHORROSALAVISTA` | **no**: 0,21 en las dos clases | por debajo del piso |

Lectura: las siete primeras son exactamente las variables con señal construida, en un orden que sigue el tamaño
de la diferencia que les impuso el generador; las que no la tienen quedan en el piso de ruido. Por eso **ninguna
importancia, umbral ni métrica del campeón es extrapolable al fondo**: describen la regla del generador. Lo que sí
se lleva al extracto es el pipeline (contrato, preparación, validación, umbrales anidados), listo para reentrenar
el día que exista una etiqueta real.

Mientras tanto, `make campeon` valida cada candidato, con su umbral, en escenarios menos favorables
([`validacion.py`](../src/proyecto_ml/validacion.py)): otra tasa de retiro y una señal más débil
(`GeneradorAsociados(fuerza_senal=...)`: 1 = el generador de siempre, 0,5 = retirados la mitad de distintos).
La tabla muestra solo al campeón actual (Random Forest). Estos escenarios cambian la regla sintética; no simulan
ni estiman el rendimiento sobre el fondo:

| Escenario sintético (horizonte de 3 meses) | PR-AUC · precisión · recall |
|---|---|
| tasa 30 %, señal 100 % (generador) | 0,760 · 0,604 · 0,703 |
| tasa 10 %, señal 100 % | 0,563 · 0,277 · 0,701 |
| tasa 10 %, señal 50 % | 0,315 · 0,210 · 0,488 |
| tasa 10 %, señal 25 % | 0,168 · 0,149 · 0,322 |

Las dos primeras filas difieren solo en la prevalencia: con el mismo modelo y la misma señal, la PR-AUC pasa de
0,563 a 0,760 y la precisión de 0,277 a 0,604. Por eso la PR-AUC solo se compara a igual tasa de retiro, y los
asociados nuevos de todos los scripts salen con la de desarrollo (`generador.asociados_nuevos`).

**Si la tasa de retiro del fondo no es 10 %.** Los umbrales salen de la prevalencia sintética. Reponderar las
predicciones fuera de partición (`campeon.sensibilidad_prevalencia`: positivos × π'/π, negativos × (1 − π')/(1 − π))
da la precisión esperada a otra tasa sin etiquetas individuales: basta la tasa trimestral de retiro del fondo, un
agregado. Supone que cambia **cuántos** se retiran y no **cómo es** quien se retira (cambio de prevalencia); sin
etiquetas reales ese supuesto no se puede comprobar con el fondo. Campeón actual (`campeon.json`, clave
`prevalencia`):

| Tasa trimestral de retiro | Precisión de la alerta (umbral 0,117) | Precisión de "se retira" (umbral 0,410) | Umbral alto para 80 % |
|---|---:|---:|---:|
| 2 % | 0,075 | 0,423 | 0,757 |
| 5 % | 0,172 | 0,654 | 0,572 |
| 10 % (sintética) | 0,305 | 0,800 | 0,411 |
| 20 % | 0,497 | 0,900 | 0,270 |
| 30 % | 0,629 | 0,939 | 0,197 |

El recall (0,716) no cambia: depende solo de quienes se retiran. **Comprobación** con el escenario del generador al
29,1 %, que es un cambio de prevalencia puro: precisión esperada 0,619, medida 0,604. Con una tasa real del 2 %, los
umbrales actuales prometerían que 8 de cada 10 asociados en "se retira" se van, y serían 4. Para una probabilidad
calibrada, `campeon.ajustar_probabilidad` (Saerens, Latinne y Decaestecker, 2002) la lleva a la nueva tasa; el
campeón usa `class_weight="balanced"` y su puntaje no es una probabilidad calibrada, así que con él se usan los
umbrales recalculados. Las dos funciones tienen pruebas con el generador a otra tasa
(`test_precision_esperada_a_otra_tasa_coincide_con_la_medida`).

El deterioro al debilitar la señal ilustra la dependencia del modelo respecto a las reglas programadas. No permite
acotar el rendimiento real: la tasa real futura tampoco está disponible y no hay etiquetas con las que calibrar
umbrales o probabilidades.

**¿Se parecen los asociados reales a los sintéticos con que se entrenó el campeón?** Sin etiqueta no se puede
medir el desempeño real, pero sí si el campeón se aplica dentro de lo que vio. `validacion.desplazamiento` hace una
**validación adversaria**: un clasificador intenta distinguir filas sintéticas de reales con las variables del
modelo (ROC-AUC fuera de partición; 0,5 = indistinguibles). Lo imprime `make predecir-real`.

| Comparación | ROC-AUC | Lectura |
|---|---|---|
| Sintéticos frente a otros sintéticos (control; 3 semillas) | ROC-AUC 0,501–0,506; comportamiento 0,485–0,492 | control cercano al azar tras normalizar ambos lotes con el mismo contrato |
| Sintéticos frente al extracto real, todas las variables | ROC-AUC 1,00 | dominios claramente separables; no mide abandono |
| … solo variables de comportamiento | ROC-AUC ≈ 0,60 | desplazamiento moderado; no es una métrica del clasificador de abandono |
| … variables que distinguen | geografía ROC-AUC 1,00; empresa 1,00; profesión ≈ 0,95 | categorías principales fuera del soporte sintético |

Distribuciones agregadas sintético-real: KS de salario 0,060, edad 0,049, antigüedad en el fondo 0,043 y
antigüedad laboral 0,083. Fuera del intervalo sintético P1–P99 queda 1,1 %, 2,5 %, 5,6 % y 9,5 %,
respectivamente. Los conteos de comportamiento tienen KS ≤ 0,020. En categóricas, la distancia de variación
total es 1,00 para empresa y códigos geográficos; 100 % de empresas y ciudades/departamentos reales y 81,8 %
de profesiones quedan fuera del soporte sintético. Sexo, tipo de contrato y estrato tienen distancia ≤ 0,017 y
ningún valor real fuera de soporte.

Lectura: el campeón **extrapola en categorías de perfil** y hay diferencias cuantitativas en salario/antigüedad;
las marginales de comportamiento son más próximas. No se modificó el generador para reducir estas diferencias:
no había justificación independiente de proceso ni etiquetas reales que validaran el cambio. `predecir-real` sigue
siendo un orden basado en reglas sintéticas, no una probabilidad ni validación del fondo.

## KFold, StratifiedKFold y TimeSeriesSplit: `make validacion`

[`scripts/comparar_validacion.py`](../scripts/comparar_validacion.py) parte los mismos datos del campeón de las
tres formas y entrena la logística en cada partición (`validacion.comparar_validaciones`):

| Estrategia | PR-AUC | ± entre particiones | retirados por prueba | filas de entrenamiento |
|---|---|---|---|---|
| KFold (al azar) | 0,544 | 0,026 | 78 a 122 | 4.000 |
| **StratifiedKFold** (la del campeón) | 0,526 | **0,053** | 100 a 101 | 4.000 |
| TimeSeriesSplit (por fecha de ingreso) | 0,542 | 0,036 | 76 a 97 | 835 → 4.167 |

- **KFold**: la prevalencia de prueba varía más (78–122 eventos sintéticos) que en la estratificada. No se debe interpretar el promedio algo mayor como ventaja concluyente.
- **StratifiedKFold**: deja la misma tasa de retiro en cada prueba. Por eso la usan el campeón, Optuna y la
  grilla.
- **TimeSeriesSplit**: solo prueba con filas posteriores a las de su entrenamiento. La foto no tiene una
  fecha por fila, así que se ordena por fecha de ingreso: entrena con los más antiguos y prueba con los más
  nuevos. Las primeras particiones entrenan con pocos datos y por eso varía tanto. Con un solo extracto,
  `particion_temporal` (foto vieja → foto reciente) es una prueba de estrés, no una validación fuera de tiempo: las
  dos fotos son las mismas personas con los mismos conteos (ver README, "Resultados").

```bash
make validacion                                  # logística, 5 particiones
make validacion ARGS="--modelo random_forest --folds 10"
```

## Regresión de ejemplo: el salario (`make regresion`)

El abandono es una clasificación. [`regresion.py`](../src/proyecto_ml/regresion.py) es la contraparte de
regresión, con un uso real: **estimar el salario** de quien no lo reporta (el 5 % en el generador). Se entrena
con quienes reportan un salario plausible (`SALARIO_PLAUSIBLE`: el relleno 987.654 y los de miles de millones no
se aprenden) y se valida con **KFold**, porque en regresión no hay clases que estratificar. SALARIO no entra como
predictor (`test_regresion_sin_fuga_y_referencia`).

**Elección y evaluación separadas.** Antes de elegir se aparta una **reserva** del 20 %. La validación cruzada
compara la referencia (`DummyRegressor`, predecir el promedio) con Ridge, Random Forest, HistGB y dos modelos que
aprenden el logaritmo del salario (Ridge y HistGB en logaritmo) sobre el resto, y elige por R²; ese R² es optimista para el elegido, porque es el mejor de varios sobre las mismas particiones. La
cifra que se reporta es la de la **reserva**: el elegido y la referencia, entrenados sin ella, medidos una sola vez
(`supera_referencia` se decide ahí). En los datos sintéticos, Ridge no supera al promedio en la reserva. En el
extracto real, HistGradientBoosting sí supera al promedio para **estimar salario** en esa reserva; esto no valida
el clasificador de abandono.

**Error relativo y por tramo.** Además de MAE, RMSE y R², se reporta `MdAPE` (mediana del error relativo) y el error
por quintil del salario predicho (`regresion.error_por_tramo`). Con el extracto real el error es **heterocedástico**:
crece con el salario, y con el HistGB en pesos el error relativo casi se duplicaba en el quintil más alto. Por eso
compiten los modelos en logaritmo: el error que minimizan es relativo, como en la ecuación de salarios de Mincer.

**Intervalo de predicción por tramo.** Cada estimación lleva un intervalo multiplicativo [ŷ ÷ f, ŷ × f] que debería
contener 8 de cada 10 salarios reales (`regresion.calibrar_intervalo`). f se calibra con los errores relativos
**fuera de partición** de la parte de elección (conformal con corrección de muestra finita), nunca con la reserva, y
hay **un f por quintil** del salario predicho (conformal "Mondrian"). La reserva mide la cobertura en total y por
quintil: un intervalo que cubre el 80 % en promedio pero no en los salarios altos no sirve para esos asociados.

**Es una estimación transversal.** Salario y predictores se miden en el mismo corte: el modelo describe cómo van
juntos en esa foto y sirve para estimar el salario de quien no lo reporta en ese corte. No pronostica salarios
futuros ni mide efectos causales.

CV sintética del 80 % de elección (5 folds; valores de consola del `make pipeline` del 25-nov):

| Métrica | Referencia (promedio) | Ridge | Ridge en log | Random Forest | HistGB en log | HistGB |
|---|---:|---:|---:|---:|---:|---:|
| MAE | 1.170 M | 1.195 M | 1.096 M | 1.217 M | 1.154 M | 1.278 M |
| RMSE | 1.924 M | 1.945 M | 1.957 M | 1.963 M | 1.992 M | 2.021 M |
| R² | −0,001 | −0,023 | −0,035 | −0,042 | −0,073 | −0,103 |

En la reserva sintética, Ridge obtiene R² −0,022 y MAE mayor que la referencia (1,252 M frente a 1,234 M); en
1.000 asociados nuevos (952 con salario), R² es −0,016 frente a −0,003 del promedio. No supera el baseline. El
intervalo del 80 % cubre el 78,1 % de la reserva y entre 74,0 % y 83,5 % por quintil (200 asociados por quintil),
con f de 1,61 a 1,70 (`reports/metrics/regresion_20261125_192920.json`). El generador
muestrea el salario mediante cuantiles y aplica además un factor sintético asociado a la etiqueta; por tanto, no es
correcto describirlo como lognormal independiente ni usarlo como evidencia de una relación real.

En `make real`, la validación cruzada elige **HistGB en logaritmo** (R² 0,222 frente a 0,216 del HistGB en pesos; la
diferencia es menor que la desviación entre particiones, 0,02). Sobre la misma reserva real (2.891 asociados):

| En la reserva real | Promedio (referencia) | HistGB en pesos (antes) | HistGB en logaritmo (ahora) |
|---|---:|---:|---:|
| R² | 0,000 | 0,190 | 0,189 |
| MAE frente al promedio | — | −16 % | −21 % |
| MdAPE | 0,210 | 0,159 | 0,148 |
| MdAPE en el quintil más alto | — | 0,280 | 0,225 |
| Intervalo del 80 %: cobertura total | — | — | 79,8 % |
| Cobertura por quintil (un f para todos → un f por quintil) | — | — | 66,8 %–88,8 % → 78,2 %–82,2 % |

El R² no cambia; lo que mejora es el error relativo, sobre todo en los salarios altos (MAE 5,7 % menor que el del
HistGB en pesos). Con un solo f (1,37), el intervalo cubría el 80 % en promedio pero el 66,8 % en el quintil más
alto; con un f por quintil (de 1,30 a 1,60), todos los quintiles quedan entre 78,2 % y 82,2 %. En los salarios más
altos la estimación sigue siendo la menos precisa: su intervalo va de ÷1,60 a ×1,60. Es evidencia agregada para
estimación transversal del salario, no predicción de retiro.
No se publican salarios absolutos del extracto. Resultados completos locales en `reports/metrics/` y
`reports/figures/regresion_salario.html`.

## Supervisado vs. no supervisado: segmentación con KMeans y DBSCAN (`make segmentos`)

| | Supervisado | No supervisado |
|---|---|---|
| Aprende de | `EVENTO_RETIRO_3M_SINTETICO` | solo el comportamiento: nunca ve la etiqueta sintética |
| Modelos | Ridge, logística, Random Forest, HistGB (el campeón) | KMeans y DBSCAN ([`segmentacion.py`](../src/proyecto_ml/segmentacion.py)) |
| Responde | ¿este asociado se retira? (puntaje y zona) | ¿qué tipos de asociado hay? (segmentos) |
| Se valida con | PR-AUC, recall, precisión (validación cruzada) | silueta; y, solo al final, la tasa de retiro de cada segmento |

[`scripts/segmentar.py`](../scripts/segmentar.py) agrupa a los asociados de la foto por sus 6 variables de
comportamiento, imputadas y estandarizadas:

- **KMeans** parte a todos en `k` grupos alrededor de `k` centros. En la ejecución actual, k = 2 tiene la
  silueta más alta (0,666); el primer grupo queda dominado por las señales que el generador condiciona a la
  etiqueta. Es una estructura sintética, no un segmento de riesgo real.
- **DBSCAN** deja 14 de 5.000 como atípicos; solo captura 1,6 % de las etiquetas sintéticas y no se considera
  un detector útil en esta ejecución.

| Marcar "en riesgo" a… | asociados | se retira | recall | ARI con el retiro |
|---|---|---|---|---|
| KMeans: segmento 1 | 93 (1,9 %) | 86,0 % | 15,9 % | 0,221 |
| DBSCAN: atípicos | 14 (0,3 %) | 57,1 % | 1,6 % | 0,023 |
| (todos los asociados) | 5.000 | 10,1 % | | |

Aunque la segmentación no recibe la etiqueta, sus entradas incluyen señales que el generador crea condicionadas
al evento; las tasas altas no son descubrimiento independiente. DBSCAN no recupera casi ningún evento. Estas
pruebas describen el comportamiento del generador y no sustentan campañas sobre el fondo.

**Mapa interactivo** (`reports/figures/segmentos.html` y, en el simulador, "Mapa de segmentos"):
- Cada asociado va en dos ejes que se leen directo: **señales de salida** (cancelaciones, disminuciones,
  intención de retiro) y **vínculo con el fondo** (ahorros, créditos, servicios).
- Cada segmento tiene su color y su contorno.
- Un botón cambia entre KMeans y DBSCAN; los atípicos van en gris con ✕.
- Al pasar el mouse sale el segmento, su tasa de retiro y el comportamiento del asociado.
- En el simulador, una estrella marca dónde cae el asociado que se está simulando.

Con PCA, en cambio, dos componentes solo explican el 28 % de la variación y los segmentos se enciman.

**Con el extracto real: modo perfil** (`--perfil`, lo usa `make segmentos-real`). En el extracto casi todos los
conteos valen 0, así que solo con comportamiento los segmentos salen por qué producto tiene cada uno. El modo
perfil agrupa por comportamiento (en `log1p`, para que un conteo raro no domine) y por quién es el asociado: edad,
antigüedad en el fondo (log), salario (log) y si le falta el estrato. Con el extracto salen cinco segmentos que se
le pueden contar al fondo: los que registran **solicitudes de cancelación de ahorros** (segmento 1, la señal de
salida más fuerte), los que registran **solicitudes de retiro parcial** (2), los **nuevos sin estrato** y con pocos
productos (3), los **nuevos con estrato** (4) y los **veteranos** vinculados, de más edad, antigüedad y salario
(5). Una solicitud no equivale a una operación completada: el extracto cuenta solicitudes, no retiros ni
cancelaciones hechas. `segmentacion.perfil` describe cada segmento con medianas y porcentajes en sus unidades
(edad, antigüedad, salario, % sin estrato, % sin movimientos).

**Por qué k = 5: decisión exploratoria, no estadística.** Auditoría sobre el extracto real (KMeans del modo
perfil; solo agregados). Las filas de modelo nulo, semillas y bootstrap las reproduce `make segmentos-real`
(`Segmentacion.auditar`, con el mismo `n_init` = 20 que la solución); quedan en el JSON local de segmentos:

| Prueba | Resultado | Lectura |
|---|---|---|
| Índices por k (3 a 7) | silueta 0,175–0,197; Calinski-Harabasz máximo en k = 3; Davies-Bouldin baja al subir k | ningún índice elige k = 5; no hay un número "natural" de grupos |
| Modelo nulo: cada columna desordenada por separado (mismas distribuciones, sin relación entre variables), 5 veces | silueta 0,179 frente a 0,161 de media (máximo 0,168) | hay estructura conjunta más allá de las distribuciones de cada variable, pero débil: la silueta supera al nulo por 0,011–0,018 |
| 10 semillas (`n_init` = 20) | ARI con la base ≥ 0,998 | la solución con k = 5 no depende del inicio |
| Bootstrap 80 % × 20 (Jaccard por segmento, criterio de Hennig) | Jaccard medio 0,953–0,995 en los cinco; retiro parcial y los tres grandes nunca se disuelven (mínimo 0,81–0,95); cancelación (305 asociados, el 2 %) se disuelve en 1 de 20 submuestras (mínimo 0,06) | los cinco son estables en el sentido de Hennig (medio > 0,85); el de cancelación, por pequeño, a veces se funde con otro |
| Silueta por segmento | veteranos 0,08; los demás 0,19–0,23 | los veteranos son un extremo de un continuo de edad y antigüedad, no un grupo separado |
| k = 4 y k = 6 | con k = 4 el segmento de cancelación se reparte; con k = 6 se parten los nuevos con estrato | k = 5 es el menor k en el que las dos solicitudes de salida quedan en segmentos propios |
| Interpretabilidad de las variables | `SIN_ESTRATO` (binaria) separa por sí sola los nuevos sin estrato (100 %) de los nuevos con estrato (0 %) | uno de los cinco segmentos lo define un dato faltante: describe un perfil incompleto, no una condición económica |

Conclusión: los cinco segmentos son una **segmentación exploratoria orientada a la interpretación del negocio**, no
grupos que los datos impongan: la estructura supera al modelo nulo, pero por poco, y ningún índice elige k = 5. Son
estables (no dependen de la semilla ni, casi nunca, de la muestra), así que sirven para describir. El de solicitudes
de cancelación, el más útil para actuar, es pequeño y en 1 de 20 submuestras se funde con otro: conviene leerlo junto
con su conteo de solicitudes, no solo como etiqueta de segmento. Una auditoría anterior, hecha a mano y sin código,
reportaba lo contrario (segmentos de salida robustos y grandes inestables); no se pudo reproducir y se reemplazó.

```bash
make segmentos                                   # k por silueta, eps por vecinos
make segmentos ARGS="--k 4"                      # 4 segmentos (los dos de riesgo: 10,2 % y 9,9 % de retiro)
make segmentos-real                              # extracto real, modo perfil, k = 5
```

## Mapa de evaluación (para la sustentación)

| Concepto | Dónde está |
|---|---|
| Probabilidad vs. etiqueta | `abandono.puntaje` (probabilidad) vs. `ModeloZonas.predict` (zona) y `.alerta` (0/1) |
| Umbral de decisión | `campeon.elegir_umbrales`: alerta (máximo F2) y alto (precisión ≥ 0,8) |
| Matriz de confusión | `campeon.matrices_confusion` / `campeon_confusion.html` (con los umbrales del campeón) |
| Métricas de evaluación | `evaluate.compute_metrics` y `SCORING`: accuracy, precisión, recall, F1, F2, especificidad, balanced accuracy, AUC, PR-AUC, MCC, Kappa, recall@10 % y precision@10 %, Brier |
| Calibración | `campeon.figura_calibracion` / `campeon_calibracion.html` y el Brier |
| Datos sintéticos vs. reales | `config.origen_datos` (`ORIGEN_DATOS`), [README: "Datos sintéticos y datos reales"](../README.md#datos-sintéticos-y-datos-reales) |
| Trade-off precisión vs. recall | `campeon.figura_umbrales` / `campeon_umbrales.html` |
| AUC-ROC | `campeon.figura_roc` / `campeon_roc.html` (todos los candidatos) |
| Qué métrica elegir | sección "Qué métrica usamos y por qué" |
| MAE, MSE, RMSE, R² | `regresion.py` / `make regresion` |
| Hiperparámetros comunes | `busqueda_optuna.espacio_random_forest`: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features |
| KFold, StratifiedKFold, TimeSeriesSplit | `validacion.comparar_validaciones` / `make validacion` |
| Supervisado vs. no supervisado | campeón (`campeon.py`) vs. KMeans y DBSCAN (`segmentacion.py`, `make segmentos`) |
| Segmentación y su gráfico interactivo | `segmentacion.figura_mapa` / `segmentos.html` y el simulador |
| Predicción con el modelo guardado | `predict.py` / `make predecir` |
| Importancia de variables | permutación (`evaluate.importancia_variables`), MDI del bosque (`importancia_arbol`), coeficientes (`abandono.coeficientes`) y SHAP (`explicar.py`) |

## Comparación automática con PyCaret (opcional)

`make pycaret` ([`scripts/comparar_pycaret.py`](../scripts/comparar_pycaret.py)) le da las mismas variables a
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

Con los datos de desarrollo (`reports/metrics/pycaret_20261125_185438.json`), los boosting de PyCaret encabezan la
validación cruzada por AUC (`gbc` 0,874; `ada` 0,873) frente a 0,850 de nuestro HistGB por defecto. Con los 1.000
nuevos (9,0 % de eventos), la PR-AUC más alta es la de LightGBM (0,623) y `gbc` (0,609), frente a 0,586 del HistGB.
Es una comparación exploratoria con una sola partición de nuevos y sin intervalos: no pasa por la selección del
campeón ni indica un ganador general.

## Afinar hiperparámetros: Ridge y `pycaret_lr`

`make afinar` ([`scripts/afinar_modelos.py`](../scripts/afinar_modelos.py)) busca en grilla (`GridSearchCV`) los
mejores hiperparámetros de los dos mejores de la tabla comparativa, con las mismas particiones y métricas:

| Modelo | Hiperparámetros | Valores |
|---|---|---|
| Ridge | `alpha` (más alto = más regularización), `class_weight` | 0.001 … 1000 (13, escala log) × None/balanced |
| `pycaret_lr` | `C` (más alto = menos regularización), `l1_ratio` (0 = L2, 1 = L1), `class_weight` | 0.001 … 1000 × L2/L1 × None/balanced |
| los dos | preprocesamiento: vacíos numéricos, codificación, recorte de atípicos | mediana/media × OneHot/ordinal × sin/IQR |

El preprocesamiento entra a la búsqueda con `features.Preprocesador` (parámetros `numeric_strategy`,
`encoding`, `clip_outliers`). `pycaret_lr` usa la LogisticRegression de PyCaret con su preprocesamiento
(media + ordinal) reproducido con `build_preprocessor`: predice igual, y así también se puede cambiar.
El recorte solo toca `SALARIO`, `EDAD`, `ANTIGUEDAD_F_MESES` y `ANTIGUEDAD_LA_MESES` (`a.COLUMNAS_RECORTE`): en los conteos de
comportamiento el IQR vale [0, 0] y recortarlos borraría la señal.

**Tabla por partición** (`pipeline | fold | Accuracy | AUC`): para cada preprocesamiento de cada modelo
(`Ridge_median_onehot`, `LogisticRegression_mean_ordinal_iqr`...), su mejor combinación del resto en cada
una de las 5 particiones. Debajo, el resumen por preprocesamiento: promedio y desviación entre particiones
y, la prueba honesta, el resultado sobre asociados **nuevos**. Con los datos de desarrollo
(`reports/metrics/afinamiento_20261125_185058.json`; nuevos: 1.000 con 9,0 % de eventos sintéticos):

- La logística rinde mejor con OneHot y recorte IQR (AUC 0,862 ± 0,008 en validación cruzada; 0,873 con nuevos), y
  Ridge con ordinal e IQR (0,854 ± 0,010; 0,879). Entre preprocesamientos de un mismo modelo, el AUC de validación
  cruzada varía a lo sumo 0,009, menos que la desviación entre particiones (0,008–0,014): ninguno gana de forma
  concluyente. Mediana y media empatan
  porque los datos de desarrollo no traen vacíos numéricos.
- Afinar sube el AUC con nuevos (Ridge 0,849 → 0,879; `pycaret_lr` 0,863 → 0,873), pero los dos eligen
  `class_weight=None`: con el umbral fijo de 0,5 el recall de Ridge cae de 0,756 a 0,278. Maximizar AUC no dice
  nada del umbral; el campeón lo fija aparte (F2 y precisión ≥ 0,8).

El **gráfico de bigotes**
(`reports/figures/afinamiento_bigotes.png`) muestra lo mismo: una caja por pipeline con sus 5 particiones
y un rombo con el resultado sobre nuevos. `afinamiento_importancia.png` muestra las variables que más
aportan a los dos afinados.

Imprime las 5 mejores combinaciones de cada uno, la tabla "por defecto vs. afinado" y, lo que cuenta,
la comparación con asociados **nuevos** (el puntaje de validación cruzada del mejor es optimista: se
eligió mirando esos mismos datos). Guarda las curvas en `reports/figures/afinamiento_curvas.png` y cada
modelo afinado en MLflow (corrida "afinar", una hija por afinado con sus hiperparámetros y el modelo). Para
guardar un pipeline por partes (completo, solo preprocesamiento y solo estimador) sigue `a.guardar_modelo`.

También arma una **tabla de predicciones** de los dos afinados (como `predict_model` de PyCaret) sobre los
asociados nuevos: variables, evento futuro sintético y, por modelo, `<modelo>_label` (1 = abandona) y `<modelo>_score`
(riesgo de abandono: probabilidad en la logística; en Ridge, distancia a la frontera, > 0 = abandona). Muestra
unas filas en la terminal (`ARGS="--muestra 10"`) y guarda todas en `data/processed/predicciones_afinados.csv`
(no va a Git). Con regularización fuerte las probabilidades se acercan a 0.5: sirven para **ordenar** el riesgo,
no como probabilidad literal. En código: `a.tabla_predicciones({"ridge": modelo}, df)`.

```bash
make afinar                         # elige por AUC
make afinar ARGS="--metrica F1"     # o Recall, MCC... (las columnas de la tabla comparativa)
```

## Afinar los árboles con Optuna: Random Forest y HistGB

`make optuna` ([`scripts/afinar_optuna.py`](../scripts/afinar_optuna.py), lógica en
[`busqueda_optuna.py`](../src/proyecto_ml/busqueda_optuna.py)) afina los dos modelos de árboles, que tienen
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

Con los datos de desarrollo (5.000 asociados, 10,1 % de etiquetas sintéticas), 60 intentos por modelo, unos
3 minutos (`reports/metrics/optuna_20261125_185402.json`; nuevos: 1.000 con 9,0 % de eventos sintéticos):

- **Ganan poco, y en validación cruzada la ganancia es optimista** (cada estudio eligió mirando esas particiones).
  HistGB: AUC de validación cruzada 0,850 → 0,875; con nuevos, ROC-AUC 0,842 → 0,887 y PR-AUC 0,586 → 0,620.
  Random Forest: 0,849 → 0,871; con nuevos, 0,865 → 0,880 y PR-AUC 0,584 → 0,594. Con 90 eventos en los nuevos,
  diferencias de PR-AUC de 0,01 a 0,03 no son concluyentes.
- **La métrica que se maximiza importa.** Por defecto se maximiza ROC-AUC, que no depende del umbral. Random
  Forest elige `balanced_subsample` y, con el umbral fijo de 0,5, pasa de recall 0,411 y precisión 0,822 a recall
  0,756 y precisión 0,300; HistGB elige `class_weight=None` y queda en recall 0,444 y precisión 0,851.
- **Por qué no reemplazan al campeón.** El campeón se elige por PR-AUC en validación cruzada repetida, con
  intervalo y prueba de Nadeau-Bengio, y su umbral sale de las predicciones fuera de partición. Para que los árboles
  afinados compitan con ese criterio: `make campeon ARGS="--optuna 20"` (afina con `average_precision`).

```bash
make optuna                                         # 60 intentos por modelo, maximiza AUC
make optuna ARGS="--metrica F1"                     # o Recall, MCC... (columnas de la tabla comparativa)
make optuna ARGS="--modelos hist_gb --intentos 200" # un solo modelo, más intentos
make optuna ARGS="--timeout 60"                     # máximo 60 segundos por modelo
make optuna ARGS="--storage sqlite:///optuna.db"    # guarda el estudio; al repetir, continúa donde quedó
```

## Técnicas de preprocesamiento y feature engineering: `make tecnicas`

`make tecnicas` compara las técnicas de preprocesamiento sobre la foto del asociado, con los mismos datos que
`make campeon`. Cada técnica cambia **una sola cosa** respecto a la base (el preprocesador de los supervisados
con una logística con `class_weight="balanced"`). Todas se miden con PR-AUC en las mismas particiones
(5 × 3), así que la diferencia con la base es lo que aporta esa técnica. Todo se aprende dentro de cada
partición de entrenamiento: SMOTE y ADASYN solo re-muestrean en `fit`, y TargetEncoder usa validación cruzada
interna. Así nada se filtra de la validación.

Los datos generados no traen vacíos, así que para comparar imputadores el script pone un 10 % de vacíos al azar
(`--vacios`) en SALARIO, ESTRATO, EDAD, ANTIGUEDAD_LA_MESES y #CANTIDAD_SERVICIOS.

| Técnica | Dónde | PR-AUC | Dif. con la base | IC 95 % de la diferencia |
|---|---|---|---|---|
| **Base**: mediana, OneHot, StandardScaler, recorte IQR, `class_weight` | `features.build_preprocessor` | 0,531 | — | — |
| Sin balancear | `tecnicas.clasificador` | 0,541 | **0,009** | 0,002 a 0,017 ✔ |
| SMOTE | `imblearn` | 0,531 | −0,000 | −0,015 a 0,014 |
| ADASYN | `imblearn` | 0,521 | −0,010 | −0,032 a 0,011 |
| SimpleImputer(mean) | `numeric_strategy="mean"` | 0,533 | 0,001 | −0,003 a 0,005 |
| KNNImputer | `numeric_strategy="knn"` | 0,531 | −0,000 | −0,005 a 0,004 |
| IterativeImputer | `numeric_strategy="iterative"` | 0,532 | 0,001 | −0,004 a 0,006 |
| SimpleImputer + MissingIndicator | `missing_indicator=True` | 0,533 | 0,001 | −0,002 a 0,005 |
| OrdinalEncoder | `encoding="ordinal"` | 0,545 | 0,014 | −0,003 a 0,030 |
| TargetEncoder | `encoding="target"` | 0,552 | **0,021** | 0,004 a 0,037 ✔ |
| MinMaxScaler | `scaler="minmax"` | 0,524 | −0,007 | −0,016 a 0,003 |
| MaxAbsScaler | `scaler="maxabs"` | 0,525 | −0,007 | −0,016 a 0,003 |
| RobustScaler | `scaler="robust"` | 0,531 | −0,000 | −0,001 a 0,001 |
| Normalizer | `scaler="normalizer"` | 0,544 | 0,013 | −0,002 a 0,027 |
| Variables nuevas (creación y transformación) | `abandono.CrearVariables` | 0,536 | 0,005 | −0,005 a 0,014 |
| Binning (KBinsDiscretizer, 5 rangos) | `bins=5, bin_columns=...` | 0,530 | −0,002 | −0,010 a 0,007 |
| PCA (95 % de la varianza) | `pca_components=0.95` | 0,496 | −0,036 | −0,068 a −0,003 ✔ |
| RFE (15 columnas) | `rfe_features=15` | 0,241 | −0,290 | −0,349 a −0,232 ✔ |

*Datos de desarrollo sincronizados: 5.000 asociados y 503 etiquetas sintéticas. Logística, 5 particiones × 3
repeticiones; se generaron vacíos aleatorios al 10 % para comparar imputadores. Los IC son diferencias pareadas
por partición y no ajustan por probar múltiples técnicas. ✔ indica que el intervalo de esta comparación no toca 0.*

**Cómo leerlo.** En esta semilla, sin balancear mejora +0,009 y TargetEncoder +0,021 frente a la base; el resto
de los cambios pequeños tiene intervalos que incluyen cero. La repetición con seeds 7, 42 y 2026 muestra que la
ganancia de sin balancear es positiva (+0,007 a +0,013), pero no concluyente en todos los seeds. TargetEncoder
empeora en un seed y Normalizer mejora numéricamente en los tres, aunque sus intervalos no son concluyentes
salvo una comparación. El baseline Dummy (prior) da AP igual a la prevalencia, 0,094–0,101, y ROC-AUC 0,5.
No se promueve Normalizer solo por su métrica: no hay justificación de dominio independiente.

La selección de candidatos es, por tanto, exploratoria. Sin balancear se conserva como baseline simple;
TargetEncoder se conserva como alternativa para categóricas; ningún resultado permite recomendar técnica para el
extracto real. SMOTE no aporta en esta ejecución; PCA y RFE empeoran la PR-AUC sintética.

**Feature engineering** (`abandono.CrearVariables`, opcional; no cambia el campeón):

- **Creación por agregación:** `SENALES_SALIDA` (suma de las solicitudes de salida) y `PRODUCTOS` (ahorros a la
  vista + contractuales + servicios).
- **Creación por interacción:** `SALIDA_POR_PRODUCTO = SENALES_SALIDA / (1 + PRODUCTOS)`, es decir, pedir salir
  teniendo poco que perder.
- **Transformación logarítmica:** `SALARIO_LOG = log(1 + SALARIO)`. EDAD y las antigüedades se
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
