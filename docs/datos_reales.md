# Datos reales: requisitos y validación fuera de tiempo

> Parte de la documentación de [proyecto-ml](../README.md). **Hoy no hay datos reales**: todo lo de este documento
> está listo para cuando lleguen, y nada de lo que se reporta en el README sale de datos reales.

## Por qué hacen falta varios extractos

Con **un solo extracto**, los conteos de comportamiento (`abandono.CONTEOS_SIN_FECHA`) y atributos como salario o
cargo (`abandono.ATRIBUTOS_SIN_FECHA`) están medidos a la fecha de extracción, no a t0. La foto del asociado los usa
como si fueran de t0, y la prueba temporal de `make campeon` compara **las mismas personas con los mismos datos**:
es una prueba de estrés, no una validación fuera de tiempo (README, "Resultados").

Con **extractos sacados en fechas distintas**, cada uno trae sus conteos medidos a su propia fecha.
`extractos.foto_entre_extractos` arma la foto sin ambigüedad:

| | Sale de |
|---|---|
| Población | los activos del extracto viejo, en su fecha |
| Variables (conteos, salario, cargo, EDAD, ANTIGUEDAD) | el extracto **viejo**, a su fecha |
| Objetivo (`#RETIROS`) | el extracto **siguiente**: ¿se retiró entre la fecha del viejo y `horizonte` meses después? |

El objetivo nunca sale del mismo archivo que las variables, así que no hay fuga por conteos sin fecha.

## Qué se valida y cómo se evita la fuga

**El uso real:** predecir qué asociados **actuales** se retiran en los **próximos** `horizonte` meses. Por eso la
validación es **temporal** (entrenar con el pasado, medir en un período posterior) y no "por individuo": en
producción el modelo puntúa a asociados que ya vio en fotos anteriores, así que verlos de nuevo en la prueba no
es una fuga. Lo que sí sería fuga es usar el futuro, o la misma fila, para elegir.

| Paso | Con qué datos | Cómo |
|---|---|---|
| Elegir el candidato (mayor PR-AUC) | solo las fotos de entrenamiento | 1 foto (3 extractos): `StratifiedKFold`, cada asociado aparece una vez. 2+ fotos (4+ extractos): **hacia adelante** (`campeon.puntajes_hacia_adelante`): cada foto se puntúa con un modelo entrenado solo con las anteriores |
| Elegir el umbral (F2 y precisión mínima) | los mismos puntajes de la selección | igual que arriba: nunca con la prueba |
| Entrenar el modelo final | todas las fotos de entrenamiento | |
| **Evaluación final** | **el último par de extractos, reservado** | se mide una sola vez, con el modelo y el umbral ya fijos |

**Por qué no una partición al azar con varias fotos.** El mismo asociado está en todas las fotos de entrenamiento
(con datos sintéticos, 2.754 de 3.000 en dos fotos seguidas). Una `StratifiedKFold` sobre las fotos juntas lo ponía
a la vez en entrenamiento y validación y, en cada partición, entrenaba con una foto posterior a la que validaba.
**Por qué no agrupar por individuo** (`StratifiedGroupKFold`): evitaría repetir personas, pero seguiría entrenando
con el futuro, y simularía asociados nuevos, que no es el caso de uso.

**Controles:** si un identificador se repite dentro de una misma foto, o un período no tiene retiros y no retiros,
o la prueba tiene menos retiros que particiones, el proceso se detiene con un error (no se relaja en silencio).
El seudónimo solo se usa para cruzar extractos y revisar repetidos: no es una variable del modelo (las variables son
una lista fija, `abandono.NUMERICAS` + `CATEGORICAS`). Los resultados no lo incluyen.

**Límite:** con 3 extractos, la selección usa una sola foto (no hay período anterior con qué ir hacia adelante).
Con 4 o más, la selección hacia adelante usa menos filas (la primera foto solo entrena), a cambio de no mirar el
futuro.

**Pendiente científico: la validación cruzada interna de TargetEncoder** (`features.py`, `StratifiedKFold(5)`
sin grupos; candidato `logreg_target`). Al ajustar, TargetEncoder codifica cada fila de entrenamiento con la media
del objetivo de su categoría calculada en las OTRAS particiones internas. Cuando el modelo se entrena con 2 o más
fotos (el modelo final con 4+ extractos; la selección hacia adelante con 5+), el mismo asociado puede estar en otra
partición con otra foto, y su etiqueta de esa foto entra en la codificación de su fila. **Es una fuga interna del
ajuste, verificada por estructura** (no por métricas): con datos sintéticos y 2 fotos, el 77 % de las filas de
entrenamiento tiene al mismo asociado en otra foto y otra partición, y en 354 la otra foto tiene otra etiqueta. Su
peso es el de una fila en la media de su categoría: a lo sumo 0,024 (la categoría más chica, 42 asociados) y
0,0015 o menos en las demás.

- **No llega a la evaluación externa:** la validación hacia adelante y la prueba final se codifican con lo aprendido
  en entrenamiento (`transform`), sin sus etiquetas.
- **Efecto observado** con datos sintéticos: agrupar la CV interna por asociado no cambia la PR-AUC fuera de tiempo
  (0,5798 en los dos casos; puntajes con correlación 1,00000). Eso dice que el efecto es chico aquí, no que no exista.
- **Corrección posible:** sklearn 1.9 deja pasar grupos a esa CV (`fit_transform(**params)` con *metadata routing*),
  pero exige activar el *routing* global y declarar los grupos en cada paso de cada pipeline, en todos los
  reentrenamientos y clones. No se hace sin un beneficio medido.
- **Revisarlo con los datos reales**, sobre todo si traen categorías con pocos asociados (cargos o agencias raros),
  donde una sola etiqueta pesa más.

## Requisitos mínimos de los datos

1. **Al menos 3 extractos** en fechas distintas, separados por al menos el horizonte (3 meses). Con 3 (E1, E2, E3)
   se entrena con el par E1 → E2 y se mide con E2 → E3. Con más, se entrena con todos los pares menos el último.
2. **Los conteos de cada extracto medidos a su fecha de extracción**, y esa fecha conocida (la que se pasa como
   `RUTA:AAAA-MM-DD`). Confirmarlo con el fondo; después, `TEMPORALIDAD_CONFIRMADA=si` en el `.env`.
3. **Cada extracto trae también a los retirados** (con `FECHARETIRO`), no solo a los activos: si no, no se sabe
   quién se fue. `revisar_par` exige que al menos el 90 % de los activos del extracto viejo aparezca en el siguiente.
4. **La misma `ANON_SALT` para todos** (`make anonimizar`): así `CEDULA_SEUDO` coincide entre extractos y se pueden
   cruzar. Con otra clave los seudónimos no coinciden y `revisar_par` lo detecta (cobertura baja).
5. **Cada extracto cumple el contrato** del dataset anonimizado (`make contrato-ml DATA=...`).
6. **Retiros suficientes** en cada par: al menos tantos como particiones (`--folds`, 5) en entrenamiento y en
   prueba; para cifras estables, del orden de 100 retiros por par (con 40 el umbral alto quedaba sobre 31 aciertos).

`extractos.revisar_par` verifica 1, 3 y 4 y la coherencia: sin fechas de retiro posteriores a la fecha del extracto,
sin seudónimos repetidos y sin asociados retirados que reaparecen activos. Los mensajes no muestran valores.

## Procedimiento (reproducible)

```bash
# 1. El custodio anonimiza cada extracto con la MISMA clave (ANON_SALT del .env)
make anonimizar DATA=extracto_2026-04.xlsx SALIDA=data/raw/anon_2026-04.parquet
make anonimizar DATA=extracto_2026-07.xlsx SALIDA=data/raw/anon_2026-07.parquet
make anonimizar DATA=extracto_2026-10.xlsx SALIDA=data/raw/anon_2026-10.parquet

# 2. En el .env: ORIGEN_DATOS=real_anonimizado, FECHA_CORTE = la del extracto más reciente,
#    TEMPORALIDAD_CONFIRMADA=si (solo cuando el fondo lo confirme)

# 3. La validación fuera de tiempo (elige el candidato y el umbral solo con los pares viejos)
make fuera-de-tiempo EXTRACTOS="data/raw/anon_2026-04.parquet:2026-04-30 data/raw/anon_2026-07.parquet:2026-07-31 data/raw/anon_2026-10.parquet:2026-10-31"

# 4. El campeón para producción, con el extracto más reciente
make campeon DATA=data/raw/anon_2026-10.parquet
```

El resultado queda en `reports/metrics/fuera_de_tiempo_<fecha>.json`, solo con agregados. No se versiona: puede
revelar cifras del fondo. Lleva el origen y la semilla.

## Qué sigue sin resolver aun con varios extractos

- La **lista** de candidatos y los hiperparámetros por defecto se armaron con datos sintéticos (`make tecnicas`
  sobre la foto de desarrollo), así que no vieron ningún extracto real: la validación fuera de tiempo elige entre
  ellos sin selección indirecta sobre la prueba real. **Si se rehace con datos reales** (`make tecnicas`,
  `--optuna`), hay que hacerlo **solo con los extractos de entrenamiento**, nunca con el último par: si no, la lista
  ya habría mirado la prueba.
- Un cambio real en el fondo (otra tasa de retiro, una crisis) entre extractos se mide, no se corrige: la cifra
  fuera de tiempo es justamente la que lo muestra.
- El modelo **no está validado para producción** hasta tener esta validación con datos reales y revisarla con el
  fondo.
