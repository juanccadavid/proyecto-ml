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

- La **lista** de candidatos y los hiperparámetros por defecto se armaron con datos sintéticos; la validación
  fuera de tiempo elige entre ellos, no los rediseña.
- Un cambio real en el fondo (otra tasa de retiro, una crisis) entre extractos se mide, no se corrige: la cifra
  fuera de tiempo es justamente la que lo muestra.
- El modelo **no está validado para producción** hasta tener esta validación con datos reales y revisarla con el
  fondo.
