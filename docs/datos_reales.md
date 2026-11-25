# Datos reales: estado del extracto y límites

> El extracto anonimizado existe solo en el entorno local del custodio. No se versiona, no se adjunta a PR y no se
> publica. El repositorio conserva únicamente el esquema autorizado de 44 columnas y datos sintéticos.

## Respuestas del fondo (custodio)

| Pregunta | Respuesta | Fecha de la respuesta | Qué cambia en el proyecto |
|---|---|---|---|
| ¿Cuándo se miden los conteos de comportamiento? | Confirmado: se miden al corte; `ULT3MES` corresponde a los tres meses anteriores al corte. | 2026-11-21 | Con un solo corte no se puede reconstruir una foto anterior. La temporalidad se documenta y no se presenta como validación fuera de tiempo. |
| ¿El extracto incluye retirados con fecha y motivo? | No. `#RETIROS` cuenta los retiros acumulados durante la permanencia del asociado; no corresponde a los últimos tres meses. | 2026-11-23 | `#RETIROS` queda como predictor histórico. No se deriva de él la etiqueta futura y se bloquea el entrenamiento real mientras esa etiqueta no exista. |
| ¿Habrá tres o más extractos? | No; solo se trabajará con el último corte. | 2026-11-21 | No se ofrece validación fuera de tiempo con datos reales. |
| ¿Se conoce y se puede publicar la tasa anual? | Pendiente: el dato no está disponible y no se publicará. | 2026-11-21 | No se define una regla de tasa real en el contrato. |
| ¿Se puede publicar el esquema? | Se acordó un esquema definitivo reducido. | 2026-11-23 | El contrato público contiene solo las 44 columnas seleccionadas por el equipo. |
| ¿Se deben agrupar cuasi-identificadores? | No. | 2026-11-21 | No se crean rangos adicionales; los campos de identificación y contacto se excluyen del modelo. |
| ¿Cuándo estará el extracto anonimizado? | Confirmado: disponible localmente. | 2026-11-21 | Se valida desde `data/raw/DataAnonimizada.xlsx`; el archivo permanece ignorado por Git. |
| ¿Se puede saber quién de este corte se retiró en los tres meses siguientes? | No. | 2026-11-25 | No habrá etiqueta futura real: el abandono no se entrena ni se evalúa con datos reales. Con el extracto corren la segmentación, la regresión del salario y el puntaje del campeón sintético (`make real`). |
| ¿Por qué `CARGO` llega idéntico a `CODPROFESION`? | Es la conversión que usa el fondo. | 2026-11-25 | `CARGO` sigue en el contrato, pero no entra al modelo (`abandono.DUPLICADAS`): repetido no agrega información y partiría la importancia de la profesión. |
| ¿Qué significa `ESTRATO` en blanco? | No se sabe. | 2026-11-25 | Supuesto del equipo, revisado con agregados: no indica retiro de ahorros (esos asociados piden menos retiros y cancelaciones), sino un perfil incompleto (más recientes, con menos productos y datos actualizados hace más tiempo). Es una categoría propia, `SIN_DATO`, y no se imputa con el estrato más común. |
| ¿`987654` en `SALARIO` es "no informado"? | Probablemente un valor anonimizado. | 2026-11-25 | En los dos casos el salario verdadero se desconoce: se trata como vacío (`SALARIO_PLAUSIBLE`) y la regresión no lo aprende. |

## Esquema definitivo

El archivo contiene 44 columnas: 29 sociodemográficas, 7 laborales, 7 de productos y 1 conteo histórico. La fuente de
verdad es [`contratos/asociados.yaml`](../contratos/asociados.yaml). Las tres variables derivadas aprobadas son
`EDAD`, `ANTIGUEDAD_F_MESES` y `ANTIGUEDAD_LA_MESES`.

El contrato conserva las 44 columnas para validar la entrega, pero el modelo no usa identificadores, nombres,
dirección, correo, celular, fechas crudas, `ESTADO` ni `FECHAULTACTUALIZACION`. `#RETIROS` sí puede entrar como
predictor histórico, pero no es el objetivo.

Las equivalencias de códigos comunicadas para `CODESTUDIOS`, `CODESTADOCIVIL` y `TIPOCONTRATO` están en el
[diccionario de datos](diccionario_datos.md). En el extracto llegan en `ESTUDIOS`, `ESTADOCIVIL` y `TIPOCONTRATO`,
y el contrato los exige como valores válidos.

## Validación local

```bash
make contrato-ml          # valida data/raw/DataAnonimizada.xlsx (o DATA=otro archivo)
```

La validación revisa columnas, tipos, rangos, códigos, unicidad, volumen y coherencia entre fechas. Los errores del
modo real informan columna y cantidad, sin imprimir valores. Un incumplimiento se corrige en la fuente o se
documenta con una regla confirmada; no se relaja el contrato para ocultarlo.

## Cómo llega el extracto y cómo lo lee el proyecto (contrato v2.2.0)

El primer extracto (corte del 31 de marzo de 2026) se revisó con agregados, sin imprimir filas. Todo el proyecto
(Python, Supabase, formulario y generador sintético) habla ahora los mismos campos que ese archivo:

| Qué trae el extracto | Qué hace el proyecto |
|---|---|
| Cuatro nombres de columna distintos: `EDAD_AÑOS`, `ANTIGÜEDAD_F_MESES`, `ANTIGÜEDAD_LA_MESES` y `FECHAINGRESALABORA` (errata) | El contrato los declara como `alias`: al leer, pasan a `EDAD`, `ANTIGUEDAD_F_MESES`, `ANTIGUEDAD_LA_MESES` y `FECHAINGRESOLABORA`. El resto del proyecto usa solo los nombres del contrato (ASCII, sin tildes, como las columnas de Supabase). |
| Códigos del [diccionario](diccionario_datos.md): `ESTUDIOS` O/U/G…, `ESTADOCIVIL` S/C/V…, `TIPOCONTRATO` 2/1/15/O/0, `#SOLICITUD_DISMINUCIONCUOTAS` SI/NO | Son los `valores` válidos del contrato (también CHECK en Supabase y listas del formulario). Excel lee algunos como números (`2`) o con espacios al final (`'O  '`): se normalizan a texto (`'2'`, `'O'`) sin cambiar el código. |
| Cerca de un tercio de los asociados con `ESTRATO` en blanco | `ESTRATO` pasa a ser opcional. Para el modelo, el blanco es la categoría `SIN_DATO` (ver respuestas del fondo). |
| Algunas `FECHAINGRESOLABORA` imposibles (antes de 1950, antes del nacimiento o en años futuros) y una digitada como texto `dd/mm/aaaa` | Las dos reglas de esa fecha son **advertencias**: se informa cuántas filas, sin bloquear, porque la fecha no entra al modelo. El texto `dd/mm/aaaa` se lee con el día primero; las fechas ISO del sistema, con su formato exacto. |
| Unos pocos salarios de relleno (`987654`) o de miles de millones | `abandono.preparar` los trata como vacíos (`SALARIO_PLAUSIBLE`, de 1 a 50 millones), igual que una antigüedad laboral mayor que la vida laboral posible. La regresión no los aprende. |
| `CARGO` igual a `CODPROFESION` en todas las filas | Es la conversión del fondo: se conserva en el contrato y queda fuera del modelo. |
| `ESTADO` = A (activo) en todos | Confirma que no hay retirados en el extracto: no sirve como etiqueta (y además sería fuga). |

`FECHA_CORTE` por defecto es `2026-03-31`: con ella `EDAD` y `ANTIGUEDAD_F_MESES` coinciden con las fechas del
archivo. Los datos sintéticos usan el mismo corte.

## Qué corre con el extracto real

```bash
make real                 # contrato-ml + segmentos-real + regresion-real + predecir-real
make segmentos-real       # KMeans y DBSCAN sobre el comportamiento (no necesita etiqueta)
make regresion-real       # estima el SALARIO (no necesita etiqueta)
make predecir-real        # puntúa a cada asociado con el campeón entrenado con sintéticos (make campeon)
```

Estos objetivos ponen `ORIGEN_DATOS=real_anonimizado` sin tocar el `.env`; los resultados quedan en
`reports/` (ignorado por Git), en experimentos de MLflow con sufijo `_real` y, las predicciones, en
`data/processed/predicciones.csv`. El mapa de segmentos con datos reales muestra solo agregados por segmento.

- **Segmentación y regresión** sí aprenden de los datos reales.
- **El abandono no**: `make campeon` con datos reales sigue bloqueado mientras no exista una etiqueta futura.
  `predecir-real` aplica a los asociados reales el campeón entrenado con los sintéticos, que ahora tienen los
  mismos códigos y una forma parecida a la del extracto. El puntaje ordena por las señales que el generador
  asocia al retiro (cancelaciones, retiros parciales, disminución de cuotas, pocos productos); **no está validado
  contra retiros reales** y no debe leerse como probabilidad de retiro del fondo.

## Pregunta abierta para el fondo

- ¿`#RETIROS` > 0 corresponde a asociados que se retiraron y volvieron a ingresar? Todos están activos.

## Limitaciones vigentes

- Solo hay un corte: no existe evaluación temporal externa con datos reales.
- Los conteos no tienen fecha propia; se conoce su ventana por la respuesta del fondo, no por una columna.
- La tasa anual real no está disponible ni autorizada para publicación.
- Confirmado: `#RETIROS` cubre toda la permanencia y no representa un retiro en los próximos tres meses.
- El extracto no contiene una etiqueta futura válida; no se entrena ni se evalúa un clasificador real con él. Las
  predicciones de `predecir-real` vienen de un modelo entrenado con sintéticos.
- Un archivo anonimizado puede conservar cuasi-identificadores. Por eso sigue siendo confidencial y no va a Git.

## Si algún día existe una etiqueta futura válida

1. Validar el archivo con `make contrato-ml`.
2. Resolver todos los errores de contrato.
3. Construir la etiqueta con seguimiento posterior al corte, sin derivarla de `#RETIROS`.
4. Actualizar el flujo contractual para esa fuente y ejecutar el entrenamiento validado.
5. Revisar únicamente resultados agregados y publicar cifras solo con autorización del fondo.
