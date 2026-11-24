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

## Esquema definitivo

El archivo contiene 44 columnas: 29 sociodemográficas, 7 laborales, 7 de productos y 1 conteo histórico. La fuente de
verdad es [`contratos/asociados.yaml`](../contratos/asociados.yaml). Las tres variables derivadas aprobadas son
`EDAD`, `ANTIGUEDAD_F_MESES` y `ANTIGUEDAD_LA_MESES`.

El contrato conserva las 44 columnas para validar la entrega, pero el modelo no usa identificadores, nombres,
dirección, correo, celular, fechas crudas, `ESTADO` ni `FECHAULTACTUALIZACION`. `#RETIROS` sí puede entrar como
predictor histórico, pero no es el objetivo.

## Validación local

```bash
make contrato-ml DATA=data/raw/DataAnonimizada.xlsx
```

La validación revisa columnas, tipos, rangos, unicidad, volumen y coherencia entre fechas. Los errores del modo
real informan columna y cantidad, sin imprimir valores. Un incumplimiento se corrige en la fuente o se documenta
con una regla confirmada; no se relaja el contrato para ocultarlo.

## Limitaciones vigentes

- Solo hay un corte: no existe evaluación temporal externa con datos reales.
- Los conteos no tienen fecha propia; se conoce su ventana por la respuesta del fondo, no por una columna.
- La tasa anual real no está disponible ni autorizada para publicación.
- Confirmado: `#RETIROS` cubre toda la permanencia y no representa un retiro en los próximos tres meses.
- El extracto no contiene una etiqueta futura válida; no se entrena ni se evalúa un clasificador real con él.
- Un archivo anonimizado puede conservar cuasi-identificadores. Por eso sigue siendo confidencial y no va a Git.

## Cuando exista una etiqueta futura válida

1. Validar el archivo con `make contrato-ml`.
2. Resolver todos los errores de contrato.
3. Construir la etiqueta con seguimiento posterior al corte, sin derivarla de `#RETIROS`.
4. Actualizar el flujo contractual para esa fuente y ejecutar el entrenamiento validado.
5. Revisar únicamente resultados agregados y publicar cifras solo con autorización del fondo.
