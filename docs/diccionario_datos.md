# Diccionario de códigos de datos

Equivalencias de las nomenclaturas comunicadas para interpretar los valores de cada columna. Son descripciones para lectura; no cambian los valores almacenados, las reglas del contrato ni las variables del modelo.

## `CODESTUDIOS`

| Código | Significado |
|---|---|
| `O` | OTROS |
| `U` | UNIVERSITARIO |
| `G` | POSGRADO |
| `S` | SECUNDARIA |
| `E` | TECNOLOGIA |
| `P` | PRIMARIA |
| `T` | TECNICO |

## `CODESTADOCIVIL`

| Código | Significado |
|---|---|
| `C` | CASADO |
| `T` | DIVORCIADO |
| `V` | UNION LIBRE |
| `S` | SOLTERO |
| `D` | VIUDO |
| `U` | SEPARADO |
| `F` | MADRE SOLTERA |
| `P` | MUJER CABEZA FAMILIA |

## `TIPOCONTRATO`

| Código | Significado |
|---|---|
| `2` | INDEFINIDO |
| `1` | TERMINO FIJO |
| `15` | PENSIONADO |
| `O` | OTRO |
| `0` | SIN CONTRATO |

**Uso en el contrato:** el extracto real trae estos códigos en las columnas `ESTUDIOS`, `ESTADOCIVIL` y `TIPOCONTRATO`. Desde la v2.2.0, el [contrato](../contratos/asociados.yaml) los exige como `valores` válidos de esas tres columnas (también `SEXO` F/M, `ESTRATO` 1 a 6 y `#SOLICITUD_DISMINUCIONCUOTAS` SI/NO), y el generador sintético y el formulario usan los mismos.
