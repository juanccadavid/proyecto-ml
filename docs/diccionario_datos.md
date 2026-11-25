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

**Alcance de los nombres:** el [contrato vigente](../contratos/asociados.yaml) contiene `ESTUDIOS` y `ESTADOCIVIL`, no `CODESTUDIOS` ni `CODESTADOCIVIL`. Estas equivalencias se registran con los nombres de columna proporcionados y no implican que los campos del contrato usen necesariamente esos códigos. `TIPOCONTRATO` sí figura en el contrato, que no restringe aquí sus valores a esta tabla.
