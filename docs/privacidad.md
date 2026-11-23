# Privacidad del extracto anonimizado

El esquema v2 describe directamente el archivo anonimizado definitivo de 44 columnas. El custodio mantiene el
archivo en `data/raw/DataAnonimizada.xlsx`; `data/` y `*.xlsx` están excluidos de Git. Un PR puede ser público,
pero el extracto y sus predicciones individuales nunca se adjuntan.

## Controles aplicados

- El contrato funciona como lista blanca: un campo desconocido detiene el modo `real_anonimizado`.
- `NROIDENTIFICACION`, nombres, dirección, correo y celular se validan, pero no entran al modelo.
- `ESTADO`, `FECHAULTACTUALIZACION` y `#RETIROS` tampoco son predictores.
- Las fechas crudas no entran al modelo; se usan las variables aprobadas `EDAD`, `ANTIGUEDAD_F_MESES` y
  `ANTIGUEDAD_LA_MESES`.
- La salida de predicción contiene solo `fila`, puntaje, zona, alerta y grupo; no copia identificadores.
- En modo real, los errores de contrato se resumen sin valores del extracto.

## Seudonimización y anonimía

Seudonimizar reemplaza un identificador por otro valor estable. No es anonimizar de forma irreversible: quien
conserve el mecanismo y la clave puede recalcular o relacionar el seudónimo. Además, combinaciones como edad,
ciudad, cargo y empresa pueden actuar como cuasi-identificadores.

Por eso el archivo sigue tratándose como confidencial aunque sus campos hayan sido anonimizados. La ubicación y
acceso a cualquier clave pertenecen al procedimiento interno del custodio; el valor nunca se registra en Git,
documentación, logs ni chats.

## Procedimiento con el archivo definitivo

1. Guardar el archivo en `data/raw/DataAnonimizada.xlsx`.
2. Ejecutar `make contrato-ml DATA=data/raw/DataAnonimizada.xlsx`.
3. Revisar columnas, tipos, vacíos, rangos y coherencia de fechas sin imprimir filas.
4. Corregir en la fuente cualquier discrepancia confirmada; no reescribir valores para forzar el contrato.
5. Compartirlo solo por el canal privado autorizado por la organización.
6. Mantener fuera de la organización el archivo original, el anonimizado, las claves y las predicciones
   individuales.

La utilidad histórica `scripts/anonimizar.py` permanece para contratos con marcas `pii`, pero no debe volver a
anonimizar el archivo v2: este ya llega anonimizado y el contrato no define transformaciones adicionales.

## Revisión humana

- Confirmar que no haya columnas adicionales ni hojas auxiliares con datos no autorizados.
- Verificar que los campos anonimizados no conserven nombres, correos, teléfonos o direcciones reconocibles.
- Revisar valores cortos y transformados que un detector automático puede omitir.
- Confirmar que ningún reporte, figura o log incluya filas o identificadores.
- Revisar el estado de Git antes de cada commit: el `.xlsx` no debe aparecer como archivo para subir.
