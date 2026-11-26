# Privacidad del extracto anonimizado

El esquema v2 describe directamente el archivo anonimizado definitivo de 44 columnas. El custodio mantiene el
archivo en `data/raw/DataAnonimizada.xlsx`; `data/` y `*.xlsx` están excluidos de Git. Un PR puede ser público,
pero el extracto y sus predicciones individuales nunca se adjuntan.

## Controles aplicados

- El contrato funciona como lista blanca: un campo desconocido detiene el modo `real_anonimizado`.
- `NROIDENTIFICACION`, nombres, dirección, correo y celular se validan, pero no entran al modelo.
- `ESTADO` y `FECHAULTACTUALIZACION` no son predictores. `#RETIROS` sí es un conteo histórico predictor, nunca la
  etiqueta de los próximos tres meses.
- Las fechas crudas no entran al modelo; se usan las variables aprobadas `EDAD`, `ANTIGUEDAD_F_MESES` y
  `ANTIGUEDAD_LA_MESES`.
- La salida de predicción contiene solo `fila`, puntaje, zona, alerta y grupo; no copia identificadores.
- En modo real, los errores y advertencias de contrato se resumen sin valores del extracto (solo columna y
  cantidad de filas).
- Leer el extracto no reescribe datos: los `alias` del contrato solo renombran columnas y la normalización de
  códigos (`2` → `'2'`, `'O  '` → `'O'`) no cambia su significado. Los valores imposibles (salario de relleno,
  antigüedad laboral imposible) se tratan como vacíos dentro del modelo, sin tocar el archivo.
- El mapa de segmentos con datos reales muestra solo agregados por segmento, no el detalle de cada asociado.
- **MLflow con datos reales recibe solo agregados.** De cada dataset registra la huella, el esquema y el número de
  filas (`registro.registrar_datos`), nunca las filas. La figura de la regresión real es por quintiles
  (`regresion.figura_por_tramo`), sin salarios individuales; el mapa de segmentos real, que tiene una coordenada
  por asociado, se queda en `reports/` (ignorado por Git) y no se sube como artefacto.
- **Riesgos que quedan, en local:** `reports/figures/segmentos.html` y `data/processed/predicciones.csv` tienen una
  fila por asociado (sin identificadores) y existen solo en el equipo del custodio. `docker-compose.yml` monta
  `./data` (solo lectura) y pasa el `.env` al contenedor: no compartir ese contenedor ni su volumen de MLflow.

## Seudonimización y anonimía

Seudonimizar reemplaza un identificador por otro valor estable. No es anonimizar de forma irreversible: quien
conserve el mecanismo y la clave puede recalcular o relacionar el seudónimo. Además, combinaciones como edad,
ciudad, cargo y empresa pueden actuar como cuasi-identificadores.

Por eso el archivo sigue tratándose como confidencial aunque sus campos hayan sido anonimizados. La ubicación y
acceso a cualquier clave pertenecen al procedimiento interno del custodio; el valor nunca se registra en Git,
documentación, logs ni chats.

## Procedimiento con el archivo definitivo

1. Guardar el archivo en `data/raw/DataAnonimizada.xlsx`.
2. Ejecutar `make contrato-ml` (valida ese archivo por defecto) y, para trabajar con él, `make real`.
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
