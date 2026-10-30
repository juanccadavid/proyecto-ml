# data/ — NO se versiona

Todo el contenido de esta carpeta está en `.gitignore` y un hook de pre-commit bloquea archivos
`.csv/.xlsx/.parquet/...`. Los datos se comparten **solo** por el canal privado acordado por el equipo.

- `raw/`: extractos tal como llegan (ya anonimizados por quien los custodia).
- `processed/`: salida de `scripts/ingesta.py` lista para modelar.

Reglas:
1. La anonimización (`scripts/anonimizar.py`) la ejecuta el custodio **antes** de compartir.
2. Nada de nombres, cédulas, correos, teléfonos ni direcciones.
3. Edad y salario en rangos.
4. Si un archivo con datos reales llega a un commit, avisar de inmediato: no basta con borrarlo,
   hay que reescribir el historial.

## Excepción: `data/ejemplo/` (sí se versiona)

Contiene `asociados_ejemplo.parquet` / `.xlsx`: 50 asociados **ficticios** con el esquema real
(SOCIODEMOGRAFICO, LABORAL, PRODUCTOS, RETIRO), para probar ingesta y modelos sin datos reales.
Se regenera con `uv run python scripts/generar_asociados_ejemplo.py --rows 50`.
Es la única carpeta exceptuada en `.gitignore` y en el hook `bloquear-datos`: **nunca** guardar datos reales aquí.
