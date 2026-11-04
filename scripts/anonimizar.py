"""Anonimiza el extracto de asociados ANTES de compartirlo con el equipo (make anonimizar).

Lo ejecuta quien custodia los datos (dentro de su organización), nunca después de compartir.

La configuración de qué es personal es el contrato de origen (contratos/asociados.yaml, campo ``pii``):
``eliminar`` se borra y ``seudonimizar`` sale como ``<COLUMNA>_SEUDO`` (HMAC-SHA256 de 64 caracteres hex con la
clave ANON_SALT del .env). Una columna del extracto que el contrato no conoce detiene el proceso hasta que se
decida qué hacer con ella: --conservar, --eliminar o --seudonimizar.

Orden (nada se escribe hasta el paso 5):
  1. carga el extracto (y lo limpia, salvo --sin-limpiar);
  2. lo anonimiza según el contrato;
  3. verifica que no quede PII: columnas personales, seudónimos inválidos, valores personales copiados o
     incrustados en otra columna (anonymize.verificar: qué detecta y qué no);
  4. valida la salida contra el contrato del dataset anonimizado para ML (contrato.para_ml), el mismo que exige
     make campeon con ORIGEN_DATOS=real_anonimizado;
  5. solo si 3 y 4 pasan, escribe (a un temporal que se renombra al final: o el archivo completo, o nada).
Si algo falla, termina con error y NO escribe ni reemplaza la salida existente. Los errores dicen columnas,
reglas y cantidades, nunca valores. Las advertencias del contrato (p. ej. una tasa de retiro distinta de la
esperada) se muestran y no bloquean.

Como el contrato para ML es estricto, una columna conservada que no está en el contrato (--conservar,
--seudonimizar de una columna extra) o agrupada en rangos (--bin) hace que la salida no lo cumpla: hay que
agregarla primero al contrato (con revisión del equipo).

Seudonimizar no es anonimizar de forma irreversible: con la clave se puede recalcular el seudónimo de una cédula
conocida. La clave la guarda solo quien anonimiza; nunca va al repositorio, a MLflow ni a los logs.

Ejemplo:
    uv run python scripts/anonimizar.py --input extracto.xlsx --output data/raw/extracto_anon.parquet
    uv run python scripts/anonimizar.py --input extracto.xlsx --output data/raw/extracto_anon.parquet \\
        --eliminar OBSERVACIONES --conservar CODSUCURSAL --bin "SALARIO:0,2000000,4000000,8000000,1000000000"

Antes de anonimizar aplica contrato.limpiar_extracto (espacios, textos tipo 'N/A' -> vacío, números escritos
como texto, filas repetidas) sin romper el contrato de asociados. --sin-limpiar lo apaga.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from proyecto_ml import contrato as k
from proyecto_ml.anonymize import AnonimizacionInsegura, anonimizar_extracto
from proyecto_ml.contrato import limpiar_extracto
from proyecto_ml.data import load_dataset


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _bin(value: str) -> tuple[str, list[float]]:
    col, edges = value.split(":", 1)
    return col.strip(), [float(e) for e in edges.split(",")]


def escribir_atomico(df, destino: Path) -> None:
    """Escribe en un temporal de la misma carpeta y lo renombra: o queda el archivo completo, o no queda nada."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporal = tempfile.mkstemp(dir=destino.parent, prefix=f".{destino.name}.", suffix=".tmp")
    os.close(descriptor)
    try:
        df.to_parquet(temporal, index=False)
        os.replace(temporal, destino)
    except BaseException:
        Path(temporal).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contrato", type=Path, default=k.CONTRATO_ASOCIADOS, help="Contrato de origen (con pii)")
    parser.add_argument("--conservar", type=_csv, default=[], help="Columnas fuera del contrato que NO son personales")
    parser.add_argument("--eliminar", "--drop-cols", type=_csv, default=[], help="Columnas personales extra a borrar")
    parser.add_argument(
        "--seudonimizar", "--id-cols", type=_csv, default=[], help="Identificadores extra a seudonimizar"
    )
    parser.add_argument("--bin", type=_bin, action="append", default=[], help='Rangos: "COLUMNA:0,10,20"')
    parser.add_argument("--sin-limpiar", action="store_true", help="No limpiar el extracto antes de anonimizar")
    args = parser.parse_args(argv)

    if args.input.resolve() == args.output.resolve():
        sys.exit("--output no puede ser el mismo archivo que --input")
    contrato = k.cargar(args.contrato)
    df = load_dataset(args.input)
    if not args.sin_limpiar:
        df = limpiar_extracto(df, contrato)
    try:
        salida = anonimizar_extracto(
            df,
            contrato,
            conservar=args.conservar,
            eliminar=args.eliminar,
            seudonimizar_columnas=args.seudonimizar,
            bins=dict(args.bin),
        )
    except AnonimizacionInsegura as e:  # paso 3: quedaría PII
        sys.exit(f"No se escribió nada. {e}")
    contrato_ml = k.para_ml(contrato)
    if errores := k.sin_valores(k.validar(salida, contrato_ml)):  # paso 4, sin valores por si quedó algo personal
        detalle = "\n".join(f"  - {e}" for e in errores[:10])
        sys.exit(
            "No se escribió nada. La salida no cumple el contrato del dataset anonimizado para ML "
            f"({len(errores)} incumplimientos):\n{detalle}"
        )
    escribir_atomico(salida, args.output)  # paso 5
    eliminadas = sorted(set(df.columns) - set(salida.columns) - set(k.columnas_pii(contrato, "seudonimizar")))
    seudo = [c for c in salida.columns if c.endswith(k.SUFIJO_SEUDONIMO)]
    print(f"{len(salida)} filas, {salida.shape[1]} columnas -> {args.output}")
    print(f"Eliminadas ({len(eliminadas)}): {', '.join(eliminadas)}")
    print(f"Seudonimizadas: {', '.join(seudo)}")
    print("La salida cumple el contrato del dataset anonimizado para ML (contrato.para_ml).")
    for aviso in k.advertencias(salida, contrato_ml):
        print(f"Advertencia del contrato (no bloquea): {aviso}")


if __name__ == "__main__":
    main()
