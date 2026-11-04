"""Anonimiza el extracto de asociados ANTES de compartirlo con el equipo (make anonimizar).

Lo ejecuta quien custodia los datos (dentro de su organización), nunca después de compartir.

La configuración de qué es personal es el contrato de origen (contratos/asociados.yaml, campo ``pii``):
``eliminar`` se borra y ``seudonimizar`` sale como ``<COLUMNA>_SEUDO`` (HMAC-SHA256 de 64 caracteres hex con la
clave ANON_SALT del .env). Una columna del extracto que el contrato no conoce detiene el proceso hasta que se
decida qué hacer con ella: --conservar, --eliminar o --seudonimizar.

Antes de escribir verifica que no quede ninguna columna personal, que los seudónimos sean válidos y que ningún
valor personal se haya colado en otra columna; si algo falla, NO escribe nada (ni un archivo a medias) y el
error dice columnas y cantidades, nunca valores. Al final valida la salida contra el contrato del dataset
anonimizado para ML (contrato.para_ml), el mismo que exige make campeon con ORIGEN_DATOS=real_anonimizado.

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
    except AnonimizacionInsegura as e:
        sys.exit(f"No se escribió nada. {e}")
    escribir_atomico(salida, args.output)
    eliminadas = sorted(set(df.columns) - set(salida.columns) - set(k.columnas_pii(contrato, "seudonimizar")))
    seudo = [c for c in salida.columns if c.endswith(k.SUFIJO_SEUDONIMO)]
    print(f"{len(salida)} filas, {salida.shape[1]} columnas -> {args.output}")
    print(f"Eliminadas ({len(eliminadas)}): {', '.join(eliminadas)}")
    print(f"Seudonimizadas: {', '.join(seudo)}")
    errores = k.sin_valores(k.validar(salida, k.para_ml(contrato)))
    if errores:
        print(f"Ojo: la salida no cumple el contrato del dataset anonimizado para ML ({len(errores)}):")
        print("\n".join(f"  - {e}" for e in errores[:10]))
    else:
        print("La salida cumple el contrato del dataset anonimizado para ML (contrato.para_ml).")


if __name__ == "__main__":
    main()
