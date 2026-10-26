"""Anonimiza un extracto ANTES de compartirlo con el equipo.

Lo ejecuta quien custodia los datos (dentro de su organización), nunca después de compartir.

Ejemplo:
    uv run python scripts/anonimizar.py --input extracto.xlsx --output data/raw/extracto_anon.parquet \
        --id-cols cedula --drop-cols nombre,correo,telefono,direccion \
        --bin "edad:18,25,35,45,55,65,100" --bin "salario:0,2000000,4000000,8000000,1000000000"

La sal se toma de ANON_SALT (.env). Guárdala solo quien anonimiza: sin ella no se revierte.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from proyecto_ml.anonymize import anonymize
from proyecto_ml.data import load_dataset


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _bin(value: str) -> tuple[str, list[float]]:
    col, edges = value.split(":", 1)
    return col.strip(), [float(e) for e in edges.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-cols", type=_csv, default=[])
    parser.add_argument("--drop-cols", type=_csv, default=[])
    parser.add_argument("--bin", type=_bin, action="append", default=[])
    args = parser.parse_args()

    df = load_dataset(args.input)
    out = anonymize(df, id_columns=args.id_cols, drop_columns=args.drop_cols, bins=dict(args.bin))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"{len(out)} filas, {out.shape[1]} columnas -> {args.output}")


if __name__ == "__main__":
    main()
