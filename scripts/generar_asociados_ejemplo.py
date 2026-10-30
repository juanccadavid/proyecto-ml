"""Genera el dataset FICTICIO de asociados (la lógica vive en proyecto_ml.generador).

Uso:
    uv run python scripts/generar_asociados_ejemplo.py                 # los 50 de data/ejemplo/ (make ejemplo)
    uv run python scripts/generar_asociados_ejemplo.py --rows 1000 --seed 7 --output-dir data/processed --nombre nuevos
    uv run python scripts/generar_asociados_ejemplo.py --faltantes --output-dir data/processed --nombre sucios
Luego: uv run python scripts/ingesta.py --input data/ejemplo/asociados_ejemplo.parquet --target "#RETIROS"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from proyecto_ml.generador import CARPETA_EJEMPLO, FALTANTES, GeneradorAsociados


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasa-retiro", type=float, default=0.3, help="Probabilidad de retiro de cada asociado")
    parser.add_argument("--output-dir", type=Path, default=CARPETA_EJEMPLO)
    parser.add_argument("--nombre", default="asociados_ejemplo", help="Nombre base de los archivos")
    parser.add_argument("--faltantes", action="store_true", help="Vaciar datos al azar (ver generador.FALTANTES)")
    args = parser.parse_args()
    if args.faltantes and args.output_dir == CARPETA_EJEMPLO and args.nombre == "asociados_ejemplo":
        parser.error("con --faltantes usa otro --output-dir o --nombre: data/ejemplo debe cumplir el contrato")

    generador = GeneradorAsociados(args.rows, args.seed, args.tasa_retiro, FALTANTES if args.faltantes else None)
    parquet, xlsx = generador.guardar(args.output_dir, args.nombre)
    for columna, vacias in generador.reporte_faltantes_.items():
        print(f"  {vacias:5d} vacíos en {columna} ({vacias / args.rows:.1%}; esperado {FALTANTES[columna]:.0%})")
    retiros = pd.read_parquet(parquet)["#RETIROS"].mean()
    print(f"{args.rows} filas (retiros {retiros:.0%}) -> {parquet}, {xlsx}")


if __name__ == "__main__":
    main()
