"""Genera el dataset FICTICIO de asociados (la lógica vive en proyecto_ml.generador).

Uso:
    uv run python scripts/generar_asociados_ejemplo.py                 # los 50 de data/ejemplo/ (make ejemplo)
    uv run python scripts/generar_asociados_ejemplo.py --rows 1000 --seed 7 --output-dir data/processed --nombre nuevos
    uv run python scripts/generar_asociados_ejemplo.py --faltantes --output-dir data/processed --nombre sucios
    uv run python scripts/generar_asociados_ejemplo.py --desarrollo   # los 2.000 de desarrollo (make datos-desarrollo)
    uv run python scripts/generar_asociados_ejemplo.py --rows 1000 --tasa-retiro 0.1 --fuerza-senal 0.5 \\
        --output-dir data/processed --nombre senal_debil                # retirados menos distinguibles
Luego: uv run python scripts/ingesta.py --input data/ejemplo/asociados_ejemplo.parquet --target "#RETIROS"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from proyecto_ml.config import PROCESSED_DIR
from proyecto_ml.generador import (
    CARPETA_EJEMPLO,
    DESARROLLO,
    FALTANTES,
    NOMBRE_DESARROLLO,
    GeneradorAsociados,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasa-retiro", type=float, default=0.3, help="Probabilidad de retiro de cada asociado")
    parser.add_argument(
        "--fuerza-senal",
        type=float,
        default=1.0,
        help="Qué tanto se diferencian retirados y activos: 1 = como siempre, 0.5 = la mitad, 0 = nada",
    )
    parser.add_argument("--output-dir", type=Path, default=CARPETA_EJEMPLO)
    parser.add_argument("--nombre", default="asociados_ejemplo", help="Nombre base de los archivos")
    parser.add_argument("--faltantes", action="store_true", help="Vaciar datos al azar (ver generador.FALTANTES)")
    parser.add_argument(
        "--desarrollo",
        action="store_true",
        help=f"Los datos de desarrollo: {DESARROLLO} en {PROCESSED_DIR.name}/{NOMBRE_DESARROLLO}.parquet",
    )
    args = parser.parse_args()
    if args.desarrollo:
        args.rows, args.seed, args.tasa_retiro = DESARROLLO["n"], DESARROLLO["seed"], DESARROLLO["tasa_retiro"]
        args.output_dir, args.nombre = PROCESSED_DIR, NOMBRE_DESARROLLO
    if args.faltantes and args.output_dir == CARPETA_EJEMPLO and args.nombre == "asociados_ejemplo":
        parser.error("con --faltantes usa otro --output-dir o --nombre: data/ejemplo debe cumplir el contrato")

    generador = GeneradorAsociados(
        args.rows, args.seed, args.tasa_retiro, FALTANTES if args.faltantes else None, args.fuerza_senal
    )
    parquet, xlsx = generador.guardar(args.output_dir, args.nombre)
    for columna, vacias in generador.reporte_faltantes_.items():
        print(f"  {vacias:5d} vacíos en {columna} ({vacias / args.rows:.1%}; esperado {FALTANTES[columna]:.0%})")
    retiros = pd.read_parquet(parquet)["#RETIROS"].mean()
    print(f"{args.rows} filas (retiros {retiros:.0%}) -> {parquet}, {xlsx}")


if __name__ == "__main__":
    main()
