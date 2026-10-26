"""Genera un dataset sintético para probar el pipeline sin datos reales.

Uso: uv run python scripts/generar_datos_sinteticos.py --rows 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from proyecto_ml.config import PROCESSED_DIR
from proyecto_ml.synthetic import make_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "sintetico.parquet")
    args = parser.parse_args()

    df = make_synthetic(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"{len(df)} filas -> {args.output} (tasa positiva {df['target'].mean():.1%})")


if __name__ == "__main__":
    main()
