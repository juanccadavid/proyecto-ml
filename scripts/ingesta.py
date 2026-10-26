"""Valida un archivo crudo, genera un perfil de calidad y lo deja en data/processed.

Uso: uv run python scripts/ingesta.py --input data/raw/archivo.csv --target target
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proyecto_ml.config import PROCESSED_DIR, REPORTS_DIR
from proyecto_ml.data import load_dataset


def profile(df, target: str) -> dict:
    return {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "duplicados": int(df.duplicated().sum()),
        "nulos_pct": {c: round(float(v), 4) for c, v in df.isna().mean().items() if v > 0},
        "tipos": {c: str(t) for c, t in df.dtypes.items()},
        "distribucion_target": {str(k): int(v) for k, v in df[target].value_counts(dropna=False).items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "dataset.parquet")
    args = parser.parse_args()

    df = load_dataset(args.input)
    if args.target not in df.columns:
        raise SystemExit(f"Falta la columna objetivo '{args.target}'")
    if df[args.target].nunique(dropna=True) != 2:
        print("ADVERTENCIA: el objetivo no es binario; revisa la definición de la etiqueta.")

    info = profile(df, args.target)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "perfil_datos.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    df = df.drop_duplicates()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(json.dumps({k: info[k] for k in ("filas", "columnas", "duplicados", "distribucion_target")}, indent=2))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
