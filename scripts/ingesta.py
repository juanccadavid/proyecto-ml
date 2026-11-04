"""Valida un archivo crudo, genera un perfil de calidad y lo deja en data/processed.

Uso: uv run python scripts/ingesta.py --input data/raw/archivo.csv --target target

La ingesta trabaja con Polars (lectura, perfil, duplicados, escritura): es la etapa que crece con el
volumen de datos. El resto del proyecto (contrato con pandera, modelos con scikit-learn) sigue en
pandas; el archivo de salida es un parquet normal que pandas lee igual que antes.
Al lado de cada paso de Polars va su equivalente en pandas, para leerlo si solo conoces pandas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from proyecto_ml import contrato
from proyecto_ml.config import PROCESSED_DIR, REPORTS_DIR
from proyecto_ml.data import load_dataset


def leer(path: Path) -> pl.DataFrame:
    """Parquet lo lee Polars directo; CSV/Excel pasan por load_dataset (pandas), que ya los maneja."""
    if path.suffix.lower() == ".parquet" and path.exists():
        return pl.read_parquet(path)  # pandas: pd.read_parquet(path)
    return pl.from_pandas(load_dataset(path))  # convierte; NaN de pandas -> null de Polars


def profile(df: pl.DataFrame, target: str) -> dict:
    nulos = df.null_count().row(0, named=True)  # pandas: df.isna().sum()  -> {columna: n_nulos}
    return {
        "filas": df.height,  # pandas: len(df)
        "columnas": df.width,  # pandas: df.shape[1]
        "duplicados": df.height - df.unique().height,  # pandas: df.duplicated().sum()
        "nulos_pct": {c: round(n / df.height, 4) for c, n in nulos.items() if n > 0},
        "tipos": {c: str(t) for c, t in df.schema.items()},  # pandas: df.dtypes (nombres Polars: Int64, String...)
        # pandas: df[target].value_counts(dropna=False); en Polars devuelve una tabla (valor, count)
        "distribucion_target": {str(k): int(v) for k, v in df[target].value_counts(sort=True).iter_rows()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "dataset.parquet")
    parser.add_argument("--contrato", type=Path, help="Contrato de datos a exigir (ej. contratos/asociados.yaml)")
    parser.add_argument(
        "--ml", action="store_true", help="Con --contrato: exigir su versión para el dataset anonimizado (sin PII)"
    )
    args = parser.parse_args()

    df = leer(args.input)
    if args.contrato:
        # el contrato está escrito con pandera sobre pandas: se le pasa una copia en pandas
        exigido = contrato.cargar(args.contrato)
        exigido = contrato.para_ml(exigido) if args.ml else exigido
        errores = contrato.validar(df.to_pandas(), exigido)
        errores = contrato.sin_valores(errores) if args.ml else errores  # anonimizado: sin valores en la consola
        if errores:
            raise SystemExit("El archivo no cumple el contrato:\n  - " + "\n  - ".join(errores))
    if args.target not in df.columns:
        raise SystemExit(f"Falta la columna objetivo '{args.target}'")
    if df[args.target].drop_nulls().n_unique() != 2:  # pandas: df[target].nunique(dropna=True)
        print("ADVERTENCIA: el objetivo no es binario; revisa la definición de la etiqueta.")

    info = profile(df, args.target)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "perfil_datos.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    df = df.unique(keep="first", maintain_order=True)  # pandas: df.drop_duplicates()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)  # pandas: df.to_parquet(args.output, index=False)
    print(json.dumps({k: info[k] for k in ("filas", "columnas", "duplicados", "distribucion_target")}, indent=2))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
