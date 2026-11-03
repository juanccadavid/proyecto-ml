"""Predicción con el campeón guardado: filas crudas -> puntaje, zona, alerta y grupo de KMeans (make predecir).

Es el paso de "producción" de la plantilla: ``train`` (aquí, ``make campeon``) deja el modelo en models/ y este
módulo lo usa con asociados nuevos, sin reentrenar. Las filas pueden venir tal como salen del extracto, de
Supabase o del formulario: el pipeline del campeón prepara las variables (``abandono.PrepararVariables``).

Uso:
    uv run python -m proyecto_ml.predict --datos data/raw/supabase_asociados.parquet
    uv run python -m proyecto_ml.predict --datos nuevos.csv --salida data/processed/predicciones.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml.config import MODELS_DIR, PROCESSED_DIR

SALIDA = PROCESSED_DIR / "predicciones.csv"  # data/ no va a Git: las predicciones son de asociados reales
COLUMNAS = ["puntaje", "zona", "alerta", "grupo_kmeans"]
IDENTIFICADORES = ["IDASOCIADOS", "CEDULA"]  # se copian a la salida para cruzarla (la cédula, si ya viene anonimizada)


def cargar(carpeta: Path = MODELS_DIR):
    """(modelo con zonas, KMeans, info) del campeón guardado; si no hay, pide correr ``make campeon``."""
    guardado = c.cargar(carpeta)
    if guardado is None:
        raise FileNotFoundError(f"No hay campeón en {carpeta}: corre primero `make campeon`.")
    return guardado


def predecir(df: pd.DataFrame, carpeta: Path = MODELS_DIR) -> pd.DataFrame:
    """Una fila por asociado de ``df``: puntaje del campeón, zona, alerta (0/1) y grupo de KMeans."""
    modelo, agrupador, _ = cargar(carpeta)
    X = df.drop(columns=a.TARGET, errors="ignore")
    salida = df[[col for col in IDENTIFICADORES if col in df]].copy()
    salida["puntaje"] = a.puntaje(modelo, X)
    salida["zona"] = modelo.predict(X)
    salida["alerta"] = modelo.alerta(X)
    salida["grupo_kmeans"] = pd.Series(agrupador.predict(X), index=df.index).map({1: "riesgo", 0: "estable"})
    return salida


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--datos", type=Path, required=True, help="Parquet, CSV o Excel con asociados")
    parser.add_argument("--salida", type=Path, default=SALIDA, help=f"CSV de salida (por defecto {SALIDA})")
    parser.add_argument("--modelos", type=Path, default=MODELS_DIR, help="Carpeta del campeón")
    args = parser.parse_args(argv)

    df = a.cargar(args.datos)
    predicciones = predecir(df, args.modelos)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    predicciones.to_csv(args.salida, index=False)
    conteo = predicciones["zona"].value_counts().reindex(list(c.ZONAS), fill_value=0)
    print(f"{len(predicciones)} asociados de {args.datos.name}:")
    print(conteo.to_string())
    print(f"Predicciones -> {args.salida}")


if __name__ == "__main__":
    main()
