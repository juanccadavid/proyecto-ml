"""Predicción con el campeón guardado: filas crudas -> puntaje, zona, alerta y grupo de KMeans (make predecir).

Es el paso de "producción" de la plantilla: ``train`` (aquí, ``make campeon``) deja el modelo en models/ y este
módulo lo usa con asociados nuevos, sin reentrenar. Las filas pueden venir tal como salen del extracto, de
Supabase o del formulario: el pipeline del campeón prepara las variables (``abandono.PrepararVariables``).

Privacidad: la salida NUNCA copia cédulas ni otros datos personales. Para cruzar las predicciones con los
registros quedan ``fila`` (la posición en el archivo de entrada) y, si el archivo viene anonimizado, sus seudónimos
``*_SEUDO`` (HMAC de 64 caracteres hex; se rechaza uno que no tenga esa forma, p. ej. una cédula cruda renombrada).
Con ``ORIGEN_DATOS=real_anonimizado`` se rechaza además un archivo que traiga columnas personales del contrato.
Volver del seudónimo a la persona es una tabla de correspondencia que guarda el custodio, fuera de este pipeline.

Uso:
    uv run python -m proyecto_ml.predict --datos data/raw/supabase_asociados.parquet
    uv run python -m proyecto_ml.predict --datos nuevos.csv --salida data/processed/predicciones.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import contrato as k
from proyecto_ml.config import MODELS_DIR, ORIGEN_REAL, PROCESSED_DIR, origen_datos

SALIDA = PROCESSED_DIR / "predicciones.csv"  # data/ no va a Git: las predicciones son de asociados reales
COLUMNAS = ["puntaje", "zona", "alerta", "grupo_kmeans"]


class DatosPersonales(ValueError):
    """El archivo o la salida tendrían datos personales: no se predice ni se escribe."""


def seudonimos(df: pd.DataFrame) -> list[str]:
    """Las columnas ``*_SEUDO`` de ``df``, verificadas: todas sus filas deben ser un HMAC de 64 caracteres hex."""
    columnas = [col for col in df.columns if str(col).endswith(k.SUFIJO_SEUDONIMO)]
    patron = re.compile(k.PATRON_SEUDONIMO)
    for col in columnas:
        malos = int((~df[col].astype("string").fillna("").map(lambda v: bool(patron.fullmatch(v)))).sum())
        if malos:
            raise DatosPersonales(f"{col}: {malos} valor(es) no son un seudónimo (¿una cédula sin seudonimizar?)")
    return columnas


def cargar(carpeta: Path = MODELS_DIR):
    """(modelo con zonas, KMeans, info) del campeón guardado; si no hay, pide correr ``make campeon``."""
    guardado = c.cargar(carpeta)
    if guardado is None:
        raise FileNotFoundError(f"No hay campeón en {carpeta}: corre primero `make campeon`.")
    return guardado


def predecir(df: pd.DataFrame, carpeta: Path = MODELS_DIR, origen: str | None = None) -> pd.DataFrame:
    """Una fila por asociado de ``df``: ``fila``, los seudónimos que traiga, puntaje del campeón, zona, alerta
    (0/1) y grupo de KMeans. Sin datos personales (ver el docstring del módulo)."""
    if origen_datos(origen) == ORIGEN_REAL and (personales := [col for col in k.columnas_pii() if col in df]):
        raise DatosPersonales(f"El archivo trae columnas personales {personales}: anonimízalo antes (make anonimizar).")
    ids = seudonimos(df)
    modelo, agrupador, _ = cargar(carpeta)
    X = df.drop(columns=a.TARGET, errors="ignore")
    salida = df[ids].copy()
    salida.insert(0, "fila", range(len(df)))
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
