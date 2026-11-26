"""Predicción con el campeón guardado: filas crudas -> puntaje, zona, alerta y grupo de KMeans (make predecir).

Es el paso de "producción" de la plantilla: ``train`` (aquí, ``make campeon``) deja el modelo en models/ y este
módulo lo usa con asociados nuevos, sin reentrenar. Las filas pueden venir tal como salen del extracto, de
Supabase o del formulario: el pipeline del campeón prepara las variables (``abandono.PrepararVariables``).

Privacidad: la salida nunca copia identificadores ni campos personales, aunque el archivo de entrada ya esté
anonimizado. Para cruzar resultados conserva únicamente ``fila`` (la posición en el archivo de entrada).

Uso:
    uv run python -m proyecto_ml.predict --datos data/raw/supabase_asociados.parquet
    uv run python -m proyecto_ml.predict --datos nuevos.csv --salida data/processed/predicciones.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import contrato as k
from proyecto_ml.config import MODELS_DIR, ORIGEN_REAL, ORIGEN_SINTETICO, PROCESSED_DIR, origen_datos

SALIDA = PROCESSED_DIR / "predicciones.csv"  # data/ no va a Git: las predicciones son de asociados reales
COLUMNAS = ["puntaje", "zona", "alerta", "grupo_kmeans"]


class DatosPersonales(ValueError):
    """El archivo o la salida tendrían datos personales: no se predice ni se escribe."""


def cargar(carpeta: Path = MODELS_DIR):
    """(modelo con zonas, KMeans, info) del campeón guardado; si no hay, pide correr ``make campeon``."""
    guardado = c.cargar(carpeta)
    if guardado is None:
        raise FileNotFoundError(f"No hay campeón en {carpeta}: corre primero `make campeon`.")
    return guardado


def revisar_real(df: pd.DataFrame) -> None:
    """Modo ``real_anonimizado``: solo columnas del contrato definitivo y todos los predictores requeridos.

    No exige el contrato completo: un lote para predecir puede ser chico (``volumen_minimo``) o traer solo
    asociados activos; lo que sí se exige es que no traiga nada que no deba salir.
    """
    contrato_ml = k.para_ml()
    if desconocidas := [str(col) for col in df.columns if col not in contrato_ml.nombres]:
        raise DatosPersonales(
            f"Columnas que el contrato del dataset anonimizado no conoce {desconocidas}: podrían ser personales. "
            "Quítalas o agrégalas al contrato (con revisión del equipo)."
        )
    if faltan := sorted(set(a.NUMERICAS + a.CATEGORICAS) - set(df.columns)):
        raise DatosPersonales(f"Faltan variables requeridas por el modelo: {faltan}")


def activos(df: pd.DataFrame) -> pd.Series:
    """El esquema definitivo no trae fecha de retiro; todas las filas recibidas se pueden puntuar."""
    return pd.Series(True, index=df.index)


def predecir(
    df: pd.DataFrame, carpeta: Path = MODELS_DIR, origen: str | None = None, solo_activos: bool = True
) -> pd.DataFrame:
    """Una fila por registro: posición, puntaje, zona, alerta y grupo. Nunca copia identificadores.

    En modo ``real_anonimizado`` también aplica la lista blanca del contrato definitivo. ``solo_activos`` se
    conserva por compatibilidad; el esquema ya no trae fecha de retiro y por tanto no filtra filas.
    """
    if origen_datos(origen) == ORIGEN_REAL:
        revisar_real(df)
    modelo, agrupador, _ = cargar(carpeta)
    filas = np.arange(len(df))  # posición en el archivo de entrada, también para los que quedan fuera
    if solo_activos:
        vigentes = activos(df).to_numpy()
        df, filas = df[vigentes], filas[vigentes]
    if df.empty:  # todos ya se retiraron (o el archivo viene vacío): nada que predecir
        return pd.DataFrame(columns=["fila", *COLUMNAS])
    X = df.drop(columns=a.TARGET, errors="ignore")
    salida = pd.DataFrame({"fila": filas}, index=df.index)
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
    fuera = len(df) - len(predicciones)
    print(f"{len(predicciones)} registros de {args.datos.name} ({fuera} excluidos):")
    print(conteo.to_string())
    print(f"Predicciones -> {args.salida}")
    if aviso := aviso_campeon_sintetico(args.modelos):
        print(aviso)
        print(resumen_desplazamiento(df))


def resumen_desplazamiento(lote: pd.DataFrame, top: int = 5) -> str:
    """Qué tan distinto es ``lote`` de los sintéticos de desarrollo (``validacion.desplazamiento``): una línea con
    los ROC-AUC de la validación adversaria y las variables que más distinguen. Solo agregados."""
    from proyecto_ml import validacion as v
    from proyecto_ml.config import ORIGEN_SINTETICO

    referencia, _ = c.datos_eleccion(origen=ORIGEN_SINTETICO)  # los de make campeon por defecto
    d = v.desplazamiento(referencia, lote)
    distinguen = ", ".join(f"{col} {auc:.2f}" for col, auc in list(d["por_variable"].items())[:top])
    return (
        "Parecido con los sintéticos de entrenamiento (validación adversaria, ROC-AUC: 0,5 = indistinguibles, "
        f"1 = fuera de lo que vio el modelo): todas las variables {d['roc_auc']:.2f}; solo comportamiento "
        f"{d['roc_auc_comportamiento']:.2f}. Las que más distinguen: {distinguen}."
    )


def aviso_campeon_sintetico(carpeta: Path = MODELS_DIR) -> str | None:
    """El aviso para cuando se puntúan datos reales con un campeón entrenado con sintéticos (``predecir-real``):
    el puntaje sigue la regla del generador y no está validado contra retiros reales. None en los demás casos."""
    if origen_datos() != ORIGEN_REAL:
        return None
    info = json.loads((carpeta / c.ARCHIVO_INFO).read_text(encoding="utf-8"))
    if info.get("origen_datos", ORIGEN_SINTETICO) == ORIGEN_REAL:
        return None
    return (
        "AVISO: el campeón se entrenó con datos SINTÉTICOS. Sobre el extracto real, el puntaje y las zonas ordenan "
        "según la regla del generador y NO están validados contra retiros reales: no son una probabilidad de retiro "
        "del fondo (docs/datos_reales.md)."
    )


if __name__ == "__main__":
    main()
