"""Los datos con que se elige el campeón (``datos_eleccion``) y los de los scripts exploratorios
(``datos_exploracion``): de dónde salen según el origen y el contrato que deben cumplir.

Separado de ``campeon`` (que elige el modelo) porque es otra responsabilidad: entrada y validación de datos. Se
sigue usando como ``campeon.datos_eleccion`` / ``campeon.DatosInvalidos``: ``campeon`` los reexporta.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from proyecto_ml import abandono as a


class DatosInvalidos(ValueError):
    """El lote no cumple el contrato de datos: no se entrena con él."""


def incumplimientos(archivo: str | Path, errores: list[str], real: bool) -> DatosInvalidos:
    """El error de un archivo que no cumple su contrato: cuántos incumplimientos, los 10 primeros y con qué
    comando revisarlo. Con datos reales, sin valores ni filas de ejemplo (``contrato.sin_valores``)."""
    from proyecto_ml import contrato

    mostrar = contrato.sin_valores(errores) if real else errores
    detalle = "\n".join(f"  - {e}" for e in mostrar[:10])
    return DatosInvalidos(
        f"{Path(archivo).name} no cumple el contrato de datos ({len(errores)} incumplimientos):\n{detalle}\n"
        f"Revísalo con: make {'contrato-ml' if real else 'contrato'} DATA={archivo}"
    )


def datos_eleccion(ruta: Path | None = None, origen: str | None = None) -> tuple[pd.DataFrame, str]:
    """Los asociados con que se elige el campeón y de dónde salieron, según el origen (``config.origen_datos``).

    - **sintetico** (por defecto): ``ruta``, o el archivo de ``make datos-desarrollo``, o los mismos 5.000
      generados en memoria (Streamlit Community Cloud, CI).
    - **real_anonimizado**: solo ``ruta`` (el extracto anonimizado). Nunca cae en los sintéticos: sin ``ruta``
      falla, para no etiquetar como reales unos datos generados.

    Todo archivo debe cumplir el contrato (``contrato.validar``); si no, ``DatosInvalidos``. Con datos reales el
    mensaje no lleva valores ni filas de ejemplo (``contrato.sin_valores``). Los generados en memoria no se
    validan aquí: el generador cumple el contrato por construcción (lo prueba tests/data).
    """
    from proyecto_ml import contrato
    from proyecto_ml.config import ORIGEN_REAL, PROCESSED_DIR, origen_datos
    from proyecto_ml.generador import NOMBRE_DESARROLLO, datos_desarrollo

    origen = origen_datos(origen)
    if origen == ORIGEN_REAL and ruta is None:
        raise DatosInvalidos(
            "ORIGEN_DATOS=real_anonimizado: indica el extracto anonimizado con --datos (make ... DATA=...). "
            "En este modo no se usan datos sintéticos."
        )
    desarrollo = PROCESSED_DIR / f"{NOMBRE_DESARROLLO}.parquet"
    if ruta is None and not desarrollo.exists():
        # La misma limpieza que a.cargar aplica al leer el archivo: así los dos caminos dan los mismos datos.
        return contrato.limpiar_extracto(datos_desarrollo()), "datos_desarrollo() en memoria"
    archivo = ruta or desarrollo
    df = a.cargar(archivo)
    # Datos reales: el contrato del dataset ANONIMIZADO (sin PII, con seudónimos); sintéticos: el de origen.
    exigido = contrato.para_ml() if origen == ORIGEN_REAL else contrato.cargar()
    errores = contrato.validar(df, exigido)
    for aviso in contrato.advertencias(df, exigido):  # heurísticas sin regla de negocio: se avisa, no se rechaza
        warnings.warn(aviso, contrato.AdvertenciaContrato, stacklevel=2)
    if origen == ORIGEN_REAL:  # fechas después del corte: FECHA_CORTE no es la del extracto y la foto queda corrida
        for columna in ("FECHARETIRO", "FECHAULTACTUALIZACION", "FECHAINGRESOLABORA"):
            if columna in df and (pd.to_datetime(df[columna], errors="coerce") > a.FECHA_CORTE).any():
                errores.append(
                    f"{columna} tiene fechas después de FECHA_CORTE ({a.FECHA_CORTE.date()}): pon en el .env "
                    "FECHA_CORTE = la fecha de extracción del archivo"
                )
    if errores:
        raise incumplimientos(archivo, errores, real=origen == ORIGEN_REAL)
    return df, archivo.name


def datos_exploracion(ruta: Path, script: str, limpiar: bool = True) -> pd.DataFrame:
    """Los datos de los scripts EXPLORATORIOS (make abandono, afinar, optuna, pycaret), o ``DatosInvalidos``.

    Esos scripts validan contra asociados "nuevos" del generador: solo tienen sentido con datos sintéticos. Con
    ``ORIGEN_DATOS=real_anonimizado`` mezclarían orígenes (y registrarían en el experimento real de MLflow), así
    que se niegan; la evaluación con datos reales es ``make campeon``. Con sintéticos, el archivo debe cumplir el
    contrato de origen, igual que en ``datos_eleccion``.
    """
    from proyecto_ml import contrato
    from proyecto_ml.config import ORIGEN_REAL, origen_datos

    if origen_datos() == ORIGEN_REAL:
        raise DatosInvalidos(
            f"{script} es exploratorio: compara contra asociados generados y solo tiene sentido con "
            "ORIGEN_DATOS=sintetico. Con datos reales usa make campeon DATA=<extracto anonimizado>."
        )
    df = a.cargar(ruta, limpiar)
    # Con --sin-limpieza el script entrena con el archivo tal cual, pero el contrato se mide sobre su versión
    # limpia: la limpieza no cambia el significado (espacios, 'N/A', repetidos), así que no rechaza de más.
    if errores := contrato.validar(df if limpiar else contrato.limpiar_extracto(df)):
        raise incumplimientos(ruta, errores, real=False)
    return df
