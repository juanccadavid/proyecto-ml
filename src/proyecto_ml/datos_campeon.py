"""Los datos con que se elige el campeón (``datos_eleccion``) y los de los scripts exploratorios
(``datos_exploracion``): de dónde salen según el origen y el contrato que deben cumplir.

Separado de ``campeon`` (que elige el modelo) porque es otra responsabilidad: entrada y validación de datos. Se
sigue usando como ``campeon.datos_eleccion`` / ``campeon.DatosInvalidos``: ``campeon`` los reexporta.
"""

from __future__ import annotations

import json
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


def archivo_eleccion(ruta: Path | None = None) -> Path | None:
    """El archivo que lee ``datos_eleccion``: ``ruta``, o el de ``make datos-desarrollo`` si existe; None si los
    datos se generan en memoria."""
    from proyecto_ml.config import PROCESSED_DIR
    from proyecto_ml.generador import NOMBRE_DESARROLLO

    if ruta is not None:
        return Path(ruta)
    desarrollo = PROCESSED_DIR / f"{NOMBRE_DESARROLLO}.parquet"
    return desarrollo if desarrollo.exists() else None


def nombre_fuente(ruta: Path | None = None) -> str:
    """El nombre de lo que leen ``datos_eleccion`` / ``datos_exploracion`` con ``ruta``: para consola y MLflow."""
    archivo = archivo_eleccion(ruta)
    return archivo.name if archivo is not None else "datos_desarrollo() en memoria"


def marca_datos(ruta: Path | None = None) -> str:
    """Huella del CONTENIDO real de los datos que leería ``datos_eleccion(ruta)``: cambia si el archivo se reescribe
    aunque conserve la ruta, o si cambian los parámetros de ``datos_desarrollo`` en memoria."""
    import hashlib

    from proyecto_ml import generador as g

    archivo = archivo_eleccion(ruta)
    if archivo is None:
        parametros = {nombre: valor for nombre, valor in vars(g).items() if nombre.isupper()}
        contenido = json.dumps(parametros, sort_keys=True, default=str)
        return "en_memoria:" + hashlib.sha256(contenido.encode()).hexdigest()[:12]
    if not archivo.exists():
        return f"ausente:{archivo.name}"
    return hashlib.sha256(archivo.read_bytes()).hexdigest()[:16]


def datos_eleccion(
    ruta: Path | None = None, origen: str | None = None, supervisado: bool = True
) -> tuple[pd.DataFrame, str]:
    """Los asociados con que se elige el campeón y de dónde salieron, según el origen (``config.origen_datos``).

    - **sintetico** (por defecto): ``ruta``, o el archivo de ``make datos-desarrollo``, o los mismos 5.000
      generados en memoria (Streamlit Community Cloud, CI).
    - **real_anonimizado**: solo ``ruta`` (el extracto anonimizado). Nunca cae en los sintéticos: sin ``ruta``
      falla, para no etiquetar como reales unos datos generados.

    Las 44 columnas del extracto deben cumplir el contrato. Los sintéticos llevan además
    ``EVENTO_RETIRO_3M_SINTETICO``. Un extracto real no tiene esa etiqueta: con ``supervisado=True`` (el
    abandono) se rechaza, porque ``#RETIROS`` es historia acumulada y no el resultado de los próximos tres meses.
    Con ``supervisado=False`` (segmentación y regresión del salario, que no usan esa etiqueta) se acepta.
    """
    from proyecto_ml import contrato
    from proyecto_ml.config import ORIGEN_REAL, origen_datos
    from proyecto_ml.generador import datos_desarrollo

    origen = origen_datos(origen)
    if origen == ORIGEN_REAL and ruta is None:
        raise DatosInvalidos(
            "ORIGEN_DATOS=real_anonimizado: indica el extracto anonimizado con --datos (make ... DATA=...). "
            "En este modo no se usan datos sintéticos."
        )
    archivo = archivo_eleccion(ruta)
    if archivo is None:
        # La misma limpieza que a.cargar aplica al leer el archivo: así los dos caminos dan los mismos datos.
        return contrato.limpiar_extracto(datos_desarrollo()), "datos_desarrollo() en memoria"
    if not archivo.exists():
        raise DatosInvalidos(f"No existe el archivo de datos: {archivo.name}")
    df = a.cargar(archivo)
    # El contrato describe únicamente las 44 columnas reales. La etiqueta 3M es un artefacto del generador.
    exigido = contrato.para_ml() if origen == ORIGEN_REAL else contrato.cargar()
    base = df.drop(columns=[a.TARGET], errors="ignore")
    errores = contrato.validar(base, exigido)
    extras = set(df) - set(exigido.nombres) - ({a.TARGET} if origen != ORIGEN_REAL else set())
    if extras:
        errores.append(f"columnas no previstas por el contrato: {sorted(extras)}")
    if origen == ORIGEN_REAL and supervisado:
        errores.append(
            f"falta una etiqueta futura válida: #RETIROS es acumulado durante la permanencia y no puede usarse "
            f"como {a.TARGET}"
        )
    elif origen != ORIGEN_REAL and supervisado and a.TARGET not in df:
        errores.append(f"falta la etiqueta sintética {a.TARGET}")
    for aviso in contrato.advertencias(base, exigido):  # heurísticas sin regla de negocio: se avisa, no se rechaza
        warnings.warn(aviso, contrato.AdvertenciaContrato, stacklevel=2)
    if origen == ORIGEN_REAL:  # fechas después del corte: FECHA_CORTE no es la del extracto y la foto queda corrida
        # Solo las fechas que registra el fondo. La de ingreso laboral la reporta la empresa y trae errores de
        # digitación (años futuros): esos los informa el contrato como advertencia. Se compara el día, no la hora:
        # una actualización a las 23:00 del día del corte es del corte.
        corte_actual = a.fecha_observacion()
        for columna in ("FECHAULTACTUALIZACION", "FECHAINGRESO"):
            if columna in df and (pd.to_datetime(df[columna], errors="coerce").dt.normalize() > corte_actual).any():
                errores.append(
                    f"{columna} tiene fechas después de FECHA_CORTE ({corte_actual.date()}): pon en el .env "
                    "FECHA_CORTE = la fecha de extracción del archivo"
                )
    if errores:
        raise incumplimientos(archivo, errores, real=origen == ORIGEN_REAL)
    return df, archivo.name


def datos_exploracion(ruta: Path | None, script: str, limpiar: bool = True) -> pd.DataFrame:
    """Los datos de los scripts EXPLORATORIOS (make abandono, afinar, optuna, pycaret), o ``DatosInvalidos``.

    Esos scripts validan contra asociados "nuevos" del generador: solo tienen sentido con datos sintéticos. Con
    ``ORIGEN_DATOS=real_anonimizado`` mezclarían orígenes (y registrarían en el experimento real de MLflow), así
    que se niegan; la evaluación con datos reales es ``make campeon``. Con sintéticos, el archivo debe cumplir el
    contrato de origen, igual que en ``datos_eleccion``.

    Sin ``ruta``, los mismos datos de desarrollo que ``make campeon`` (``archivo_eleccion``, o el generador en
    memoria): así los experimentos se comparan con el campeón. Antes el valor por defecto era ``DATOS_ASOCIADOS``,
    que desde el contrato v2 nunca trae la etiqueta sintética (data/ejemplo y Supabase tienen las 44 columnas del
    extracto), y los cuatro comandos fallaban sin argumentos.
    """
    from proyecto_ml import contrato
    from proyecto_ml.config import ORIGEN_REAL, origen_datos
    from proyecto_ml.generador import datos_desarrollo

    if origen_datos() == ORIGEN_REAL:
        raise DatosInvalidos(
            f"{script} es exploratorio: compara contra asociados generados y solo tiene sentido con "
            "ORIGEN_DATOS=sintetico. El extracto real no tiene una etiqueta futura para entrenamiento."
        )
    archivo = archivo_eleccion(ruta)
    if archivo is None:  # sin archivo de desarrollo: los mismos 5.000 de make campeon, generados en memoria
        desarrollo = datos_desarrollo()
        return contrato.limpiar_extracto(desarrollo) if limpiar else desarrollo
    ruta = archivo
    df = a.cargar(ruta, limpiar)
    # Con --sin-limpieza el script entrena con el archivo tal cual, pero el contrato se mide sobre su versión
    # limpia: la limpieza no cambia el significado (espacios, 'N/A', repetidos), así que no rechaza de más.
    medido = df if limpiar else contrato.limpiar_extracto(df)
    base = medido.drop(columns=[a.TARGET], errors="ignore")
    errores = contrato.validar(base)
    if a.TARGET not in medido:
        errores.append(f"falta la etiqueta sintética {a.TARGET}")
    if extras := set(medido) - set(contrato.cargar().nombres) - {a.TARGET}:
        errores.append(f"columnas no previstas por el contrato: {sorted(extras)}")
    if errores:
        raise incumplimientos(ruta, errores, real=False)
    return df
