"""Varios extractos de fechas distintas: la base de una validación FUERA DE TIEMPO de verdad.

Con un solo extracto, las dos fotos de ``validacion.particion_temporal`` son las mismas personas con los mismos
conteos (no traen fecha): es una prueba de estrés, no una validación fuera de tiempo (README, "Resultados").
Con extractos sacados en fechas distintas, cada conteo está medido a la fecha de SU extracto, y la foto se arma
sin esa ambigüedad:

- **variables**: las del extracto viejo, tal como estaban en su fecha (``FECHA_OBSERVACION`` = esa fecha, así
  EDAD y ANTIGUEDAD también);
- **población**: los activos del extracto viejo en su fecha;
- **objetivo**: 1 si el extracto siguiente muestra que se retiró entre la fecha del viejo y ``horizonte_meses``
  después; 0 si sigue activo. El objetivo nunca sale del mismo archivo que las variables.

Las filas se cruzan por el seudónimo (``CEDULA_SEUDO``, o ``IDASOCIADOS_SEUDO``; con sintéticos, ``CEDULA``): todos
los extractos deben anonimizarse con la MISMA ``ANON_SALT``. Los requisitos están en docs/datos_reales.md.

``campeon.validacion_fuera_de_tiempo`` usa esto con 3 o más extractos: entrena con los pares viejos y mide con el
par más reciente. Este módulo no genera datos ni resultados: espera los extractos reales.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from proyecto_ml import abandono as a

LLAVES = ("CEDULA_SEUDO", "IDASOCIADOS_SEUDO", "CEDULA")  # la primera que traigan los dos extractos
COBERTURA_MIN = 0.9  # de los activos del viejo, al menos esta fracción debe aparecer en el nuevo


@dataclass(frozen=True)
class Extracto:
    """Un archivo de asociados y la fecha en que se sacó (sus conteos y atributos están medidos a esa fecha)."""

    datos: pd.DataFrame
    fecha: pd.Timestamp
    nombre: str = ""

    def __post_init__(self) -> None:
        # Una fecha vacía (None, "NaT") apagaría en silencio los controles de revisar_par (horizonte cubierto,
        # retiros posteriores al extracto): toda comparación con NaT da False.
        if pd.isna(self.fecha):
            raise ValueError(f"El extracto {self.nombre or '(sin nombre)'} no tiene fecha: indica AAAA-MM-DD.")
        # Las fechas de los extractos (FECHARETIRO, FECHAINGRESOLABORA) no tienen zona horaria: compararlas con una
        # que sí la tiene daría un TypeError críptico en revisar_par.
        if getattr(self.fecha, "tzinfo", None) is not None:
            raise ValueError(f"La fecha del extracto {self.nombre or '(sin nombre)'} no debe tener zona horaria.")

    @classmethod
    def de(cls, datos: pd.DataFrame, fecha: str | pd.Timestamp, nombre: str = "") -> Extracto:
        return cls(datos, pd.Timestamp(fecha), nombre or str(pd.Timestamp(fecha).date()))


def llave(viejo: Extracto, nuevo: Extracto) -> str:
    """La columna con que se cruzan los dos extractos, o ``ValueError`` si no comparten ninguna."""
    for columna in LLAVES:
        if columna in viejo.datos and columna in nuevo.datos:
            return columna
    raise ValueError(f"{viejo.nombre} y {nuevo.nombre} no comparten ninguna llave {LLAVES}: no se pueden cruzar.")


def _retiro(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["FECHARETIRO"], errors="coerce")


def activos(extracto: Extracto) -> pd.DataFrame:
    """Los asociados activos en la fecha del extracto: ingresaron antes y no tienen retiro hasta esa fecha."""
    df, fecha = extracto.datos, extracto.fecha
    retiro = _retiro(df)
    vigentes = (pd.to_datetime(df["FECHAINGRESOLABORA"], errors="coerce") <= fecha) & (retiro.isna() | (retiro > fecha))
    return df[vigentes]


def revisar_par(viejo: Extracto, nuevo: Extracto, horizonte_meses: int = a.HORIZONTE_MESES) -> list[str]:
    """Problemas que impiden armar la foto ``viejo -> nuevo`` (lista vacía si se puede). Sin valores personales."""
    problemas = []
    fin = viejo.fecha + pd.DateOffset(months=horizonte_meses)
    if nuevo.fecha < fin:
        problemas.append(
            f"{nuevo.nombre} ({nuevo.fecha.date()}) no cubre el horizonte: hace falta un extracto de {fin.date()} "
            "o posterior para saber quién se retiró en esos meses."
        )
    for e in (viejo, nuevo):
        if (posteriores := int((_retiro(e.datos) > e.fecha).sum())) > 0:
            problemas.append(f"{e.nombre}: {posteriores} fecha(s) de retiro posteriores a la fecha del extracto.")
    try:
        columna = llave(viejo, nuevo)
    except ValueError as error:
        return [*problemas, str(error)]
    base = activos(viejo)
    if base[columna].duplicated().any() or nuevo.datos[columna].duplicated().any():
        problemas.append(f"{columna} repetido en algún extracto: no se puede cruzar fila a fila.")
    cobertura = float(base[columna].isin(nuevo.datos[columna]).mean()) if len(base) else 0.0
    if cobertura < COBERTURA_MIN:
        problemas.append(
            f"Solo el {cobertura:.0%} de los activos de {viejo.nombre} aparece en {nuevo.nombre} (mínimo "
            f"{COBERTURA_MIN:.0%}): ¿la misma ANON_SALT? ¿el nuevo trae también a los retirados?"
        )
    retirados_antes = viejo.datos.loc[_retiro(viejo.datos).notna(), columna]
    reactivados = nuevo.datos[nuevo.datos[columna].isin(retirados_antes) & _retiro(nuevo.datos).isna()]
    if len(reactivados):
        problemas.append(
            f"{len(reactivados)} asociado(s) retirados en {viejo.nombre} aparecen activos en {nuevo.nombre}."
        )
    return problemas


def foto_entre_extractos(viejo: Extracto, nuevo: Extracto, horizonte_meses: int = a.HORIZONTE_MESES) -> pd.DataFrame:
    """La foto de los activos de ``viejo`` en su fecha, con el objetivo tomado de ``nuevo`` (ver el módulo).

    ``ValueError`` si ``revisar_par`` encuentra problemas. Quien está en ``viejo`` y no en ``nuevo`` sale (no se
    sabe qué le pasó); su cantidad queda en ``foto.attrs["sin_seguimiento"]``.
    """
    if problemas := revisar_par(viejo, nuevo, horizonte_meses):
        raise ValueError("No se puede armar la foto entre extractos:\n  - " + "\n  - ".join(problemas))
    columna = llave(viejo, nuevo)
    base = activos(viejo)
    retiro_nuevo = _retiro(nuevo.datos).set_axis(nuevo.datos[columna])
    seguidos = base[columna].isin(retiro_nuevo.index)
    vista = base[seguidos].copy()
    fin = viejo.fecha + pd.DateOffset(months=horizonte_meses)
    retiro = vista[columna].map(retiro_nuevo)
    vista[a.TARGET] = ((retiro > viejo.fecha) & (retiro <= fin)).astype(int).to_numpy()
    vista[a.FECHA_OBSERVACION] = viejo.fecha
    vista.attrs["sin_seguimiento"] = int((~seguidos).sum())
    return vista
