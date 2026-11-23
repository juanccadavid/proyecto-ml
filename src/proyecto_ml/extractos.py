"""Compatibilidad para un flujo temporal que no está disponible con el esquema definitivo.

El fondo entregará un solo corte y el contrato v2 no incluye fecha de retiro. Por tanto no es posible construir
etiquetas entre extractos sin inventar información. Las funciones conservan una interfaz explícita para que un
llamador antiguo reciba un error claro.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

LLAVES = ("NROIDENTIFICACION",)


@dataclass(frozen=True)
class Extracto:
    datos: pd.DataFrame
    fecha: pd.Timestamp
    nombre: str = "extracto"

    @classmethod
    def de(cls, datos: pd.DataFrame, fecha, nombre: str = "extracto") -> Extracto:
        return cls(datos=datos, fecha=pd.Timestamp(fecha), nombre=nombre)


def foto_entre_extractos(viejo: Extracto, nuevo: Extracto, horizonte_meses: int = 3) -> pd.DataFrame:
    del viejo, nuevo, horizonte_meses
    raise ValueError(
        "El contrato v2 no permite construir una foto entre extractos: solo hay un corte y no existe fecha de retiro."
    )
