"""Validación más allá de los "nuevos" del generador: escenarios menos favorables y partición temporal.

Las métricas sobre asociados nuevos generados miden si el modelo recupera la regla del generador, no si
funciona en el fondo real (validación circular). Mientras llega el extracto, dos cosas la hacen defendible:

- **Escenarios** (``evaluar_escenarios``): validar con asociados nuevos con otra tasa de retiro y una señal
  más débil (``GeneradorAsociados(fuerza_senal=...)``). En vez de una cifra perfecta, muestra cuánto se
  degrada el modelo si el fondo real no se parece al generador.
- **Partición temporal** (``particion_temporal``): con el extracto real, entrenar con una foto vieja y
  probar con la más reciente. Es la validación honesta; aquí queda lista y probada con datos generados.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from proyecto_ml import abandono as a
from proyecto_ml.evaluate import compute_metrics
from proyecto_ml.generador import GeneradorAsociados

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Escenario:
    nombre: str
    tasa_retiro: float
    fuerza_senal: float


# La tasa del generador (30 %) infla la precisión; un fondo real está más cerca del 10 %. La señal al
# 50 % y al 25 % simula que los retirados reales se diferencian menos que en el generador.
ESCENARIOS = (
    Escenario("tasa 30 %, señal 100 % (generador)", 0.30, 1.0),
    Escenario("tasa 10 %, señal 100 %", 0.10, 1.0),
    Escenario("tasa 10 %, señal 50 %", 0.10, 0.5),
    Escenario("tasa 10 %, señal 25 %", 0.10, 0.25),
)
METRICAS_ESCENARIO = ["base_rate", "roc_auc", "pr_auc", "precision", "recall"]


def datos_escenario(escenario: Escenario, n: int = 1000, seed: int = 2026, horizonte: int | None = None):
    """Asociados nuevos del escenario. Con ``horizonte``, su foto (como el entrenamiento)."""
    df = GeneradorAsociados(n, seed=seed, tasa_retiro=escenario.tasa_retiro, fuerza_senal=escenario.fuerza_senal)
    df = df.generar()
    return a.foto(df, horizonte) if horizonte else df


def evaluar_escenarios(
    modelos: dict[str, Pipeline],
    umbrales: dict[str, float] | None = None,
    escenarios=ESCENARIOS,
    n: int = 1000,
    seed: int = 2026,
    horizonte: int | None = None,
) -> pd.DataFrame:
    """Métricas de cada modelo ya entrenado en cada escenario: una fila por (escenario, modelo).

    ``umbrales`` da el umbral de cada modelo (el del campeón, por ejemplo); si falta, el de siempre
    (``a.UMBRAL``: 0,5 en probabilidad, 0 en Ridge). Los mismos asociados para todos los modelos.
    """
    umbrales = umbrales or {}
    filas = []
    for escenario in escenarios:
        df = datos_escenario(escenario, n, seed, horizonte)
        X, y = a.separar(df)
        for nombre, modelo in modelos.items():
            umbral = umbrales.get(nombre, a.UMBRAL[a.metodo_puntaje(modelo)])
            m = compute_metrics(y, a.puntaje(modelo, X), threshold=umbral)
            filas.append({"escenario": escenario.nombre, "modelo": nombre, **{k: m[k] for k in METRICAS_ESCENARIO}})
    return pd.DataFrame(filas)


def tabla_escenarios(resultados: pd.DataFrame, metricas=("roc_auc", "precision", "recall")) -> pd.DataFrame:
    """Como la tabla de la presentación: filas = escenario, columnas = (modelo, métrica)."""
    orden = list(dict.fromkeys(resultados["escenario"]))
    tabla = resultados.pivot_table(index="escenario", columns="modelo", values=list(metricas)).reindex(orden)
    modelos = list(dict.fromkeys(resultados["modelo"]))
    return tabla.swaplevel(axis=1).reindex(columns=pd.MultiIndex.from_product([modelos, list(metricas)]))


# ---------------------------------------------------------------- partición temporal (extracto real)


def particion_temporal(
    df: pd.DataFrame,
    horizonte_meses: int = a.HORIZONTE_MESES,
    meses_prueba: int = 3,
    corte: pd.Timestamp = a.FECHA_CORTE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrenamiento = foto vieja, prueba = foto más reciente. Las etiquetas de entrenamiento son anteriores.

    - Prueba: ``a.foto(df, H, corte)``: activos en corte − H, ¿se retiran antes del corte?
    - Entrenamiento: la misma foto ``meses_prueba`` antes: activos en corte − meses_prueba − H,
      ¿se retiran antes de corte − meses_prueba? Quien se retira después cuenta como activo en esa foto.

    Un asociado puede estar en las dos fotos (en fechas distintas): es lo normal en una validación
    temporal. Ojo, como en ``a.foto``: si los conteos de solicitudes son a la fecha de extracción y no a
    cada t0, la foto vieja ve comportamiento posterior (fuga). Hay que confirmarlo con el fondo.
    """
    if meses_prueba < 1:
        raise ValueError("meses_prueba debe ser al menos 1")
    corte = pd.Timestamp(corte)
    entrenamiento = a.foto(df, horizonte_meses, corte - pd.DateOffset(months=meses_prueba))
    prueba = a.foto(df, horizonte_meses, corte)
    return entrenamiento, prueba
