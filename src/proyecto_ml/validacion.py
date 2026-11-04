"""Validación más allá de los "nuevos" del generador: escenarios menos favorables y partición temporal.

Las métricas sobre asociados nuevos generados miden si el modelo recupera la regla del generador, no si
funciona en el fondo real (validación circular). Mientras llega el extracto, dos cosas la hacen defendible:

- **Escenarios** (``evaluar_escenarios``): validar con asociados nuevos con otra tasa de retiro y una señal
  más débil (``GeneradorAsociados(fuerza_senal=...)``). En vez de una cifra perfecta, muestra cuánto se
  degrada el modelo si el fondo real no se parece al generador.
- **Partición temporal** (``particion_temporal``): con el extracto real, entrenar con una foto vieja y
  probar con la más reciente. Es la validación honesta; aquí queda lista y probada con datos generados.
- **Estrategias de validación cruzada** (``comparar_validaciones``): KFold, StratifiedKFold y TimeSeriesSplit
  sobre los mismos datos, para ver por qué el campeón usa la estratificada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

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
            m = compute_metrics(y, a.puntaje(modelo, X), threshold=umbral, metodo=a.metodo_puntaje(modelo))
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


# ---------------------------------------------------------------- estrategias de validación cruzada

# TimeSeriesSplit necesita las filas en orden de tiempo. La foto no tiene una fecha por fila (todos se observan
# en t0), así que se ordena por fecha de ingreso: entrena con los asociados más antiguos y prueba con los
# más nuevos (cohortes). Con el extracto real, la validación temporal de verdad es ``particion_temporal``.
ORDEN_TEMPORAL = "FECHAINGRESOLABORA"


def estrategias_cv(folds: int = 5, random_state: int = 0) -> dict:
    """Las tres formas de partir: KFold (al azar), StratifiedKFold (misma tasa de retiro en cada partición)
    y TimeSeriesSplit (cada partición prueba con filas posteriores a todas las de su entrenamiento)."""
    return {
        "KFold": KFold(folds, shuffle=True, random_state=random_state),
        "StratifiedKFold": StratifiedKFold(folds, shuffle=True, random_state=random_state),
        "TimeSeriesSplit": TimeSeriesSplit(folds),
    }


def _puntuar(y, puntaje, metrica) -> float:
    return float(metrica(y, puntaje)) if 0 < y.sum() < len(y) else float("nan")  # sin retirados no se define


def comparar_validaciones(
    modelo: Pipeline, df: pd.DataFrame, folds: int = 5, random_state: int = 0, orden: str = ORDEN_TEMPORAL
) -> pd.DataFrame:
    """Una fila por estrategia y partición: tamaños, retirados en la prueba, PR-AUC y ROC-AUC.

    ``df`` son filas crudas con el objetivo (p. ej. ``a.foto(...)``); se ordenan por ``orden`` para
    TimeSeriesSplit (KFold y StratifiedKFold barajan, así que el orden no les cambia nada).
    """
    df = df.sort_values(orden, kind="stable").reset_index(drop=True)
    X, y = a.separar(df)
    filas = []
    for estrategia, cv in estrategias_cv(folds, random_state).items():
        for k, (entrena, prueba) in enumerate(cv.split(X, y), start=1):
            y_entrena, y_prueba = y.iloc[entrena], y.iloc[prueba]
            fila = {
                "estrategia": estrategia,
                "particion": k,
                "filas_entrenamiento": len(entrena),
                "filas_prueba": len(prueba),
                "retirados_prueba": int(y_prueba.sum()),
                "tasa_prueba": float(y_prueba.mean()),
                "pr_auc": float("nan"),
                "roc_auc": float("nan"),
            }
            if y_entrena.nunique() == 2:  # sin las dos clases en el entrenamiento no hay modelo
                puntaje = a.puntaje(clone(modelo).fit(X.iloc[entrena], y_entrena), X.iloc[prueba])
                fila["pr_auc"] = _puntuar(y_prueba, puntaje, average_precision_score)
                fila["roc_auc"] = _puntuar(y_prueba, puntaje, roc_auc_score)
            filas.append(fila)
    return pd.DataFrame(filas)


def resumen_validaciones(por_particion: pd.DataFrame) -> pd.DataFrame:
    """Por estrategia: PR-AUC media, su desviación entre particiones y cuántos retirados tuvo cada prueba."""
    g = por_particion.groupby("estrategia", sort=False)
    return pd.DataFrame(
        {
            "pr_auc": g["pr_auc"].mean(),
            "pr_auc_std": g["pr_auc"].std(),
            "roc_auc": g["roc_auc"].mean(),
            "retirados_min": g["retirados_prueba"].min(),
            "retirados_max": g["retirados_prueba"].max(),
            "tasa_prueba_std": g["tasa_prueba"].std(),
            "entrenamiento_medio": g["filas_entrenamiento"].mean().round().astype(int),
        }
    )


def figura_validaciones(por_particion: pd.DataFrame, titulo: str):
    """Plotly: la PR-AUC de cada partición (puntos) y su promedio (raya), una fila por estrategia."""
    import plotly.graph_objects as go

    colores = {"KFold": "#2a78d6", "StratifiedKFold": "#eb6834", "TimeSeriesSplit": "#1baf7a"}
    fig = go.Figure()
    for estrategia, datos in por_particion.groupby("estrategia", sort=False):
        color = colores.get(estrategia, "#898781")
        fig.add_scatter(
            x=datos["pr_auc"],
            y=[estrategia] * len(datos),
            mode="markers",
            name=estrategia,
            marker={"size": 11, "color": color, "line": {"color": "#fcfcfb", "width": 2}},
            customdata=np.column_stack([datos["particion"], datos["retirados_prueba"], datos["filas_prueba"]]),
            hovertemplate=f"{estrategia} · partición %{{customdata[0]}}<br>PR-AUC=%{{x:.3f}}"
            "<br>%{customdata[1]} retirados de %{customdata[2]}<extra></extra>",
        )
        fig.add_scatter(
            x=[datos["pr_auc"].mean()],
            y=[estrategia],
            mode="markers",
            showlegend=False,
            marker={"symbol": "line-ns", "size": 26, "line": {"color": "#0b0b0b", "width": 2}},
            hovertemplate=f"{estrategia}: promedio %{{x:.3f}}<extra></extra>",
        )
    fig.update_layout(
        title=titulo,
        xaxis={"title": "PR-AUC de la partición (raya negra = promedio)", "gridcolor": "#e1e0d9"},
        yaxis={"autorange": "reversed"},
        template="plotly_white",
        height=360,
        legend={"orientation": "h", "y": -0.25},
    )
    return fig
