"""Evaluación de correlación de variables: entre ellas (redundancia) y con el objetivo (señal).

- Entre numéricas: Spearman (por rangos), que capta relaciones monótonas aunque no sean rectas y no se deja
  llevar por atípicos como Pearson. Dos variables con |r| alto dicen casi lo mismo: una sobra.
- Con el objetivo (0/1): Spearman para las numéricas y V de Cramér para las categóricas (0 = independiente,
  1 = la categoría determina el objetivo). Ojo: correlación no es causalidad, y una variable con correlación
  baja puede servir combinada con otras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

if TYPE_CHECKING:
    import plotly.graph_objects as go

UMBRAL_REDUNDANCIA = 0.8

# Divergente azul <-> rojo con punto medio gris (paleta de referencia de los reportes).
_ESCALA = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]


def matriz(X: pd.DataFrame, metodo: str = "spearman") -> pd.DataFrame:
    """Correlación entre las columnas numéricas de ``X`` (por pares, ignorando los vacíos)."""
    return X.select_dtypes("number").corr(method=metodo)


def pares_redundantes(m: pd.DataFrame, umbral: float = UMBRAL_REDUNDANCIA) -> pd.DataFrame:
    """Pares de variables con |correlación| >= ``umbral``, del más fuerte al más débil (cada par una vez)."""
    arriba = m.where(np.triu(np.ones(m.shape, dtype=bool), k=1)).rename_axis("variable_a").reset_index()
    pares = arriba.melt(id_vars="variable_a", var_name="variable_b", value_name="correlacion").dropna()
    pares = pares[pares["correlacion"].abs() >= umbral]
    return pares.reindex(pares["correlacion"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def cramer_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér entre dos categóricas (con la corrección de sesgo de Bergsma); 0 si no hay variación."""
    tabla = pd.crosstab(x, y)
    n = tabla.to_numpy().sum()
    if n == 0 or min(tabla.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(tabla, correction=False)[0]
    filas, columnas = tabla.shape
    phi2 = max(0.0, chi2 / n - (filas - 1) * (columnas - 1) / (n - 1))
    filas_c, columnas_c = filas - (filas - 1) ** 2 / (n - 1), columnas - (columnas - 1) ** 2 / (n - 1)
    divisor = min(filas_c - 1, columnas_c - 1)
    return float(np.sqrt(phi2 / divisor)) if divisor > 0 else 0.0


def con_objetivo(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Relación de cada variable con el objetivo, de la más fuerte a la más débil.

    Columnas: ``variable``, ``tipo``, ``medida`` (spearman o cramer_v), ``valor`` (con signo en Spearman:
    + = más alta, más retiro) y ``fuerza`` (|valor|, para ordenar).
    """
    y = pd.Series(np.asarray(y), index=X.index)
    filas = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            valor = X[c].corr(y, method="spearman")
            filas.append({"variable": c, "tipo": "numérica", "medida": "spearman", "valor": valor})
        else:
            valor = cramer_v(X[c].fillna("(vacío)"), y)
            filas.append({"variable": c, "tipo": "categórica", "medida": "cramer_v", "valor": valor})
    tabla = pd.DataFrame(filas)
    tabla["valor"] = tabla["valor"].fillna(0.0)  # una columna constante no se correlaciona
    tabla["fuerza"] = tabla["valor"].abs()
    return tabla.sort_values("fuerza", ascending=False).reset_index(drop=True)


def figura_matriz(m: pd.DataFrame, titulo: str = "Correlación de Spearman entre variables numéricas") -> go.Figure:
    """Mapa de calor de la matriz (-1 rojo, 0 gris, +1 azul); el valor exacto en el tooltip."""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Heatmap(
            z=m.to_numpy(),
            x=list(m.columns),
            y=list(m.index),
            zmin=-1,
            zmax=1,
            colorscale=_ESCALA,
            xgap=2,
            ygap=2,
            colorbar={"title": "r", "thickness": 12},
            hovertemplate="%{y}<br>%{x}<br>r = %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=titulo,
        template="plotly_white",
        height=max(420, 28 * len(m) + 160),
        yaxis={"autorange": "reversed"},
        xaxis={"tickangle": -45},
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )
    return fig
