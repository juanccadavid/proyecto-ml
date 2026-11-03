"""Regresión: estimar el SALARIO del asociado con sus demás variables (MAE, MSE, RMSE y R²).

El abandono es una clasificación (se retira sí/no); esta es la contraparte de regresión, con un uso real: el
5 % de los asociados no reporta salario (nómina no reportada) y el modelo da una estimación para ellos. Se
entrena solo con quienes sí lo reportan y se valida con **KFold** (en regresión no hay clases que estratificar).

Las métricas, en pesos:

- **MAE** (error absoluto medio): cuánto se equivoca en promedio. Se lee directo: "±X pesos".
- **MSE** (error cuadrático medio): castiga más los errores grandes. Está en pesos², no se lee directo.
- **RMSE** (raíz del MSE): vuelve a pesos; si es mucho mayor que el MAE, hay pocos errores muy grandes.
- **R²**: fracción de la variación del salario que explica el modelo. 0 = igual que predecir el promedio
  (``DummyRegressor``, la referencia), 1 = perfecto, negativo = peor que el promedio.

Las variables son las del abandono (``abandono.preparar``) sin SALARIO y sin TIPOPACTO: el pacto integral se
define por el salario (>= 10 SMMLV), así que delataría el objetivo (fuga).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from proyecto_ml import abandono as a
from proyecto_ml.features import build_preprocessor

OBJETIVO = "SALARIO"
FUGA = ["TIPOPACTO"]
NUMERICAS = [c for c in a.NUMERICAS if c != OBJETIVO]
CATEGORICAS = [c for c in a.CATEGORICAS if c not in FUGA]
RECORTE = [c for c in a.COLUMNAS_RECORTE if c != OBJETIVO]
ERRORES = ["MAE", "MSE", "RMSE"]
SCORING = {
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}
TITULOS = {
    "media": "Promedio (referencia)",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "hist_gb": "HistGradientBoosting",
}


def modelos(random_state: int = 42) -> dict[str, Pipeline]:
    """La referencia (predice siempre el promedio) y tres regresores, con el preprocesamiento del abandono."""
    estimadores = {
        "media": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=random_state
        ),
        "hist_gb": HistGradientBoostingRegressor(random_state=random_state),
    }
    # PrepararVariables entrega también SALARIO y TIPOPACTO; el ColumnTransformer los descarta (remainder="drop").
    return {
        nombre: Pipeline(
            [
                ("preparar", a.PrepararVariables()),
                ("preprocess", build_preprocessor(NUMERICAS, CATEGORICAS, clip_outliers="iqr", clip_columns=RECORTE)),
                ("model", estimador),
            ]
        )
        for nombre, estimador in estimadores.items()
    }


def separar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X (filas crudas, sin el abandono) e y (SALARIO), solo de quienes reportan salario."""
    con_salario = df[pd.to_numeric(df[OBJETIVO], errors="coerce").notna()]
    return con_salario.drop(columns=[a.TARGET], errors="ignore"), pd.to_numeric(con_salario[OBJETIVO]).astype(float)


def metricas(y, prediccion) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y, prediccion)),
        "MSE": float(mean_squared_error(y, prediccion)),
        "RMSE": float(root_mean_squared_error(y, prediccion)),
        "R2": float(r2_score(y, prediccion)),
    }


def comparar(candidatos: dict[str, Pipeline], X, y, folds: int = 5, random_state: int = 0) -> pd.DataFrame:
    """Validación cruzada KFold: promedio y desviación de cada métrica, mejor R² arriba."""
    cv = KFold(folds, shuffle=True, random_state=random_state)
    filas = {}
    for nombre, modelo in candidatos.items():
        r = cross_validate(modelo, X, y, cv=cv, scoring=SCORING)
        promedio = {m: float(r[f"test_{m}"].mean()) for m in SCORING}
        filas[nombre] = {
            **{m: -promedio[m] for m in ERRORES},  # sklearn maximiza: los errores llegan con signo negativo
            "R2": promedio["R2"],
            "R2_std": float(r["test_R2"].std()),
        }
    return pd.DataFrame.from_dict(filas, orient="index").sort_values("R2", ascending=False)


def prediccion_fuera_de_particion(modelo: Pipeline, X, y, folds: int = 5, random_state: int = 0) -> np.ndarray:
    """La predicción de cada asociado hecha por un modelo que no lo vio (para la figura real vs. predicho)."""
    return cross_val_predict(clone(modelo), X, y, cv=KFold(folds, shuffle=True, random_state=random_state))


def figura_real_vs_predicho(y, prediccion, titulo: str):
    """Plotly: salario real vs. predicho; sobre la diagonal, predicción perfecta."""
    import plotly.graph_objects as go

    y, prediccion = np.asarray(y), np.asarray(prediccion)
    bajo, alto = float(min(y.min(), prediccion.min())), float(max(y.max(), prediccion.max()))
    fig = go.Figure()
    fig.add_scatter(
        x=[bajo, alto],
        y=[bajo, alto],
        mode="lines",
        name="predicción perfecta",
        line={"color": "#898781", "width": 1, "dash": "dot"},
    )
    fig.add_scatter(
        x=y,
        y=prediccion,
        mode="markers",
        name="asociado",
        marker={"size": 8, "color": "#2a78d6", "opacity": 0.5, "line": {"color": "#fcfcfb", "width": 1}},
        hovertemplate="real=$%{x:,.0f}<br>predicho=$%{y:,.0f}<extra></extra>",
    )
    fig.update_layout(
        title=titulo,
        xaxis={"title": "Salario real", "tickformat": "$,.0f", "gridcolor": "#e1e0d9"},
        yaxis={"title": "Salario predicho (fuera de partición)", "tickformat": "$,.0f", "gridcolor": "#e1e0d9"},
        template="plotly_white",
        height=560,
        legend={"orientation": "h", "y": -0.18},
    )
    return fig
