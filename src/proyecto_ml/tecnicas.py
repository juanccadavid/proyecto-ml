"""Comparación de técnicas de preprocesamiento y feature engineering en el modelo de abandono (make tecnicas).

Cada técnica cambia UNA cosa respecto a la base (el preprocesador de los supervisados + el clasificador con
class_weight) y se mide con PR-AUC y ROC-AUC en validación cruzada estratificada repetida. Todas usan las
mismas particiones, así que la diferencia con la base es lo que aporta (o quita) esa técnica.

Todo se aprende dentro de cada partición de entrenamiento (imputadores, escaladores, TargetEncoder, PCA, RFE,
y SMOTE/ADASYN, que solo re-muestrean en ``fit``): nada se filtra de las particiones de validación.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from proyecto_ml import abandono as a
from proyecto_ml.models import get_models

if TYPE_CHECKING:
    import plotly.graph_objects as go

CLASIFICADORES = ("logreg", "random_forest", "hist_gb")
# Para que comparar imputadores tenga sentido: vacíos al azar en estas columnas (los datos generados no traen).
COLUMNAS_VACIOS = ["SALARIO", "ESTRATO", "EDAD", "ANTIGUEDAD", "#CANTIDAD_SERVICIOS"]
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


@dataclass(frozen=True)
class Tecnica:
    """Una variante: ``preproceso`` cambia parámetros del preprocesador de ``abandono.armar_pipeline``."""

    grupo: str
    nombre: str
    descripcion: str
    preproceso: dict = field(default_factory=dict)
    muestreo: str | None = None  # "smote" o "adasyn"
    balance: bool = True  # class_weight del clasificador
    crear_variables: bool = False


TECNICAS = [
    Tecnica("Base", "base", "Mediana, OneHot, StandardScaler, recorte IQR y class_weight='balanced'"),
    Tecnica("Desbalance", "sin balancear", "Sin class_weight ni re-muestreo", balance=False),
    Tecnica("Desbalance", "SMOTE", "Retirados sintéticos entre vecinos de la minoría", muestreo="smote", balance=False),
    Tecnica("Desbalance", "ADASYN", "Como SMOTE, más donde la minoría es difícil", muestreo="adasyn", balance=False),
    Tecnica("Imputación", "media", "SimpleImputer(mean)", {"numeric_strategy": "mean"}),
    Tecnica("Imputación", "KNN", "KNNImputer: promedio de 5 vecinos", {"numeric_strategy": "knn"}),
    Tecnica(
        "Imputación", "iterativa", "IterativeImputer: cada columna con las demás", {"numeric_strategy": "iterative"}
    ),
    Tecnica("Imputación", "mediana + indicador", "SimpleImputer + MissingIndicator", {"missing_indicator": True}),
    Tecnica("Codificación", "ordinal", "OrdinalEncoder: un número por categoría", {"encoding": "ordinal"}),
    Tecnica("Codificación", "target", "TargetEncoder: tasa de retiro por categoría", {"encoding": "target"}),
    Tecnica("Escalado", "min-max", "MinMaxScaler: a [0, 1]", {"scaler": "minmax"}),
    Tecnica("Escalado", "max-abs", "MaxAbsScaler: a [-1, 1] sin centrar", {"scaler": "maxabs"}),
    Tecnica("Escalado", "robusto", "RobustScaler: mediana e IQR", {"scaler": "robust"}),
    Tecnica("Escalado", "normalizer", "Normalizer: cada fila a norma 1", {"scaler": "normalizer"}),
    Tecnica(
        "Feature engineering",
        "variables nuevas",
        "Agregaciones, interacción y log del salario (CrearVariables)",
        crear_variables=True,
    ),
    Tecnica(
        "Feature engineering",
        "binning",
        "EDAD, ANTIGUEDAD y SALARIO en 5 rangos (KBinsDiscretizer)",
        {"bins": 5, "bin_columns": a.COLUMNAS_RECORTE},
    ),
    Tecnica("Extracción", "PCA 95 %", "Componentes que guardan el 95 % de la varianza", {"pca": 0.95}),
    Tecnica("Selección", "RFE 15", "RFE: las 15 columnas más útiles para una logística", {"rfe_features": 15}),
]


def con_vacios(
    X: pd.DataFrame, fraccion: float = 0.1, columnas: list[str] = COLUMNAS_VACIOS, seed: int = 0
) -> pd.DataFrame:
    """Copia de ``X`` con ``fraccion`` de vacíos al azar (MCAR) en ``columnas``."""
    X = X.copy()
    rng = np.random.default_rng(seed)
    for c in columnas:
        X.loc[rng.random(len(X)) < fraccion, c] = np.nan
    return X


def clasificador(nombre: str = "logreg", balance: bool = True, random_state: int = 42):
    """Un modelo de ``models.get_models``; sin ``balance``, sin class_weight (lo compensa el re-muestreo)."""
    if nombre not in CLASIFICADORES:
        raise ValueError(f"clasificador debe ser uno de {CLASIFICADORES}, no '{nombre}'")
    modelo = get_models(random_state)[nombre]
    return modelo if balance else modelo.set_params(class_weight=None)


def modelo(tecnica: Tecnica, nombre_clasificador: str = "logreg", random_state: int = 42):
    """El pipeline de una técnica (``abandono.armar_pipeline``): preparar -> [crear variables] -> preprocesar
    -> [SMOTE/ADASYN] -> clasificador."""
    return a.armar_pipeline(
        clasificador(nombre_clasificador, tecnica.balance, random_state),
        crear_variables=tecnica.crear_variables,
        muestreo=tecnica.muestreo,
        random_state=random_state,
        **tecnica.preproceso,
    )


def comparar(
    X: pd.DataFrame,
    y: pd.Series,
    tecnicas: list[Tecnica] = TECNICAS,
    nombre_clasificador: str = "logreg",
    folds: int = 5,
    repeticiones: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Una fila por técnica: PR-AUC y ROC-AUC (media y desviación entre particiones), diferencia de PR-AUC
    con la base y segundos de entrenamiento por partición."""
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeticiones, random_state=random_state)
    filas = []
    for t in tecnicas:
        inicio = time.perf_counter()
        r = cross_validate(modelo(t, nombre_clasificador, random_state), X, y, cv=cv, scoring=SCORING)
        filas.append(
            {
                "grupo": t.grupo,
                "tecnica": t.nombre,
                "descripcion": t.descripcion,
                "pr_auc": r["test_pr_auc"].mean(),
                "pr_auc_std": r["test_pr_auc"].std(),
                "roc_auc": r["test_roc_auc"].mean(),
                "roc_auc_std": r["test_roc_auc"].std(),
                "segundos": (time.perf_counter() - inicio) / (folds * repeticiones),
            }
        )
    tabla = pd.DataFrame(filas)
    base = tabla.loc[tabla["tecnica"] == "base", "pr_auc"]
    tabla["dif_pr_auc"] = tabla["pr_auc"] - (base.iloc[0] if len(base) else np.nan)
    return tabla


def figura(tabla: pd.DataFrame, titulo: str = "PR-AUC por técnica (validación cruzada)") -> go.Figure:
    """Un punto por técnica con su desviación, por grupo; la línea punteada es la base."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    grupos = list(dict.fromkeys(tabla["grupo"]))
    alturas = [max(1, (tabla["grupo"] == g).sum()) for g in grupos]
    fig = make_subplots(
        rows=len(grupos), cols=1, shared_xaxes=True, row_heights=alturas, subplot_titles=grupos, vertical_spacing=0.04
    )
    base = tabla.loc[tabla["tecnica"] == "base", "pr_auc"]
    for i, g in enumerate(grupos, start=1):
        parte = tabla[tabla["grupo"] == g]
        fig.add_trace(
            go.Scatter(
                x=parte["pr_auc"],
                y=parte["tecnica"],
                mode="markers",
                marker={"size": 10, "color": "#2a78d6", "line": {"width": 2, "color": "#fcfcfb"}},
                error_x={"type": "data", "array": parte["pr_auc_std"], "thickness": 2, "width": 0, "color": "#86b6ef"},
                customdata=np.stack([parte["descripcion"], parte["roc_auc"], parte["dif_pr_auc"]], axis=-1),
                hovertemplate=(
                    "<b>%{y}</b><br>%{customdata[0]}<br>PR-AUC %{x:.3f} (dif. con la base %{customdata[2]:+.3f})"
                    "<br>ROC-AUC %{customdata[1]:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=i,
            col=1,
        )
        if len(base):
            fig.add_vline(x=base.iloc[0], line={"dash": "dot", "width": 1, "color": "#898781"}, row=i, col=1)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="PR-AUC (la línea punteada es la base)", row=len(grupos), col=1)
    fig.update_layout(
        title=titulo, template="plotly_white", height=110 + 34 * len(tabla) + 40 * len(grupos), margin={"l": 40}
    )
    return fig
