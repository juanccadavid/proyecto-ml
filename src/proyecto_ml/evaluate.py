"""Métricas para clasificación binaria desbalanceada."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler

# Columnas de la tabla comparativa, con los mismos nombres que compare_models de PyCaret.
TIEMPO = "TT (Sec)"
SCORING = {
    "Accuracy": "accuracy",
    "AUC": "roc_auc",  # usa predict_proba o, si no hay (Ridge), decision_function
    "Recall": make_scorer(recall_score, zero_division=0),
    "Prec.": make_scorer(precision_score, zero_division=0),
    "F1": make_scorer(f1_score, zero_division=0),
    "Kappa": make_scorer(cohen_kappa_score),
    "MCC": make_scorer(matthews_corrcoef),
}
COLUMNAS_COMPARATIVA = [*SCORING, TIEMPO]


def recall_at_k(y_true, y_score, k_frac: float = 0.1) -> float:
    """Proporción de positivos capturados en el top ``k_frac`` de puntajes.

    Responde: "si el negocio solo puede gestionar al 10 % con mayor riesgo,
    ¿qué fracción de los casos reales encuentra?".
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    positives = y_true.sum()
    if positives == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(y_score) * k_frac)))
    top_idx = np.argsort(-y_score, kind="stable")[:k]
    return float(y_true[top_idx].sum() / positives)


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    return {
        "base_rate": float(y_true.mean()),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_score)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall_at_10pct": recall_at_k(y_true, y_score, 0.10),
    }


def tabla_comparativa(modelos: dict, X, y, folds: int = 5, ordenar: str = "AUC", random_state: int = 0) -> pd.DataFrame:
    """Tabla al estilo de ``compare_models`` de PyCaret: promedio de la validación cruzada por modelo.

    Todos los modelos se evalúan con las mismas particiones. ``TT (Sec)`` es el tiempo promedio de
    entrenamiento por partición (depende del equipo: sirve para comparar, no como valor absoluto).
    Las métricas usan ``predict``, es decir, el umbral propio de cada modelo.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    filas = {}
    for nombre, modelo in modelos.items():
        r = cross_validate(modelo, X, y, cv=cv, scoring=SCORING)  # clona: no toca `modelo`
        final = modelo.steps[-1][1] if hasattr(modelo, "steps") else modelo  # el estimador, no el Pipeline
        filas[nombre] = {
            "Model": type(final).__name__,
            **{m: float(r[f"test_{m}"].mean()) for m in SCORING},
            TIEMPO: float(r["fit_time"].mean()),
        }
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    return tabla.sort_values(ordenar, ascending=ordenar == TIEMPO)  # tiempo: el más rápido arriba


def resaltar(tabla: pd.DataFrame) -> Styler:
    """Como PyCaret en un notebook: el mejor de cada métrica en amarillo (en tiempo, el más rápido)."""
    amarillo = "background-color: #ffff00; color: black"
    metricas = [c for c in tabla.columns if c in SCORING]
    return (
        tabla.style.format(precision=4, subset=[*metricas, TIEMPO])
        .highlight_max(subset=metricas, props=amarillo)
        .highlight_min(subset=[TIEMPO], props=amarillo)
    )
