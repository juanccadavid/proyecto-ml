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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

if TYPE_CHECKING:
    from matplotlib.figure import Figure
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


# ---------------------------------------------------------------- búsqueda de hiperparámetros


def afinar(modelo, grilla: dict | list[dict], X, y, folds: int = 5, metrica: str = "AUC", random_state: int = 0):
    """Búsqueda en grilla (GridSearchCV) con las mismas particiones y métricas que ``tabla_comparativa``.

    Prueba todas las combinaciones de ``grilla`` (nombres ``paso__parametro`` del Pipeline), elige la de
    mejor ``metrica`` promedio y la reentrena con todo ``X`` (``best_estimator_``). Ojo: ``best_score_``
    es optimista, porque se eligió mirando esas mismas particiones; la prueba honesta son datos nuevos.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    return GridSearchCV(modelo, grilla, scoring=SCORING, refit=metrica, cv=cv).fit(X, y)


def _sin_paso(params: dict) -> dict:
    """{"model__alpha": 0.1} -> {"alpha": 0.1}: el nombre del paso del Pipeline no aporta al leer."""
    return {k.split("__")[-1]: v for k, v in params.items()}


def formato_params(params: dict) -> str:
    return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in _sin_paso(params).items())


def resultados_busqueda(busqueda, top: int = 5) -> pd.DataFrame:
    """Las ``top`` combinaciones de una búsqueda, por su métrica, con las columnas de la tabla comparativa."""
    r = busqueda.cv_results_
    tabla = pd.DataFrame(
        {
            "params": [formato_params(p) for p in r["params"]],
            **{m: r[f"mean_test_{m}"] for m in SCORING},
            TIEMPO: r["mean_fit_time"],
        }
    )
    return tabla.sort_values(busqueda.refit, ascending=False, kind="stable").head(top).reset_index(drop=True)


_LEGIBLE = {"l1_ratio=0.0": "L2", "l1_ratio=1.0": "L1"}  # nombres de la leyenda


def figura_busqueda(busquedas: dict, eje_x: dict[str, str]) -> Figure:
    """Un panel por búsqueda: métrica de validación cruzada según el hiperparámetro ``eje_x[nombre]``.

    Una curva por combinación de los demás hiperparámetros: el color indica ``class_weight`` y el
    tipo de línea el resto (por ejemplo L2 / L1). El punto elegido va con un anillo y la línea punteada
    gris es el valor por defecto (1, tanto para alpha como para C).
    """
    from matplotlib.figure import Figure  # sin pyplot: no abre ventanas ni depende del backend

    colores = {"None": "#2a78d6", "balanced": "#eb6834"}  # paleta de referencia: azul y naranja
    tinta, gris = "#1f2937", "#9ca3af"
    fig = Figure(figsize=(6 * len(busquedas), 4.4), layout="constrained")
    for ax, (titulo, b) in zip(fig.subplots(1, len(busquedas), squeeze=False).ravel(), busquedas.items(), strict=True):
        x = eje_x[titulo]
        r = pd.DataFrame([_sin_paso(p) for p in b.cv_results_["params"]], dtype=object).astype({x: float})
        y = b.cv_results_[f"mean_test_{b.refit}"]
        # dtype=object y map(str): en pandas 3, None se volvería NaN o vacío y groupby lo descartaría.
        texto = r.drop(columns=x).map(str)
        otros = [c for c in texto.columns if (texto[c] != texto[c].iloc[0]).any()]  # sin los constantes (solver)
        estilos: dict[tuple, str] = {}
        for clave, filas in texto.groupby(otros, sort=False).groups.items():
            valores = dict(zip(otros, clave if isinstance(clave, tuple) else (clave,), strict=True))
            etiqueta = {k: _LEGIBLE.get(f"{k}={v}", f"{k}={v}") for k, v in valores.items()}
            resto = tuple(e for k, e in etiqueta.items() if k != "class_weight")
            estilo = estilos.setdefault(resto, ["-", "--", ":", "-."][len(estilos) % 4])
            ax.plot(
                r.loc[filas, x],
                y[filas],
                estilo,
                color=colores.get(valores.get("class_weight", "None"), gris),
                lw=2,
                marker="o",
                ms=6,
                label=", ".join(etiqueta.values()),
            )
        mejor = _sin_paso(b.best_params_)
        ax.plot(mejor[x], b.best_score_, "o", ms=14, mfc="none", mec=tinta, mew=2)  # anillo en el elegido
        ax.axvline(1, color=gris, lw=1, ls=":", label="por defecto")
        ax.set_xscale("log")
        ax.set_title(f"{titulo} · mejor: {formato_params({x: float(mejor[x])})} ({b.refit} = {b.best_score_:.3f})")
        ax.set_xlabel(f"{x} (escala log; más alto = {'menos' if x == 'C' else 'más'} regularización)")
        ax.set_ylabel(f"{b.refit} promedio (validación cruzada)")
        ax.grid(alpha=0.3, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="best")
    return fig
