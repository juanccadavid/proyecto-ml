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


def afinar(
    modelo,
    grilla: dict | list[dict],
    X,
    y,
    folds: int = 5,
    metrica: str = "AUC",
    random_state: int = 0,
    n_jobs: int | None = None,
):
    """Búsqueda en grilla (GridSearchCV) con las mismas particiones y métricas que ``tabla_comparativa``.

    Prueba todas las combinaciones de ``grilla`` (nombres ``paso__parametro`` del Pipeline), elige la de
    mejor ``metrica`` promedio y la reentrena con todo ``X`` (``best_estimator_``). Ojo: ``best_score_``
    es optimista, porque se eligió mirando esas mismas particiones; la prueba honesta son datos nuevos.
    ``n_jobs=-1`` reparte los entrenamientos entre los núcleos (el tiempo TT de cada uno sube un poco).
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    return GridSearchCV(modelo, grilla, scoring=SCORING, refit=metrica, cv=cv, n_jobs=n_jobs).fit(X, y)


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


def _mejor_de_cada_grupo(busqueda, agrupar: list[str], nombre: str) -> pd.Series:
    """Etiqueta del grupo -> posición en ``cv_results_`` de su mejor combinación."""
    r = busqueda.cv_results_
    params = pd.DataFrame([_sin_paso(p) for p in r["params"]], dtype=object)
    etiquetas = params[agrupar].apply(lambda f: "_".join([nombre, *(str(v) for v in f if v is not None)]), axis=1)
    return pd.Series(r[f"mean_test_{busqueda.refit}"]).groupby(etiquetas, sort=False).idxmax()


def mejores_por_grupo(busqueda, agrupar: list[str], nombre: str) -> dict[str, dict]:
    """La mejor combinación de cada grupo de la búsqueda: ``{"Ridge_median_onehot_iqr": params, ...}``.

    Un grupo por cada combinación de los hiperparámetros de ``agrupar`` (p. ej. el preprocesamiento);
    dentro de cada uno, la de mejor métrica de la búsqueda (p. ej. el mejor alpha con ese preprocesamiento).
    El nombre une ``nombre`` y los valores del grupo (los ``None`` se omiten).
    """
    mejores = _mejor_de_cada_grupo(busqueda, agrupar, nombre)
    return {etiqueta: busqueda.cv_results_["params"][i] for etiqueta, i in mejores.items()}


def resultados_por_fold(
    busqueda, agrupar: list[str], nombre: str, metricas: tuple[str, ...] = ("Accuracy",)
) -> pd.DataFrame:
    """Tabla larga ``pipeline | fold | métricas``: el puntaje de cada partición de la validación cruzada.

    Una fila por partición para cada grupo de ``mejores_por_grupo`` (p. ej. cada preprocesamiento con
    su mejor alpha). Las mismas particiones para todos, así que se puede comparar fila a fila.
    """
    r = busqueda.cv_results_
    filas = [
        {"pipeline": pipeline, "fold": k + 1, **{m: float(r[f"split{k}_test_{m}"][i]) for m in metricas}}
        for pipeline, i in _mejor_de_cada_grupo(busqueda, agrupar, nombre).items()
        for k in range(busqueda.n_splits_)
    ]
    return pd.DataFrame(filas)


_LEGIBLE = {"l1_ratio=0.0": "L2", "l1_ratio=1.0": "L1"}  # nombres de la leyenda


def figura_busqueda(busquedas: dict, eje_x: dict[str, str], fijar: tuple[str, ...] = ()) -> Figure:
    """Un panel por búsqueda: métrica de validación cruzada según el hiperparámetro ``eje_x[nombre]``.

    Una curva por combinación de los demás hiperparámetros: el color indica ``class_weight`` y el
    tipo de línea el resto (por ejemplo L2 / L1). El punto elegido va con un anillo y la línea punteada
    gris es el valor por defecto (1, tanto para alpha como para C). Los hiperparámetros de ``fijar``
    (p. ej. el preprocesamiento) se dejan en el valor del mejor, para no llenar el panel de curvas.
    """
    from matplotlib.figure import Figure  # sin pyplot: no abre ventanas ni depende del backend

    colores = {"None": "#2a78d6", "balanced": "#eb6834"}  # paleta de referencia: azul y naranja
    tinta, gris = "#1f2937", "#9ca3af"
    fig = Figure(figsize=(6 * len(busquedas), 4.8 if fijar else 4.4), layout="constrained")
    for ax, (titulo, b) in zip(fig.subplots(1, len(busquedas), squeeze=False).ravel(), busquedas.items(), strict=True):
        x = eje_x[titulo]
        r = pd.DataFrame([_sin_paso(p) for p in b.cv_results_["params"]], dtype=object).astype({x: float})
        y = pd.Series(b.cv_results_[f"mean_test_{b.refit}"], index=r.index)
        mejor = _sin_paso(b.best_params_)
        fijos = {c: mejor[c] for c in fijar if c in r}
        for c, v in fijos.items():  # solo las combinaciones con el valor del mejor
            r = r[r[c].map(str) == str(v)]
        y = y[r.index]
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
        ax.plot(mejor[x], b.best_score_, "o", ms=14, mfc="none", mec=tinta, mew=2)  # anillo en el elegido
        ax.axvline(1, color=gris, lw=1, ls=":", label="por defecto")
        ax.set_xscale("log")
        titulo_panel = f"{titulo} · mejor: {formato_params({x: float(mejor[x])})} ({b.refit} = {b.best_score_:.3f})"
        if fijos:  # solo los valores, para que quepa: "median · ordinal" (None = paso apagado, se omite)
            titulo_panel += "\npreprocesamiento: " + " · ".join(str(v) for v in fijos.values() if v is not None)
        ax.set_title(titulo_panel)
        ax.set_xlabel(f"{x} (escala log; más alto = {'menos' if x == 'C' else 'más'} regularización)")
        ax.set_ylabel(f"{b.refit} promedio (validación cruzada)")
        ax.grid(alpha=0.3, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="best")
    return fig


_COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # paleta de referencia, en orden fijo
_TINTA, _GRIS = "#1f2937", "#9ca3af"


def figura_por_fold(
    por_fold: pd.DataFrame, metricas: list[str], nuevos: pd.DataFrame | None = None, ordenar: str | None = None
) -> Figure:
    """Gráfico de bigotes: la distribución de cada métrica entre las particiones, un pipeline por fila.

    ``por_fold`` es la tabla larga de ``resultados_por_fold``. La caja va del cuartil 1 al 3 con la mediana,
    los bigotes hasta el mínimo y el máximo, y encima cada partición como un punto. Si se pasa ``nuevos``
    (pipeline x métrica), un rombo marca el resultado sobre datos que el modelo no vio. El color es el
    modelo (lo que va antes del primer ``_`` en el nombre del pipeline).
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    ordenar = ordenar or metricas[0]
    orden = por_fold.groupby("pipeline", sort=False)[ordenar].mean().sort_values().index.tolist()  # mejor arriba
    familias = list(dict.fromkeys(p.split("_")[0] for p in orden[::-1]))
    color = {f: _COLORES[i % len(_COLORES)] for i, f in enumerate(familias)}
    rng = np.random.default_rng(0)

    fig = Figure(figsize=(5.2 * len(metricas) + 1.6, 0.36 * len(orden) + 1.6), layout="constrained")
    ejes = fig.subplots(1, len(metricas), sharey=True, squeeze=False).ravel()
    for ax, m in zip(ejes, metricas, strict=True):
        datos = [por_fold.loc[por_fold["pipeline"] == p, m].to_numpy() for p in orden]
        cajas = ax.boxplot(datos, orientation="horizontal", whis=(0, 100), widths=0.55, patch_artist=True)
        for i, p in enumerate(orden):
            c = color[p.split("_")[0]]
            cajas["boxes"][i].set(facecolor=c + "33", edgecolor=c, linewidth=1.5)
            cajas["medians"][i].set(color=c, linewidth=2)
            for linea in (*cajas["whiskers"][2 * i : 2 * i + 2], *cajas["caps"][2 * i : 2 * i + 2]):
                linea.set(color=c, linewidth=1.2)
            jitter = rng.uniform(-0.15, 0.15, len(datos[i]))
            ax.plot(datos[i], i + 1 + jitter, "o", ms=4, color=c, alpha=0.8, mec="white", mew=0.5)
            if nuevos is not None and p in nuevos.index:
                ax.plot(nuevos.loc[p, m], i + 1, "D", ms=7, mfc="white", mec=_TINTA, mew=1.5)
        ax.set_yticks(range(1, len(orden) + 1), orden, fontsize=8)
        ax.set_xlabel(f"{m} por partición")
        ax.set_title(m)
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    leyenda = [Line2D([], [], color=c, lw=6, alpha=0.5, label=f) for f, c in color.items()]
    leyenda.append(Line2D([], [], ls="", marker="o", ms=4, color=_GRIS, label="cada partición (CV)"))
    if nuevos is not None:
        leyenda.append(Line2D([], [], ls="", marker="D", ms=7, mfc="white", mec=_TINTA, label="asociados nuevos"))
    fig.legend(handles=leyenda, loc="outside lower center", ncols=len(leyenda), frameon=False, fontsize=9)
    fig.suptitle(f"Distribución por pipeline ({por_fold['fold'].nunique()} particiones)", fontsize=13)
    return fig


def importancia_variables(
    modelos: dict, X: pd.DataFrame, y, scoring: str = "roc_auc", n_repeats: int = 10, random_state: int = 0
) -> pd.DataFrame:
    """Qué variables aportan más a cada modelo: permutation importance sobre las variables originales.

    Desordena una columna de ``X`` a la vez y mide cuánto cae ``scoring``: si el modelo depende de esa
    variable, el puntaje cae mucho. Sirve igual para lineales, bosques y boosting (HistGB no trae
    ``feature_importances_``), y mide la variable completa (no cada columna del OneHot). Conviene usar
    datos que el modelo no vio. Formato largo: modelo, variable, importancia (caída promedio), std.
    """
    from sklearn.inspection import permutation_importance

    filas = []
    for nombre, modelo in modelos.items():
        r = permutation_importance(modelo, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state)
        filas += [
            {"modelo": nombre, "variable": v, "importancia": float(mu), "std": float(sd)}
            for v, mu, sd in zip(X.columns, r.importances_mean, r.importances_std, strict=True)
        ]
    return pd.DataFrame(filas)


def top_importancia(importancias: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Las ``top`` variables de cada modelo, una columna por modelo: "variable (caída)"."""
    columnas = {}
    for nombre, grupo in importancias.groupby("modelo", sort=False):
        mejores = grupo.nlargest(top, "importancia")
        columnas[nombre] = [f"{v} ({i:.3f})" for v, i in zip(mejores["variable"], mejores["importancia"], strict=True)]
    return pd.DataFrame(columnas, index=pd.RangeIndex(1, top + 1, name="puesto"))


def figura_importancia(
    importancias: pd.DataFrame, top: int = 10, titulos: dict | None = None, metrica: str = "AUC"
) -> Figure:
    """Un panel por modelo con sus ``top`` variables: barra = caída de la métrica al desordenarla (± std)."""
    from matplotlib.figure import Figure

    modelos = list(dict.fromkeys(importancias["modelo"]))
    columnas = min(2, len(modelos))
    filas = -(-len(modelos) // columnas)
    fig = Figure(figsize=(6.2 * columnas, 0.32 * top * filas + 1.2 * filas), layout="constrained")
    ejes = fig.subplots(filas, columnas, squeeze=False).ravel()
    for ax in ejes[len(modelos) :]:
        ax.remove()
    for ax, nombre in zip(ejes, modelos, strict=False):
        datos = importancias[importancias["modelo"] == nombre].nlargest(top, "importancia")[::-1]  # mayor arriba
        ax.barh(
            datos["variable"],
            datos["importancia"],
            xerr=datos["std"],
            color=_COLORES[0],
            height=0.6,
            error_kw={"ecolor": _GRIS, "lw": 1, "capsize": 2},
        )
        ax.axvline(0, color=_GRIS, lw=1)
        ax.set_title((titulos or {}).get(nombre, nombre))
        ax.set_xlabel(f"caída de {metrica} al desordenar la variable")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Variables que más aportan a cada modelo (permutation importance)", fontsize=13)
    return fig


def _column_transformer(estimador):
    """El ColumnTransformer dentro de un preprocesador (puede venir anidado: ColumnDropper -> columns)."""
    from sklearn.compose import ColumnTransformer

    if isinstance(estimador, ColumnTransformer):
        return estimador
    for _, paso in getattr(estimador, "steps", []):
        encontrado = _column_transformer(paso)
        if encontrado is not None:
            return encontrado
    return None


def importancia_arbol(modelo) -> pd.DataFrame:
    """``feature_importances_`` de un modelo de árboles (Random Forest), por columna ya transformada.

    Es la importancia propia del bosque (MDI): cuánto reduce, en promedio, la impureza de los árboles
    cada columna. Va por columna transformada (``cat__SEXO_F``, ``num__EDAD``...), con el nombre de su
    rama como prefijo. Ojo: favorece a las numéricas continuas, que ofrecen más cortes; la
    ``importancia_variables`` (por permutación) no tiene ese sesgo.
    """
    estimador = modelo.steps[-1][1]
    ct = _column_transformer(modelo[:-1])
    nombres = [
        f"{rama}__{columna}"
        for rama, transformador, _ in ct.transformers_
        if rama != "remainder" and transformador != "drop"
        for columna in transformador.get_feature_names_out()
    ]
    importancias = estimador.feature_importances_
    if len(nombres) != len(importancias):  # p. ej. con PCA después del ColumnTransformer
        raise ValueError("Las columnas del preprocesador no coinciden con las del modelo")
    tabla = pd.DataFrame({"variable": nombres, "importancia": importancias})
    return tabla.sort_values("importancia", ascending=False, ignore_index=True)


def figura_importancia_interactiva(importancias: pd.DataFrame, titulo: str, top: int = 20):
    """Barras de Plotly con las ``top`` columnas más importantes; al pasar el mouse muestra variable e importancia.

    Se guarda con ``fig.write_html(ruta)`` y se abre en el navegador; en un notebook basta con ``fig``.
    """
    import plotly.graph_objects as go

    datos = importancias.head(top)
    fig = go.Figure(
        go.Bar(
            x=datos["variable"],
            y=datos["importancia"],
            marker_color=_COLORES[0],
            hovertemplate="Variable=%{x}<br>Importancia=%{y:.6f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=titulo,
        xaxis_title="Variables",
        yaxis_title="Importancia",
        xaxis_tickangle=35,
        template="plotly_white",
        height=560,
        margin={"b": 200},
    )
    return fig
