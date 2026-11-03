"""Abandono de asociados del fondo de empleados: modelos supervisados y KMeans (sin objetivo).

Supervisados: Ridge, regresión logística, Random Forest (bagging) y HistGradientBoosting (boosting).
Todos usan las mismas variables, elegidas para no hacer trampa: se excluyen las columnas que
solo existen PORQUE el asociado ya se retiró (fecha y motivo de retiro, ESTADO, fechas de
actualización que coinciden con el retiro...). Con ellas cualquier modelo acierta el 100 %
pero no sirve para anticipar un retiro.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import adjusted_rand_score, confusion_matrix, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from proyecto_ml.evaluate import compute_metrics
from proyecto_ml.features import build_preprocessor
from proyecto_ml.models import build_pipeline, get_models

if TYPE_CHECKING:
    from matplotlib.figure import Figure

TARGET = "#RETIROS"
FECHA_CORTE = pd.Timestamp("2026-10-31")

# Columnas que delatan el resultado (fuga de información): nunca entran a los modelos.
FUGA = [
    TARGET,
    "FECHARETIRO",
    "MOTIVORETIRO",
    "ESTADO",
    "ESTADO_LABORAL",
    "FECHAULTACTUALIZACION",
    "ULTIMAACTUALIZACION",
    "FECHATERMINACONTRATO",
]

# Comportamiento con el fondo en los últimos meses: la señal principal de abandono.
COMPORTAMIENTO = [
    "#SOLICITUD_DISMINUCIONCUOTAS",
    "#SOLICITUD_CANCELACIONAHORROS",
    "#SOLICITUD_RETIROPARCIALAHORROS",
    "#CANTIDAD_AHORROSALAVISTA",
    "#CANTIDAD_AHORROSCONTRACTUALES",
    "CANCELACIONCREDITOS_ULT3MESES",
    "#SOLICITUDCREDITOS_ULT3MESES",
    "#CANTIDAD_SERVICIOS",
    "#INTENCIONRETIROS_ULT3MESES",
]
NUMERICAS = [*COMPORTAMIENTO, "SALARIO", "ESTRATO", "EDAD", "ANTIGUEDAD"]
CATEGORICAS = ["SEXO", "TIPOCONTRATO", "TIPOPACTO", "PERIODODEDUCE", "TIPOCUENTA", "CARGO", "CODCIUDAD", "AGENCIA"]

# Señales que, sin mirar el objetivo, indican intención de irse (+) o de quedarse (-).
# Se usan para decidir cuál de los dos grupos de KMeans es el de "riesgo".
SENAL_RIESGO = {
    "#SOLICITUD_DISMINUCIONCUOTAS": 1,
    "#SOLICITUD_CANCELACIONAHORROS": 1,
    "#SOLICITUD_RETIROPARCIALAHORROS": 1,
    "CANCELACIONCREDITOS_ULT3MESES": 1,
    "#INTENCIONRETIROS_ULT3MESES": 1,
    "#CANTIDAD_AHORROSCONTRACTUALES": -1,
    "#SOLICITUDCREDITOS_ULT3MESES": -1,
}


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Variables del modelo: edad y antigüedad (años al corte) + columnas elegidas. Sin fuga ni PII."""
    X = pd.DataFrame(index=df.index)
    X["EDAD"] = (FECHA_CORTE - pd.to_datetime(df["FECHANACIMIENTO"])).dt.days / 365.25
    X["ANTIGUEDAD"] = (FECHA_CORTE - pd.to_datetime(df["FECHAINGRESOLABORA"])).dt.days / 365.25
    for c in NUMERICAS:
        if c not in X:
            X[c] = pd.to_numeric(df[c]).astype(float)  # Int64 con vacíos -> float con NaN (lo imputa el pipeline)
    for c in CATEGORICAS:
        # Los códigos (ciudad, agencia) son categorías, no cantidades. El vacío sigue vacío (no el texto "nan").
        X[c] = df[c].map(lambda v: np.nan if pd.isna(v) else str(v))
    return X[NUMERICAS + CATEGORICAS]


# ---------------------------------------------------------------- supervisados

TITULOS = {
    "ridge": "Ridge",
    "logreg": "Regresión logística",
    "random_forest": "Random Forest (bagging)",
    "hist_gb": "HistGradientBoosting (boosting)",
}
# Umbral de "abandona" según el puntaje que da cada modelo.
UMBRAL = {"predict_proba": 0.5, "decision_function": 0.0}


def modelo_ridge(alpha: float = 1.0) -> Pipeline:
    """Ridge para clasificación: regresión Ridge sobre la clase codificada como -1/+1.

    class_weight="balanced" compensa que los retiros son minoría. El puntaje
    (decision_function) es positivo cuando el modelo predice abandono.
    """
    return Pipeline(
        [
            ("preprocess", build_preprocessor(NUMERICAS, CATEGORICAS)),
            ("model", RidgeClassifier(alpha=alpha, class_weight="balanced")),
        ]
    )


def modelos_supervisados(alpha: float = 1.0, random_state: int = 42) -> dict[str, Pipeline]:
    """Ridge + el catálogo de models.py (logística, Random Forest, HistGB), con el mismo preprocesador.

    Random Forest es *bagging*: muchos árboles independientes, cada uno con una muestra bootstrap,
    que votan. HistGradientBoosting es *boosting*: árboles en secuencia, cada uno corrige los errores
    de los anteriores (la versión de scikit-learn inspirada en LightGBM).
    """
    catalogo = {
        nombre: build_pipeline(build_preprocessor(NUMERICAS, CATEGORICAS), modelo)
        for nombre, modelo in get_models(random_state).items()
    }
    return {"ridge": modelo_ridge(alpha), **catalogo}


def metodo_puntaje(modelo: Pipeline) -> str:
    """Probabilidad si el modelo la da; si no (Ridge), la distancia a la frontera."""
    return "predict_proba" if hasattr(modelo, "predict_proba") else "decision_function"


def puntaje(modelo: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Puntaje de abandono: más alto = más riesgo. Se compara con ``UMBRAL[metodo_puntaje(modelo)]``."""
    valores = getattr(modelo, metodo_puntaje(modelo))(X)
    return valores[:, 1] if valores.ndim == 2 else valores


def metricas_supervisado(modelo: Pipeline, df: pd.DataFrame) -> dict[str, float]:
    return compute_metrics(df[TARGET], puntaje(modelo, preparar(df)), threshold=UMBRAL[metodo_puntaje(modelo)])


def coeficientes(modelo: Pipeline, top: int = 10) -> pd.Series:
    """Variables con más peso (positivo = empuja hacia abandono)."""
    nombres = modelo.named_steps["preprocess"].get_feature_names_out()
    coef = pd.Series(np.ravel(modelo.named_steps["model"].coef_), index=nombres)  # 1D o (1, n) según versión
    return coef.reindex(coef.abs().sort_values(ascending=False).index)[:top]


# ---------------------------------------------------------------- sin objetivo: KMeans


class AgrupadorAbandono:
    """KMeans con 2 grupos sobre el comportamiento del asociado, entrenado SIN el objetivo.

    Después de agrupar, el grupo de "riesgo" se elige mirando solo los centroides: el que
    tiene más solicitudes de cancelación/disminución e intención de retiro (SENAL_RIESGO).
    El objetivo real solo se usa después, para medir qué tan bien quedó la separación.
    """

    def __init__(self, random_state: int = 42):
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),  # KMeans no admite vacíos; mediana de train
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=2, n_init=20, random_state=random_state)),
            ]
        )

    def fit(self, df: pd.DataFrame) -> AgrupadorAbandono:
        self.pipeline.fit(preparar(df)[COMPORTAMIENTO])
        centros = pd.DataFrame(self.pipeline.named_steps["kmeans"].cluster_centers_, columns=COMPORTAMIENTO)
        senal = sum(centros[c] * peso for c, peso in SENAL_RIESGO.items())
        self.grupo_riesgo_ = int(senal.idxmax())
        self.perfil_ = pd.DataFrame(
            self.pipeline.named_steps["scaler"].inverse_transform(centros), columns=COMPORTAMIENTO
        ).rename(index={self.grupo_riesgo_: "riesgo", 1 - self.grupo_riesgo_: "estable"})
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """1 = grupo de riesgo de abandono, 0 = grupo estable."""
        return (self.pipeline.predict(preparar(df)[COMPORTAMIENTO]) == self.grupo_riesgo_).astype(int)


def metricas_agrupacion(agrupador: AgrupadorAbandono, df: pd.DataFrame) -> dict[str, float]:
    """Compara los grupos con el abandono real (que el modelo nunca vio)."""
    y, grupo = df[TARGET].to_numpy(), agrupador.predict(df)
    X = agrupador.pipeline[:-1].transform(preparar(df)[COMPORTAMIENTO])  # imputado y escalado, sin KMeans
    tn, fp, fn, tp = confusion_matrix(y, grupo, labels=[0, 1]).ravel()
    return {
        "base_rate": float(y.mean()),
        "pct_en_grupo_riesgo": float(grupo.mean()),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,  # de los marcados en riesgo, cuántos se fueron
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,  # de los que se fueron, cuántos quedaron en riesgo
        "accuracy": float((tp + tn) / len(y)),
        "ari": float(adjusted_rand_score(y, grupo)),  # 0 = azar, 1 = grupos idénticos al abandono real
        "silhouette": float(silhouette_score(X, grupo)) if len(set(grupo)) > 1 else float("nan"),
    }


# ---------------------------------------------------------------- gráfico: matrices de confusión


def figura_matrices_confusion(
    supervisados: dict[str, Pipeline], agrupador: AgrupadorAbandono, df: pd.DataFrame, columnas: int = 3
) -> Figure:
    """Matriz de confusión de cada modelo sobre ``df`` (filas = real, columnas = predicción).

    Un panel por modelo supervisado (en el orden del dict) y uno final para KMeans.
    Cada celda muestra el conteo y el % de su fila: en la fila 1, el % de la columna 1 es el recall.
    """
    from matplotlib import colormaps
    from matplotlib.figure import Figure  # sin pyplot: no abre ventanas ni depende del backend

    y, X = df[TARGET].to_numpy(), preparar(df)
    reales = ["0\nse queda", "1\nabandona"]
    paneles = {TITULOS.get(nombre, nombre): (modelo.predict(X), reales) for nombre, modelo in supervisados.items()}
    paneles["KMeans (sin objetivo)"] = (agrupador.predict(df), ["0\nestable", "1\nriesgo"])
    columnas = min(columnas, len(paneles))
    filas = -(-len(paneles) // columnas)  # división hacia arriba
    fig = Figure(figsize=(4.5 * columnas, 4.4 * filas), layout="constrained")
    fig.suptitle(f"Matriz de confusión (n={len(y)})", fontsize=14)
    ejes = fig.subplots(filas, columnas, squeeze=False).ravel()
    for ax in ejes[len(paneles) :]:
        ax.remove()  # cuadrícula incompleta: sin paneles vacíos
    for ax, (titulo, (pred, etiquetas)) in zip(ejes, paneles.items(), strict=False):
        cm = confusion_matrix(y, pred, labels=[0, 1])
        ax.imshow(cm, cmap=colormaps["Blues"], vmin=0, vmax=cm.max())
        por_fila = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        for (i, j), n in np.ndenumerate(cm):
            color = "white" if n > cm.max() / 2 else "#1f2937"  # texto legible sobre celda oscura o clara
            ax.text(j, i, f"{n}\n({por_fila[i, j]:.0%})", ha="center", va="center", fontsize=12, color=color)
        ax.set_title(titulo)
        ax.set_xticks([0, 1], etiquetas)
        ax.set_yticks([0, 1], reales)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.tick_params(length=0)
        for borde in ax.spines.values():
            borde.set_visible(False)
    return fig
