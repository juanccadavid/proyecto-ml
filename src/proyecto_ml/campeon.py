"""El modelo campeón del abandono: el candidato con mejor PR-AUC en validación cruzada repetida, con su umbral.

Los afinados, Optuna y PyCaret son evidencia de experimentos (van a MLflow); el campeón es el único
"producto": lo usa el simulador y queda en ``models/abandono_campeon.joblib`` + ``models/campeon.json``.

- **Criterio:** PR-AUC promedio en validación cruzada repetida (no en los "nuevos" del generador, que
  miden si el modelo recupera la regla del generador). PR-AUC y no AUC: con pocos retiros, el AUC se ve
  alto aunque la mitad de las alertas sean falsas.
- **Umbrales**, elegidos con las predicciones fuera de partición (cada asociado puntuado por un modelo que
  no lo vio), en la escala del puntaje de cada modelo (probabilidad, o distancia a la frontera en Ridge):

  - ``umbral``: maximiza F2 (el recall pesa el doble: perder a quien se va cuesta más que una llamada de más).
  - ``umbral_alto``: el menor puntaje desde el cual al menos ``precision_min`` (0,8) de las alertas son
    retiros reales. Si ningún umbral la alcanza, no hay zona "se retira": todas las alertas son zona gris.

  Tres zonas: **se retira** (>= umbral_alto), **zona gris** (entre los dos) y **se queda** (< umbral).
- **KMeans** (sin objetivo) se entrena con los mismos datos: el simulador muestra en qué grupo cae el
  asociado y la tasa de retiro real de ese grupo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from proyecto_ml import abandono as a

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.pipeline import Pipeline

CRITERIO = "pr_auc"
FOLDS = 5
REPETICIONES = 3
BETA = 2.0  # F2
PRECISION_MIN = 0.8  # zona "se retira": 8 de cada 10 alertas son retiros reales (fuera de partición)
ARCHIVO_MODELO = "abandono_campeon.joblib"
ARCHIVO_INFO = "campeon.json"
ARCHIVO_GRUPOS = "abandono_kmeans.joblib"
ZONAS = ("se retira", "zona gris", "se queda")


def datos_eleccion(ruta: Path | None = None) -> tuple[pd.DataFrame, str]:
    """Los asociados con que se elige el campeón y de dónde salieron: ``ruta``, o el archivo de
    ``make datos-desarrollo``, o los mismos 2.000 generados en memoria (Streamlit Community Cloud)."""
    from proyecto_ml.config import PROCESSED_DIR
    from proyecto_ml.generador import NOMBRE_DESARROLLO, datos_desarrollo

    if ruta:
        return a.cargar(ruta), ruta.name
    desarrollo = PROCESSED_DIR / f"{NOMBRE_DESARROLLO}.parquet"
    if desarrollo.exists():
        return a.cargar(desarrollo), desarrollo.name
    return datos_desarrollo(), "datos_desarrollo() en memoria"


def puntajes_fuera_de_particion(modelo: Pipeline, X, y, folds: int = FOLDS, repeticiones: int = REPETICIONES):
    """Matriz (repeticiones x filas): el puntaje de cada asociado dado por un modelo que no lo vio.

    Cada repetición baraja las particiones con otra semilla. ``cross_val_predict`` clona el modelo.
    """
    metodo = a.metodo_puntaje(modelo)
    filas = []
    for r in range(repeticiones):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=r)
        p = cross_val_predict(modelo, X, y, cv=cv, method=metodo)
        filas.append(p[:, 1] if p.ndim == 2 else p)
    return np.vstack(filas)


def elegir_umbrales(y, puntaje, beta: float = BETA, precision_min: float = PRECISION_MIN) -> dict:
    """``umbral`` (máximo F-beta) y ``umbral_alto`` (precisión >= precision_min), con sus métricas."""
    precision, recall, cortes = precision_recall_curve(y, puntaje)
    precision, recall = precision[:-1], recall[:-1]  # el último punto (recall 0) no tiene corte
    fbeta = (1 + beta**2) * precision * recall / np.maximum(beta**2 * precision + recall, 1e-12)
    i = int(np.argmax(fbeta))
    resultado = {
        "umbral": float(cortes[i]),
        "fbeta": float(fbeta[i]),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "umbral_alto": None,
        "precision_alto": None,
        "recall_alto": None,
    }
    seguros = np.flatnonzero((precision >= precision_min) & (cortes >= cortes[i]))
    if len(seguros):
        j = int(seguros[0])  # cortes crece: el primero es el más bajo que cumple
        resultado |= {
            "umbral_alto": float(cortes[j]),
            "precision_alto": float(precision[j]),
            "recall_alto": float(recall[j]),
        }
    return resultado


def zona(puntaje: float, umbral: float, umbral_alto: float | None) -> str:
    """'se retira', 'zona gris' o 'se queda' según los dos umbrales del campeón."""
    if umbral_alto is not None and puntaje >= umbral_alto:
        return "se retira"
    return "zona gris" if puntaje >= umbral else "se queda"


class ModeloZonas(BaseEstimator):
    """El campeón listo para usar: su pipeline y sus dos umbrales en un solo objeto.

    ``predict(df)`` da la zona de cada asociado ("se retira", "zona gris" o "se queda") a partir de la fila cruda
    (o ya preparada). ``alerta(df)`` da 1 desde el umbral de alerta. ``predict_proba`` o ``decision_function``
    (el que tenga el modelo) dan el puntaje, así que ``abandono.puntaje`` también funciona con él. Es lo que se
    guarda en models/ y en MLflow (``models:/abandono@campeon``).
    """

    def __init__(self, modelo=None, umbral: float = 0.5, umbral_alto: float | None = None):
        self.modelo = modelo
        self.umbral = umbral
        self.umbral_alto = umbral_alto

    def fit(self, X, y):
        """Reentrena el pipeline con los mismos umbrales (normalmente llega ya entrenado)."""
        self.modelo.fit(X, y)
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return check_is_fitted(self.modelo) is None

    @property
    def metodo(self) -> str:
        return a.metodo_puntaje(self.modelo)

    def puntaje(self, X) -> np.ndarray:
        return a.puntaje(self.modelo, X)

    def predict(self, X) -> np.ndarray:
        return np.array([zona(p, self.umbral, self.umbral_alto) for p in self.puntaje(X)], dtype=object)

    def alerta(self, X) -> np.ndarray:
        return (self.puntaje(X) >= self.umbral).astype(int)

    @available_if(lambda self: hasattr(self.modelo, "predict_proba"))
    def predict_proba(self, X):
        return self.modelo.predict_proba(X)

    @available_if(lambda self: hasattr(self.modelo, "decision_function"))
    def decision_function(self, X):
        return self.modelo.decision_function(X)


@dataclass
class Candidato:
    nombre: str
    modelo: Pipeline  # sin entrenar
    puntajes: np.ndarray  # fuera de partición, repeticiones x filas
    umbrales: dict
    metricas: dict  # cv_pr_auc (media y std entre repeticiones), cv_roc_auc, mcc y kappa al umbral...

    @property
    def promedio(self) -> np.ndarray:
        """El puntaje fuera de partición de cada asociado, promedio de las repeticiones."""
        return self.puntajes.mean(axis=0)


def evaluar(nombre: str, modelo: Pipeline, X, y, folds: int = FOLDS, repeticiones: int = REPETICIONES, **kw):
    puntajes = puntajes_fuera_de_particion(modelo, X, y, folds, repeticiones)
    pr = [average_precision_score(y, p) for p in puntajes]
    roc = [roc_auc_score(y, p) for p in puntajes]
    promedio = puntajes.mean(axis=0)  # el umbral se elige sobre el promedio de las repeticiones
    umbrales = elegir_umbrales(y, promedio, **kw)
    alerta = (promedio >= umbrales["umbral"]).astype(int)
    metricas = {
        "pr_auc": float(np.mean(pr)),
        "pr_auc_std": float(np.std(pr)),
        "roc_auc": float(np.mean(roc)),
        **{k: v for k, v in umbrales.items() if v is not None},
        "mcc": float(matthews_corrcoef(y, alerta)),  # al umbral de alerta, corrigen el desbalance
        "kappa": float(cohen_kappa_score(y, alerta)),
    }
    return Candidato(nombre, modelo, puntajes, umbrales, metricas)


@dataclass
class Campeon:
    nombre: str
    modelo: Pipeline  # entrenado con todos los datos
    umbral: float
    umbral_alto: float | None
    metodo: str  # predict_proba o decision_function
    candidatos: dict[str, Candidato]
    agrupador: a.AgrupadorAbandono
    grupos: dict  # perfil y tasa de retiro real de cada grupo de KMeans
    info: dict = field(default_factory=dict)  # datos, filas, horizonte, fecha...
    y: np.ndarray | None = None  # objetivo de los datos de elección (para la figura de umbrales)

    def tabla(self) -> pd.DataFrame:
        """Candidatos x métricas de la validación cruzada, ordenados por el criterio."""
        tabla = pd.DataFrame({n: c.metricas for n, c in self.candidatos.items()}).T
        return tabla.sort_values(CRITERIO, ascending=False)

    def con_zonas(self) -> ModeloZonas:
        """El pipeline entrenado con sus dos umbrales: lo que se guarda y se registra en MLflow."""
        return ModeloZonas(self.modelo, self.umbral, self.umbral_alto)

    def zona(self, X: pd.DataFrame) -> list[str]:
        return self.con_zonas().predict(X).tolist()


def perfil_grupos(agrupador: a.AgrupadorAbandono, df: pd.DataFrame) -> dict:
    """Por grupo (riesgo/estable): tamaño, tasa de retiro real (que KMeans nunca vio) y perfil promedio."""
    grupo = pd.Series(agrupador.predict(df), index=df.index).map({1: "riesgo", 0: "estable"})
    salida = {}
    for nombre in ("riesgo", "estable"):
        en_grupo = grupo == nombre
        salida[nombre] = {
            "asociados": int(en_grupo.sum()),
            "tasa_retiro": float(df.loc[en_grupo, a.TARGET].mean()) if en_grupo.any() else float("nan"),
            "perfil": {c: float(v) for c, v in agrupador.perfil_.loc[nombre].items()},
        }
    return salida


def elegir(
    df: pd.DataFrame,
    candidatos: dict[str, Pipeline] | None = None,
    folds: int = FOLDS,
    repeticiones: int = REPETICIONES,
    beta: float = BETA,
    precision_min: float = PRECISION_MIN,
    info: dict | None = None,
) -> Campeon:
    """Evalúa los candidatos (por defecto los 4 supervisados), elige el de mejor PR-AUC y lo entrena con todo."""
    X, y = a.separar(df)  # filas crudas: el pipeline prepara las variables (PrepararVariables)
    candidatos = candidatos or a.modelos_supervisados()
    evaluados = {
        n: evaluar(n, m, X, y, folds, repeticiones, beta=beta, precision_min=precision_min)
        for n, m in candidatos.items()
    }
    nombre = max(evaluados, key=lambda n: evaluados[n].metricas[CRITERIO])
    elegido = evaluados[nombre]
    agrupador = a.AgrupadorAbandono().fit(df)
    info = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "filas": len(df),
        "tasa_retiro": float(y.mean()),
        "folds": folds,
        "repeticiones": repeticiones,
        "beta": beta,
        "precision_min": precision_min,
        **(info or {}),
    }
    return Campeon(
        nombre=nombre,
        modelo=clone(elegido.modelo).fit(X, y),
        umbral=elegido.umbrales["umbral"],
        umbral_alto=elegido.umbrales["umbral_alto"],
        metodo=a.metodo_puntaje(elegido.modelo),
        candidatos=evaluados,
        agrupador=agrupador,
        grupos=perfil_grupos(agrupador, df),
        info=info,
        y=y.to_numpy(),
    )


def resumen(campeon: Campeon) -> dict:
    """Lo que va en campeon.json: modelo, umbrales, métricas de todos los candidatos, grupos y datos."""
    return {
        "modelo": campeon.nombre,
        "titulo": a.TITULOS.get(campeon.nombre, campeon.nombre),
        "metodo": campeon.metodo,
        "umbral": campeon.umbral,
        "umbral_alto": campeon.umbral_alto,
        "criterio": CRITERIO,
        "metricas_cv": campeon.candidatos[campeon.nombre].metricas,
        "candidatos": {n: c.metricas for n, c in campeon.candidatos.items()},
        "confusion_cv": {  # fuera de partición, con cada umbral: tn, fp, fn, tp
            n: dict(zip(("tn", "fp", "fn", "tp"), map(int, m.ravel()), strict=True))
            for n, m in (matrices_confusion(campeon).items() if campeon.y is not None else ())
        },
        "grupos": campeon.grupos,
        **campeon.info,
    }


def guardar(campeon: Campeon, carpeta: Path, extra: dict | None = None) -> dict[str, Path]:
    """``abandono_campeon.joblib`` (ModeloZonas), ``abandono_kmeans.joblib`` y ``campeon.json`` en ``carpeta``."""
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {
        "modelo": carpeta / ARCHIVO_MODELO,
        "grupos": carpeta / ARCHIVO_GRUPOS,
        "info": carpeta / ARCHIVO_INFO,
    }
    joblib.dump(campeon.con_zonas(), rutas["modelo"])
    joblib.dump(campeon.agrupador, rutas["grupos"])
    texto = json.dumps({**resumen(campeon), **(extra or {})}, indent=2, ensure_ascii=False, default=str)
    rutas["info"].write_text(texto + "\n", encoding="utf-8", newline="\n")
    return rutas


def cargar(carpeta: Path):
    """(modelo, agrupador, info) desde ``carpeta``; None si falta alguno de los tres archivos.

    ``modelo`` es un ModeloZonas. Un campeón guardado antes (solo el pipeline) se envuelve con los umbrales de
    ``campeon.json``.
    """
    rutas = [carpeta / ARCHIVO_MODELO, carpeta / ARCHIVO_GRUPOS, carpeta / ARCHIVO_INFO]
    if not all(r.exists() for r in rutas):
        return None
    modelo, info = joblib.load(rutas[0]), json.loads(rutas[2].read_text(encoding="utf-8"))
    if not isinstance(modelo, ModeloZonas):
        modelo = ModeloZonas(modelo, info["umbral"], info["umbral_alto"])
    return modelo, joblib.load(rutas[1]), info


def figura_umbrales(campeon: Campeon):
    """Plotly: precisión y recall del campeón según el umbral (fuera de partición), con las tres zonas."""
    import plotly.graph_objects as go

    candidato = campeon.candidatos[campeon.nombre]
    y = campeon.y
    precision, recall, cortes = precision_recall_curve(y, candidato.promedio)
    fig = go.Figure()
    for valores, nombre, color in ((precision[:-1], "Precisión", "#2a78d6"), (recall[:-1], "Recall", "#eb6834")):
        fig.add_scatter(
            x=cortes,
            y=valores,
            name=nombre,
            mode="lines",
            line={"width": 2, "color": color},
            hovertemplate=f"{nombre}=%{{y:.2f}}<br>umbral=%{{x:.3f}}<extra></extra>",
        )
    bajo, alto = float(cortes.min()), float(cortes.max())
    tope = campeon.umbral_alto if campeon.umbral_alto is not None else alto
    bandas = [("se queda", bajo, campeon.umbral), ("zona gris", campeon.umbral, tope)]
    if campeon.umbral_alto is not None:
        bandas.append(("se retira", campeon.umbral_alto, alto))
    for (texto, x0, x1), opacidad in zip(bandas, (0.0, 0.08, 0.16), strict=False):
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="#6b7280",
            opacity=opacidad,
            line_width=0,
            annotation_text=texto,
            annotation_position="top left",
        )
    escala = "probabilidad" if campeon.metodo == "predict_proba" else "puntaje (distancia a la frontera)"
    fig.update_layout(
        title=f"Campeón: {a.TITULOS.get(campeon.nombre, campeon.nombre)} — precisión y recall según el umbral",
        xaxis_title=f"Umbral ({escala})",
        yaxis_title="",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


# ---------------------------------------------------------------- curva ROC y matriz de confusión del campeón

_COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # paleta de referencia, un color fijo por candidato
_GRIS, _TINTA = "#898781", "#0b0b0b"


def matrices_confusion(campeon: Campeon) -> dict[str, np.ndarray]:
    """Matriz de confusión del campeón (filas = real, columnas = predicción) con cada uno de sus umbrales.

    Con los puntajes fuera de partición: cada asociado lo clasificó un modelo que no lo vio, así que cuadra con
    la precisión y el recall de ``metricas_cv``. "alerta" = zona gris o se retira; "se retira" = umbral alto.
    """
    puntaje = campeon.candidatos[campeon.nombre].promedio
    cortes = {"alerta": campeon.umbral}
    if campeon.umbral_alto is not None:
        cortes["se retira"] = campeon.umbral_alto
    return {
        nombre: confusion_matrix(campeon.y, (puntaje >= corte).astype(int), labels=[0, 1])
        for nombre, corte in cortes.items()
    }


def figura_confusion(campeon: Campeon):
    """Plotly: un panel por umbral del campeón con el conteo y el % de su fila (en la fila 'se retira', el recall)."""
    from plotly.subplots import make_subplots

    matrices = matrices_confusion(campeon)
    corte = {"alerta": campeon.umbral, "se retira": campeon.umbral_alto}
    formato = ".1%" if campeon.metodo == "predict_proba" else "+.2f"
    titulos = [f"Umbral {n} ({corte[n]:{formato}})" for n in matrices]
    fig = make_subplots(rows=1, cols=len(matrices), subplot_titles=titulos, horizontal_spacing=0.12)
    reales = ["se queda (0)", "se retira (1)"]
    for k, (nombre, cm) in enumerate(matrices.items(), start=1):
        por_fila = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        predichas = ["sin alerta", "alerta"] if nombre == "alerta" else ["no 'se retira'", "se retira"]
        celdas = [
            [f"<b>{n}</b><br>{p:.1%} de la fila" for n, p in zip(fila, pf, strict=True)]
            for fila, pf in zip(cm, por_fila, strict=True)
        ]
        fig.add_heatmap(
            z=por_fila,
            x=predichas,
            y=reales,
            text=celdas,
            texttemplate="%{text}",
            customdata=cm,
            colorscale=[[0, "#cde2fb"], [1, "#104281"]],  # azul secuencial de la paleta, claro -> oscuro
            zmin=0,
            zmax=1,
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate="real: %{y}<br>predicción: %{x}<br>%{customdata} asociados (%{z:.1%} de la fila)"
            "<extra></extra>",
            row=1,
            col=k,
        )
        fig.update_xaxes(title_text="Predicción", side="bottom", row=1, col=k)
    fig.update_yaxes(title_text="Real", autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=f"Campeón: {a.TITULOS.get(campeon.nombre, campeon.nombre)} — matriz de confusión "
        f"(fuera de partición, n={len(campeon.y)})",
        template="plotly_white",
        height=420,
        width=440 * len(matrices) + 80,
        font={"color": _TINTA},
    )
    return fig


def figura_roc(campeon: Campeon):
    """Plotly: curva ROC fuera de partición de cada candidato, con los umbrales del campeón marcados.

    Eje x: falsos positivos entre los que se quedan (1 − especificidad); eje y: recall. La diagonal es el azar
    (AUC 0,5). Con pocos retiros el AUC se ve alto aunque muchas alertas sean falsas: por eso el criterio es
    la PR-AUC y esta curva se lee junto con la de precisión y recall (``figura_umbrales``).
    """
    import plotly.graph_objects as go

    y = campeon.y
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="azar (AUC 0,50)", line={"color": _GRIS, "width": 1, "dash": "dot"}
    )
    for color, (nombre, candidato) in zip(_COLORES, campeon.candidatos.items(), strict=False):
        fpr, tpr, cortes = roc_curve(y, candidato.promedio)
        es_campeon = nombre == campeon.nombre
        titulo = a.TITULOS.get(nombre, nombre)
        fig.add_scatter(
            x=fpr,
            y=tpr,
            customdata=cortes,
            mode="lines",
            name=f"{titulo}{' (campeón)' if es_campeon else ''} · AUC {roc_auc_score(y, candidato.promedio):.3f}",
            line={"color": color, "width": 3 if es_campeon else 2},
            opacity=1 if es_campeon else 0.6,
            hovertemplate=f"{titulo}<br>falsos positivos=%{{x:.1%}}<br>recall=%{{y:.1%}}<br>umbral=%{{customdata:.3f}}"
            "<extra></extra>",
        )
    puntaje = campeon.candidatos[campeon.nombre].promedio
    negativos, positivos = max(int((y == 0).sum()), 1), max(int((y == 1).sum()), 1)
    umbrales = {"alerta (zona gris)": (campeon.umbral, "circle")}
    if campeon.umbral_alto is not None:
        umbrales["se retira"] = (campeon.umbral_alto, "diamond")
    for texto, (corte, simbolo) in umbrales.items():
        alerta = puntaje >= corte
        fpr, tpr = (alerta & (y == 0)).sum() / negativos, (alerta & (y == 1)).sum() / positivos
        fig.add_scatter(
            x=[fpr],
            y=[tpr],
            mode="markers",
            name=f"umbral {texto}",
            marker={"size": 12, "symbol": simbolo, "color": _COLORES[0], "line": {"color": "#fcfcfb", "width": 2}},
            hovertemplate=f"umbral {texto} = {corte:.3f}<br>falsos positivos=%{{x:.1%}}<br>recall=%{{y:.1%}}"
            "<extra></extra>",
        )
    fig.update_layout(
        title="Curva ROC fuera de partición (cada asociado puntuado por un modelo que no lo vio)",
        xaxis={
            "title": "Falsos positivos (de los que se quedan, % con alerta)",
            "tickformat": ".0%",
            "range": [0, 1],
            "gridcolor": "#e1e0d9",
        },
        yaxis={
            "title": "Recall (de los que se retiran, % con alerta)",
            "tickformat": ".0%",
            "range": [0, 1.02],
            "gridcolor": "#e1e0d9",
        },
        template="plotly_white",
        height=620,
        width=720,
        legend={"x": 0.98, "y": 0.02, "xanchor": "right", "yanchor": "bottom"},
        font={"color": _TINTA},
    )
    return fig
