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
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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


@dataclass
class Candidato:
    nombre: str
    modelo: Pipeline  # sin entrenar
    puntajes: np.ndarray  # fuera de partición, repeticiones x filas
    umbrales: dict
    metricas: dict  # cv_pr_auc (media y std entre repeticiones), cv_roc_auc...


def evaluar(nombre: str, modelo: Pipeline, X, y, folds: int = FOLDS, repeticiones: int = REPETICIONES, **kw):
    puntajes = puntajes_fuera_de_particion(modelo, X, y, folds, repeticiones)
    pr = [average_precision_score(y, p) for p in puntajes]
    roc = [roc_auc_score(y, p) for p in puntajes]
    promedio = puntajes.mean(axis=0)  # el umbral se elige sobre el promedio de las repeticiones
    umbrales = elegir_umbrales(y, promedio, **kw)
    metricas = {
        "pr_auc": float(np.mean(pr)),
        "pr_auc_std": float(np.std(pr)),
        "roc_auc": float(np.mean(roc)),
        **{k: v for k, v in umbrales.items() if v is not None},
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

    def zona(self, X: pd.DataFrame) -> list[str]:
        return [zona(p, self.umbral, self.umbral_alto) for p in a.puntaje(self.modelo, X)]


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
        "grupos": campeon.grupos,
        **campeon.info,
    }


def guardar(campeon: Campeon, carpeta: Path, extra: dict | None = None) -> dict[str, Path]:
    """``abandono_campeon.joblib`` (pipeline), ``abandono_kmeans.joblib`` y ``campeon.json`` en ``carpeta``."""
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = {
        "modelo": carpeta / ARCHIVO_MODELO,
        "grupos": carpeta / ARCHIVO_GRUPOS,
        "info": carpeta / ARCHIVO_INFO,
    }
    joblib.dump(campeon.modelo, rutas["modelo"])
    joblib.dump(campeon.agrupador, rutas["grupos"])
    texto = json.dumps({**resumen(campeon), **(extra or {})}, indent=2, ensure_ascii=False, default=str)
    rutas["info"].write_text(texto + "\n", encoding="utf-8", newline="\n")
    return rutas


def cargar(carpeta: Path):
    """(modelo, agrupador, info) desde ``carpeta``; None si falta alguno de los tres archivos."""
    rutas = [carpeta / ARCHIVO_MODELO, carpeta / ARCHIVO_GRUPOS, carpeta / ARCHIVO_INFO]
    if not all(r.exists() for r in rutas):
        return None
    return joblib.load(rutas[0]), joblib.load(rutas[1]), json.loads(rutas[2].read_text(encoding="utf-8"))


def figura_umbrales(campeon: Campeon):
    """Plotly: precisión y recall del campeón según el umbral (fuera de partición), con las tres zonas."""
    import plotly.graph_objects as go

    candidato = campeon.candidatos[campeon.nombre]
    y = campeon.y
    precision, recall, cortes = precision_recall_curve(y, candidato.puntajes.mean(axis=0))
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
