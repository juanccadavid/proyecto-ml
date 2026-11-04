"""Figuras y tablas del campeón (Plotly): umbrales, matriz de confusión, ROC y calibración.

Separado de ``campeon`` (que elige el modelo y lo guarda) porque es otra responsabilidad: reportes. Lo usan
``make campeon`` (scripts/elegir_campeon.py) y las pruebas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

from proyecto_ml import campeon as c


def figura_umbrales(campeon: c.Campeon):
    """Plotly: precisión y recall del campeón según el umbral (fuera de partición), con las tres zonas."""
    if campeon.y is None:
        raise ValueError("El campeón no trae y (cargado de disco): no hay predicciones fuera de partición.")
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
        title=f"Campeón: {c.TITULOS.get(campeon.nombre, campeon.nombre)} — precisión y recall según el umbral",
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

# Paleta de referencia en orden fijo, un color por candidato (hasta 8: los de lista_candidatos y --optuna)
_COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_GRIS, _TINTA = "#898781", "#0b0b0b"


def figura_confusion(campeon: c.Campeon):
    """Plotly: un panel por umbral del campeón con el conteo y el % de su fila (en la fila 'se retira', el recall)."""
    if campeon.y is None:
        raise ValueError("El campeón no trae y (cargado de disco): no hay predicciones fuera de partición.")
    from plotly.subplots import make_subplots

    matrices = c.matrices_confusion(campeon)
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
        title=f"Campeón: {c.TITULOS.get(campeon.nombre, campeon.nombre)} — matriz de confusión "
        f"(fuera de partición, n={len(campeon.y)})<br><sup>El umbral se eligió con estas mismas predicciones: "
        "las cifras son optimistas (la validación anidada está en campeon.json, umbrales_honestos)</sup>",
        template="plotly_white",
        height=420,
        width=440 * len(matrices) + 80,
        font={"color": _TINTA},
    )
    return fig


def figura_roc(campeon: c.Campeon):
    """Plotly: curva ROC fuera de partición de cada candidato, con los umbrales del campeón marcados.

    Eje x: falsos positivos entre los que se quedan (1 − especificidad); eje y: recall. La diagonal es el azar
    (AUC 0,5). Con pocos retiros el AUC se ve alto aunque muchas alertas sean falsas: por eso el criterio es
    la PR-AUC y esta curva se lee junto con la de precisión y recall (``figura_umbrales``).
    """
    if campeon.y is None:
        raise ValueError("El campeón no trae y (cargado de disco): no hay predicciones fuera de partición.")
    import plotly.graph_objects as go

    y = campeon.y
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="azar (AUC 0,50)", line={"color": _GRIS, "width": 1, "dash": "dot"}
    )
    for i, (nombre, candidato) in enumerate(campeon.candidatos.items()):
        color = _COLORES[i % len(_COLORES)]
        fpr, tpr, cortes = roc_curve(y, candidato.promedio)
        es_campeon = nombre == campeon.nombre
        titulo = c.TITULOS.get(nombre, nombre)
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


CORTES_CALIBRACION = (0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1)  # finos cerca de 0: casi todos están ahí


def tabla_calibracion(y, puntaje, cortes=CORTES_CALIBRACION) -> pd.DataFrame:
    """Por grupo de probabilidad: asociados, probabilidad media del modelo y retiro real (sin grupos vacíos).

    Con un 2 % de retiro casi todas las probabilidades están cerca de 0: los deciles quedarían todos ahí y la
    curva tendría dos puntos. Por eso los cortes son fijos y más finos abajo.
    """
    datos = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(puntaje)})
    grupo = pd.cut(datos["p"], bins=list(cortes), include_lowest=True)
    tabla = datos.groupby(grupo, observed=True).agg(asociados=("y", "size"), predicho=("p", "mean"), real=("y", "mean"))
    return tabla[tabla["asociados"] > 0]


def figura_calibracion(campeon: c.Campeon):
    """Plotly: curva de calibración del campeón (fuera de partición). None si el puntaje no es una probabilidad.

    Compara la probabilidad media que da el modelo en cada grupo (``tabla_calibracion``) con la tasa de retiro
    real del grupo: sobre la diagonal, el "30 %" del modelo es un 30 % de verdad. Un modelo puede ordenar bien
    (PR-AUC alta) y aun así exagerar o subestimar la probabilidad; el Brier lo resume.
    """
    if campeon.y is None:
        raise ValueError("El campeón no trae y (cargado de disco): no hay predicciones fuera de partición.")
    import plotly.graph_objects as go

    if campeon.metodo != "predict_proba":
        return None
    puntaje, y = campeon.candidatos[campeon.nombre].promedio, campeon.y
    tabla = tabla_calibracion(y, puntaje)
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="calibración perfecta", line={"color": _GRIS, "width": 1, "dash": "dot"}
    )
    fig.add_scatter(
        x=tabla["predicho"],
        y=tabla["real"],
        customdata=np.column_stack([tabla["asociados"], tabla.index.astype(str)]),
        mode="lines+markers",
        name=f"{c.TITULOS.get(campeon.nombre, campeon.nombre)} (Brier {brier_score_loss(y, puntaje):.4f})",
        line={"color": _COLORES[0], "width": 2},
        # punto más grande = más asociados en el grupo: los grupos chicos (los del medio) son los más ruidosos
        marker={"size": 8 + 16 * np.sqrt(tabla["asociados"] / tabla["asociados"].max()), "line": {"color": "#fcfcfb"}},
        hovertemplate="grupo %{customdata[1]} · %{customdata[0]} asociados<br>probabilidad media del modelo=%{x:.1%}"
        "<br>retiro real del grupo=%{y:.1%}<extra></extra>",
    )
    fig.update_layout(
        title="Calibración del campeón (fuera de partición)",
        xaxis={"title": "Probabilidad que da el modelo", "tickformat": ".0%", "gridcolor": "#e1e0d9"},
        yaxis={"title": "Retiro real en ese grupo", "tickformat": ".0%", "gridcolor": "#e1e0d9"},
        template="plotly_white",
        height=560,
        width=700,
        legend={"x": 0.02, "y": 0.98},
        font={"color": _TINTA},
    )
    return fig
