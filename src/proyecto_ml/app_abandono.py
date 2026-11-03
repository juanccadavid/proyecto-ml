"""Simulador interactivo del abandono con Streamlit: ¿se retiraría este asociado?

Usa solo el modelo campeón (``make campeon``: models/abandono_campeon.joblib + campeon.json) y sus tres
zonas: **se retira** (puntaje >= umbral_alto: al menos 8 de cada 10 alertas así son retiros reales),
**zona gris** (entre el umbral de alerta y el alto) y **se queda**. Muestra también en qué grupo de KMeans
(entrenado sin el objetivo) cae el asociado y la tasa de retiro real de ese grupo.

Cada control es una variable de ``abandono.preparar``: al moverlo, la predicción se recalcula al instante.

Uso:  make simulador   (o: uv run streamlit run src/proyecto_ml/app_abandono.py)
Sin el campeón en models/ (Streamlit Community Cloud: los .joblib no van a Git), lo elige al arrancar, en
memoria, con los mismos datos de desarrollo (2.000 asociados, semilla 7) y una sola repetición.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml.config import MODELS_DIR

ETIQUETAS = {
    "EDAD": "Edad (años)",
    "ANTIGUEDAD": "Antigüedad en la empresa (años)",
    "SALARIO": "Salario mensual",
    "ESTRATO": "Estrato",
    "#SOLICITUD_DISMINUCIONCUOTAS": "Solicitudes de disminución de cuotas",
    "#SOLICITUD_CANCELACIONAHORROS": "Solicitudes de cancelación de ahorros",
    "#SOLICITUD_RETIROPARCIALAHORROS": "Solicitudes de retiro parcial de ahorros",
    "#CANTIDAD_AHORROSALAVISTA": "Ahorros a la vista",
    "#CANTIDAD_AHORROSCONTRACTUALES": "Ahorros contractuales",
    "CANCELACIONCREDITOS_ULT3MESES": "Créditos cancelados (últ. 3 meses)",
    "#SOLICITUDCREDITOS_ULT3MESES": "Créditos solicitados (últ. 3 meses)",
    "#CANTIDAD_SERVICIOS": "Servicios que usa",
    "#INTENCIONRETIROS_ULT3MESES": "Intenciones de retiro (últ. 3 meses)",
    "SEXO": "Sexo",
    "TIPOCONTRATO": "Tipo de contrato",
    "TIPOPACTO": "Tipo de pacto",
    "PERIODODEDUCE": "Periodo de deducción",
    "TIPOCUENTA": "Tipo de cuenta",
    "CARGO": "Cargo",
    "CODCIUDAD": "Código de ciudad",
    "AGENCIA": "Agencia",
}
CONTINUAS = {"EDAD": 1.0, "ANTIGUEDAD": 0.5, "SALARIO": 100_000.0}  # columna -> paso del control
PERFIL = ["SALARIO", "ESTRATO", "SEXO", "TIPOCONTRATO", "TIPOPACTO", "PERIODODEDUCE", "TIPOCUENTA"]
UBICACION = ["CARGO", "CODCIUDAD", "AGENCIA"]
ZONA = {  # emoji, tipo de mensaje, texto
    "se retira": ("😟", "error", "Se retira: el modelo está seguro. ¡Contáctalo pronto!"),
    "zona gris": ("😐", "warning", "Zona gris: hay señales de retiro, pero el modelo no está seguro. Haz seguimiento."),
    "se queda": ("😀", "success", "Se queda: este asociado probablemente sigue en el fondo."),
}

st.set_page_config(page_title="Simulador de abandono", page_icon="🏦", layout="centered")


@st.cache_resource
def cargar_campeon(carpeta: Path):
    return c.cargar(carpeta)


@st.cache_resource(show_spinner="Eligiendo el modelo campeón (una sola vez)...")
def elegir_campeon_aqui():
    """Como ``make campeon`` pero en memoria y con una repetición: (modelo, agrupador, info)."""
    df, fuente = c.datos_eleccion()
    h = a.HORIZONTE_MESES
    elegido = c.elegir(a.foto(df, h), repeticiones=1, info={"datos": fuente, "horizonte_meses": h})
    return elegido.con_zonas(), elegido.agrupador, c.resumen(elegido)


@st.cache_data
def cargar_referencia(horizonte: int | None) -> pd.DataFrame:
    """Variables de los datos del campeón, para rangos, valores iniciales y categorías."""
    df, _ = c.datos_eleccion()
    return a.preparar(a.foto(df, horizonte) if horizonte else df)


def rango(serie: pd.Series, paso: float) -> tuple[float, float]:
    """Mínimo y máximo observados, redondeados al paso y con un paso de margen a cada lado."""
    bajo = np.floor(serie.min() / paso) * paso - paso
    return max(float(bajo), 0.0), float(np.ceil(serie.max() / paso) * paso + paso)


def control(columna: str, referencia: pd.DataFrame):
    """Widget de una variable: número (continuas), slider (conteos), radio o lista (categorías)."""
    serie, etiqueta = referencia[columna], ETIQUETAS.get(columna, columna)
    if columna in a.CATEGORICAS:
        opciones = sorted(serie.dropna().unique())
        inicial = opciones.index(serie.mode().iloc[0])
        if len(opciones) <= 3:
            return st.radio(etiqueta, opciones, index=inicial, horizontal=True, key=columna)
        return st.selectbox(etiqueta, opciones, index=inicial, key=columna)
    if columna in CONTINUAS:
        paso = CONTINUAS[columna]
        bajo, alto = rango(serie, paso)
        inicial = float(np.clip(round(serie.median() / paso) * paso, bajo, alto))
        formato = "%.0f" if paso >= 1 else "%.1f"
        return st.number_input(etiqueta, bajo, alto, inicial, paso, format=formato, key=columna)
    bajo, alto = int(serie.min()), int(serie.max())
    return st.slider(etiqueta, bajo, max(alto, bajo + 1), int(serie.median()), key=columna)


def formato_puntaje(puntaje: float, metodo: str) -> str:
    return f"{puntaje:.1%}" if metodo == "predict_proba" else f"{puntaje:+.2f}"


guardado = cargar_campeon(MODELS_DIR)
elegido_aqui = guardado is None
modelo, agrupador, info = elegir_campeon_aqui() if elegido_aqui else guardado
metodo, umbral, umbral_alto = info["metodo"], info["umbral"], info["umbral_alto"]
referencia = cargar_referencia(info.get("horizonte_meses"))

st.caption("Fondo de empleados · modelo de abandono")
st.title("¿Se retiraría este asociado? 🏦")

with st.sidebar:
    st.header("Modelo campeón")
    st.markdown(f"**{info['titulo']}**")
    cv = info["metricas_cv"]
    st.caption(
        f"Elegido por PR-AUC en validación cruzada: {cv['pr_auc']:.3f} (± {cv.get('pr_auc_std', 0):.3f}). "
        f"Datos: `{info.get('datos', 'sin nombre')}` ({info['filas']} asociados, {info['tasa_retiro']:.1%} retiro), "
        f"foto a {info.get('horizonte_meses') or 0} meses."
    )
    alerta = (
        f"**Zona gris** desde {formato_puntaje(umbral, metodo)} "
        f"(encuentra al {cv['recall']:.0%} de los que se van; precisión {cv['precision']:.0%})."
    )
    if umbral_alto is None:
        st.markdown(alerta + " Ningún umbral llega a la precisión pedida: no hay zona 'se retira'.")
    else:
        st.markdown(
            alerta + f" **Se retira** desde {formato_puntaje(umbral_alto, metodo)} "
            f"(precisión {cv['precision_alto']:.0%}, recall {cv['recall_alto']:.0%})."
        )
    if elegido_aqui:
        st.info("No había campeón en `models/`: se eligió al abrir la app (como `make campeon`, una repetición).")
    if st.button("Volver a los valores iniciales"):
        for columna in a.NUMERICAS + a.CATEGORICAS:
            st.session_state.pop(columna, None)
        st.rerun()

valores: dict[str, object] = {}
izquierda, derecha = st.columns(2)
with izquierda:
    valores["EDAD"] = control("EDAD", referencia)
with derecha:
    valores["ANTIGUEDAD"] = control("ANTIGUEDAD", referencia)

st.subheader("Comportamiento con el fondo")
columnas = st.columns(2)
for i, columna in enumerate(a.COMPORTAMIENTO):
    with columnas[i % 2]:
        valores[columna] = control(columna, referencia)

st.subheader("Perfil")
columnas = st.columns(2)
for i, columna in enumerate(PERFIL):
    with columnas[i % 2]:
        valores[columna] = control(columna, referencia)
columnas = st.columns(len(UBICACION))
for col, columna in zip(columnas, UBICACION, strict=True):
    with col:
        valores[columna] = control(columna, referencia)

asociado = pd.DataFrame([valores])[a.NUMERICAS + a.CATEGORICAS]
asociado[a.NUMERICAS] = asociado[a.NUMERICAS].astype(float)  # como en preparar: los conteos son float
puntaje = float(a.puntaje(modelo, asociado)[0])
zona = modelo.predict(asociado)[0]
emoji, tipo, texto = ZONA[zona]

st.divider()
st.header(f"Riesgo de abandono: {emoji}")
st.metric(
    "Probabilidad de retiro" if metodo == "predict_proba" else "Puntaje del modelo", formato_puntaje(puntaje, metodo)
)
if metodo == "predict_proba":
    st.progress(min(max(puntaje, 0.0), 1.0))
getattr(st, tipo)(texto)

# ---- KMeans: el grupo de comportamiento (entrenado sin el objetivo)
grupo = "riesgo" if agrupador.grupos(asociado)[0] == 1 else "estable"
otro = "estable" if grupo == "riesgo" else "riesgo"
st.subheader(f"Grupo de comportamiento: {grupo}")
st.caption(
    f"KMeans agrupó a los asociados por su comportamiento sin ver quién se retiró. En el grupo **{grupo}** "
    f"({info['grupos'][grupo]['asociados']} asociados) se retiró el **{info['grupos'][grupo]['tasa_retiro']:.1%}**; "
    f"en el grupo {otro}, el {info['grupos'][otro]['tasa_retiro']:.1%}."
)
perfil = pd.DataFrame(
    {
        "Este asociado": asociado[a.COMPORTAMIENTO].iloc[0],
        f"Promedio del grupo {grupo}": pd.Series(info["grupos"][grupo]["perfil"]),
        f"Promedio del grupo {otro}": pd.Series(info["grupos"][otro]["perfil"]),
    }
).rename(index=ETIQUETAS)
st.dataframe(perfil.round(2), width="stretch")

st.subheader("¿Y si cambia una sola variable?")
variable = st.selectbox(
    "Variable", a.NUMERICAS, index=a.NUMERICAS.index("#INTENCIONRETIROS_ULT3MESES"), format_func=ETIQUETAS.get
)
if variable in CONTINUAS:
    bajo, alto = rango(referencia[variable], CONTINUAS[variable])
    malla = np.linspace(bajo, alto, 60)
else:
    malla = np.arange(int(referencia[variable].min()), int(referencia[variable].max()) + 1, dtype=float)
curva = pd.concat([asociado] * len(malla), ignore_index=True).assign(**{variable: malla})
curva["puntaje"] = a.puntaje(modelo, curva)
fig = px.line(
    curva,
    x=variable,
    y="puntaje",
    markers=variable not in CONTINUAS,
    labels={
        variable: ETIQUETAS[variable],
        "puntaje": "Probabilidad de retiro" if metodo == "predict_proba" else "Puntaje",
    },
    color_discrete_sequence=["#2a78d6"],
)
fig.add_hline(y=umbral, line_dash="dash", line_color="#6b7280", annotation_text="zona gris")
if umbral_alto is not None:
    fig.add_hline(y=umbral_alto, line_dash="dash", line_color="#6b7280", annotation_text="se retira")
fig.add_scatter(
    x=[valores[variable]],
    y=[puntaje],
    mode="markers",
    marker={"size": 12, "color": "#eb6834"},
    name="este asociado",
    showlegend=False,
)
if metodo == "predict_proba":
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
fig.update_layout(margin={"t": 20, "b": 10}, template="plotly_white")
st.plotly_chart(fig, width="stretch")
st.caption(
    "El resto de variables queda fijo en los valores de arriba. "
    "Datos sintéticos o anonimizados: no usar con información real de asociados identificables."
)
