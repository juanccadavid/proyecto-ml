"""Simulador interactivo del modelo de abandono con Streamlit: ¿se retiraría este asociado?

Cada control es una variable de ``abandono.preparar`` (las mismas con las que se entrenó): al moverlo,
la predicción se recalcula al instante. Los rangos, valores iniciales y categorías salen de los datos
de entrenamiento (DATOS_ASOCIADOS del .env).

Uso:  make simulador   (o: uv run streamlit run src/proyecto_ml/app_abandono.py)
Usa los modelos de ``make abandono`` (models/abandono_*.joblib). Si no están, como en Streamlit Community
Cloud (los .joblib no van a Git), entrena los mismos supervisados al arrancar, en memoria.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from proyecto_ml import abandono as a
from proyecto_ml.config import MODELS_DIR, get_settings
from proyecto_ml.generador import GeneradorAsociados

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

st.set_page_config(page_title="Simulador de abandono", page_icon="🏦", layout="centered")


@st.cache_resource
def cargar_modelos(carpeta: Path) -> dict[str, object]:
    rutas = {nombre: carpeta / f"abandono_{nombre}.joblib" for nombre in a.TITULOS}
    return {nombre: joblib.load(ruta) for nombre, ruta in rutas.items() if ruta.exists()}


@st.cache_data
def cargar_asociados() -> pd.DataFrame:
    """Asociados de entrenamiento (con el objetivo). Sin archivo: asociados del generador."""
    ruta = get_settings().datos_asociados
    return a.cargar(ruta) if ruta.exists() else GeneradorAsociados(500).generar()


@st.cache_resource(show_spinner="Entrenando los modelos de abandono...")
def entrenar_modelos() -> dict[str, object]:
    """Los supervisados de ``make abandono``, entrenados con ``cargar_asociados()``. No se guardan en disco."""
    df = cargar_asociados()
    X, y = a.preparar(df), df[a.TARGET]
    return {nombre: modelo.fit(X, y) for nombre, modelo in a.modelos_supervisados().items()}


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


def veredicto(puntaje: float, metodo: str) -> tuple[str, str, str]:
    """(emoji, tipo de mensaje, texto). Con probabilidad hay zona intermedia; con Ridge solo el lado de la frontera."""
    if metodo == "predict_proba":
        if puntaje < 0.3:
            return "😀", "success", "Riesgo bajo: este asociado probablemente se queda en el fondo."
        if puntaje < a.UMBRAL[metodo]:
            return "😐", "warning", "Riesgo medio: vale la pena hacerle seguimiento antes de que lo piense."
        return "😟", "error", "Riesgo alto: el modelo predice que se retira. ¡Contáctalo pronto!"
    if puntaje < a.UMBRAL[metodo]:
        return "😀", "success", "El modelo predice que este asociado se queda en el fondo."
    return "😟", "error", "El modelo predice que este asociado se retira. ¡Contáctalo pronto!"


def formato_puntaje(puntaje: float, metodo: str) -> str:
    return f"{puntaje:.1%}" if metodo == "predict_proba" else f"{puntaje:+.2f}"


modelos = cargar_modelos(MODELS_DIR)
entrenados_aqui = not modelos
if entrenados_aqui:
    modelos = entrenar_modelos()
referencia = a.preparar(cargar_asociados())

st.caption("Fondo de empleados · modelo de abandono")
st.title("¿Se retiraría este asociado? 🏦")

with st.sidebar:
    st.header("Modelo")
    nombre = st.radio("Predecir con", list(modelos), format_func=lambda n: a.TITULOS[n], key="modelo")
    st.caption(
        f"Entrenado con `{get_settings().datos_asociados.name}` ({len(referencia)} asociados). "
        "Ridge no da probabilidad: su puntaje es la distancia a la frontera (> 0 = se retira)."
    )
    if entrenados_aqui:
        st.info("No había modelos en `models/`: se entrenaron al abrir la app (como `make abandono`).")
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
modelo = modelos[nombre]
metodo = a.metodo_puntaje(modelo)
puntaje = float(a.puntaje(modelo, asociado)[0])
emoji, tipo, texto = veredicto(puntaje, metodo)

st.divider()
st.header(f"Riesgo de abandono: {emoji}")
st.metric(
    "Probabilidad de retiro" if metodo == "predict_proba" else "Puntaje de Ridge", formato_puntaje(puntaje, metodo)
)
if metodo == "predict_proba":
    st.progress(min(max(puntaje, 0.0), 1.0))
getattr(st, tipo)(texto)

with st.expander("¿Qué dicen los demás modelos de este mismo asociado?"):
    filas = []
    for otro, m in modelos.items():
        met = a.metodo_puntaje(m)
        valor = float(a.puntaje(m, asociado)[0])
        filas.append(
            {
                "Modelo": a.TITULOS[otro],
                "Puntaje": formato_puntaje(valor, met),
                "Predicción": "se retira" if valor >= a.UMBRAL[met] else "se queda",
            }
        )
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

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
)
fig.add_hline(y=a.UMBRAL[metodo], line_dash="dash", line_color="gray", annotation_text="umbral")
fig.add_scatter(
    x=[valores[variable]], y=[puntaje], mode="markers", marker={"size": 12}, name="este asociado", showlegend=False
)
if metodo == "predict_proba":
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
fig.update_layout(margin={"t": 20, "b": 10})
st.plotly_chart(fig, width="stretch")
st.caption(
    f"El resto de variables queda fijo en los valores de arriba. Modelo: {a.TITULOS[nombre]}. "
    "Datos sintéticos o anonimizados: no usar con información real de asociados identificables."
)
