"""Simulador interactivo del abandono con Streamlit: ¿se retiraría este asociado?

Usa solo el modelo campeón (``make campeon``: models/abandono_campeon.joblib + campeon.json) y sus tres
zonas: **se retira** (puntaje >= umbral_alto: al menos 8 de cada 10 alertas así son retiros reales),
**zona gris** (entre el umbral de alerta y el alto) y **se queda**. Muestra también en qué grupo de KMeans
(entrenado sin el objetivo) cae el asociado y la tasa de retiro real de ese grupo, y con SHAP, qué variables de
este asociado suben o bajan su puntaje. Un mapa interactivo ubica al asociado entre los segmentos de KMeans y
DBSCAN (no supervisados).

Cada control es una variable de ``abandono.preparar``: al moverlo, la predicción se recalcula al instante.

Uso:  make simulador   (o: uv run streamlit run src/proyecto_ml/app_abandono.py)
Sin el campeón en models/ (Streamlit Community Cloud: los .joblib no van a Git), lo elige al arrancar, en
memoria, con los mismos datos de desarrollo (5.000 asociados, semilla 7) y una sola repetición.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from proyecto_ml import abandono as a
from proyecto_ml import campeon as c
from proyecto_ml import explicar as e
from proyecto_ml import registro as r
from proyecto_ml import segmentacion as sg
from proyecto_ml.config import MODELS_DIR, ORIGEN_REAL, ORIGEN_SINTETICO, aviso_origen, get_settings

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
    df, fuente = c.datos_eleccion(origen=ORIGEN_SINTETICO)  # la demo pública: solo datos sintéticos
    h = a.HORIZONTE_MESES
    vista = a.foto(df, h)  # la huella va sobre la foto, como en make campeon
    info = {"datos": fuente, "horizonte_meses": h, **r.parametros_datos(vista, ORIGEN_SINTETICO), "experimental": True}
    elegido = c.elegir(vista, repeticiones=1, info=info, honestos=False)  # sin la anidada: arranca antes
    return elegido.con_zonas(), elegido.agrupador, c.resumen(elegido)


@st.cache_data
def datos_referencia(horizonte: int | None, origen: str, ruta: str | None = None) -> pd.DataFrame:
    """Los asociados del MISMO origen que el campeón: con un campeón real, el extracto de DATOS_ASOCIADOS (.env).
    Así el fondo de SHAP, los rangos y el mapa no mezclan datos sintéticos con un modelo real."""
    # ``ruta`` (DATOS_ASOCIADOS) es parte de la llave de las cachés: si cambia, se recalcula todo.
    df, _ = c.datos_eleccion(Path(ruta) if ruta else None, origen=origen)
    return a.foto(df, horizonte) if horizonte else df


@st.cache_data
def huella_referencia(horizonte: int | None, origen: str, ruta: str | None = None) -> str:
    return r.huella(datos_referencia(horizonte, origen, ruta))


@st.cache_data
def cargar_referencia(horizonte: int | None, origen: str = ORIGEN_SINTETICO, ruta: str | None = None) -> pd.DataFrame:
    """Variables de los datos del campeón, para rangos, valores iniciales y categorías."""
    return a.preparar(datos_referencia(horizonte, origen, ruta))


@st.cache_resource(show_spinner="Preparando la explicación (SHAP)...")
def explicador(_modelo, horizonte: int | None, origen: str = ORIGEN_SINTETICO, ruta: str | None = None) -> e.Explicador:
    """SHAP del campeón frente a los asociados de referencia (``_modelo``: Streamlit no lo usa como llave)."""
    return e.Explicador(_modelo, cargar_referencia(horizonte, origen, ruta))


@st.cache_resource(show_spinner="Segmentando a los asociados (KMeans y DBSCAN)...")
def segmentar(horizonte: int | None, origen: str = ORIGEN_SINTETICO, ruta: str | None = None):
    """(segmentación, datos con el retiro) de los asociados de referencia, para el mapa."""
    df = datos_referencia(horizonte, origen, ruta)
    return sg.Segmentacion().fit(df), df


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
origen = info.get("origen_datos", ORIGEN_SINTETICO)  # campeones guardados antes: todos eran sintéticos
experimental = origen == ORIGEN_SINTETICO
# Con un campeón real, la referencia es el extracto anonimizado de DATOS_ASOCIADOS (.env); si no, los sintéticos.
ruta_ref = str(get_settings().datos_asociados) if origen == ORIGEN_REAL else None
try:  # los datos de referencia deben cumplir su contrato (con datos reales, el del dataset anonimizado)
    huella_ref = huella_referencia(info.get("horizonte_meses"), origen, ruta_ref) if "datos_huella" in info else None
except c.DatosInvalidos as error:  # el mensaje ya viene sin valores personales en modo real
    st.error(f"Los datos de referencia (DATOS_ASOCIADOS del .env) no se pueden usar: {error}")
    st.stop()
if huella_ref is not None and huella_ref != info["datos_huella"]:
    # Los datos de referencia (fondo de SHAP, rangos, mapa) no son los mismos con que se eligió el campeón.
    if not experimental:
        st.error(
            "Los datos de DATOS_ASOCIADOS (.env) no son los mismos con que se eligió el campeón real (su huella no "
            "coincide). Apunta DATOS_ASOCIADOS al extracto anonimizado de make campeon, o vuelve a correrlo."
        )
        st.stop()
    st.warning("Los datos de desarrollo cambiaron desde que se eligió el campeón: corre `make campeon` de nuevo.")
referencia = cargar_referencia(info.get("horizonte_meses"), origen, ruta_ref)
etiqueta = c.etiqueta_puntaje(metodo, info.get("metricas_cv", {}))

st.caption("Fondo de empleados · modelo de abandono")
st.title("¿Se retiraría este asociado? 🏦")
if experimental:
    st.warning(
        f"**{aviso_origen(origen)}** Las cifras, las explicaciones y los segmentos muestran la regla del generador "
        "de asociados ficticios, no el comportamiento del fondo."
    )
else:
    st.info(aviso_origen(origen))
    if info.get("temporalidad_confirmada") is False:
        st.warning(
            "**Temporalidad sin confirmar:** si los conteos del extracto se midieron a la fecha de extracción y no a "
            "t0, las cifras de este modelo están infladas por fuga. Pendiente de confirmar con el fondo."
        )

with st.sidebar:
    st.header("Modelo campeón")
    st.markdown(f"**{info['titulo']}**")
    cv = info["metricas_cv"]
    st.caption(
        f"Elegido por PR-AUC en validación cruzada: {cv['pr_auc']:.3f} "
        f"(± {cv.get('pr_auc_std', 0):.3f} entre particiones"
        + (f"; IC 95 % {cv['pr_auc_ic_inf']:.2f}–{cv['pr_auc_ic_sup']:.2f}" if "pr_auc_ic_inf" in cv else "")
        + "). "
        + (
            f"Empata con {len(info['empatados'])} candidato(s): la diferencia está dentro del ruido. "
            if info.get("empatados")
            else ""
        )
        + f"Datos ({'sintéticos' if experimental else 'reales anonimizados'}): `{info.get('datos', 'sin nombre')}` "
        f"({info['filas']} asociados, {info['tasa_retiro']:.1%} retiro), "
        f"foto a {info.get('horizonte_meses') or 0} meses."
    )
    version = [f"elegido el {info['fecha'][:16].replace('T', ' ')}"] if "fecha" in info else []
    version += [f"datos `{info['datos_huella']}`"] if "datos_huella" in info else []
    if version:
        st.caption("Versión del modelo: " + " · ".join(version))
    # Precisión y recall honestos (validación anidada) si están; los de metricas_cv se midieron con las mismas
    # predicciones con que se eligió el umbral, y son optimistas.
    hon = info.get("umbrales_honestos") or {}
    rec, pre = hon.get("recall", cv["recall"]), hon.get("precision", cv["precision"])
    medida = "validación anidada" if hon else "fuera de partición, optimista"
    alerta = (
        f"**Zona gris** desde {formato_puntaje(umbral, metodo)} "
        f"(encuentra al {rec:.0%} de los que se van; precisión {pre:.0%}; {medida})."
    )
    if umbral_alto is None:
        st.markdown(alerta + " Ningún umbral llega a la precisión pedida: no hay zona 'se retira'.")
    elif umbral_alto <= umbral:  # el umbral de alerta ya alcanza la precisión pedida: toda alerta es "se retira"
        st.markdown(
            f"**Se retira** desde {formato_puntaje(umbral, metodo)} (encuentra al {rec:.0%} de los que se "
            f"van; precisión {pre:.0%}; {medida}). El umbral de alerta ya alcanza la precisión pedida, así que no "
            "hay zona gris."
        )
    else:
        st.markdown(
            alerta + f" **Se retira** desde {formato_puntaje(umbral_alto, metodo)} "
            f"(precisión {hon.get('precision_alto', cv['precision_alto']):.0%}, "
            f"recall {hon.get('recall_alto', cv['recall_alto']):.0%}; {medida})."
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
st.metric(etiqueta, formato_puntaje(puntaje, metodo))
if metodo == "predict_proba":
    st.progress(min(max(puntaje, 0.0), 1.0))
getattr(st, tipo)(texto)

# ---- SHAP: de dónde sale el puntaje de este asociado
st.subheader("¿Por qué este puntaje?")
explicacion = explicador(modelo, info.get("horizonte_meses"), origen, ruta_ref)
tabla = explicacion.explicar(asociado)
st.plotly_chart(e.figura_explicacion(tabla, explicacion.base, metodo, etiquetas=ETIQUETAS), width="stretch", key="shap")
st.caption(
    f"SHAP: cada barra es cuánto sube (rojo) o baja (azul) esa variable el puntaje de este asociado frente al "
    f"promedio de {len(explicacion.fondo)} asociados de referencia. La suma da exactamente el puntaje del modelo."
    + (" Con datos sintéticos, la explicación muestra la regla del generador, no la del fondo." if experimental else "")
)

# ---- KMeans: el grupo de comportamiento (entrenado sin el objetivo)
grupo = "riesgo" if agrupador.grupos(asociado)[0] == 1 else "estable"
otro = "estable" if grupo == "riesgo" else "riesgo"
st.subheader(f"Grupo de comportamiento: {grupo}")
st.caption(
    f"KMeans agrupó a los asociados por su comportamiento sin ver quién se retiró. En el grupo **{grupo}** "
    f"({info['grupos'][grupo]['asociados']} asociados) se retiró el **{info['grupos'][grupo]['tasa_retiro']:.1%}**; "
    f"en el grupo {otro}, el {info['grupos'][otro]['tasa_retiro']:.1%}. El grupo describe el comportamiento: "
    "no es una probabilidad de retiro (esa es la del modelo, arriba)."
)
perfil = pd.DataFrame(
    {
        "Este asociado": asociado[a.COMPORTAMIENTO].iloc[0],
        f"Promedio del grupo {grupo}": pd.Series(info["grupos"][grupo]["perfil"]),
        f"Promedio del grupo {otro}": pd.Series(info["grupos"][otro]["perfil"]),
    }
).rename(index=ETIQUETAS)
st.dataframe(perfil.round(2), width="stretch")

# ---- Mapa de segmentos (no supervisado): dónde cae este asociado
st.subheader("Mapa de segmentos")
segmentacion, datos_mapa = segmentar(info.get("horizonte_meses"), origen, ruta_ref)
atipico = segmentacion.dbscan(asociado)[0] == sg.ATIPICO
st.plotly_chart(
    sg.figura_mapa(segmentacion, datos_mapa, asociado=asociado, titulo="Asociados de referencia por comportamiento"),
    width="stretch",
    key="mapa",
)
atipicos = segmentacion.dbscan(datos_mapa) == sg.ATIPICO
tasa_atipicos = datos_mapa.loc[atipicos, a.TARGET].mean()
st.caption(
    f"La estrella es este asociado. Con el botón se cambia entre KMeans (k = {segmentacion.k_}, por silueta) y "
    f"DBSCAN, que deja aparte a los **atípicos** (✕): en los datos de referencia se retira el {tasa_atipicos:.0%} "
    f"de ellos. Según DBSCAN, este asociado **{'es atípico' if atipico else 'no es atípico'}**. Ninguno de los dos "
    "vio quién se retiró (no supervisados): agrupan solo por comportamiento."
)

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
        "puntaje": etiqueta,
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
