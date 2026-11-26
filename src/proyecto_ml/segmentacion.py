"""Segmentación de asociados (no supervisada): KMeans y DBSCAN sobre el comportamiento, y su mapa interactivo.

Aprendizaje **no supervisado**: los algoritmos nunca ven ``#RETIROS``; agrupan a los asociados por cómo usan el
fondo (``abandono.COMPORTAMIENTO``: solicitudes, ahorros, créditos, intención de retiro). La etiqueta sintética solo se
mira después, para validar si los segmentos sirven. Los supervisados (``abandono.modelos_supervisados``, el
campeón) aprenden directamente del retiro.

- **KMeans** parte a todos en ``k`` grupos alrededor de ``k`` centros. ``k`` se elige con la silueta (qué tan
  separado queda cada asociado de los otros grupos frente al suyo); ``tabla_k`` da también la inercia (codo).
- **DBSCAN** busca zonas densas: un grupo es un montón de asociados con al menos ``min_samples`` vecinos a
  menos de ``eps``. Quien no cae en ninguna zona densa es **atípico** (etiqueta -1). No hay que darle ``k``.
  ``eps`` sale de la curva de distancias al vecino ``min_samples`` (cuantil ``CUANTIL_EPS``).

El mapa pone a cada asociado en dos ejes que se leen directo (promedios del comportamiento estandarizado):
**señales de salida** (``SALIDA``: cancelaciones, disminuciones, retiros parciales, intención de retiro) y
**vínculo con el fondo** (``VINCULO``: ahorros, créditos, servicios), y dibuja el contorno de cada segmento.
Con PCA, dos componentes solo explican el 28 % de la variación y los segmentos se enciman.

Dos modos (``Segmentacion(modo=...)``):

- ``"comportamiento"`` (por defecto): solo los conteos de ``abandono.COMPORTAMIENTO``. Con el extracto real, casi
  todos en 0, separa sobre todo por qué productos tiene cada uno.
- ``"perfil"``: los conteos en escala logarítmica (``log1p``, para que un conteo raro no domine la distancia) más
  quién es el asociado: ``EDAD``, la antigüedad en el fondo (log), el salario (log) y si le falta el estrato. Da
  segmentos que se le pueden contar al fondo (nuevos con y sin estrato, veteranos, los que registran solicitudes
  de retiro parcial o de cancelación de ahorros; una solicitud no es una operación completada). Con variables
  mixtas la silueta queda plana para todo ``k`` y ningún índice favorece un ``k``: ``K_PERFIL`` = 5 es una decisión
  exploratoria, el menor ``k`` en el que las dos solicitudes de salida quedan en segmentos propios (estabilidad en
  docs/metodologia.md). Se puede cambiar con ``k``.

``perfil`` describe cada segmento con medianas y porcentajes que se leen sin estandarizar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from proyecto_ml import abandono as a

if TYPE_CHECKING:
    import plotly.graph_objects as go

VARIABLES = a.COMPORTAMIENTO
# Modo "perfil": los conteos (en log) más quién es el asociado.
QUIEN = ["EDAD", "ANTIGUEDAD_F_LOG", "SALARIO_LOG", "SIN_ESTRATO"]
VARIABLES_PERFIL = [*VARIABLES, *QUIEN]
MODOS = ("comportamiento", "perfil")
K_PERFIL = 5
RANGO_K = range(2, 9)
SALIDA = [v for v, peso in a.SENAL_RIESGO.items() if peso > 0]
VINCULO = [v for v in VARIABLES if v not in SALIDA]
CUANTIL_EPS = 0.9  # eps = distancia al vecino min_samples que solo supera el 10 % de los asociados
MUESTRA_SILUETA = 5000  # la silueta es cuadrática: con más asociados se mide en una muestra fija
ATIPICO = -1
# Paleta de referencia en orden fijo; el gris queda para los atípicos de DBSCAN.
_COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_GRIS, _TINTA = "#898781", "#0b0b0b"


class VariablesPerfil(TransformerMixin, BaseEstimator):
    """Salida de ``abandono.preparar`` -> ``VARIABLES_PERFIL`` (sin estado: no aprende nada).

    Los conteos y la antigüedad van en ``log1p`` y el salario en log: los tres están sesgados a la derecha y, sin
    eso, unos pocos valores altos deciden los segmentos. ``SIN_ESTRATO`` vale 1 si el estrato está en blanco, que
    en el extracto real marca un perfil incompleto (ver ``abandono.ESTRATO_SIN_DATO``). El salario vacío o
    imposible queda vacío: lo rellena el imputador del escalador.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Z = np.log1p(X[VARIABLES].astype(float))
        Z["EDAD"] = X["EDAD"].astype(float)
        Z["ANTIGUEDAD_F_LOG"] = np.log1p(X["ANTIGUEDAD_F_MESES"].astype(float))
        Z["SALARIO_LOG"] = np.log(X["SALARIO"].astype(float))
        Z["SIN_ESTRATO"] = (X["ESTRATO"] == a.ESTRATO_SIN_DATO).astype(float)
        return Z[VARIABLES_PERFIL]

    def get_feature_names_out(self, input_features=None):
        return np.array(VARIABLES_PERFIL, dtype=object)


def escalador(modo: str = "comportamiento") -> Pipeline:
    """Fila cruda o preparada -> variables del ``modo`` imputadas (mediana) y estandarizadas: la entrada de los dos."""
    if modo not in MODOS:
        raise ValueError(f"modo debe ser uno de {MODOS}, no {modo!r}")
    variables = ColumnTransformer([("c", "passthrough", VARIABLES)]) if modo == "comportamiento" else VariablesPerfil()
    return Pipeline(
        [
            ("preparar", a.PrepararVariables()),
            ("variables", variables),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def tabla_k(Z: np.ndarray, rango=RANGO_K, random_state: int = 42) -> pd.DataFrame:
    """Silueta (más alta = grupos más separados) e inercia (para el codo) de KMeans con cada ``k``.

    La silueta compara todos contra todos: con más de ``MUESTRA_SILUETA`` asociados (el extracto real) se mide en
    una muestra fija de ese tamaño; la inercia y KMeans sí usan a todos.
    """
    muestra = MUESTRA_SILUETA if len(Z) > MUESTRA_SILUETA else None
    filas = []
    for k in rango:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Z)
        silueta = silhouette_score(Z, km.labels_, sample_size=muestra, random_state=random_state)
        filas.append({"k": k, "silueta": float(silueta), "inercia": float(km.inertia_)})
    return pd.DataFrame(filas).set_index("k")


def eps_por_vecinos(Z: np.ndarray, min_samples: int, cuantil: float = CUANTIL_EPS) -> float:
    """``eps`` de DBSCAN: el cuantil de la distancia de cada asociado a su vecino número ``min_samples``.

    Los conteos de comportamiento son discretos y en el extracto real casi todos valen 0: muchos asociados tienen
    exactamente el mismo comportamiento y su distancia al vecino es 0, lo que daría ``eps = 0`` (DBSCAN no lo
    acepta). El cuantil se toma entonces sobre las distancias positivas, que son las que dicen a qué escala se
    separan los asociados distintos.
    """
    distancias = NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)[0][:, -1]
    positivas = distancias[distancias > 0]
    return float(np.quantile(positivas, cuantil)) if len(positivas) else 1e-6


class Segmentacion:
    """KMeans y DBSCAN sobre las mismas variables escaladas, más los dos ejes del mapa.

    Los segmentos de KMeans se numeran por su señal de salida (``abandono.SENAL_RIESGO`` en el centro): el
    segmento 1 es el de más solicitudes de cancelación e intención de retiro. Sin mirar la etiqueta sintética.
    ``modo``: ``"comportamiento"`` (por defecto) o ``"perfil"`` (ver el docstring del módulo).
    """

    def __init__(
        self,
        k: int | None = None,
        eps: float | None = None,
        min_samples: int | None = None,
        random_state: int = 42,
        modo: str = "comportamiento",
    ):
        self.k = k
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state
        self.modo = modo

    def fit(self, df: pd.DataFrame) -> Segmentacion:
        self.escalador_ = escalador(self.modo).fit(df)
        self.variables_ = VARIABLES if self.modo == "comportamiento" else VARIABLES_PERFIL
        Z = self.escalador_.transform(df)
        if self.k is None:  # la silueta por k se calcula siempre, para mostrarla
            self.tabla_k_ = tabla_k(Z, random_state=self.random_state)
            # Con solo conteos, k por silueta. En modo perfil la silueta queda plana: k fijo (ver K_PERFIL).
            self.k_ = int(self.tabla_k_["silueta"].idxmax()) if self.modo == "comportamiento" else K_PERFIL
        else:
            self.tabla_k_ = None
            self.k_ = self.k
        kmeans = KMeans(n_clusters=self.k_, n_init=20, random_state=self.random_state).fit(Z)
        centros = pd.DataFrame(kmeans.cluster_centers_, columns=self.variables_)
        senal = sum(centros[c] * peso for c, peso in a.SENAL_RIESGO.items())
        orden = senal.sort_values(ascending=False).index  # segmento 1 = más señales de salida
        self.renombrar_ = {int(grupo): i + 1 for i, grupo in enumerate(orden)}
        self.kmeans_ = kmeans

        self.min_samples_ = self.min_samples or 2 * len(self.variables_)  # regla común: el doble de dimensiones
        self.eps_ = self.eps or eps_por_vecinos(Z, self.min_samples_)
        self.dbscan_ = DBSCAN(eps=self.eps_, min_samples=self.min_samples_).fit(Z)
        nucleos = self.dbscan_.core_sample_indices_
        self.nucleos_ = NearestNeighbors(n_neighbors=1).fit(Z[nucleos])
        self.etiquetas_nucleos_ = self.dbscan_.labels_[nucleos]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.escalador_.transform(df)

    def kmeans(self, df: pd.DataFrame) -> np.ndarray:
        """Segmento de KMeans (1 = más señales de salida) de cada asociado."""
        return np.vectorize(self.renombrar_.get)(self.kmeans_.predict(self.transform(df)))

    def dbscan(self, df: pd.DataFrame) -> np.ndarray:
        """Grupo de DBSCAN (0, 1, ...) o -1 si es atípico. DBSCAN no predice: se asigna el grupo del asociado
        núcleo más cercano si está a menos de ``eps`` (así queda igual para los datos de entrenamiento)."""
        distancia, cercano = self.nucleos_.kneighbors(self.transform(df))
        etiqueta = self.etiquetas_nucleos_[cercano[:, 0]]
        return np.where(distancia[:, 0] <= self.eps_, etiqueta, ATIPICO)

    def plano(self, df: pd.DataFrame) -> np.ndarray:
        """Coordenadas del mapa: (señales de salida, vínculo con el fondo), en desviaciones estándar."""
        Z = self.transform(df)
        columnas = {v: i for i, v in enumerate(self.variables_)}
        return np.column_stack(
            [Z[:, [columnas[v] for v in SALIDA]].mean(axis=1), Z[:, [columnas[v] for v in VINCULO]].mean(axis=1)]
        )


def perfil(df: pd.DataFrame, segmento: np.ndarray) -> pd.DataFrame:
    """Por segmento: asociados, %, tasa del evento sintético (que el algoritmo no vio; solo con sintéticos), quién es
    (medianas de edad, antigüedad en el fondo y salario; % sin estrato y % sin ningún producto ni solicitud) y el
    comportamiento medio. Todo en sus unidades, sin estandarizar: es la tabla para contarle al fondo cada segmento."""
    X = a.preparar(df) if "FECHANACIMIENTO" in df else df
    grupos = X.groupby(segmento)
    quien = pd.DataFrame(
        {
            "edad_mediana": grupos["EDAD"].median(),
            "antiguedad_f_mediana": grupos["ANTIGUEDAD_F_MESES"].median(),
            "salario_mediano": grupos["SALARIO"].median(),
            "pct_sin_estrato": (X["ESTRATO"] == a.ESTRATO_SIN_DATO).groupby(segmento).mean(),
            "pct_sin_movimientos": (X[VARIABLES].fillna(0).sum(axis=1) == 0).groupby(segmento).mean(),
        }
    )
    tabla = pd.concat([quien, grupos[VARIABLES].mean()], axis=1)
    if a.TARGET in df:
        tabla.insert(0, "tasa_retiro", a.objetivo_binario(df).groupby(segmento).mean())
    tabla.insert(0, "pct", pd.Series(segmento).value_counts(normalize=True))
    tabla.insert(0, "asociados", pd.Series(segmento).value_counts())
    return tabla.rename_axis("segmento")


def comparar_con_retiro(df: pd.DataFrame, segmentacion: Segmentacion) -> pd.DataFrame:
    """No supervisado vs. evento futuro sintético: si marcar "en riesgo" al segmento 1 o a los atípicos de DBSCAN
    encuentra a los que se van (recall), cuántas alertas aciertan (precisión) y el ARI con el retiro."""
    y = a.objetivo_binario(df).to_numpy()
    marcas = {
        "KMeans: segmento 1": segmentacion.kmeans(df) == 1,
        "DBSCAN: atípicos": segmentacion.dbscan(df) == ATIPICO,
    }
    filas = {}
    for nombre, riesgo in marcas.items():
        tp = int((riesgo & (y == 1)).sum())
        filas[nombre] = {
            "marcados": int(riesgo.sum()),
            "tasa_retiro": float(y[riesgo].mean()) if riesgo.any() else float("nan"),
            "precision": tp / riesgo.sum() if riesgo.any() else 0.0,
            "recall": tp / max(int(y.sum()), 1),
            "ari": float(adjusted_rand_score(y, riesgo.astype(int))),
        }
    return pd.DataFrame(filas).T


def _contorno(puntos: np.ndarray, nucleo: float = 0.85) -> np.ndarray:
    """Envolvente convexa del ``nucleo`` (85 %) más central del segmento, cerrada y con un poco de holgura.

    Sin recortar, unos pocos asociados lejanos estiran el contorno sobre todo el mapa.
    """
    from scipy.spatial import ConvexHull, QhullError

    distancia = np.linalg.norm(puntos - np.median(puntos, axis=0), axis=1)
    p = np.unique(puntos[distancia <= np.quantile(distancia, nucleo)], axis=0)
    try:
        casco = p[ConvexHull(p).vertices]  # en orden alrededor del contorno
    except (QhullError, ValueError):  # menos de 3 puntos distintos o todos en línea
        return np.vstack([p, p[:1]])
    centro = casco.mean(axis=0)
    casco = centro + (casco - centro) * 1.06  # que la línea no pase encima de los puntos del borde
    return np.vstack([casco, casco[:1]])


def _nombre_segmento(metodo: str, segmento: int, k: int) -> str:
    if metodo == "DBSCAN":
        return "atípicos" if segmento == ATIPICO else f"grupo denso {segmento + 1}"
    if k == 2:
        return {1: "segmento 1 (riesgo)", 2: "segmento 2 (estable)"}[segmento]
    return f"segmento {segmento}" + (" (más señales de salida)" if segmento == 1 else "")


def figura_mapa(
    segmentacion: Segmentacion,
    df: pd.DataFrame,
    asociado: pd.DataFrame | None = None,
    titulo: str | None = None,
    detalle_individual: bool = True,
) -> go.Figure:
    """Mapa interactivo (señales de salida x vínculo) con un botón para ver KMeans o DBSCAN.

    Cada segmento es un color con su contorno; al pasar el mouse sale el segmento, su tasa del evento sintético y el
    comportamiento del asociado. Con ``asociado`` (una fila) se marca dónde cae, con una estrella.

    ``detalle_individual=False`` (datos reales): el mouse muestra solo el segmento y su tasa (agregados), no los
    conteos ni si esa persona se retiró.
    """
    import plotly.graph_objects as go

    # Los conteos dejan a muchos asociados en el mismo punto: se separan un poco solo para dibujarlos.
    plano = segmentacion.plano(df) + np.random.default_rng(0).normal(0, 0.03, (len(df), 2))
    X = a.preparar(df) if "FECHANACIMIENTO" in df else df
    retiro = a.objetivo_binario(df).to_numpy() if a.TARGET in df else None
    etiquetas = {"KMeans": segmentacion.kmeans(df), "DBSCAN": segmentacion.dbscan(df)}
    detalle = np.column_stack(
        [
            X["#SOLICITUD_RETIROPARCIALAHORROS"].fillna(-1),
            X["#SOLICITUD_CANCELACIONAHORROS"].fillna(-1),
            X["#SOLICITUD_DISMINUCIONCUOTAS"].fillna(-1),
            retiro if retiro is not None else np.full(len(X), np.nan),
        ]
    )
    fig = go.Figure()
    visibles: dict[str, list[bool]] = {"KMeans": [], "DBSCAN": []}
    for metodo, segmento in etiquetas.items():
        orden = sorted(set(segmento), key=lambda s: (s == ATIPICO, s))
        for i, s in enumerate(orden):
            en = segmento == s
            nombre = _nombre_segmento(metodo, int(s), segmentacion.k_)
            color = _GRIS if s == ATIPICO else _COLORES[i % len(_COLORES)]
            tasa = f" · se retira el {retiro[en].mean():.1%}" if retiro is not None else ""
            visible = metodo == "KMeans"
            if s != ATIPICO and en.sum() >= 3:  # contorno del segmento (los atípicos no forman grupo)
                borde = _contorno(plano[en])
                fig.add_scatter(
                    x=borde[:, 0],
                    y=borde[:, 1],
                    mode="lines",
                    line={"color": color, "width": 3, "shape": "spline", "smoothing": 0.6},
                    hoverinfo="skip",
                    showlegend=False,
                    visible=visible,
                )
                for m in visibles:
                    visibles[m].append(m == metodo)
            fig.add_scatter(
                x=plano[en, 0],
                y=plano[en, 1],
                mode="markers",
                name=f"{nombre} ({int(en.sum())}{tasa})",
                marker={
                    "size": 7 if s != ATIPICO else 8,
                    "color": color,
                    "opacity": 0.75,
                    "symbol": "x" if s == ATIPICO else "circle",
                    "line": {"color": "#fcfcfb", "width": 0.5},
                },
                customdata=detalle[en] if detalle_individual else None,
                hovertemplate=(
                    f"<b>{nombre}</b>{tasa}<br>retiros parciales: %{{customdata[0]}}"
                    "<br>cancelaciones de ahorros: %{customdata[1]}<br>disminuciones de cuotas: %{customdata[2]}"
                    + ("<br>se retiró: %{customdata[3]}" if retiro is not None else "")
                    if detalle_individual
                    else f"<b>{nombre}</b>{tasa}"
                )
                + "<extra></extra>",
                visible=visible,
            )
            for m in visibles:
                visibles[m].append(m == metodo)
    if asociado is not None:
        punto = segmentacion.plano(asociado)
        fig.add_scatter(
            x=punto[:, 0],
            y=punto[:, 1],
            mode="markers",
            name="este asociado",
            marker={"size": 20, "symbol": "star", "color": _TINTA, "line": {"color": "#fcfcfb", "width": 2}},
            hovertemplate="este asociado<extra></extra>",
        )
        for m in visibles:
            visibles[m].append(True)
    botones = [
        {
            "label": f"KMeans (k = {segmentacion.k_})",
            "method": "update",
            "args": [{"visible": visibles["KMeans"]}],
        },
        {
            "label": f"DBSCAN (eps = {segmentacion.eps_:.2f})",
            "method": "update",
            "args": [{"visible": visibles["DBSCAN"]}],
        },
    ]
    fig.update_layout(
        title=titulo or "Segmentos de asociados por comportamiento (sin ver el retiro)",
        xaxis={
            "title": "señales de salida → (cancelaciones, disminuciones, intención de retiro; en desv. estándar)",
            "zeroline": False,
            "gridcolor": "#e1e0d9",
        },
        yaxis={
            "title": "vínculo con el fondo → (ahorros, créditos, servicios)",
            "zeroline": False,
            "gridcolor": "#e1e0d9",
        },
        template="plotly_white",
        height=620,
        legend={"orientation": "h", "y": -0.18},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.12,
                "xanchor": "left",
                "buttons": botones,
                "showactive": True,
            }
        ],
        margin={"t": 110},
        font={"color": _TINTA},
    )
    return fig


def figura_k(tabla: pd.DataFrame, elegido: int) -> go.Figure:
    """Silueta según ``k`` (el elegido, marcado): por qué KMeans usa ese número de segmentos."""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(
            x=tabla.index,
            y=tabla["silueta"],
            mode="lines+markers",
            line={"color": _COLORES[0], "width": 2},
            marker={"size": 9},
            hovertemplate="k=%{x}<br>silueta=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_scatter(
        x=[elegido],
        y=[tabla.loc[elegido, "silueta"]],
        mode="markers",
        marker={"size": 16, "color": "rgba(0,0,0,0)", "line": {"color": _TINTA, "width": 2}},
        hoverinfo="skip",
        showlegend=False,
    )
    fig.update_layout(
        title=f"KMeans: silueta según el número de segmentos (elegido k = {elegido})",
        xaxis={"title": "k (número de segmentos)", "dtick": 1},
        yaxis={"title": "silueta (más alto = más separados)", "gridcolor": "#e1e0d9"},
        template="plotly_white",
        height=380,
        showlegend=False,
    )
    return fig
